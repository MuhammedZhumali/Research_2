"""Receiver-side synthetic erasure of already planned research frames.

The loss filter sees only current batch positions and a modelled OFF-context
marker. No signal values, target labels, future contact or future batch enter.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import floor
from time import perf_counter

from context_telemetry_v2.diagnosis import diagnose
from context_telemetry_v2.policies import PublicBuffer, plan
from context_telemetry_v2.wire import ACK_BYTES, encode_ack


def draw(seed, schedule, index, domain):
    key = f"R2-frame-loss-v1|{seed}|{schedule}|{index}|{domain}"
    return int.from_bytes(sha256(key.encode("ascii")).digest()[:8], "big")


@dataclass(frozen=True)
class ContactDraw:
    seed: int
    schedule: str
    index: int

    @property
    def exposed(self):
        return 3 * draw(self.seed, self.schedule, self.index, "opportunity") < 2**64

    def selected(self, mode, markers):
        if mode not in ("none", "random", "selective"):
            raise ValueError(mode)
        if not markers or mode == "none" or not self.exposed:
            return None
        positions = range(len(markers))
        if mode == "selective" and any(markers):
            positions = [position for position, marked in enumerate(markers) if marked]
        return min(positions, key=lambda j: (draw(self.seed,self.schedule,self.index,f"rank_{j}"),j))


def simulate(public_all, contacts, method, rate, config, settings, mode, draws):
    received, contexts = {}, {}
    attempted_times = set()
    source_times = [m.time_s for m in public_all.measurements]
    decisions, contact_rows, frame_rows = [], [], []
    sequence = 0
    for index, contact in enumerate(contacts):
        start, end = contact["start_s"], contact["end_s"]
        count = bisect_right(source_times, start)
        public = PublicBuffer(public_all.measurements[:count],
                              tuple(e for e in public_all.contexts if e.time_s <= start), start)
        reserved = settings["planner_reservation_s"]
        budget = max(0, floor((end-start-reserved)*rate/8))
        began = perf_counter()
        frames = plan(method, public, received, contexts, budget, settings, sequence)
        planner_wall = perf_counter()-began
        delay = max(reserved, planner_wall)
        budget = max(0, floor((end-start-delay)*rate/8))
        while frames and sum(frame.wire_bytes for frame in frames) > budget:
            frames.pop()
        cursor = start + delay
        markers = [any(not context.heater_on for context in frame.contexts) for frame in frames]
        selected = draws[index].selected(mode, markers)
        rows = []
        for position, frame in enumerate(frames):
            slot_start = cursor
            lost = position == selected
            cursor += frame.wire_bytes * 8 / rate  # Reserve data + ACK slot even on erasure.
            data_air_end = slot_start + len(frame.blob) * 8 / rate
            assert cursor <= end + 1e-7
            assert all(m.time_s <= start for m in frame.measurements)
            assert all(c.time_s <= start for c in frame.contexts)
            assert frame.sequence == sequence
            times = {m.time_s for m in frame.measurements}
            assert not times.intersection(received)  # previously ACKed records are never retried
            repeated = len(times.intersection(attempted_times))
            attempted_times.update(times)
            ack_count = 0
            if not lost:
                ack_count = len(encode_ack(frame.sequence, cursor))
                assert ack_count == ACK_BYTES
                received.update((m.time_s,m) for m in frame.measurements)
                contexts.update((e.epoch,e) for e in frame.contexts)
            rows.append(dict(contact_number=index+1, position=position, sequence=sequence,
                             slot_start_s=slot_start, data_end_s=data_air_end, slot_end_s=cursor,
                             marker=int(markers[position]), erased=int(lost),
                             data_frame_bytes=len(frame.blob), ack_bytes=ack_count,
                             reserved_bytes=frame.wire_bytes, wire_bytes=len(frame.blob)+ack_count,
                             observations=len(frame.measurements), retry_observations=repeated,
                             retried_frame=int(repeated>0)))
            sequence += 1
        began = perf_counter()
        certs = diagnose(tuple(received.values()),tuple(contexts.values()),config=config,decision_time_s=end)
        ground_wall = perf_counter()-began
        ground_reserve = settings["ground_diagnosis_reservation_s"]
        decisions.append(dict(time_s=end+max(ground_reserve,ground_wall),contact_number=index+1,
                              certificates=[asdict(c) for c in certs]))
        totals = dict(attempted_frames=len(rows), erased_frames=sum(r["erased"] for r in rows),
                      marker_frames=sum(markers),
                      erased_marker_frames=sum(r["erased"]*r["marker"] for r in rows),
                      erased_data_bytes=sum(r["erased"]*r["data_frame_bytes"] for r in rows))
        for column in ("wire_bytes","data_frame_bytes","ack_bytes","reserved_bytes",
                       "retry_observations","retried_frame"):
            totals[column] = sum(row[column] for row in rows)
        assert totals["reserved_bytes"] <= budget
        assert totals["reserved_bytes"] - totals["wire_bytes"] == ACK_BYTES*totals["erased_frames"]
        assert totals["erased_frames"] == int(bool(rows) and draws[index].exposed and mode != "none")
        contact_rows.append(dict(contact_number=index+1, start_s=start,end_s=end, budget_bytes=budget,
                                 opportunity=int(draws[index].exposed),active_contact=int(bool(rows)),
                                 exposed_active_contact=int(draws[index].exposed and bool(rows)),
                                 planner_wall_s=planner_wall,ground_wall_s=ground_wall,
                                 planner_overrun=int(planner_wall>reserved),
                                 ground_overrun=int(ground_wall>ground_reserve),**totals))
        frame_rows.extend(rows)
    aggregate = {key:sum(r[key] for r in contact_rows) for key in
                 ("attempted_frames","erased_frames","marker_frames","erased_marker_frames",
                  "erased_data_bytes","wire_bytes","data_frame_bytes","ack_bytes","reserved_bytes",
                  "retry_observations","retried_frame","opportunity","active_contact",
                  "exposed_active_contact","planner_wall_s","ground_wall_s","planner_overrun","ground_overrun")}
    aggregate.update(n_contacts=len(contacts),receiver_count=len(received),
                     receiver_context_count=len(contexts),airtime_s=aggregate["wire_bytes"]*8/rate)
    return decisions, contact_rows, frame_rows, aggregate


def bytes_to(frames, cutoff, rate):
    """Channel bytes, including erased forward data; idle ACK slots use no airtime."""
    consumed = 0.
    for frame in frames:
        # On success the ACK occupies the immediately following reserved slot.
        elapsed = max(0., cutoff-frame["slot_start_s"])
        consumed += min(frame["wire_bytes"], elapsed*rate/8)
    return consumed


def evaluate(ep, decisions, frames, rate, target_decisions):
    for target in ep.targets:
        timeline = target_decisions(target, decisions)
        first = next((d for d in timeline if d[1]["label"] != "insufficient"), None)
        correct = next((d for d in timeline if d[1]["label"] == target.label), None)
        cutoff = correct[0]["time_s"] if correct else target.deadline_s
        wire = bytes_to(frames, cutoff, rate)
        wrong = any(c["label"] not in (target.label,"insufficient") for _,c in timeline)
        yield dict(target_epoch=target.command_epoch, truth_label=target.label,
                   start_s=target.start_s,end_s=target.end_s,deadline_s=target.deadline_s,
                   first_label=first[1]["label"] if first else "insufficient",
                   first_time_s=first[0]["time_s"] if first else None,
                   first_correct_time_s=cutoff if correct else None,ever_wrong=int(wrong),
                   no_decision=int(first is None),resolved=int(correct is not None),
                   capped_delay_s=cutoff-target.start_s,
                   final_label=timeline[-1][1]["label"] if timeline else "insufficient",
                   wire_to_correct_bytes=wire if correct else None,
                   airtime_to_correct_s=wire*8/rate if correct else None,
                   wire_to_cutoff_bytes=wire,cutoff_s=cutoff,
                   contacts_after_target_to_correct=sum(target.start_s<=d["time_s"]<=cutoff
                                                        for d in decisions) if correct else None)
