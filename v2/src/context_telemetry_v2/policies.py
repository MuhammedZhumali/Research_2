"""Causal delivery policies, v2: inputs contain public observations only.

The event summary selects measured records, not model-generated sufficient
statistics.  Presence of a current observation is used, never its value.
"""
from dataclasses import dataclass
from .model import CommandEvent, Measurement
from .wire import encode, encoded_wire_size

METHODS = ("fresh_raw", "fifo_raw", "fresh_compact", "fifo_compact",
           "snapshot_events", "fixed_groups", "greedy_groups", "events_thinned")

@dataclass(frozen=True)
class PublicBuffer:
    measurements: tuple[Measurement, ...]
    contexts: tuple[CommandEvent, ...]
    cutoff_s: float
    def __post_init__(self):
        if any(x.time_s > self.cutoff_s for x in (*self.measurements, *self.contexts)):
            raise ValueError("Future observation supplied to delivery")

def plan(method, public, received_times, received_epochs, budget_bytes, settings, sequence=0):
    if method not in METHODS:
        raise ValueError(method)
    by_epoch = {c.epoch:c for c in public.contexts}
    known_t, known_c = set(received_times), set(received_epochs)
    remaining, frames = int(budget_bytes), []
    max_records = settings["max_records_per_frame"]

    def send_records(records, compact=True, self_contained=False):
        nonlocal sequence, remaining
        pending = [m for m in records if m.time_s not in known_t]
        while pending:
            accepted, contexts = [], {}
            for m in pending[:max_records]:
                trial = dict(contexts)
                if compact and m.command_epoch in by_epoch:
                    c=by_epoch[m.command_epoch]
                    if self_contained or c.epoch not in known_c:
                        trial[c.epoch]=c
                if encoded_wire_size(accepted+[m], tuple(trial.values()), include_epoch=compact)>remaining:
                    break
                accepted.append(m)
                contexts=trial
            if not accepted:
                break
            frame=encode(sequence,public.cutoff_s,accepted,tuple(contexts.values()),include_epoch=compact)
            frames.append(frame)
            sequence+=1
            remaining-=frame.wire_bytes
            known_t.update(m.time_s for m in accepted)
            known_c.update(contexts)
            pending=pending[len(accepted):]

    candidates=[m for m in public.measurements if m.time_s not in known_t]
    if method in METHODS[:4]:
        send_records(sorted(candidates,key=lambda m:m.time_s,reverse=method.startswith("fresh")),
                     compact=method.endswith("compact"),self_contained=method.endswith("compact"))
    elif method=="events_thinned":
        step=settings["thinning_interval_s"]
        selected=[m for m in candidates if m.time_s % step==0 or m.current_a is not None]
        send_records(sorted(selected,key=lambda m:m.time_s))
    elif method=="snapshot_events":
        # V2 amendment: pack event and its representative history together.
        # No singleton snapshot frame followed by a second undersized fragment.
        # Each OFF event receives its first and latest available observation,
        # an intermediate record near the declared diagnostic span, and sparse
        # current when available. No observation values enter selection.
        selected=[]
        for c in sorted(public.contexts,key=lambda c:c.time_s):
            if c.heater_on:
                continue
            group=sorted((m for m in public.measurements if m.command_epoch==c.epoch),key=lambda m:m.time_s)
            if not group:
                continue
            current=[m for m in group if m.current_a is not None]
            summary=[group[0]]
            if current:
                summary.append(current[0])
            target=group[0].time_s+settings["summary_span_s"]
            summary.append(min(group,key=lambda m:abs(m.time_s-target)))
            latest_partner=group[-1].time_s-settings["summary_span_s"]
            summary.append(min(group,key=lambda m:abs(m.time_s-latest_partner)))
            summary.append(group[-1])
            selected.extend(dict((m.time_s,m) for m in summary).values())
        send_records(selected)
        # Summary-only: no full-history fallback. Later contacts may update
        # latest endpoints, so the persistent receiver can accumulate more than
        # five distinct records per epoch across the whole day.
    else:
        groups=[]
        for event in public.contexts:
            if event.heater_on or event.time_s+settings["group_post_s"]>public.cutoff_s:
                continue
            group=[m for m in public.measurements
                   if event.time_s-settings["group_pre_s"]<=m.time_s<=event.time_s+settings["group_post_s"]]
            if group:
                groups.append((event,group))
        while groups:
            def score(pair):
                event,group=pair
                missing=[m for m in group if m.time_s not in known_t]
                if not missing:
                    return (-1,-event.time_s)
                ctx={m.command_epoch:by_epoch[m.command_epoch] for m in missing
                     if m.command_epoch in by_epoch and m.command_epoch not in known_c}
                # Conservative approximate fragment cost, deterministic and
                # independent of temperatures, currents and hidden truth.
                cost=sum(encoded_wire_size(missing[i:i+max_records],tuple(ctx.values()) if i==0 else (),include_epoch=True)
                         for i in range(0,len(missing),max_records))
                span=max(m.time_s for m in group)-min(m.time_s for m in group)
                utility=3+int(any(m.current_a is not None for m in group))+min(span/1800,1)
                return utility/cost,-event.time_s
            chosen=min(range(len(groups)),key=lambda i:groups[i][0].time_s) if method=="fixed_groups" else max(range(len(groups)),key=lambda i:score(groups[i]))
            _,group=groups.pop(chosen)
            send_records(sorted(group,key=lambda m:m.time_s))
    assert sum(f.wire_bytes for f in frames)<=budget_bytes
    return frames
