"""Causal certificates for historical OFF outcomes, not permanent health labels.

The same detector and declared profile bounds are used by every delivery policy
and every robustness condition. No schedule, future command, truth, actual plant
parameter, fault time, or evaluation target is an input.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from .model import CommandEvent, Measurement, Label, PROFILES


@dataclass(frozen=True, slots=True)
class DiagnosisConfig:
    ambient_min_c: float
    ambient_max_c: float
    tau_min_s: float
    tau_max_s: float
    heater_rise_min_c: float
    heater_rise_max_c: float
    temperature_noise_bound_c: float
    current_off_max_abs_a: float = .05
    current_on_min_a: float = 1.
    current_on_max_a: float = 1.4
    min_pair_interval_s: float = 300.
    max_pair_interval_s: float = 3600.

    def __post_init__(self) -> None:
        if not all(math.isfinite(getattr(self, key)) for key in self.__dataclass_fields__):
            raise ValueError("Diagnostic bounds must be finite")
        if not (0 < self.tau_min_s <= self.tau_max_s and
                self.ambient_min_c <= self.ambient_max_c and
                0 <= self.heater_rise_min_c <= self.heater_rise_max_c and
                0 <= self.temperature_noise_bound_c and
                0 <= self.current_off_max_abs_a < self.current_on_min_a <= self.current_on_max_a and
                0 < self.min_pair_interval_s <= self.max_pair_interval_s):
            raise ValueError("Invalid diagnostic bounds")


def diagnosis_config(profile: str) -> DiagnosisConfig:
    p = PROFILES[profile]
    return DiagnosisConfig(8. - p["ambient_amplitude"], 12. + p["ambient_amplitude"],
                           *p["tau"], *p["rise"], p["noise"])


@dataclass(frozen=True, slots=True)
class Diagnosis:
    label: Label
    reason: str
    evidence_times_s: tuple[float, ...]
    command_epoch: int | None
    as_of_s: float


def diagnose(measurements: Sequence[Measurement], contexts: Sequence[CommandEvent], *,
             config: DiagnosisConfig, decision_time_s: float | None = None) -> tuple[Diagnosis, ...]:
    """Return earliest causal certificates per observed epoch and label.

    A raw low-current certificate has no epoch and says that the actuator was
    OFF at its acquisition time. Evaluator attribution is never fed back here.
    Every raw low-current time is retained because unseen target boundaries must
    not be used for thinning. Thermal certificates require a delivered OFF event
    and matching opaque identifiers on both measurements: mode constancy must
    be evidenced, not assumed from a missing event or an epoch's numeric value.

    Opposing certificates under model mismatch are retained for error auditing.
    ``latest_outcomes`` reports their combined outcome as insufficient.
    """
    if any(type(m) is not Measurement for m in measurements):
        raise TypeError("Only public Measurement records are accepted")
    if any(type(c) is not CommandEvent for c in contexts):
        raise TypeError("Only delivered public CommandEvent records are accepted")
    if any(not math.isfinite(m.time_s) or m.time_s < 0 or not math.isfinite(m.temperature_c)
           or (m.current_a is not None and not math.isfinite(m.current_a)) for m in measurements):
        raise ValueError("Invalid measurement")
    if any(not math.isfinite(c.time_s) or c.time_s < 0 for c in contexts):
        raise ValueError("Invalid context time")
    if decision_time_s is None:
        decision_time_s = max((m.time_s for m in measurements), default=0.)
    if not math.isfinite(decision_time_s) or decision_time_s < 0:
        raise ValueError("Invalid decision time")
    available = sorted((m for m in measurements if m.time_s <= decision_time_s), key=lambda m: m.time_s)
    by_epoch: dict[int, CommandEvent] = {}
    conflicts: set[int] = set()
    for event in contexts:
        if event.time_s <= decision_time_s:
            if event.epoch in by_epoch and by_epoch[event.epoch] != event:
                conflicts.add(event.epoch)
            by_epoch[event.epoch] = event
    for epoch in conflicts:
        by_epoch.pop(epoch, None)
    certificates: dict[tuple, Diagnosis] = {}

    def add(label: Label, reason: str, evidence: tuple[float, ...], epoch: int | None) -> None:
        result = Diagnosis(label, reason, evidence, epoch, max(evidence))
        key = (epoch, label) if epoch is not None else (None, label, result.as_of_s)
        previous = certificates.get(key)
        if previous is None or (result.as_of_s, evidence) < (previous.as_of_s, previous.evidence_times_s):
            certificates[key] = result

    grouped: dict[int, list[Measurement]] = {}
    for m in available:
        epoch = m.command_epoch if m.command_epoch >= 0 else None
        event = by_epoch.get(m.command_epoch)
        if event is not None and event.time_s > m.time_s:
            event = None
        if m.current_a is not None:
            if abs(m.current_a) <= config.current_off_max_abs_a:
                add("nominal", "observed_low_current", (m.time_s,), epoch)
            elif (event is not None and not event.heater_on and
                  config.current_on_min_a <= m.current_a <= config.current_on_max_a):
                add("fault", "high_current_during_observed_OFF", (m.time_s,), epoch)
        if event is not None and not event.heater_on:
            grouped.setdefault(m.command_epoch, []).append(m)
    for epoch, points in grouped.items():
        if len(points) < 2:
            continue
        times = np.array([m.time_s for m in points], dtype=float)
        temp = np.array([m.temperature_c for m in points], dtype=float)
        first, second = np.triu_indices(len(points), k=1)
        dt = times[second] - times[first]
        keep = (dt >= config.min_pair_interval_s) & (dt <= config.max_pair_interval_s)
        first, second, dt = first[keep], second[keep], dt[keep]
        if not len(first):
            continue
        decay_a, decay_b = np.exp(-dt / config.tau_min_s), np.exp(-dt / config.tau_max_s)
        initial_low = temp[first] - config.temperature_noise_bound_c
        initial_high = temp[first] + config.temperature_noise_bound_c
        observed_low = temp[second] - config.temperature_noise_bound_c
        observed_high = temp[second] + config.temperature_noise_bound_c

        def compatible(on: bool) -> np.ndarray:
            eq_low = config.ambient_min_c + (config.heater_rise_min_c if on else 0.)
            eq_high = config.ambient_max_c + (config.heater_rise_max_c if on else 0.)
            lower = np.minimum(eq_low + (initial_low - eq_low) * decay_a,
                               eq_low + (initial_low - eq_low) * decay_b)
            upper = np.maximum(eq_high + (initial_high - eq_high) * decay_a,
                               eq_high + (initial_high - eq_high) * decay_b)
            # Allow roundoff in serialized float32 values and exponential code.
            return (observed_low <= upper + 1e-5) & (observed_high >= lower - 1e-5)

        off_possible, on_possible = compatible(False), compatible(True)
        for label, mask in (("nominal", off_possible & ~on_possible), ("fault", on_possible & ~off_possible)):
            indices = np.flatnonzero(mask)
            if len(indices):
                selected = min(indices, key=lambda i: (times[second[i]], times[first[i]]))
                add(label, "same_epoch_thermal_" + label,
                    (float(times[first[selected]]), float(times[second[selected]])), epoch)
    return tuple(sorted(certificates.values(), key=lambda c: (c.as_of_s, c.command_epoch or -1, c.label, c.evidence_times_s)))


def latest_outcomes(certificates: Sequence[Diagnosis]) -> dict[int, Label]:
    """Historical per-epoch display; no carry-forward to a new unseen epoch."""
    labels: dict[int, set[str]] = {}
    for result in certificates:
        if result.command_epoch is not None:
            labels.setdefault(result.command_epoch, set()).add(result.label)
    return {epoch: next(iter(values)) if len(values) == 1 else "insufficient"
            for epoch, values in labels.items()}
