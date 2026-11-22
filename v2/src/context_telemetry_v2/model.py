"""Synthetic v2 plant; observable records are separated from evaluator truth.

No numerical plant parameter in this module is a measured KazSTSAT parameter.
Local ON/OFF events are not ground telecommands withheld from the receiver.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import bisect
import math
import random
from typing import Literal

Label = Literal["nominal", "fault", "insufficient"]
PROFILES = {
    "clear_current": dict(tau=(900., 1500.), rise=(20., 32.), ambient_amplitude=1., noise=.15, current_interval=300.),
    "slow_sparse_current": dict(tau=(2400., 3600.), rise=(8., 14.), ambient_amplitude=2., noise=.3, current_interval=3600.),
    "thermal_only": dict(tau=(1200., 2400.), rise=(18., 30.), ambient_amplitude=1., noise=.2, current_interval=None),
}
CONDITIONS = ("base", "thermal_mismatch", "temperature_gaussian", "current_bias")


@dataclass(frozen=True, slots=True)
class ThermalParams:
    tau_s: float
    heater_rise_c: float
    ambient_mean_c: float
    ambient_amplitude_c: float
    ambient_period_s: float
    ambient_phase_rad: float
    current_on_a: float
    temperature_noise_bound_c: float
    current_noise_bound_a: float = .04
    current_bias_a: float = 0.
    temperature_noise_distribution: str = "uniform"

    def ambient_at(self, time_s: float) -> float:
        return self.ambient_mean_c + self.ambient_amplitude_c * math.sin(
            2 * math.pi * time_s / self.ambient_period_s + self.ambient_phase_rad)


@dataclass(frozen=True, slots=True)
class CommandEvent:
    time_s: float
    heater_on: bool
    epoch: int


@dataclass(frozen=True, slots=True)
class Measurement:
    time_s: float
    temperature_c: float
    current_a: float | None
    command_epoch: int


@dataclass(frozen=True, slots=True)
class PublicBuffer:
    measurements: tuple[Measurement, ...]
    contexts: tuple[CommandEvent, ...]
    cutoff_s: float


@dataclass(frozen=True, slots=True)
class TruthSample:
    """Evaluator only: never accepted by diagnostic/transport functions."""
    time_s: float
    temperature_c: float
    ambient_c: float
    commanded_on: bool
    actuator_on: bool
    current_a: float
    fault_active: bool
    command_epoch: int


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    """One actual OFF transition; these boundaries are evaluator-only."""
    command_epoch: int
    start_s: float
    end_s: float
    deadline_s: float
    label: Label


@dataclass(frozen=True, slots=True)
class ThermalEpisode:
    seed: int
    profile: str
    condition: str
    duration_s: float
    params: ThermalParams
    commands: tuple[CommandEvent, ...]
    measurements: tuple[Measurement, ...]
    truth: tuple[TruthSample, ...]
    fault_start_s: float | None
    targets: tuple[EvaluationTarget, ...]

    def public_prefix(self, cutoff_s: float) -> PublicBuffer:
        if not math.isfinite(cutoff_s) or cutoff_s < 0 or cutoff_s > self.duration_s:
            raise ValueError("Public cutoff outside the observed episode")
        return PublicBuffer(tuple(x for x in self.measurements if x.time_s <= cutoff_s),
                            tuple(x for x in self.commands if x.time_s <= cutoff_s), cutoff_s)

    def truth_at(self, time_s: float) -> TruthSample:
        if not math.isfinite(time_s) or not 0 <= time_s <= self.duration_s:
            raise ValueError("Truth query outside the episode")
        index = bisect.bisect_right([x.time_s for x in self.truth], time_s) - 1
        return self.truth[index]


def generate_episode(
    seed: int, *, profile: str = "slow_sparse_current", condition: str = "base",
    faulty: bool = False, duration_s: float = 86400., sample_interval_s: float = 60.,
    integration_step_s: float = 5.,
) -> ThermalEpisode:
    """Generate three unknown local OFF transitions and delayed persistent failure.

    OFF starts are sampled in [3600,7200], [28800,32400], [54000,57600] s.
    OFF dwell is uniform 1800..18000 s, shorter at its minimum than start jitter:
    unlike a 3--5 hour dwell, this does not create a clock-only guaranteed OFF
    interval. All event times are snapped to the 60 s acquisition grid. The
    detector receives neither these windows nor the selected actual times.
    Separate RNG streams keep future schedules/fault selection out of noise.
    """
    if profile not in PROFILES or condition not in CONDITIONS:
        raise ValueError("Unknown profile or condition")
    if duration_s != 86400.:
        raise ValueError("Registered v2 episodes have a 24 h duration")
    if sample_interval_s != 60. or not 0 < integration_step_s <= 60:
        raise ValueError("Registered acquisition interval is 60 s; integration must be positive")
    cfg = PROFILES[profile]
    parameter_rng = random.Random(seed ^ 0xB19C77)
    schedule_rng = random.Random(seed ^ 0x39CA1)
    epoch_rng = random.Random(seed ^ 0x7E27AD)
    fault_rng = random.Random(seed ^ 0x19FA01)
    temperature_rng = random.Random(seed ^ 0x5EED17)
    current_rng = random.Random(seed ^ 0x4C0FFE)
    params = ThermalParams(
        parameter_rng.uniform(*cfg["tau"]), parameter_rng.uniform(*cfg["rise"]),
        parameter_rng.uniform(8., 12.), cfg["ambient_amplitude"], 5400.,
        parameter_rng.uniform(0., 2 * math.pi), parameter_rng.uniform(1.1, 1.3), cfg["noise"],
    )
    if condition == "thermal_mismatch":
        params = replace(params, tau_s=params.tau_s * 1.35)
    elif condition == "temperature_gaussian":
        params = replace(params, temperature_noise_distribution="gaussian_equal_variance")
    elif condition == "current_bias":
        params = replace(params, current_bias_a=.15)
    initial_temperature = params.ambient_at(0.) + parameter_rng.uniform(0., 8.)
    used_epochs: set[int] = set()

    def opaque_epoch() -> int:
        while True:
            candidate = epoch_rng.randrange(1, 2 ** 32)
            if candidate not in used_epochs:
                used_epochs.add(candidate)
                return candidate

    events = [CommandEvent(0., True, opaque_epoch())]
    off_events = []
    for earliest, latest in ((3600, 7200), (28800, 32400), (54000, 57600)):
        start = float(schedule_rng.randrange(earliest // 60, latest // 60 + 1) * 60)
        dwell = float(schedule_rng.randrange(1800 // 60, 18000 // 60 + 1) * 60)
        event = CommandEvent(start, False, opaque_epoch())
        off_events.append(event)
        events.extend((event, CommandEvent(start + dwell, True, opaque_epoch())))
    events = tuple(events)
    fault_start = off_events[fault_rng.choice((1, 2))].time_s if faulty else None
    targets = tuple(EvaluationTarget(
        event.epoch, event.time_s, events[events.index(event) + 1].time_s,
        event.time_s + 21600.,
        "fault" if fault_start is not None and event.time_s >= fault_start else "nominal",
    ) for event in off_events)
    measurements, truth = [], []
    temperature, current_time, event_index = initial_temperature, 0., 0
    current_interval = cfg["current_interval"]
    current_every = int(current_interval / sample_interval_s) if current_interval is not None else None
    for sample_index in range(int(duration_s / sample_interval_s) + 1):
        sample_time = sample_index * sample_interval_s
        while current_time < sample_time - 1e-9:
            while event_index + 1 < len(events) and events[event_index + 1].time_s <= current_time + 1e-9:
                event_index += 1
            step_end = min(sample_time, current_time + integration_step_s)
            if event_index + 1 < len(events):
                step_end = min(step_end, events[event_index + 1].time_s)
            stuck = fault_start is not None and current_time >= fault_start
            actuator_on = events[event_index].heater_on or stuck
            equilibrium = params.ambient_at((current_time + step_end) / 2) + params.heater_rise_c * actuator_on
            temperature = equilibrium + (temperature - equilibrium) * math.exp(-(step_end - current_time) / params.tau_s)
            current_time = step_end
        while event_index + 1 < len(events) and events[event_index + 1].time_s <= sample_time + 1e-9:
            event_index += 1
        event = events[event_index]
        stuck = fault_start is not None and sample_time >= fault_start
        actuator_on = event.heater_on or stuck
        true_current = params.current_on_a if actuator_on else 0.
        if params.temperature_noise_distribution == "uniform":
            noise = temperature_rng.uniform(-params.temperature_noise_bound_c, params.temperature_noise_bound_c)
        else:
            noise = temperature_rng.gauss(0., params.temperature_noise_bound_c / math.sqrt(3.))
        current_noise = current_rng.uniform(-params.current_noise_bound_a, params.current_noise_bound_a)
        observed_current = true_current + params.current_bias_a + current_noise
        measurements.append(Measurement(sample_time, temperature + noise,
            observed_current if current_every is not None and sample_index % current_every == 0 else None, event.epoch))
        truth.append(TruthSample(sample_time, temperature, params.ambient_at(sample_time),
            event.heater_on, actuator_on, true_current, stuck, event.epoch))
    return ThermalEpisode(seed, profile, condition, duration_s, params, events,
                          tuple(measurements), tuple(truth), fault_start, targets)
