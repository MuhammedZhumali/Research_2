"""Offline geometric KazSTSAT visibility from frozen public elements.

These are modeled visibility windows, not actual ground-station bookings or
measured RF availability. Run from any directory with the pilot's Python venv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import skyfield
from skyfield.api import EarthSatellite, load, wgs84

ROOT = Path(__file__).resolve().parents[1]
ORBIT = ROOT / "data" / "orbit"
SCHEDULE_ID = "kazstsat_astana_10deg"
LATITUDE_DEG, LONGITUDE_DEG, ELEVATION_M = 51.1694, 71.4491, 350.0
THRESHOLD_DEG, DAYS, ROOT_TOLERANCE_S = 10.0, 7, 0.001
UTC = timezone.utc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tle_checksum(line: str) -> bool:
    return len(line) == 69 and sum(int(c) if c.isdigit() else int(c == "-") for c in line[:68]) % 10 == int(line[68])


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    return {"count": int(len(a)), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "contacts")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Output must remain inside R2_ContextTelemetry_Pilot")
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ORBIT / "source_manifest.json").read_text(encoding="utf-8"))
    source_checks = {}
    for source in manifest["sources"]:
        if source["status"] == "retrieved":
            path = ROOT / source["path"]
            source_checks[source["id"]] = sha256(path) == source["sha256"]
    assert source_checks and all(source_checks.values()), "Frozen input hash mismatch"
    omm = json.loads((ORBIT / "celestrak_43783_omm.json").read_text(encoding="utf-8"))[0]
    sn = json.loads((ORBIT / "satnogs_43783_tle.json").read_text(encoding="utf-8"))[0]
    c_lines = (ORBIT / "celestrak_43783.tle").read_text(encoding="ascii").splitlines()
    assert (omm["OBJECT_NAME"], omm["OBJECT_ID"], omm["NORAD_CAT_ID"]) == ("KAZSTSAT", "2018-099AB", 43783)
    assert sn["norad_cat_id"] == 43783 and sn["tle0"].removeprefix("0 ").strip() == "KAZSTSAT"
    assert c_lines[0].strip() == "KAZSTSAT"
    checksum_checks = {"celestrak_line1": tle_checksum(c_lines[1]), "celestrak_line2": tle_checksum(c_lines[2]),
                       "satnogs_line1": tle_checksum(sn["tle1"]), "satnogs_line2": tle_checksum(sn["tle2"])}
    assert all(checksum_checks.values()), "Invalid TLE checksum"
    ts = load.timescale(builtin=True)  # No download and no mutable external Earth-orientation data.
    satellites = {
        "celestrak_omm": EarthSatellite.from_omm(ts, omm),
        "celestrak_tle": EarthSatellite(c_lines[1], c_lines[2], c_lines[0].strip(), ts),
        "satnogs_tle": EarthSatellite(sn["tle1"], sn["tle2"], "KAZSTSAT", ts),
    }
    for sat in satellites.values():
        assert sat.model.satnum == 43783 and sat.model.intldesg == "18099AB"
    epoch = datetime.fromisoformat(omm["EPOCH"]).replace(tzinfo=UTC)
    origin = epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    end = origin + timedelta(days=DAYS)
    horizon_s = float(DAYS * 86400)
    t0, t1 = ts.from_datetime(origin), ts.from_datetime(end)
    station = wgs84.latlon(LATITUDE_DEG, LONGITUDE_DEG, elevation_m=ELEVATION_M)

    def calc(sat):
        difference = sat - station

        def altitude(seconds):
            return difference.at(ts.utc(origin.year, origin.month, origin.day, second=seconds)).altaz()[0].degrees

        def crossing(estimate, rising):
            lo, hi = max(0.0, estimate - 5.0), min(horizon_s, estimate + 5.0)
            a, b = float(altitude(lo)) - THRESHOLD_DEG, float(altitude(hi)) - THRESHOLD_DEG
            assert a * b <= 0, f"Crossing was not bracketed at {estimate}"
            assert (a <= 0 <= b) if rising else (b <= 0 <= a)
            while hi - lo > ROOT_TOLERANCE_S:
                mid = (lo + hi) / 2
                above = float(altitude(mid)) >= THRESHOLD_DEG
                if above == rising:
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) / 2

        def maximum(lo, hi):
            ratio = (math.sqrt(5) - 1) / 2
            x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
            y1, y2 = float(altitude(x1)), float(altitude(x2))
            while hi - lo > ROOT_TOLERANCE_S:
                if y1 < y2:
                    lo, x1, y1 = x1, x2, y2
                    x2 = lo + ratio * (hi - lo)
                    y2 = float(altitude(x2))
                else:
                    hi, x2, y2 = x2, x1, y1
                    x1 = hi - ratio * (hi - lo)
                    y1 = float(altitude(x1))
            return max(y1, y2)

        times, events = sat.find_events(station, t0, t1, altitude_degrees=THRESHOLD_DEG)
        raw_events = [(float((dt - origin).total_seconds()), int(ev)) for dt, ev in zip(times.utc_datetime(), events)]
        windows, active_start = [], (0.0 if float(altitude(0.0)) >= THRESHOLD_DEG else None)
        for seconds, event in raw_events:
            if event == 0:
                assert active_start is None
                active_start = crossing(seconds, True)
            elif event == 2:
                assert active_start is not None
                finish = crossing(seconds, False)
                windows.append((active_start, finish, maximum(active_start, finish)))
                active_start = None
        if active_start is not None:
            windows.append((active_start, horizon_s, maximum(active_start, horizon_s)))
        assert windows
        for i, (start, finish, peak) in enumerate(windows):
            assert 0 <= start < finish <= horizon_s and THRESHOLD_DEG <= peak <= 90
            if i:
                assert windows[i-1][1] < start
        return windows, altitude, raw_events

    windows, altitude, events = calc(satellites["celestrak_omm"])
    alternatives = {}
    for key in ("celestrak_tle", "satnogs_tle"):
        other, _, _ = calc(satellites[key])
        assert len(other) == len(windows)
        deltas = np.abs(np.asarray(windows) - np.asarray(other))
        alternatives[key] = {"count": len(other), "max_start_delta_s": float(deltas[:, 0].max()),
                             "max_end_delta_s": float(deltas[:, 1].max()), "max_peak_delta_deg": float(deltas[:, 2].max()),
                             "epoch_utc": iso(satellites[key].epoch.utc_datetime()),
                             "epoch_delta_from_omm_s": float((satellites[key].epoch.utc_datetime() - epoch).total_seconds())}
        assert max(alternatives[key]["max_start_delta_s"], alternatives[key]["max_end_delta_s"]) < 1.0

    # Independent dense-grid membership check, separate from find_events.
    grid = np.arange(0.0, horizon_s + 1, 10.0)
    grid_elev = np.asarray(altitude(grid))
    expected_above = np.zeros(len(grid), dtype=bool)
    for start, finish, _ in windows:
        expected_above |= (grid >= start) & (grid <= finish)
    membership_errors = int(np.count_nonzero(expected_above != (grid_elev >= THRESHOLD_DEG)))
    assert membership_errors == 0
    for start, finish, peak in windows:
        local_grid = grid_elev[(grid >= start) & (grid <= finish)]
        assert not len(local_grid) or peak + 0.001 >= float(local_grid.max())
    boundary_error_deg = []
    for start, finish, _ in windows:
        for seconds in (start, finish):
            if 0 < seconds < horizon_s:
                boundary_error_deg.append(abs(float(altitude(seconds)) - THRESHOLD_DEG))
        if start >= 0.5:
            assert float(altitude(start - 0.5)) < THRESHOLD_DEG < float(altitude(start + 0.5))
        if finish <= horizon_s - 0.5:
            assert float(altitude(finish - 0.5)) > THRESHOLD_DEG > float(altitude(finish + 0.5))
    assert float(np.max(boundary_error_deg)) < 0.001
    # Check SGP4 error flags over the full grid, rather than trusting contact rows alone.
    grid_times = ts.utc(origin.year, origin.month, origin.day, second=grid)
    propagation = satellites["celestrak_omm"].at(grid_times)
    propagation_errors = sorted({str(x) for x in np.asarray(propagation.message, dtype=object).ravel() if x is not None})
    assert not propagation_errors

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["schedule_id", "contact_id", "start_s", "end_s", "start_utc", "end_utc", "max_elevation_deg"], lineterminator="\n")
    writer.writeheader()
    for i, (start, finish, peak) in enumerate(windows, 1):
        writer.writerow({"schedule_id": SCHEDULE_ID, "contact_id": f"KAZSTSAT_{i:03d}",
                         "start_s": f"{start:.6f}", "end_s": f"{finish:.6f}",
                         "start_utc": iso(origin + timedelta(seconds=start)), "end_utc": iso(origin + timedelta(seconds=finish)),
                         "max_elevation_deg": f"{peak:.6f}"})
    csv_path = out / "kazstsat.csv"
    csv_path.write_bytes(buffer.getvalue().encode("utf-8"))
    durations = [b-a for a, b, _ in windows]
    gaps = [windows[i+1][0] - windows[i][1] for i in range(len(windows)-1)]
    counts = {str((origin + timedelta(days=d)).date()): sum(int(a // 86400) == d for a, _, _ in windows) for d in range(DAYS)}
    statistics = {"window_count": len(windows), "duration_s": describe(durations), "inter_window_gap_s": describe(gaps),
                  "start_to_start_s": describe([windows[i+1][0]-windows[i][0] for i in range(len(windows)-1)]),
                  "maximum_elevation_deg": describe([c for _, _, c in windows]), "windows_starting_per_utc_day": counts,
                  "total_visible_s": float(sum(durations)), "visible_fraction": float(sum(durations) / horizon_s),
                  "initial_horizon_gap_s": windows[0][0], "terminal_horizon_gap_s": horizon_s-windows[-1][1]}
    data_dir = Path(skyfield.__file__).parent / "data"
    package_data_hashes = {p.name: sha256(p) for p in sorted(data_dir.glob("*.npz"))}
    metadata = {
        "schedule_id": SCHEDULE_ID, "data_kind": "calculated_geometric_visibility_not_observed_RF_contacts",
        "satellite": {"name": "KAZSTSAT", "norad_cat_id": 43783, "cospar_id": "2018-099AB"},
        "primary_elements": "data/orbit/celestrak_43783_omm.json", "epoch_utc": iso(epoch),
        "saved_orbital_fields": {"inclination_deg": omm["INCLINATION"], "eccentricity": omm["ECCENTRICITY"],
                                 "mean_motion_rev_per_day": omm["MEAN_MOTION"], "derived_period_minutes": 1440.0 / omm["MEAN_MOTION"]},
        "origin_utc": iso(origin), "horizon_end_utc_exclusive": iso(end), "horizon_s": horizon_s,
        "horizon_rule": "Seven UTC days starting at 00:00 on the saved element epoch's UTC date, not retrieval/current date",
        "station_assumptions": {"name": "model Astana station (not surveyed antenna coordinates)", "latitude_deg": LATITUDE_DEG,
                                "longitude_deg": LONGITUDE_DEG, "height_m": ELEVATION_M, "height_reference": "WGS84 ellipsoid",
                                "minimum_geometric_elevation_deg": THRESHOLD_DEG, "refraction": False,
                                "terrain_or_antenna_mask": False, "weather_or_link_budget": False},
        "model": {"propagation": "SGP4 via Skyfield EarthSatellite.from_omm", "geodetic_station": "WGS84",
                  "timescale": "Skyfield packaged builtin timescale, no downloaded EOP; no polar-motion table installed",
                  "time_data_sha256": package_data_hashes, "crossing_refinement_tolerance_s": ROOT_TOLERANCE_S,
                  "peak_search": "golden-section within each short LEO pass; assumes a single elevation maximum"},
        "versions": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "skyfield", "sgp4", "jplephem")}},
        "source_manifest": "data/orbit/source_manifest.json", "output_csv_sha256": sha256(csv_path), "statistics": statistics,
        "generator_script": "scripts/build_contacts.py", "generator_script_sha256": sha256(Path(__file__)),
        "interpretation": ["Orbit identification/epoch were cross-checked; CelesTrak and SatNOGS can share the underlying Space-Track GP source, so agreement is not independent orbit truth.",
                           "No RF rate, key policy, telemetry generation rate, detector, payload power, buffer size, or actual availability is inferred for KazSTSAT.",
                           "SSTL describes a technology-demonstration ground station in Astana, but the coordinates, altitude, elevation mask, and operation in this pilot are explicit assumptions.",
                           "Numerical refinement precision is not physical orbit accuracy; TLE/OMM propagation does not establish measured satellite access."]}
    checks = {"passed": True, "frozen_source_sha256": source_checks, "tle_checksums": checksum_checks,
              "identity_matches": True, "epoch_matches_between_sources": all(abs(x["epoch_delta_from_omm_s"]) < 0.001 for x in alternatives.values()),
              "source_comparison": alternatives, "dense_grid_spacing_s": 10.0, "dense_grid_points": len(grid),
              "dense_grid_membership_errors": membership_errors, "sgp4_errors_on_dense_grid": propagation_errors,
              "peaks_not_below_dense_grid": True,
              "max_contact_boundary_error_deg": float(max(boundary_error_deg)), "ordered_nonoverlapping_within_horizon": True,
              "csv_rows": len(windows), "unclipped_rise_set_checks_0_5s": True,
              "physical_validation": "Not performed; checks establish numerical consistency only"}
    (out / "kazstsat_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "kazstsat_quality_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "epoch_utc": iso(epoch), "origin_utc": iso(origin), "statistics": statistics, "checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
