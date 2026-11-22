"""Export revision evidence tables and statements from the paper's compact inputs."""
import argparse
import csv
from pathlib import Path
from shutil import copyfile

from export_tables import BASE, rows, table, write

STRESS_FIELDS = ("profile", "condition", "schedule_id", "rate_bps", "stress", "n_targets",
                 "resolved_fraction_mean", "ever_wrong_fraction_mean", "unknown_fraction_mean",
                 "capped_delay_s_mean", "wire_bytes_mean")


def refresh(pilot):
    data = BASE / "data"
    copyfile(pilot/"revision_2026/attribution/outputs/outcomes.csv", data/"deadline_outcomes.csv")
    copyfile(pilot/"v2/tables/full_history.csv", data/"stress_detector.csv")
    outputs = pilot / "revision_2026/experiments/outputs"
    for source, local in (("aggregate", "composition_cells"), ("contrasts", "composition_contrasts")):
        copyfile(outputs / (source + ".csv"), data / (local + ".csv"))
    with (pilot/"v2/tables/robustness_32.csv").open(newline="", encoding="utf-8") as source:
        summary = [row for row in csv.DictReader(source) if row["method"] == "snapshot_events"]
    with (data/"stress_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, STRESS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)


def attribution():
    by_truth = {row["truth"]:row for row in rows("deadline_outcomes")}
    metrics = (
        ("OFF targets", "targets"), ("Correct within six hours", "timely"),
        ("Any incorrect decision", "ever_wrong"), ("No decision", "no_decision"),
        (r"\quad No contact overlaps the horizon", "no_window_intersects_horizon"),
        (r"\quad Contact misses usable evidence", "window_but_no_timed_opportunity"),
        (r"\quad Resolved only by contact history", "contact_reference_only"),
        (r"\quad Continuous history insufficient", "insufficient_continuous_evidence"))
    nfmt = lambda x: f"{int(x):,}".replace(",",r"\,")
    body = [[label, *(nfmt(by_truth[t][k]) for t in ("fault", "nominal", "all"))]
            for label, k in metrics]
    table("attribution", "Validation outcomes at 32 bit/s under calculated KazSTSAT windows. "
          "Indented rows partition all missed outcomes at the six-hour deadline.",
          "attribution", ["Outcome", "Failed OFF", "Nominal OFF", "All"], body, "lrrr")
    a = by_truth["all"]
    pct = 100 * int(a["timely"]) / int(a["targets"])
    write("attribution.tex",
          rf"Table~\ref{{tab:attribution}} explains why only {pct:.2f}\% of historical OFF outcomes "
          rf"were resolved. Of {nfmt(a['no_decision'])} missed deadlines, only {a['contact_reference_only']} "
          "were recoverable by sending unlimited history at the same contacts. The timing category "
          "has a contact overlapping the horizon but no contact starting after evidence was acquired "
          "and ending with time for diagnosis. Three nominal OFF intervals ended between sparse "
          "current samples and their thermal pairs fit both hypotheses. This post hoc partition "
          "attributes failure to the earliest limiting stage of the fixed model.")


def previous_stress():
    history = {r["condition"]:r for r in rows("stress_detector") if r["profile"]=="slow_sparse_current"}
    summary = {r["condition"]:r for r in rows("stress_summary")}
    labels = (("base","Baseline"),("current_bias","Current bias +0.15 A"),
              ("temperature_gaussian","Gaussian noise"),("thermal_mismatch",r"Plant $\tau\times1.35$"))
    body=[]
    for key,label in labels:
        n=int(history[key]["n_targets"])
        hcount=lambda k: str(round(n*float(history[key][k])))
        scount=lambda k: str(round(n*float(summary[key][k])))
        body.append([label, hcount("final_correct_fraction"), hcount("ever_wrong_fraction"),
                     scount("resolved_fraction_mean"), scount("unknown_fraction_mean")])
    table("stress", "Earlier one-factor stresses, 120 OFF targets per row with sparse current. "
          "$^a$Continuous full history. $^b$Summaries use orbital windows at 32 bit/s.",
          "stress", ["Condition","Final correct$^a$","Ever wrong$^a$","Timely$^b$","Unknown$^b$"],
          body, "lrrrr")


def wire():
    body = [["Acquisition time / temperature / flags","8 / 4 / 1"],
            ["Interval ID / optional current","4 / 4"],
            ["Command time / ON--OFF / interval ID","8 / 1 / 4"],
            ["Header / record counts / CRC","20 / 4 / 4"],
            ["Nonce / tag (modeled)","12 / 16"],
            ["Complete ACK","56"]]
    table("wire","Serialized fields. Each data frame incurs 56 forward overhead bytes. "
          "Every received frame also incurs a 56-byte ACK.", "wire",["Field","Bytes"],body,"lr")

def composition():
    data = {(r["profile"], int(r["lag_s"]), int(r["rate_bps"]), r["method"]):r
            for r in rows("composition_cells")}
    contrasts = {(r["method"], r["reference"], r["metric"]):r for r in rows("composition_contrasts")
                 if r["comparison"]=="composition" and r["profile"]=="all"
                 and r["lag_s"]=="0" and r["rate_bps"]=="32"}
    methods = {"endpoints":"Endpoints", "endpoints_current":"Endpoints + current",
               "full_summary":"Full summary"}
    pct = contrasts["full_summary","endpoints_current","correct_pct"]
    extra_bytes = float(contrasts["full_summary","endpoints_current","bytes_day"]["difference"])
    body=[]
    for lag in (0,120):
        for method,label in methods.items():
            r = data["all",lag,32,method]
            body.append([str(lag),label,r["correct"],r["ever_wrong"],r["no_decision"],
                         f'{float(r["capped_delay_h"]):.2f}',f'{float(r["bytes_day"]):.0f}'])
    table("ablation","Summary composition on the same 1080 historical OFF commands at 32 bit/s. "
          "Correct, wrong and no-decision counts refer to the six-hour horizon. Delay is capped at six hours.",
          "ablation",["Lag (s)","Records","Correct","Ever wrong","None","Delay (h)","B/day"],body,"rlrrrrr")
    t1 = data["thermal_only",0,32,"endpoints_current"]; t2 = data["thermal_only",0,32,"full_summary"]
    continuous = data["all",120,32,"continuous_history"]; contact = data["all",120,32,"contact_history"]
    lagged = data["all",120,32,"full_summary"]
    gain = int(t2["correct"]) - int(t1["correct"])
    current_gain = contrasts["endpoints_current","endpoints","correct_pct"]
    body = (r"With zero lag, adding the first available current-bearing record improved resolution by "
            rf'{float(current_gain["difference"]):.2f}~pp. Adding both thermal-offset selection rules gained '
            rf'{float(pct["difference"]):.2f}~pp [{float(pct["lo95"]):.2f}, {float(pct["hi95"]):.2f}] '
            rf'for {extra_bytes:.0f} extra bytes/day (Table~\ref{{tab:ablation}}). '
            rf'All {gain} net additional resolutions came from the temperature-only profile '
            rf'({t1["correct"]} to {t2["correct"]} of {t2["targets"]}). Both current-equipped profiles '
            r'matched on resolution and mean capped delay. The comparison tests the two offsets jointly. '
            r'Intervals are descriptive pointwise 95\% paired '
            r'percentile intervals from 5000 whole-episode resamples within profile and fault-assignment strata. '
            rf'At 4~bit/s the three rules resolved '
            + ", ".join(data["all",0,4,m]["correct"] for m in methods)
            + rf' targets in table order. At 8~bit/s their counts matched those at 32~bit/s.'
            + "\n\n"
            + rf'With a 120~s sensor lag, the summary resolved {lagged["correct"]} commands in time and '
            rf'full history at the same contacts resolved {contact["correct"]}. The detector encountered '
            rf'conflicting evidence for {lagged["conflict"]} commands with summaries and {contact["conflict"]} '
            r'with full history. Omitting one of two contradictory readings can leave an apparently '
            r'unambiguous diagnosis. The lower conflict count alone cannot establish greater diagnostic '
            r'reliability. With continuous history the same detector issued a wrong decision for '
            rf'{continuous["ever_wrong"]} commands, reached a correct decision at least once for '
            rf'{continuous["correct"]}, and ended with a correct decision for {continuous["final_correct"]}.')
    write("ablation-results.tex",body)
    write("composition-abstract.tex", 'In the zero-lag test at 32~bit/s, extra temperature pairs '
          'improved timely resolution only when current measurements were unavailable.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,help="refresh compact inputs from a full research tree")
    args=parser.parse_args()
    if args.pilot:
        refresh(args.pilot)
    wire()
    attribution()
    previous_stress()
    composition()


if __name__=="__main__":
    main()
