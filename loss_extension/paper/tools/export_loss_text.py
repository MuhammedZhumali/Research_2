"""Generate the frame-loss manuscript passages from the archived analyses."""
from export_tables import BASE, num, rows, write
import json


def main():
    summary, thinned, fresh = "snapshot_events", "events_thinned", "fresh_compact"
    rates={(r["loss_mode"], r["method"]): r for r in rows("primary_rates")}
    pairs={(r["loss_mode"],r["candidate"],r["comparator"]): r for r in rows("primary_contrasts")}
    def interval(arm, method, comparator):
        r=pairs[arm,method,comparator]
        return f'{num(r["difference_pp"])} [{num(r["family_adjusted_low_pp"])}, {num(r["family_adjusted_high_pp"])}]'
    def rate(arm, method):
        return num(rates[arm,method]["fault_detection_pct"])
    exposed = json.loads((BASE/"data/independent-check.json").read_text())["primary"]
    summary_loss = next(r for r in exposed if r["method"]==summary and r["loss_mode"]=="selective")
    periodic = {r["method"]: num(r["fault_detection_pct"]) for r in rows("all_cells")
                if (r["schedule_id"], r["rate_bps"], r["loss_mode"])==("synthetic_90min","32","random")}
    write("loss-abstract.tex",
        'A separate simulation at 32~bit/s on the orbital schedule erased at most one frame per contact, '
        'chosen uniformly or preferentially from frames carrying OFF context. Summaries detected more '
        'faults within six hours than thinned history under both rules. Their advantage over the latest '
        'contextualized readings remained uncertain.')
    write("loss-results.tex",
        r'Of 480 episodes, 240 had late faults and included 357 failed OFF commands. '
        rf'Summary minus thinned-history detection was {interval("random",summary,thinned)}~pp under uniform '
        rf'and {interval("selective",summary,thinned)}~pp under selective erasure (adjusted paired intervals). '
        rf'Summary minus fresh compact was {interval("random",summary,fresh)} and {interval("selective",summary,fresh)}~pp. '
        rf'Thinned minus fresh was {interval("random",thinned,fresh)} and {interval("selective",thinned,fresh)}~pp. '
        r'No healthy episode raised a false alarm; missed faults remained unresolved.'
        '\n\n'
        rf'All {summary_loss["singleton_erasure_contacts"]} affected summary batches contained one frame. The '
        rf'two rules therefore erased the same frames, accounting for the identical {rate("random",summary)}\% '
        rf'estimates. Fresh compact had lower capped delay but transmitted roughly nine times as many bytes. '
        rf'On periodic contacts at 32~bit/s under uniform erasure, thinned history resolved {periodic[thinned]}\% '
        rf'of faults, fresh compact {periodic[fresh]}\% and summaries {periodic[summary]}\%. This reversal is descriptive.')
    write("loss-conclusion.tex",
          'Under the uniform and selective erasure rules on the orbital schedule at 32~bit/s, summaries '
          'detected more faults in time than thinned history at lower traffic. '
          'The paired comparisons with fresh contextualized measurements were inconclusive. '
          'On the periodic schedule under uniform erasure, thinned history had the highest observed detection rate.')


if __name__=="__main__":
    main()

