"""Generate the paper's compact tables from archived CSVs (Python standard library)."""
import csv
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def rows(name):
    with (BASE/"data"/(name+".csv")).open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))
def write(name,text):
    (BASE/name).write_text(text.strip()+"\n",encoding="utf-8")
def table(name,caption,label,headers,body,spec):
    return write("tables/"+name+".tex","\n".join([
        r"\begin{table}[!htb]\centering\small",r"\caption{"+caption+r"}\label{tab:"+label+"}",
        r"\begin{tabular}{@{}"+spec+r"@{}}\toprule"," & ".join(headers)+r"\\\midrule",
        *[" & ".join(r)+r"\\" for r in body],r"\bottomrule\end{tabular}\end{table}"]))
def num(x,d=2):
    return f"{(0 if abs(float(x))<.5*10**-d else float(x)):.{d}f}"

methods={"fresh_raw":"Fresh raw","fresh_compact":"Fresh compact","fifo_raw":"FIFO raw","fifo_compact":"FIFO compact","fixed_groups":"Fixed groups","greedy_groups":"Greedy groups","events_thinned":"Thinned history","snapshot_events":"Event summaries"}
profiles={"clear_current":"Frequent","slow_sparse_current":"Sparse","thermal_only":"Temperature"}
def main():
    body=[]
    for r in rows("plant_parameters"):
        rg=lambda a,b: num(r[a],0)+"--"+num(r[b],0)
        body.append([profiles[r["profile"]],rg("tau_min_s","tau_max_s"),rg("rise_min_c","rise_max_c"),num(r["ambient_amplitude_c"],1),num(r["temperature_noise_bound_c"]),num(r["current_interval_s"],0) if r["current_interval_s"] else "---"])
    table("plant","Assumed synthetic profiles. Temperatures are in degrees Celsius. The temperature-only profile has no current channel.","plant",["Profile",r"$\tau$ (s)","Rise",r"Ambient $a$",r"Noise $\epsilon$","Current period (s)"],body,"lrrrrr")
    rmap={r["method"]:r for r in rows("kazstsat_32_mixture")}
    body=[]
    for key,m in methods.items():
        r=rmap[key]; body.append([m,num(r["resolved_targets"],0),num(float(r["capped_delay_s_mean"])/3600),num(r["wire_bytes_mean"],0)])
    body.append(["Contact full history",str(round(float(r["oracle_resolved_fraction_mean"])*int(r["n_targets"]))),num(float(r["oracle_capped_delay_s_mean"])/3600),"Unlimited"])
    table("exploratory","Lossless exploratory KazSTSAT windows at 32 bit/s. All 360 OFF outcomes enter the six-hour capped delay. Daily bytes include framing, context and ACKs.","explore",["Representation","Correct / 360","Delay (h)","Bytes/day"],body,"lrrr")
    primary=rows("primary_summary")
    body=[]
    for schedule in ("kazstsat_astana_10deg","synthetic_90min"):
        for rate in ("8","32"):
            cells=[r for r in primary if r["schedule_id"]==schedule and r["rate_bps"]==rate]
            assert len(cells)==3
            def bounds(metric,scale,digits):
                lo=min(float(r[metric+"_ci_low"]) for r in cells)
                hi=max(float(r[metric+"_ci_high"]) for r in cells)
                return "["+num(scale*lo,digits)+", "+num(scale*hi,digits)+"]"
            body.append(["KazSTSAT" if schedule.startswith("kaz") else "Periodic",rate,
                         bounds("coverage_loss",100,2),bounds("delay_gap_s",1,1)])
    table("validation","Outer limits of the three profile-specific simultaneous intervals at each setting "
          "(2000 paired episodes per profile). Positive values favor contact history. Error-risk intervals "
          "are [$-0.40$, 0.40] pp throughout.","validation",
          ["Schedule","bit/s","Resolution loss (pp)","Delay excess (s)"],body,"lrrr")
    arm={"none":"No loss","random":"Uniform","selective":"Selective"}
    body=[]
    for r in rows("primary_rates"):
        body.append([arm[r["loss_mode"]],methods[r["method"]],r["correct_fault_targets"],num(r["fault_detection_pct"]),num(float(r["fault_capped_delay_s"])/3600),num(r["daily_wire_bytes"],0)])
    table("loss","New KazSTSAT-window experiment at 32 bit/s. The raw number of correct fault decisions is out of 357 failed transitions. The primary percentage averages within episodes. Delay includes unresolved failures at six hours. Bytes/day average actual traffic over all 480 episodes.","loss",["Loss","Delivery","Correct","Detected (\\%)","Delay (h)","Bytes/day"],body,"llrrrr")

if __name__=="__main__": main()
