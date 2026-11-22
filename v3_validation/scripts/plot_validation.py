"""Scientific validation figures from completed CSV/JSON files only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.tools/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from export_validation_tables import read,load,num,boolean,PROFILES,SCHEDULES,verified_run,file_record,resolve,validate_manuscript

COLORS={'snapshot_events':'#009e73','fresh_compact':'#0072b2','events_thinned':'#6f3a94'}

def save(fig,name,out,manifest,description):
    for extension in ['png','pdf']:
        fig.savefig(out/(name+'.'+extension),dpi=190,bbox_inches='tight')
    manifest.append({'name':name,'description':description,'png':file_record(out/(name+'.png')),'pdf':file_record(out/(name+'.pdf'))})
    plt.close(fig)

def lab_figure(lab,out,manifest):
    rows=read(lab/'predictions.csv');result=load(lab/'results.json')
    if result['status']!='completed':raise ValueError('Uncompleted laboratory scoring')
    if load(lab/'audit.json').get('status')!='passed':raise ValueError('Laboratory audit has not passed')
    time=np.array([num(r['time_s']) for r in rows]);actual=np.array([num(r['measured_t1_c']) for r in rows])
    first=np.array([num(r['first_order_c']) for r in rows]);lag=np.array([num(r['sensor_lag_c']) for r in rows]);power=np.array([num(r['heater1_percent']) for r in rows])
    split=min(num(r['time_s']) for r in rows if r['split']=='held_out');off=result['commanded_off_observation']['start_s']
    fig=plt.figure(figsize=(9.5,6.5));gs=fig.add_gridspec(2,2,height_ratios=[3,1],hspace=.10,wspace=.22)
    full=fig.add_subplot(gs[0,0]);command=fig.add_subplot(gs[1,0],sharex=full);detail=fig.add_subplot(gs[:,1])
    for axis in [full,detail]:
        axis.plot(time,actual,color='black',linewidth=1.3,label='Measured temperature')
        axis.plot(time,first,color='#d55e00',linewidth=1.2,label='First-order prediction')
        axis.plot(time,lag,color='#0072b2',linewidth=1.3,label='Two-state lag prediction')
        axis.grid(alpha=.18);axis.set_ylabel('Temperature (°C)')
    full.axvspan(split,time[-1],color='#56b4e9',alpha=.12);full.axvline(split,color='#666666',linestyle='--',linewidth=1)
    full.set_title('Calibration-only free-run predictions',fontsize=12)
    full.text(split+5,full.get_ylim()[0]+1,'Held-out',fontsize=12,color='#325a72')
    full.tick_params(labelbottom=False)
    command.step(time,power,where='post',color='#009e73');command.axvspan(split,time[-1],color='#56b4e9',alpha=.12);command.set_ylim(-5,105)
    command.set_ylabel('Command (%)');command.set_xlabel('Elapsed time (s)');command.grid(alpha=.18)
    detail.set_xlim(off-20,time[-1]);visible=(time>=off-20)
    low=min(np.min(actual[visible]),np.min(first[visible]),np.min(lag[visible]));high=max(np.max(actual[visible]),np.max(first[visible]),np.max(lag[visible]))
    detail.set_ylim(low-1,high+2);detail.axvline(off,color='#009e73',linestyle='--',label='Logged OFF')
    detail.set_xlabel('Elapsed time (s)');detail.set_title('One held-out OFF: measured lag',fontsize=12)
    peak_delay=result['commanded_off_observation']['observed_peak_after_off_s'];peak_time=off+peak_delay;idx=np.argmin(abs(time-peak_time))
    detail.annotate('Measured post-OFF peak',xy=(time[idx],actual[idx]),xytext=(off+25,high+1),fontsize=12,arrowprops={'arrowstyle':'->','color':'black'})
    handles,labels=detail.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=12)
    fig.suptitle('Public hardware record — nominal commands, no measured fault labels',fontsize=13)
    fig.subplots_adjust(top=.90,bottom=.17)
    save(fig,'lab_physics',out,manifest,'Measured APMonitor command/temperature record and calibration-only predictions; one held-out OFF, no population confidence interval.')

def primary_figure(run,out,manifest):
    verified_run(run);rows=read(run/'tables/primary_endpoints.csv')
    keys=sorted(set((r['profile'],r['schedule_id'],num(r['rate_bps'])) for r in rows))
    labels=[PROFILES[p]+' | '+('Syn' if s=='synthetic_90min' else 'Kaz')+' | '+str(int(rate)) for p,s,rate in keys]
    fig,axes=plt.subplots(1,3,figsize=(10.5,7.5),sharey=True)
    specs=[('coverage_loss',.01,'Resolution loss (pp)'),('excess_any_wrong',.01,'Any-error excess (pp)'),('delay_gap_s',1,'Delay excess (s)')]
    for axis,(endpoint,scale,title) in zip(axes,specs):
        selected=[next(r for r in rows if (r['profile'],r['schedule_id'],num(r['rate_bps']))==k and r['endpoint']==endpoint) for k in keys]
        mean=np.array([num(r['mean_difference'])/scale for r in selected]);low=np.array([num(r['ci_low'])/scale for r in selected]);high=np.array([num(r['ci_high'])/scale for r in selected]);margin=num(selected[0]['margin'])/scale
        axis.axvspan(-margin,margin,color='#009e73',alpha=.09);axis.axvline(-margin,color='#d55e00',linestyle='--',linewidth=.9);axis.axvline(margin,color='#d55e00',linestyle='--',linewidth=.9);axis.axvline(0,color='#666666',linewidth=.7)
        for i,r in enumerate(selected):
            color=COLORS['snapshot_events'] if boolean(r['eq_supported']) else '#d55e00'
            axis.errorbar(mean[i],i,xerr=[[mean[i]-low[i]],[high[i]-mean[i]]],fmt='o',markersize=4,capsize=3,color=color)
        extent=max(margin,max(abs(low)),max(abs(high)));axis.set_xlim(-extent*1.12,extent*1.12)
        axis.set_xticks([-margin,0,margin])
        axis.set_xlabel(title,fontsize=12);axis.grid(axis='y',alpha=.15);axis.set_ylim(len(keys)-.5,-.5)
    axes[0].set_yticks(range(len(keys)),labels,fontsize=12)
    fig.suptitle('Event summaries: simultaneous finite-sample intervals\nShaded bands are equivalence margins; rows show profile, schedule, bit/s',fontsize=12)
    fig.subplots_adjust(left=.32,right=.99,top=.88,bottom=.12,wspace=.14)
    save(fig,'primary_intervals',out,manifest,'All twelve primary comparisons and all three endpoints; simultaneous 95% family confidence, no selection of favorable cells.')

def boundary_figure(run,out,manifest):
    rows=read(run/'tables/matched_budget.csv')
    fig,axes=plt.subplots(2,3,figsize=(9.5,7),sharex=True,sharey='row')
    colors={'synthetic_90min':'#0072b2','kazstsat_astana_10deg':'#009e73'}
    handles=[]
    for col,profile in enumerate(PROFILES):
        for schedule in SCHEDULES:
            rr=sorted([r for r in rows if r['profile']==profile and r['schedule_id']==schedule],key=lambda r:num(r['rate_bps']))
            x=np.array([num(r['rate_bps']) for r in rr])
            for row,(metric,reference,scale) in enumerate([('resolved_fraction','oracle_resolved_fraction',.01),('capped_delay_s','oracle_capped_delay_s',3600)]):
                mean=np.array([num(r[metric+'_mean'])/scale for r in rr]);low=np.array([num(r[metric+'_ci_low'])/scale for r in rr]);high=np.array([num(r[metric+'_ci_high'])/scale for r in rr])
                line=axes[row,col].errorbar(x,mean,yerr=[mean-low,high-mean],color=colors[schedule],marker='o',linewidth=1.3,capsize=3,label=SCHEDULES[schedule]+' summary')
                reference_line=axes[row,col].plot(x,[num(r[reference+'_mean'])/scale for r in rr],color=colors[schedule],linestyle='--',label=SCHEDULES[schedule]+' full history')[0]
                if col==0 and row==0:handles.extend([line,reference_line])
                axes[row,col].grid(alpha=.18);axes[row,col].set_xscale('log',base=2);axes[row,col].set_xticks(x,[str(int(v)) for v in x])
        axes[0,col].set_title(PROFILES[profile],fontsize=12);axes[0,col].set_ylim(-3,103);axes[1,col].set_ylim(0,6.2)
        axes[1,col].set_xlabel('Allocation (bit/s)')
    axes[0,0].set_ylabel('Resolved OFF outcomes (%)');axes[1,0].set_ylabel('Mean capped delay (h)')
    counts=sorted({int(num(r['n_episodes'])) for r in rows})
    if len(counts)!=1:raise ValueError('Budget figure must use matched sample sizes')
    fig.suptitle('Matched budget comparison\n'+str(counts[0])+' identical episodes/profile at every allocation',fontsize=12)
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=12);fig.subplots_adjust(top=.89,bottom=.19,wspace=.11,hspace=.16)
    save(fig,'budget_boundary',out,manifest,'Summary and bandwidth-unlimited contact reference on the same registered episode subset at every allocation; descriptive marginal bootstrap bounds, not additional simultaneous claims at 4 bit/s.')

def secondary_figure(run,out,manifest):
    rows=read(run/'tables/paired_secondary.csv');fig,axes=plt.subplots(2,3,figsize=(9.5,8));has_tie=False
    for col,profile in enumerate(PROFILES):
        for row,schedule in enumerate(SCHEDULES):
            ax=axes[row,col];rr=[r for r in rows if r['profile']==profile and r['schedule_id']==schedule]
            for method in ['fresh_compact','events_thinned']:
                points=[]
                for rate,marker in [(8,'o'),(32,'s')]:
                    metrics={r['metric']:r for r in rr if r['candidate']==method and num(r['rate_bps'])==rate}
                    x=num(metrics['wire_bytes']['mean_difference'])/1024;y=num(metrics['capped_delay_s']['mean_difference'])/60;coverage=num(metrics['resolved_fraction']['mean_difference'])*100
                    points.append(dict(x=x,y=y,coverage=coverage,rate=rate,marker=marker))
                if np.allclose([points[0][k] for k in ('x','y','coverage')],[points[1][k] for k in ('x','y','coverage')],rtol=0,atol=1e-10):
                    points=[dict(points[0],marker='D',rate='both')];has_tie=True
                for point in points:
                    x,y,coverage=point['x'],point['y'],point['coverage']
                    ax.scatter(x,y,color=COLORS[method],marker=point['marker'],s=42,zorder=5)
                    if method=='events_thinned' and point['rate']==32 and profile=='clear_current' and schedule=='synthetic_90min':offset,align=(-2,-24),'right'
                    elif method=='events_thinned':offset,align=(0,8),'center'
                    elif point['rate']==32:offset,align=(-6,-28 if schedule=='kazstsat_astana_10deg' else -15),'right'
                    else:offset,align=(6,8),'left'
                    ax.annotate(f'{coverage:+.1f} pp',(x,y),xytext=offset,textcoords='offset points',fontsize=12,ha=align,color=COLORS[method])
            ax.axhline(0,color='#666666',linewidth=.7);ax.axvline(0,color='#666666',linewidth=.7);ax.grid(alpha=.15);ax.margins(x=.28,y=.30)
            ax.set_title(PROFILES[profile]+' | '+('90-minute' if row==0 else 'KazSTSAT'),fontsize=12)
            if row==1:ax.set_xlabel('Daily-byte difference (KiB)')
            if col==0:ax.set_ylabel('Delay difference (min)')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLORS['fresh_compact'],marker='o',linestyle='',label='Fresh compact'),Line2D([],[],color=COLORS['events_thinned'],marker='o',linestyle='',label='Thinned history'),Line2D([],[],color='black',marker='o',linestyle='',label='8 bit/s'),Line2D([],[],color='black',marker='s',linestyle='',label='32 bit/s')]
    if has_tie:handles.append(Line2D([],[],color='black',marker='D',linestyle='',label='8 and 32 tie'))
    fig.legend(handles=handles,loc='lower center',ncol=len(handles),frameon=False,fontsize=12)
    fig.suptitle('Matched secondary policies minus event summary\nLabels show resolution differences in percentage points',fontsize=12)
    fig.subplots_adjust(top=.86,bottom=.13,hspace=.34,wspace=.28)
    save(fig,'secondary_tradeoff',out,manifest,'Paired secondary point estimates on the registered same-episode subset; annotations retain resolution differences and costs are unconditional daily wire bytes.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=Path('outputs/confirmatory-main'));p.add_argument('--lab-dir',type=Path,default=Path('lab/main'));p.add_argument('--output-dir',type=Path,default=Path('figures'));p.add_argument('--lab-only',action='store_true');a=p.parse_args()
    a.run_dir=resolve(a.run_dir);a.lab_dir=resolve(a.lab_dir)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out=resolve(a.output_dir);out.mkdir(exist_ok=True,parents=True)
    if out.resolve() != (ROOT/'figures').resolve():
        shutil.copytree(ROOT/'figures/v2_archive',out/'v2_archive',dirs_exist_ok=True)
    manifest=[];lab_figure(a.lab_dir,out,manifest)
    sources=[a.lab_dir/'predictions.csv',a.lab_dir/'results.json',a.lab_dir/'audit.json']
    if not a.lab_only:
        primary_figure(a.run_dir,out,manifest);boundary_figure(a.run_dir,out,manifest);secondary_figure(a.run_dir,out,manifest)
        sources += [a.run_dir/'tables/primary_endpoints.csv',a.run_dir/'tables/matched_budget.csv',a.run_dir/'tables/paired_secondary.csv']
    record={'lab_only':a.lab_only,'figures':manifest,'sources':[file_record(f) for f in sources],'script':file_record(Path(__file__))}
    (out/'figure_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    if not a.lab_only:
        validate_manuscript(out.parent/'paper',out.parent/'tables',out)
    print(json.dumps({'figures':[r['name'] for r in manifest]},indent=2))

if __name__=='__main__':main()
