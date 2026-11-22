"""Lossless contact delivery and receiver state, with no truth input."""
import bisect
from dataclasses import asdict
import math
import time
from .policies import PublicBuffer, plan
from .wire import ACK_BYTES, encode_ack
from .diagnosis import diagnose

def simulate(public_all, contacts, method, rate, config, settings, trace=False):
    received, contexts = {}, {}
    source_times=[m.time_s for m in public_all.measurements]
    decisions, traces, ledger, examples=[],[],[],[]
    totals=dict(wire_bytes=0,data_bytes=0,context_bytes=0,ack_bytes=0,
                planning_wall_s=0.0,diagnosis_wall_s=0.0,planner_overruns=0,
                diagnosis_overruns=0,max_budget_excess_bytes=0,causality_violations=0)
    sequence=0
    for index,c in enumerate(contacts):
        start,end=c['start_s'],c['end_s']
        count=bisect.bisect_right(source_times,start)
        public=PublicBuffer(public_all.measurements[:count],
                            tuple(e for e in public_all.contexts if e.time_s<=start),start)
        reserved=settings['planner_reservation_s']
        budget=max(0,math.floor((end-start-reserved)*rate/8))
        began=time.perf_counter()
        frames=plan(method,public,received,contexts,budget,settings,sequence)
        elapsed=time.perf_counter()-began
        totals['planning_wall_s']+=elapsed
        totals['planner_overruns']+=int(elapsed>reserved)
        delay=max(reserved,elapsed)
        budget=max(0,math.floor((end-start-delay)*rate/8))
        while frames and sum(f.wire_bytes for f in frames)>budget:
            frames.pop()
        cursor,used=start+delay,0
        tx=[]
        for frame in frames:
            tx_start=cursor
            cursor+=frame.wire_bytes*8/rate
            if cursor>end+1e-8:
                raise AssertionError('Contact exceeded')
            ack=encode_ack(frame.sequence,cursor)
            assert len(ack)==ACK_BYTES
            if any(m.time_s>start for m in frame.measurements) or any(e.time_s>start for e in frame.contexts):
                raise AssertionError('Noncausal frame')
            received.update((m.time_s,m) for m in frame.measurements)
            contexts.update((e.epoch,e) for e in frame.contexts)
            for name,value in [('wire_bytes',frame.wire_bytes),('data_bytes',frame.data_bytes),
                               ('context_bytes',frame.context_bytes),('ack_bytes',len(ack))]:
                totals[name]+=value
            used+=frame.wire_bytes
            sequence+=1
            entry=dict(start_s=tx_start,end_s=cursor,wire_bytes=frame.wire_bytes,
                       cumulative_wire_bytes=totals['wire_bytes'],contact_number=index+1)
            ledger.append(entry)
            if trace:
                tx.append({**entry,'sequence':frame.sequence,'data_bytes':frame.data_bytes,
                           'context_bytes':frame.context_bytes,'ack_bytes':len(ack),
                           'measurement_times_s':[m.time_s for m in frame.measurements],
                           'measurement_epochs':[m.command_epoch for m in frame.measurements],
                           'context_times_s':[e.time_s for e in frame.contexts],
                           'context_epochs':[e.epoch for e in frame.contexts]})
                if len(examples)<2:
                    examples.append({'data_frame_hex':frame.blob.hex(),'ack_hex':ack.hex()})
        totals['max_budget_excess_bytes']=max(totals['max_budget_excess_bytes'],used-budget)
        began=time.perf_counter()
        certificates=diagnose(tuple(received.values()),tuple(contexts.values()),config=config,decision_time_s=end)
        diagnostic_time=time.perf_counter()-began
        totals['diagnosis_wall_s']+=diagnostic_time
        totals['diagnosis_overruns']+=int(diagnostic_time>settings['ground_diagnosis_reservation_s'])
        decision_s=end+max(settings['ground_diagnosis_reservation_s'],diagnostic_time)
        decision=dict(time_s=decision_s,contact_number=index+1,
                      certificates=[asdict(d) for d in certificates],
                      cumulative_wire_bytes=totals['wire_bytes'])
        decisions.append(decision)
        if trace:
            traces.append({**c,'contact_number':index+1,'budget_bytes':budget,'used_wire_bytes':used,
                           'planning_wall_s':elapsed,'diagnosis_wall_s':diagnostic_time,
                           'decision':decision,'frames':tx})
    return {**totals,'airtime_s':totals['wire_bytes']*8/rate,
            'receiver_count':len(received),'receiver_context_count':len(contexts),
            'frames_sent':len(ledger),'n_contacts':len(contacts),
            '_decisions':decisions,'_ledger':ledger,'_trace':traces,'_wire_examples':examples}
