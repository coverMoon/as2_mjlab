"""Offline only: classify trajectories and recompute rewards without simulation."""
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEGS=['FL','FR','RL','RR']

def segments(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def write_csv(path,columns,rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for row in rows:
            writer.writerow([format(float(x),'.8g') if isinstance(x,(float,np.floating)) else x for x in row])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--metrics-only',action='store_true')
    args=parser.parse_args();out=args.directory
    assert not (out/'metrics.csv').exists(),'Use a new analysis directory; existing results are not overwritten'
    shutil.copy2(__file__,out/'provenance/analyze_used.py')
    d=dict(np.load(out/'trajectory.npz',allow_pickle=False));cfg=yaml.safe_load((out/'config.yaml').read_text())
    dt=cfg['dt'];conditions=cfg['conditions'];weights=d['weights'];names=d['reward_names'].tolist()
    valid=d['alive'];nt,n=valid.shape
    rate=d['rate'];raw=rate/weights[None,None,:];contribution=rate*dt
    A=d['total'];fc=names.index('foot_clearance');pose=names.index('pose')
    B=A-contribution[:,:,fc];C=A-contribution[:,:,pose];D=B-contribution[:,:,pose]
    np.testing.assert_allclose(A,contribution.sum(-1),atol=1e-6,rtol=1e-5)
    t=(np.arange(nt)+1)*dt
    tilt=np.arccos(np.clip(1-2*(d['root'][:,:,4]**2+d['root'][:,:,5]**2),-1,1))*180/np.pi
    local_x=d['root'][:,:,0]-np.array([c['patch_center'][0] for c in conditions])[None,:]
    local_y=d['root'][:,:,1]-np.array([c['patch_center'][1] for c in conditions])[None,:]
    foot_local_x=d['foot_pos'][:,:,:,0]-np.array([c['patch_center'][0] for c in conditions])[None,:,None]
    stall=np.zeros((nt,n),bool);candidate=np.zeros_like(stall);lift=np.zeros_like(stall)
    phase=np.full((nt,n),'unrecorded',dtype='<U24')
    episodes=[];stall_events=[]
    term_names=d['termination_names'].tolist()
    for i,c in enumerate(conditions):
        length=int(valid[:,i].sum());sl=slice(0,length)
        assert length>0 and valid[:length,i].all()
        cmd=d['command'][:length,i]
        np.testing.assert_allclose(cmd[:,0],c['vx'],atol=1e-6)
        assert (cmd[:,1:]==0).all()
        up=c['scenario'].startswith('up_');down=c['scenario'].startswith('down_')
        if up:
            near=(foot_local_x[:length,i,:2].max(-1)>=-3.25)&(local_x[:length,i]<-1.1)&(np.abs(local_y[:length,i])<1.)
        elif down:
            near=(foot_local_x[:length,i,:2].max(-1)>=1.25)&(local_x[:length,i]<3.5)&(np.abs(local_y[:length,i])<1.)
        elif c['scenario'] in ('slope','rough'):
            near=(local_x[:length,i]>=-3.3)&(np.abs(local_y[:length,i])<1.)
        else:near=np.zeros(length,bool)
        approach_ids=np.flatnonzero(near);approach=approach_ids[0] if len(approach_ids) else None
        phase[:length,i]='locomotion' if c['scenario']=='flat' else 'pre_approach'
        if approach is not None:
            phase[approach:length,i]='post_approach_motion'
            phase[approach:min(length,approach+round(.5/dt)),i]='first_approach'
        lifted=(np.max(d['clearance'][:length,i,:2],axis=-1)>.04)&(np.max(np.abs(d['foot_vel'][:length,i,:2,2]),axis=-1)>.15)&near
        lift[:length,i]=lifted
        phase[:length,i][lifted]='lift_attempt'
        low_speed=(np.abs(d['v_b'][:length,i,0])<.1)&(cmd[:,0]>=.5)&(t[:length]>.5)
        for start,end in segments(low_speed):
            duration=(end-start)*dt
            if duration<1.-1e-8:continue
            candidate[start:end,i]=True
            movement=np.ptp(local_x[start:end,i]);max_tilt=float(tilt[start:end,i].max())
            close_fraction=float(near[start:end].mean())
            previous=max(0,start-round(.5/dt))
            body_hit=bool((d['nonfoot_force'][previous:end,i]>10).any())
            thigh_hit=bool((d['thigh_force'][previous:end,i]>10).any())
            if max_tilt>45:classification='unstable_or_fallen'
            elif body_hit:classification='post_body_collision'
            elif movement>.2:classification='moving_backward_or_drift_uncertain'
            elif close_fraction>=.8 and (up or down):classification='obstacle_stall'
            elif close_fraction>=.8:classification='terrain_stall'
            elif c['scenario']=='flat':classification='flat_stall'
            else:classification='location_uncertain'
            if classification in ('obstacle_stall','terrain_stall','flat_stall'):
                stall[start:end,i]=True;phase[start:end,i]='stall'
            stall_events.append({'env_id':i,'scenario':c['scenario'],'vx':c['vx'],'start_s':(start+1)*dt,'end_s':end*dt,
                'duration_s':duration,'classification':classification,'x_start_m':float(local_x[start,i]),'x_end_m':float(local_x[end-1,i]),
                'tilt_max_deg':max_tilt,'near_obstacle_fraction':close_fraction,'preceding_thigh_contact':thigh_hit})
        terminal=d['terms'][length-1,i]
        reasons=[name for name,flag in zip(term_names,terminal) if flag]
        failure=any(name!='time_out' for name in reasons)
        success=bool(d['success_step'][i]>=0)
        exited=bool(d.get('course_exit_step',np.full(n,-1))[i]>=0)
        if success:reasons.append('diagnostic_success')
        if exited:reasons.append('course_exit_censored')
        if not reasons:reasons.append('diagnostic_time_cap')
        passed_height=c['dimension'] if success and up else None
        summary={
            'env_id':i,'scenario':c['scenario'],'terrain':c['terrain'],'vx':c['vx'],'replicate':c['replicate'],
            'step_height_m':c['step_height_m'],'difficulty':c['difficulty'],'duration_s':length*dt,
            'success':success,'failure':failure,'course_exit':exited,'end_reason':'+'.join(reasons),
            'distance_x_m':float(local_x[length-1,i]-c['start_x']),
            'max_forward_x_m':float(local_x[:length,i].max()),'max_base_z_m':float(d['root'][:length,i,2].max()),
            'stall_s':float(stall[:length,i].sum()*dt),'candidate_stall_s':float(candidate[:length,i].sum()*dt),
            'stall_occurrence':bool(stall[:length,i].any()),'candidate_occurrence':bool(candidate[:length,i].any()),
            'max_step_height_passed_m':passed_height,
            'max_training_level_equivalent_passed':c['training_level_equivalent'] if passed_height is not None else None,
            'approach_time_s':float((approach+1)*dt) if approach is not None else None,
            'lift_attempt_s':float(lift[:length,i].sum()*dt),
            'attempt_then_failure':bool(failure and lifted.any()),
            'A_return':float(A[:length,i].sum()),'B_return':float(B[:length,i].sum()),'C_return':float(C[:length,i].sum()),'D_return':float(D[:length,i].sum()),
        }
        for j,name in enumerate(names):summary[name+'_return']=float(contribution[:length,i,j].sum())
        episodes.append(summary)
    write_csv(out/'episode_metrics.csv',list(episodes[0]),([r[k] for k in episodes[0]] for r in episodes))
    if stall_events:write_csv(out/'stall_events.csv',list(stall_events[0]),([r[k] for k in stall_events[0]] for r in stall_events))
    ti,ei=np.nonzero(valid)
    end_lengths=valid.sum(0)
    if not args.metrics_only:
        header=['env_id','time_s','scenario','vx_command','terrain','difficulty','terrain_level','training_level_equivalent','phase','terminal_window']
        terminal_window=ti>=np.maximum(0,end_lengths[ei]-round(1/dt))
        def prefix(k):
            j=ei[k];c=conditions[j]
            return [int(j),t[ti[k]],c['scenario'],c['vx'],c['terrain'],c['difficulty'],c.get('terrain_level',0),c['training_level_equivalent'],phase[ti[k],j],int(terminal_window[k])]
        def reward_rows():
            for k,(a,i) in enumerate(zip(ti,ei)):
                yield prefix(k)+[A[a,i]]+[value for j in range(len(names)) for value in (raw[a,i,j],weights[j],contribution[a,i,j])]
        write_csv(out/'reward_timeseries.csv',header+['total_reward']+[f'{name}_{suffix}' for name in names for suffix in ('raw','weight','contribution')],reward_rows())
        def robot_rows():
            for k,(a,i) in enumerate(zip(ti,ei)):
                yield prefix(k)+[local_x[a,i],local_y[a,i],tilt[a,i],int(candidate[a,i]),int(stall[a,i]),int(lift[a,i])]+d['root'][a,i].tolist()+d['v_b'][a,i].tolist()+d['v_w'][a,i].tolist()+d['omega_b'][a,i].tolist()+d['command'][a,i].tolist()+d['q'][a,i].tolist()+d['qd'][a,i].tolist()+d['action'][a,i].tolist()+d['terms'][a,i].astype(int).tolist()+[d['nonfoot_force'][a,i]]
        robot_header=header+['local_x_m','local_y_m','tilt_deg','candidate_stall','confirmed_stall','lift_attempt']+['root_'+x for x in ('x','y','z','qw','qx','qy','qz')]+['v_body_'+x for x in 'xyz']+['v_world_'+x for x in 'xyz']+['omega_body_'+x for x in 'xyz']+['command_'+x for x in ('vx','vy','yaw')]+[f'{p}_{j}' for p in ('joint_pos','joint_vel','action') for j in range(12)]+['done_'+x for x in term_names]+['nonfoot_force_peak_N']
        write_csv(out/'robot_timeseries.csv',robot_header,robot_rows())
        def foot_rows():
            for k,(a,i) in enumerate(zip(ti,ei)):
                for leg in range(4):
                    yield prefix(k)+[LEGS[leg]]+d['foot_pos'][a,i,leg].tolist()+d['foot_vel'][a,i,leg].tolist()+[d['clearance'][a,i,leg],int(d['foot_contact'][a,i].reshape(4)[leg]>0)]+d['foot_force'][a,i,leg].tolist()+[d['fc_leg'][a,i,leg],d['fc_leg'][a,i,leg]*weights[fc]*dt,int(d['rays_valid'][a,i,leg])]
        write_csv(out/'foot_timeseries.csv',header+['leg','world_x_m','world_y_m','world_z_m','world_vx_m_s','world_vy_m_s','world_vz_m_s','clearance_m','contact','force_x_N','force_y_N','force_z_N','fc_raw','fc_contribution','all_five_rays_valid'],foot_rows())
        write_csv(out/'counterfactual_rewards.csv',header+['A','B_no_fc','C_no_pose','D_no_fc_no_pose','B_minus_A','C_minus_A','D_minus_A'],(prefix(k)+[A[a,i],B[a,i],C[a,i],D[a,i],B[a,i]-A[a,i],C[a,i]-A[a,i],D[a,i]-A[a,i]] for k,(a,i) in enumerate(zip(ti,ei))))
    metrics=[];stages=[]
    for scenario in dict.fromkeys(c['scenario'] for c in conditions):
        for vx in (.6,1.):
            indices=[i for i,c in enumerate(conditions) if c['scenario']==scenario and c['vx']==vx]
            es=[episodes[i] for i in indices];c=conditions[indices[0]]
            row={'scenario':scenario,'vx':vx,'n':len(es),'step_height_m':c['step_height_m'],'difficulty':c['difficulty'],
                'success_rate':float(np.mean([e['success'] for e in es])),'failure_rate':float(np.mean([e['failure'] for e in es])),
                'course_exit_rate':float(np.mean([e['course_exit'] for e in es])),
                'stall_occurrence_rate':float(np.mean([e['stall_occurrence'] for e in es])),
                'candidate_stall_rate':float(np.mean([e['candidate_occurrence'] for e in es])),
                'mean_stall_s':float(np.mean([e['stall_s'] for e in es])),
                'mean_distance_x_m':float(np.mean([e['distance_x_m'] for e in es])),
                'mean_duration_s':float(np.mean([e['duration_s'] for e in es])),
                'attempt_then_failure_rate':float(np.mean([e['attempt_then_failure'] for e in es])),
                'max_step_height_passed_m':max((e['max_step_height_passed_m'] or 0 for e in es),default=0),
                'max_training_level_equivalent_passed':max((e['max_training_level_equivalent_passed'] or 0 for e in es),default=0)}
            metrics.append(row)
            for stage in ('all','pre_approach','first_approach','lift_attempt','stall','post_approach_motion','locomotion','end_window'):
                mask=valid[:,indices].copy()
                if stage=='end_window': mask &= np.arange(nt)[:,None]>=np.maximum(0,end_lengths[indices]-round(1/dt))[None,:]
                elif stage!='all':mask &= phase[:,indices]==stage
                count=int(mask.sum())
                if not count:continue
                seconds=count*dt
                r={'scenario':scenario,'vx':vx,'stage':stage,'samples':count,'total_seconds':seconds}
                for label,arr in [('A',A),('B',B),('C',C),('D',D)]:
                    r[label+'_return']=float(arr[:,indices][mask].sum());r[label+'_per_s']=r[label+'_return']/seconds
                r['positive_reward_per_s']=float(np.maximum(rate[:,indices][mask],0).sum(-1).mean())
                r['foot_speed_xy_mean']=float(np.linalg.norm(d['foot_vel'][:,indices][mask][...,:2],axis=-1).mean())
                r['front_clearance_mean_m']=float(d['clearance'][:,indices][mask][:,:2].mean())
                r['front_negative_clearance_fraction']=float((d['clearance'][:,indices][mask][:,:2]<0).mean())
                for j,name in enumerate(names):r[name+'_per_s']=float(rate[:,indices,j][mask].mean())
                stages.append(r)
    write_csv(out/'metrics.csv',list(metrics[0]),([r[k] for k in metrics[0]] for r in metrics))
    write_csv(out/'counterfactual_stage_summary.csv',list(stages[0]),([r[k] for k in stages[0]] for r in stages))
    # Event-aligned windows. Separate from phases; windows can overlap and are not additive.
    aligned=[]
    for event in stall_events:
        if event['classification'] not in ('obstacle_stall','terrain_stall','flat_stall'):continue
        i=event['env_id'];start=round(event['start_s']/dt)-1;end=round(event['end_s']/dt)
        for label,lo,hi in [('pre_stall_1s',max(0,start-round(1/dt)),start),('stall',start,end)]:
            if hi<=lo:continue
            r={**event,'window':label,'samples':hi-lo,'vx_mean':float(d['v_b'][lo:hi,i,0].mean()),
                'foot_speed_mean':float(np.linalg.norm(d['foot_vel'][lo:hi,i,:,:2],axis=-1).mean()),'positive_per_s':float(np.maximum(rate[lo:hi,i],0).sum(-1).mean())}
            for j,name in enumerate(names):r[name+'_per_s']=float(rate[lo:hi,i,j].mean())
            aligned.append(r)
    if aligned:write_csv(out/'stall_aligned_rewards.csv',list(aligned[0]),([r[k] for k in aligned[0]] for r in aligned))
    if not args.metrics_only:
        plots=out/'plots';plots.mkdir()
        plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
        for scenario in dict.fromkeys(c['scenario'] for c in conditions):
            for vx in (.6,1.):
                group=[e for e in episodes if e['scenario']==scenario and e['vx']==vx]
                representative=max(group,key=lambda e:(e['stall_s'],e['duration_s']))
                i=representative['env_id'];length=int(end_lengths[i]);tt=t[:length]
                fig,ax=plt.subplots(5,1,figsize=(12,15),sharex=True,layout='constrained')
                fig.suptitle(f'{scenario}, vx={vx} m/s, env={i}, height={conditions[i]["step_height_m"]} m; 0-{length*dt:.2f} s')
                ax[0].plot(tt,d['command'][:length,i,0],'k--',label='command vx');ax[0].plot(tt,d['v_b'][:length,i,0],label='body vx');ax[0].plot(tt,d['v_b'][:length,i,1],label='body vy');ax[0].set_ylabel('Velocity [m/s]');ax[0].legend(ncol=3)
                for leg in range(4):ax[1].plot(tt,d['clearance'][:length,i,leg],label=LEGS[leg])
                ax[1].axhline(.1,color='k',ls='--',label='target');ax[1].set_ylabel('Local clearance [m]');ax[1].legend(ncol=5)
                twin=ax[1].twinx();twin.plot(tt,np.linalg.norm(d['foot_vel'][:length,i,:,:2],axis=-1).mean(-1),color='gray',alpha=.5);twin.set_ylabel('Mean foot XY speed [m/s]')
                ax[2].plot(tt,contribution[:length,i,fc],color='tab:red',label='foot clearance per step');ax[2].set_ylabel('Reward / control step');ax[2].legend()
                for name in ('tracking_lin_vel','tracking_ang_vel','pose','soft_landing','thigh_contact','action_rate','termination'):
                    ax[3].plot(tt,rate[:length,i,names.index(name)],label=name)
                ax[3].set_ylabel('Reward rate [1/s]');ax[3].legend(ncol=3)
                ax[4].plot(tt,local_x[:length,i],color='k',lw=2,label='base X')
                for leg in range(4):ax[4].plot(tt,foot_local_x[:length,i,leg],alpha=.6,label=LEGS[leg]+' X')
                ax[4].axhline(conditions[i]['goal_x'],color='g',ls='--',label='success goal');ax[4].set_ylabel('Patch-relative X [m]');ax[4].set_xlabel('Time [s]');ax[4].legend(ncol=3)
                for start,end in segments(stall[:length,i]):
                    for panel in ax:panel.axvspan((start+1)*dt,end*dt,color='orange',alpha=.18)
                fig.savefig(plots/f'trace_{scenario}_vx{vx}_env{i}.png',dpi=150);plt.close(fig)
        # Scene-level success/failure/stall counts and rewards around stalling.
        fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
        labels=[f'{m["scenario"]}\n{m["vx"]}m/s' for m in metrics];x=np.arange(len(metrics))
        for offset,key,color in [(-.25,'success_rate','tab:green'),(0,'failure_rate','tab:red'),(.25,'stall_occurrence_rate','tab:orange')]:axes[0].bar(x+offset,[m[key] for m in metrics],width=.25,label=key,color=color)
        axes[0].set_xticks(x,labels,rotation=90);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Fraction of 8 initial states');axes[0].legend();axes[0].set_title('Fixed-scene outcomes (first episode only)')
        stack_stages=['pre_approach','first_approach','lift_attempt','stall']
        for j,name in enumerate(('tracking_lin_vel','tracking_ang_vel','pose','foot_clearance','action_rate','soft_landing','thigh_contact')):
            values=[]
            for stage in stack_stages:
                entries=[s for s in stages if s['stage']==stage and s['scenario'].startswith('up_')]
                total=sum(s['total_seconds'] for s in entries)
                values.append(sum(s[name+'_per_s']*s['total_seconds'] for s in entries)/total if total else np.nan)
            axes[1].plot(stack_stages,values,marker='o',label=name)
        axes[1].set_ylabel('Pooled reward rate [1/s]');axes[1].set_title('Ascending stairs: reward composition by phase');axes[1].legend()
        fig.savefig(plots/'outcomes_and_reward_composition.png',dpi=180);plt.close(fig)
    facts={'episodes':episodes,'metrics':metrics,'stages':stages,'stall_events':stall_events,
        'classification':{'candidate':'|body vx|<0.1 m/s continuously >=1s, command vx>=0.5, time>0.5s',
        'confirmed':'upright tilt<=45deg; no body collision in preceding .5s/window; X range<=.2m; >=80% samples in terrain-specific obstacle zone (flat stalls separately labelled)',
        'obstacle_zone':'up: front foot X>=-3.25m and base X<-1.1m, |Y|<1m; down: front foot X>=1.25m and base X<3.5m, |Y|<1m',
        'lift':'near obstacle, max front-foot clearance>.04m and max absolute front-foot vertical velocity>.15m/s',
        'phases':'first_approach is first .5s after entering obstacle zone; lift and stall take precedence; end_window last1s overlaps other phases'},
        'all_ray_valid_fraction':float(d['rays_valid'][valid].mean()),'samples':len(ti),'feet_samples':4*len(ti)}
    (out/'facts.json').write_text(json.dumps(facts,indent=2,allow_nan=False))
    print(json.dumps(metrics,indent=2));print('ANALYSIS COMPLETE',out,flush=True)

if __name__=='__main__':main()
