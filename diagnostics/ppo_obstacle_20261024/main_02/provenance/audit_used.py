"""Independent source/log audit, event-aligned mechanism metrics and replay check."""
import argparse,csv,difflib,json,math
from pathlib import Path
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def training_history(out):
    config=yaml.safe_load((out/'config.yaml').read_text())
    run=Path(config['checkpoint']).parent;root=run.parent;history=[]
    while run.exists():
        settings=yaml.load((run/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
        available=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
        history.append({'run':run.name,'settings':{k:settings[k] for k in ('resume','load_run','load_checkpoint','max_iterations')},
                        'last_available_checkpoint':str(available[-1]) if available else None})
        if settings['resume']!='true':break
        run=root/settings['load_run']
    if not run.exists():history.append({'missing_earlier_run':str(run),'limitation':'Earlier config and source cannot be audited locally'})
    (out/'training_history.json').write_text(json.dumps(history,indent=2))

def plot_aligned(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit=json.loads((out/'mechanism_audit.json').read_text());config=yaml.safe_load((out/'config.yaml').read_text())
    d=np.load(out/'trajectory.npz');names=d['reward_names'].tolist();dt=config['dt']
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    shown=['tracking_lin_vel','tracking_ang_vel','pose','foot_clearance','soft_landing','thigh_contact','action_rate','torques']
    colors=plt.cm.tab10(np.arange(9))
    for panel,vx in zip(axes[:2],(.6,1.)):
        selected=[r for r in audit['stall_structure'] if r['vx']==vx and r['first_stall_start_s'] is not None]
        before=[];after=[]
        for r in selected:
            i=r['env_id'];a=round(r['first_stall_start_s']/dt)-1;b=round(r['first_stall_end_s']/dt)
            before.append(d['rate'][max(0,a-round(1/dt)):a,i].mean(0));after.append(d['rate'][a:b,i].mean(0))
        values=np.stack([np.mean(before,0),np.mean(after,0)])
        positive=np.zeros(2);negative=np.zeros(2)
        for j,name in enumerate(shown+['other_penalties']):
            heights=values[:,names.index(name)] if name!='other_penalties' else values[:,[k for k,n in enumerate(names) if n not in shown]].sum(-1)
            bottom=np.where(heights>=0,positive,negative)
            panel.bar([0,1],heights,bottom=bottom,color=colors[j],label=name)
            positive+=np.maximum(heights,0);negative+=np.minimum(heights,0)
        panel.scatter([0,1],values.sum(-1),color='black',marker='D',s=35,label='net')
        panel.set_xticks([0,1],['1s before first stall','First stall interval']);panel.set_ylabel('Reward rate [1/s]')
        panel.set_title(f'Ascending stairs, vx={vx} m/s, n={len(selected)} episodes')
        panel.axhline(0,color='k',lw=.7);panel.grid(axis='y',alpha=.25)
    axes[0].legend(fontsize=8,ncol=2,loc='lower left')
    effects=[x for x in audit['paired_effects'] if x['effect']=='fc_cost_approach_minus_before']
    means=np.array([x['mean_difference_per_s'] for x in effects]);ci=np.array([x['bootstrap_95_percentile_CI'] for x in effects])
    axes[2].errorbar(means,np.arange(len(effects)),xerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=4)
    axes[2].set_yticks(np.arange(len(effects)),[f'{x["scenario"]}, {x["vx"]} m/s' for x in effects]);axes[2].axvline(0,color='k',ls='--')
    axes[2].set_xlabel('Foot Clearance cost increase [reward/s]');axes[2].set_title('Approach minus preceding 0.5s\nPaired bootstrap 95% CI, 8 initial states')
    fig.suptitle('Reward mechanisms: identical recorded trajectories; no policy retraining')
    destination=out/'plots/stall_aligned_reward_budget.png'
    assert not destination.exists()
    fig.savefig(destination,dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('primary',type=Path);p.add_argument('replay',type=Path)
    args=p.parse_args();out=args.primary
    assert not (out/'mechanism_audit.json').exists()
    f=json.loads((out/'facts.json').read_text());c=yaml.safe_load((out/'config.yaml').read_text())
    d=np.load(out/'trajectory.npz');r=np.load(args.replay/'trajectory.npz')
    replay_f=json.loads((args.replay/'facts.json').read_text())
    dt=c['dt'];names=d['reward_names'].tolist();rates=d['rate'];times=(np.arange(rates.shape[0])+1)*dt
    rows=[]
    for ep in f['episodes']:
        i=ep['env_id'];cond=c['conditions'][i]
        if not cond['scenario'].startswith('up_'):continue
        length=int(d['alive'][:,i].sum());events=[e for e in f['stall_events'] if e['env_id']==i and e['classification']=='obstacle_stall']
        row={k:ep[k] for k in ('env_id','scenario','vx','replicate','success','failure','stall_s')}
        onset=round(ep['approach_time_s']/dt)-1 if ep['approach_time_s'] is not None else None
        windows={}
        if onset is not None:
            windows['before_approach']=(max(0,onset-round(.5/dt)),onset)
            windows['approach']=(onset,min(length,onset+round(.5/dt)))
        if events:
            event=events[0];a=round(event['start_s']/dt)-1;b=round(event['end_s']/dt)
            windows['pre_stall']=(max(0,a-round(1/dt)),a)
            windows['first_stall']=(a,b)
            row['first_stall_start_s']=event['start_s'];row['first_stall_end_s']=event['end_s']
        else:row['first_stall_start_s']=None;row['first_stall_end_s']=None
        for label in ('before_approach','approach','pre_stall','first_stall'):
            bounds=windows.get(label)
            for name in ('foot_clearance','pose','tracking_lin_vel','tracking_ang_vel','action_rate','soft_landing','thigh_contact','torques'):
                row[label+'_'+name+'_per_s']=float(rates[bounds[0]:bounds[1],i,names.index(name)].mean()) if bounds and bounds[1]>bounds[0] else None
        if events:
            a,b=windows['first_stall']
            row['stall_front_force_x_mean_N']=float(d['foot_force'][a:b,i,:2,0].mean())
            row['stall_front_z_max_m']=float(d['foot_pos'][a:b,i,:2,2].max())
            row['stall_front_foot_speed_xy_mean_m_s']=float(np.linalg.norm(d['foot_vel'][a:b,i,:2,:2],axis=-1).mean())
            row['stall_positive_reward_per_s']=float(np.maximum(rates[a:b,i],0).sum(-1).mean())
            row['stall_negative_clearance_fraction']=float((d['clearance'][a:b,i,:2]<0).mean())
        else:
            for key in ('stall_front_force_x_mean_N','stall_front_z_max_m','stall_front_foot_speed_xy_mean_m_s','stall_positive_reward_per_s','stall_negative_clearance_fraction'):row[key]=None
        rows.append(row)
    with (out/'mechanism_metrics.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(20261024)
    effects=[]
    for scenario in ('up_low','up_mid','up_high'):
        for vx in (.6,1.):
            rs=[v for v in rows if v['scenario']==scenario and v['vx']==vx]
            for label,after,before in [('fc_cost_approach_minus_before','approach','before_approach'),('fc_cost_stall_minus_pre_stall','first_stall','pre_stall')]:
                ds=np.array([-v[after+'_foot_clearance_per_s']+v[before+'_foot_clearance_per_s'] for v in rs if v[after+'_foot_clearance_per_s'] is not None and v[before+'_foot_clearance_per_s'] is not None])
                if not len(ds):continue
                boot=ds[rng.integers(0,len(ds),(10000,len(ds)))].mean(-1)
                effects.append({'scenario':scenario,'vx':vx,'effect':label,'n_initial_states':len(ds),'mean_difference_per_s':float(ds.mean()),'bootstrap_95_percentile_CI':np.quantile(boot,[.025,.975]).tolist()})
    startup_a=np.load(out/'startup_dr.npz');startup_b=np.load(args.replay/'startup_dr.npz')
    comparison={
        'startup_dr_bitwise_equal':{k:bool(np.array_equal(startup_a[k],startup_b[k])) for k in startup_a},
        'first_policy_action_max_abs_difference':float(np.abs(d['action'][0]-r['action'][0]).max()),
        'first_step_root_max_abs_difference':float(np.abs(d['root'][0]-r['root'][0]).max()),
        'first_second_root_max_abs_difference':float(np.abs(d['root'][:50]-r['root'][:50]).max()),
        'different_success_outcomes':int(np.sum((d['success_step']>=0)!=(r['success_step']>=0))),
        'primary_successes':int(np.sum(d['success_step']>=0)),'replay_successes':int(np.sum(r['success_step']>=0)),
        'interpretation':'Identical initial states/DR and deterministic actor; GPU contact simulation is not bitwise reproducible over long horizons. Compare aggregate outcomes, not exact long trajectories.',
        'scene_comparison':[{ 'scenario':a['scenario'],'vx':a['vx'],'primary_success_rate':a['success_rate'],'replay_success_rate':b['success_rate'],'primary_stall_rate':a['stall_occurrence_rate'],'replay_stall_rate':b['stall_occurrence_rate'],'primary_failure_rate':a['failure_rate'],'replay_failure_rate':b['failure_rate']} for a,b in zip(f['metrics'],replay_f['metrics'])]}
    (out/'reproducibility.json').write_text(json.dumps(comparison,indent=2))
    logs=Path(c['checkpoint']).parent;ea=EventAccumulator(str(logs));ea.Reload()
    log_summary={}
    for tag in ea.Tags()['scalars']:
        if tag.startswith(('Episode_Reward/','Curriculum/','Train/mean_episode_length')):
            samples=ea.Scalars(tag);last=samples[-1]
            tail=[s.value for s in samples if s.step>=last.step-100]
            log_summary[tag]={'last_iteration':last.step,'last_value':last.value,'last_100_iterations_mean':float(np.mean(tail)),
                              'semantics':'Episode_Reward = per-episode term sum / configured 20s, averaged over reset envs; not necessarily true episode-time average' if tag.startswith('Episode_Reward/') else 'upstream scalar'}
    (out/'training_log_summary.json').write_text(json.dumps(log_summary,indent=2))
    # Source/config inference, not measured training command frequencies.
    pure_forward=(1-.05)*.25*.1*(.4/(2*math.pi))
    sampling={'definition':'vx>=0.5, |vy|<=0.1, |executed yaw rate|<=0.1 m/s or rad/s',
        'assumptions':['late curriculum','low vx group uniform [-1,1] on non-flat','vy uniform [-1,1]','heading targets uniform [-pi,pi] independent of current heading','heading stiffness .5','5% standing','uniform independent resamples; not an observed training frequency'],
        'fraction_among_nonflat_command_resamples':pure_forward,
        'fraction_all_resamples_normal_pyramid_30pct':pure_forward*.3,
        'fraction_all_resamples_inverted_pyramid_20pct':pure_forward*.2,
        'normal_pyramid_spawn':'central top platform, x/y random offsets +/-0.5 inside 3m platform: initial traversal is descending',
        'inverted_pyramid_spawn':'central bottom platform: initial traversal is ascending',
        'terrain_curriculum':'radial distance >4m promotes; distance <command speed*20*.5 demotes; no explicit successful ascent signal',
        'training_command_function_source_missing_from_archive':True}
    (out/'command_sampling_estimate.json').write_text(json.dumps(sampling,indent=2))
    src=out/'provenance/training_source/src/as2_mjlab/tasks/ppo/env_cfg.py'
    current=out/'provenance/current_source/src/as2_mjlab/tasks/ppo/env_cfg.py'
    (out/'provenance/training_vs_current_env_source.diff').write_text(''.join(difflib.unified_diff(src.read_text().splitlines(True),current.read_text().splitlines(True),fromfile='archived_training_env.py',tofile='current_env.py')))
    audit={'paired_effects':effects,'sampling_estimate':sampling,'reproducibility':comparison,
           'reward_semantics':{'dt':dt,'termination_weight':c['reward_weights']['termination'],'actual_failure_step_contribution':c['reward_weights']['termination']*dt,'raw_reconstruction':'step_reward / weight; zero-weight terms not present',
            'robot_state_timestamp':'inside reward compute before final forward; derived quantities lag one 0.005s physics step, matching reward inputs'},
           'stall_structure':rows}
    (out/'mechanism_audit.json').write_text(json.dumps(audit,indent=2))
    training_history(out)
    plot_aligned(out)
    print('Paired effects',json.dumps(effects,indent=2))
    print('Replay comparison',json.dumps(comparison,indent=2))
    print('Pure forward conditional estimated fraction',pure_forward)

if __name__=='__main__':main()
