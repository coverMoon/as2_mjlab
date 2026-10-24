"""Read-only PPO experiment; no training and no production configuration edits.

uv run python diagnostics/ppo_obstacle_20261024/diagnose.py --output <new-directory>
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
import torch
import yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from as2_mjlab.tasks.ppo.env_cfg import ppo_rough_env_cfg
from as2_mjlab.tasks.ppo.rl_cfg import ppo_runner_cfg

ROOT = Path(__file__).resolve().parents[2]
LEGS = ['FL', 'FR', 'RL', 'RR']
CHECKPOINT = ROOT/'logs/rsl_rl/as2_ppo/2026-10-22_16-09-58_as2-rough-ppo/model_10000.pt'

class FixedCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids] = 0
        self.vel_command_b[env_ids, 0] = torch.as_tensor(self.cfg.velocities, device=self.device)[env_ids]
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = False
        self.is_heading_env[env_ids] = False
        self.is_world_env[env_ids] = False

@dataclass(kw_only=True)
class FixedCommandCfg(UniformVelocityCommandCfg):
    velocities: tuple[float, ...] = ()
    def build(self, env):
        return FixedCommand(self, env)

def fixed_reset(env, env_ids):
    """Identical stored initial state on every reset; startup DR remains seeded."""
    robot = env.scene['robot']
    robot.write_root_link_pose_to_sim(env.diagnostic_pose[env_ids], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    robot.write_joint_state_to_sim(env.diagnostic_joints[env_ids], torch.zeros_like(env.diagnostic_joints[env_ids]), env_ids=env_ids)

def cases():
    result = [('flat', 'flat', 0., -3.65, 2.5)]
    for h, tag in [(0.06, 'low'), (0.12, 'mid'), (0.24, 'high')]:
        result += [(f'up_{tag}', 'pyramid_stairs', h, -3.65, -1.1),
                   (f'down_{tag}', 'pyramid_stairs', h, 0., 3.45)]
    result += [('slope', 'hf_pyramid_slope', 0.2, -3.65, 0.),
               ('rough', 'random_rough', 0.06, -3.65, 2.5)]
    return result

def flatten_dict(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items(): result.update(flatten_dict(child, f'{prefix}.{key}' if prefix else str(key)))
        return result
    return {prefix: value}

def prepare_provenance(args, out, cfg, agent):
    source = out/'provenance'; source.mkdir()
    shutil.copy2(__file__,source/'diagnose_used.py')
    for original in (ROOT/'src/as2_mjlab/tasks/ppo').rglob('*.py'):
        destination=source/'current_source'/original.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,destination)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    run = args.checkpoint.parent
    for filename in ('env.yaml', 'agent.yaml'):
        shutil.copy2(run/'params'/filename, source/f'training_{filename}')
    shutil.copy2(run/'git/as2_mjlab.diff', source/'training_git.diff')
    dump_yaml(source/'current_production_env.yaml', asdict(ppo_rough_env_cfg()))
    dump_yaml(source/'diagnostic_env.yaml', asdict(cfg))
    dump_yaml(source/'current_agent.yaml', asdict(agent))
    training = yaml.load((source/'training_env.yaml').read_text(), Loader=yaml.BaseLoader)
    current = yaml.load((source/'current_production_env.yaml').read_text(), Loader=yaml.BaseLoader)
    a,b = flatten_dict(training),flatten_dict(current)
    diff = {key:{'training':a.get(key), 'current':b.get(key)} for key in sorted(a.keys()|b.keys()) if a.get(key)!=b.get(key)}
    (source/'configuration_differences.json').write_text(json.dumps(diff,indent=2,ensure_ascii=False))
    # Reconstruct archived tracked source without a branch/worktree or repository mutation.
    text=(source/'training_git.diff').read_text(); commit=text.split('--- git commit ---\n')[1].splitlines()[0]
    reconstructed=source/'training_source'; reconstructed.mkdir()
    for name in ('src/as2_mjlab/tasks/ppo/env_cfg.py','src/as2_mjlab/tasks/ppo/mdp/rewards.py'):
        dest=reconstructed/name;dest.parent.mkdir(parents=True,exist_ok=True)
        old=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
        dest.write_bytes(old)
        parts=text.split('diff --git ')
        chunk=next((part for part in parts[1:] if part.startswith(f'a/{name} b/{name}\n')),None)
        if chunk:
            subprocess.run(['patch','-p1','--batch'],input=('diff --git '+chunk).encode(),cwd=reconstructed,check=True,capture_output=True)
    return checkpoint, {
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status_before':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'training_git_commit':commit, 'training_worktree_was_dirty':True,
        'checkpoint':str(args.checkpoint.resolve()), 'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'checkpoint_iteration':checkpoint['iter'], 'checkpoint_infos':checkpoint.get('infos'),
        'versions':{p:version(p) for p in ('mjlab','rsl-rl-lib','torch','mujoco','onnx')},
        'configuration_difference_count':len(diff),
        'archived_rewards_equal_current':(reconstructed/'src/as2_mjlab/tasks/ppo/mdp/rewards.py').read_bytes()==(ROOT/'src/as2_mjlab/tasks/ppo/mdp/rewards.py').read_bytes(),
        'untracked_training_commands_source_archived':False,
    }

class Recorder:
    def __init__(self, env):
        self.env=env;self.rows={};self.alive=np.ones(env.num_envs,dtype=bool)
        self.names=list(env.reward_manager.active_terms)
        self.weights=np.array([env.reward_manager.get_term_cfg(n).weight for n in self.names])
        assert np.all(self.weights!=0), 'Raw recovery requires nonzero weights'
        self.ids,_=env.scene['robot'].find_sites(LEGS, preserve_order=True)
        self.max_fc_error=0.;self.max_sum_error=0.
        self.original_compute=env.reward_manager.compute
        env.reward_manager.compute=self.compute
    def compute(self, dt):
        result=self.original_compute(dt)
        env=self.env;r=env.scene['robot'].data
        height=env.scene['foot_height_scan'].data.heights
        velocity=r.site_lin_vel_w[:,self.ids]
        feet=env.scene['feet_ground_contact'].data
        rate=env.reward_manager._step_reward.clone()
        cmd=env.command_manager.get_command('twist')
        fc=(height-.1).abs()*torch.linalg.vector_norm(velocity[...,:2],dim=-1)
        fc=fc*((torch.linalg.vector_norm(cmd[:,:2],dim=-1)+cmd[:,2].abs())>.1)[:,None]
        raw_fc=rate[:,self.names.index('foot_clearance')]/self.weights[self.names.index('foot_clearance')]
        self.max_fc_error=max(self.max_fc_error,(fc.sum(-1)-raw_fc).abs().max().item())
        self.max_sum_error=max(self.max_sum_error,(rate.sum(-1)*dt-result).abs().max().item())
        values={
            'root':torch.cat((r.root_link_pos_w,r.root_link_quat_w),dim=-1),
            'v_b':r.root_link_lin_vel_b,'v_w':r.root_link_lin_vel_w,'omega_b':r.root_link_ang_vel_b,
            'q':r.joint_pos,'qd':r.joint_vel,'action':env.action_manager.action,
            'foot_pos':r.site_pos_w[:,self.ids],'foot_vel':velocity,'clearance':height,
            'foot_contact':feet.found,'foot_force':feet.force,'fc_leg':fc,
            'command':cmd,'rate':rate,'total':result,
            'terms':torch.stack([env.termination_manager.get_term(n) for n in env.termination_manager.active_terms],-1),
            'nonfoot_force':torch.linalg.vector_norm(env.scene['nonfoot_ground_touch'].data.force_history,dim=-1).flatten(1).amax(-1),
            'thigh_force':torch.linalg.vector_norm(env.scene['thigh_ground_contact'].data.force_history,dim=-1).amax(-1),
            'rays_valid':(env.scene['foot_height_scan'].data.distances>=0).reshape(env.num_envs,4,5).all(-1),
        }
        for key,value in values.items():
            array=value.detach().cpu().numpy().copy()
            assert np.isfinite(array).all(),f'Nonfinite {key}'
            self.rows.setdefault(key,[]).append(array)
        self.rows.setdefault('alive',[]).append(self.alive.copy())
        return result
    def finish(self, out, success_step):
        arrays={key:np.stack(v) for key,v in self.rows.items()}
        arrays['success_step']=success_step
        arrays['reward_names']=np.array(self.names);arrays['weights']=self.weights
        arrays['termination_names']=np.array(self.env.termination_manager.active_terms)
        np.savez_compressed(out/'trajectory.npz',**arrays)
        return arrays

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--scenarios',default='all')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    selected=[x for x in cases() if args.scenarios=='all' or x[0] in args.scenarios.split(',')]
    assert selected
    conditions=[{'scenario':name,'terrain':kind,'dimension':dim,'start_x':start,'goal_x':goal,'vx':vx,'replicate':rep,
                 'difficulty':dim/.3 if 'pyramid' in kind else None,
                 'training_level_equivalent':9*dim/.3 if 'pyramid' in kind else None}
                for name,kind,dim,start,goal in selected for vx in (.6,1.) for rep in range(args.repeats)]
    cfg=ppo_rough_env_cfg();cfg.seed=args.seed;cfg.scene.num_envs=len(conditions)
    cfg.auto_reset=False;cfg.episode_length_s=args.seconds;cfg.curriculum={}
    for group in cfg.observations.values(): group.enable_corruption=False
    cfg.events.pop('push_robot')
    cfg.events.pop('reset_robot_joints');cfg.events['reset_base']=EventTermCfg(func=fixed_reset,mode='reset')
    for sensor in cfg.scene.sensors: sensor.debug_vis=False
    template=cfg.scene.terrain.terrain_generator
    terrain_cfgs={}
    for name,kind,dim,_,_ in selected:
        sub=deepcopy(template.sub_terrains[kind]);sub.proportion=1.
        if 'pyramid' in kind: sub.step_height_range=(dim,dim)
        elif kind=='hf_pyramid_slope':sub.slope_range=(dim,dim)
        elif kind=='random_rough':sub.noise_range=(.02,dim)
        terrain_cfgs[name]=sub
    cfg.scene.terrain=TerrainEntityCfg(terrain_type='generator',max_init_terrain_level=0,
        terrain_generator=TerrainGeneratorCfg(seed=args.seed,curriculum=True,size=(8.,8.),num_rows=1,num_cols=len(selected),border_width=4.,sub_terrains=terrain_cfgs,difficulty_range=(.5,.5),add_lights=False))
    cfg.commands={'twist':FixedCommandCfg(entity_name='robot',velocities=tuple(c['vx'] for c in conditions),
        resampling_time_range=(1e6,1e6),rel_standing_envs=0.,rel_heading_envs=0.,heading_command=False,debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.,0.),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.),heading=None))}
    agent=ppo_runner_cfg();saved,provenance=prepare_provenance(args,out,cfg,agent)
    print('Building',len(conditions),'environments',flush=True)
    env=ManagerBasedRlEnv(cfg,device=args.device)
    try:
        robot=env.scene['robot'];terrain=env.scene.terrain
        cols=torch.tensor([list(terrain_cfgs).index(c['scenario']) for c in conditions],device=env.device)
        centers=terrain.terrain_origins[0,cols].cpu().numpy()
        rng=np.random.default_rng(args.seed)
        pose=np.zeros((len(conditions),7),dtype=np.float32)
        yaws=rng.uniform(-.03,.03,len(conditions))
        initial_joint_noise=rng.uniform(-.015,.015,(len(conditions),robot.num_joints)).astype('float32')
        data=mujoco.MjData(env.sim.mj_model);mujoco.mj_forward(env.sim.mj_model,data)
        for i,c in enumerate(conditions):
            pose[i,:2]=centers[i,:2]+np.array([c['start_x']+rng.uniform(-.04,.04),rng.uniform(-.12,.12)])
            ray=np.array([pose[i,0],pose[i,1],5.]);geom=np.zeros(1,dtype=np.int32)
            distance=mujoco.mj_ray(env.sim.mj_model,data,ray,np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,geom)
            assert distance>=0,'Initial terrain ray missed'
            pose[i,2]=5.-distance+float(robot.data.default_root_state[i,2])
            goal_ray=np.array([centers[i,0]+c['goal_x'],centers[i,1],5.])
            goal_distance=mujoco.mj_ray(env.sim.mj_model,data,goal_ray,np.array([0.,0.,-1.]),np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,geom)
            c['goal_ground_height_m']=float(5.-goal_distance)
            c.update(env_id=i,patch_center=centers[i].tolist(),initial_pose_xyz=pose[i,:3].tolist(),initial_yaw=float(yaws[i]),step_height_m=c['dimension'] if 'pyramid' in c['terrain'] else None,step_width_m=.3 if 'pyramid' in c['terrain'] else None)
        env.diagnostic_pose=torch.as_tensor(pose,device=env.device)
        angles=torch.as_tensor(yaws,dtype=torch.float32,device=env.device)
        env.diagnostic_pose[:,3:]=quat_from_euler_xyz(torch.zeros_like(angles),torch.zeros_like(angles),angles)
        env.diagnostic_joints=robot.data.default_joint_pos+torch.as_tensor(initial_joint_noise,device=env.device)
        # Actual startup DR buffers, not merely their configured ranges.
        actual_dr={'encoder_bias':robot.data.encoder_bias.cpu().numpy()}
        for name in ('geom_friction','body_ipos'):
            value=getattr(env.sim.model,name)
            value=value.cpu().numpy().copy()
            actual_dr[name]=value
        np.savez_compressed(out/'startup_dr.npz',**actual_dr)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=MjlabOnPolicyRunner(wrapped,asdict(agent),device=args.device)
        runner.load(str(args.checkpoint),load_cfg={'actor':True},strict=True,map_location=args.device)
        policy=runner.get_inference_policy(device=args.device)
        state={k:v.clone() for k,v in policy.state_dict().items()}
        recorder=Recorder(env)
        success_step=np.full(env.num_envs,-1,dtype=np.int32)
        obs=wrapped.get_observations()
        start=time.monotonic()
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                expected=torch.tensor([c['vx'] for c in conditions],device=env.device)
                command=env.command_manager.get_command('twist')
                assert torch.allclose(command[:,0],expected) and (command[:,1:]==0).all()
                action=policy(obs)
                raw_obs,reward,terminated,truncated,_=env.step(action)
                done=(terminated|truncated).cpu().numpy()
                root=recorder.rows['root'][-1]
                reached=(root[:,0]-centers[:,0]>=np.array([c['goal_x'] for c in conditions])) & (np.abs(root[:,1]-centers[:,1])<1.)
                reached &= (1.-2.*(root[:,4]**2+root[:,5]**2)>np.cos(np.pi/4))
                reached &= root[:,2]>np.array([c['goal_ground_height_m'] for c in conditions])+.20
                reached &= recorder.alive & ~done
                success_step[reached]=step
                recorder.alive &= ~done
                recorder.alive &= ~reached
                if not recorder.alive.any():break
                if done.any():
                    env.reset(env_ids=torch.as_tensor(np.flatnonzero(done),device=env.device))
                obs=wrapped.get_observations()
                if step%100==0:print('step',step,'remaining first episodes',int(recorder.alive.sum()),'elapsed',round(time.monotonic()-start,1),flush=True)
        assert not policy.training
        assert all(torch.equal(state[k],v) for k,v in policy.state_dict().items())
        arrays=recorder.finish(out,success_step)
        assert recorder.max_fc_error<1e-5 and recorder.max_sum_error<1e-5
        metadata={**provenance,'seed':args.seed,'device':args.device,'episode_seconds':args.seconds,'dt':env.step_dt,
                  'conditions':conditions,'repeats':args.repeats,'reward_weights':dict(zip(recorder.names,recorder.weights.tolist())),
                  'reward_timing':'captured inside compute before post-step forward, identical one-physics-substep derived-state lag as production rewards',
                  'raw_rewards':'recovered from RewardManager._step_reward / weight; no repeated reward calls; all weights nonzero',
                  'scale_rewards_by_dt':cfg.scale_rewards_by_dt,'reward_total_max_abs_error':recorder.max_sum_error,
                  'foot_clearance_formula_max_abs_error':recorder.max_fc_error,
                  'domain_randomization_enabled':['foot_friction startup (0.3,1.6) shared','encoder_bias startup (-0.015,0.015)','base_com startup +/-0.05m'],
                  'domain_randomization_disabled':['observation noise','push_robot'],
                  'startup_dr_file':'startup_dr.npz','initial_state_variation':'yaw +/-0.03 rad; lateral +/-0.12 m; longitudinal +/-0.04 m; joint position +/-0.015 rad',
                  'auto_reset':False,'sampling':'one first episode per environment; dead envs reset only to satisfy manual-reset contract and are excluded thereafter',
                  'policy_deterministic':True,'policy_state_unchanged':True,'wall_seconds':time.monotonic()-start}
        metadata['success_rule']='reach goal_x within 1 m of patch centerline, tilt <45deg, base >goal ground+0.20m, before native failure; stop recording at success to avoid later terrain-border falls'
        (out/'config.yaml').write_text(yaml.safe_dump(metadata,sort_keys=False,allow_unicode=True))
        print('COLLECTION COMPLETE',out,arrays['alive'].sum(),'samples',flush=True)
    finally: env.close()

if __name__=='__main__': main()
