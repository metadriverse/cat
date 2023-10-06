import argparse
import numpy as np
from tqdm import trange
import time
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from advgen.adv_generator import AdvGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--OV_traj_num', type=int,default=32)
    parser.add_argument('--AV_traj_num', type=int,default=1)
    adv_generator = AdvGenerator(parser)

    args = parser.parse_args()

    extra_args = dict(mode="top_down", film_size=(2200, 2200))

    env = WaymoEnv(
            {
                "agent_policy": ReplayEgoCarPolicy,
                "reactive_traffic": False,
                "use_render": False,
                "data_directory": './raw_scenes_500',
                "num_scenarios": 500,
                "force_reuse_object_name" :True,
                "sequential_seed": True,
                "vehicle_config":dict(show_navi_mark=False,show_dest_mark=False,)
            }
        )

    attack_cnt = 0
    time_cost = 0.

    pbar = trange(500)
    for i in pbar:

      ######################## First Round : log the normal scenario ########################
      env.reset(force_seed=i)
      
      
      done = False
      ep_timestep = 0
      adv_generator.before_episode(env)   # initialization before each episode

      env.render(**extra_args)
      env.engine._top_down_renderer.set_adv(adv_generator.adv_agent)


      while True:
        adv_generator.log_AV_history()    # log the ego car's states at every step
        o, r, done, info = env.step([1.0, 0.]) # replace it with your own controller
        env.render(**extra_args,text={'Replay': 'Raw Scenario'})
        ep_timestep += 1
        if done:
          adv_generator.after_episode()   # post-processing after each episode
          break
      
      
      ################ Second Round : create the adversarial counterpart #####################
      
      env.reset(force_seed=i)
      env.vehicle.ego_crash_flag = False
      done = False
      ep_timestep = 0

      t0 = time.time()

      adv_generator.before_episode(env)   # initialization before each episode
      adv_generator.generate()            # Adversarial scenario generation with the logged history corresponding to the current env 
      env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj) # set the adversarial traffic
      
      t1 = time.time()
      time_cost += t1 - t0
      
      while True:
        o, r, done, info = env.step([1.0, 0.]) # replace it with your own controller
        env.render(**extra_args,text={'Generate': 'Safety-Critical Scenario'})
        ep_timestep += 1
        crash = env.vehicle.ego_crash_flag
        if done or crash:
          if crash:
            attack_cnt += 1
          adv_generator.after_episode()    # post-processing after each episode
          pbar.set_postfix(avg_attack_success_rate=attack_cnt/(i+1),avg_compute_time=time_cost/(i+1)) # benchmarking the attack success rate and computational time
          break

    env.close()