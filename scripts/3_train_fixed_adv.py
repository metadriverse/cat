import numpy as np
import torch
import gym
import argparse
import os

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from advgen.adv_generator import AdvGenerator

from saferl_algo import TD3,utils
from saferl_plotter.logger import SafeLogger

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, adv_generator, eval_episodes=100):
	
	_rewards = [0.] * eval_episodes
	_costs = [0.] * eval_episodes

	for ep_num in range(eval_episodes):
		state, done = eval_env.reset(), False
		adv_generator.before_episode(eval_env)
		while not done:
			adv_generator.log_AV_history()
			action = policy.select_action(np.array(state))
			state, reward, done, info = eval_env.step(action)

		_rewards[ep_num] = info['route_completion']
		_costs[ep_num] = float(eval_env.vehicle.crash_vehicle)
		adv_generator.after_episode(update_AV_traj=True,mode='eval')

	avg_reward_normal = sum(_rewards) / eval_episodes
	avg_cost_normal = sum(_costs) / eval_episodes


	_rewards = [0.] * eval_episodes
	_costs = [0.] * eval_episodes

	for ep_num in range(eval_episodes):
		state, done = eval_env.reset(), False
		adv_generator.before_episode(eval_env)
		adv_generator.generate(mode='eval')
		env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj)
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, info = eval_env.step(action)

		_rewards[ep_num] = info['route_completion']
		_costs[ep_num] = float(eval_env.vehicle.crash_vehicle)

	avg_reward_adv = sum(_rewards) / eval_episodes
	avg_cost_adv = sum(_costs) / eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: Reward_normal {avg_reward_normal:.3f} Cost_normal {avg_cost_normal: .3f} Reward_adv {avg_reward_adv:.3f} Cost_adv {avg_cost_adv: .3f}")
	print("---------------------------------------")
	return avg_reward_normal,avg_cost_normal,avg_reward_adv,avg_cost_adv


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_name", default='RLTrainADV',type=str)
	parser.add_argument("--env", default="MDWaymo")          
	parser.add_argument("--start_timesteps", default=10000, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=25000, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument('--OV_traj_num', type=int,default=32)
	parser.add_argument('--AV_traj_num', type=int,default=1)

	adv_generator = AdvGenerator(parser)
	args = parser.parse_args()

	
	file_name = args.exp_name
	logger = SafeLogger(exp_name=args.exp_name, env_name=args.env, seed=args.seed,
						fieldnames=['route_completion_normal','crash_rate_normal','route_completion_adv','crash_rate_adv'])

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	config_train = dict(
				data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
				start_scenario_index = 0,
				num_scenarios=400,
				sequential_seed = False,
				force_reuse_object_name = True,
				horizon = 50,
				no_light = True,
				no_static_vehicles = True,
				reactive_traffic = False,
				vehicle_config=dict(
					lidar = dict(num_lasers=30,distance=50, num_others=3),
					side_detector = dict(num_lasers=30),
					lane_line_detector = dict(num_lasers=12)),
			)
	
	config_test = dict(
				data_directory=os.path.join(os.path.dirname(__file__), "./raw_scenes_500"),
				start_scenario_index = 400,
				num_scenarios = 100,
				crash_vehicle_done=True,
				sequential_seed = True,
				force_reuse_object_name = True,
				horizon = 50,
				no_light = True,
				no_static_vehicles = True,
				reactive_traffic = False,
				vehicle_config=dict(
					lidar = dict(num_lasers=30,distance=50, num_others=3),
					side_detector = dict(num_lasers=30),
					lane_line_detector = dict(num_lasers=12)),
			)

	# Set seeds
	env = WaymoEnv(config=config_train)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq
	policy = TD3.TD3(**kwargs)
	

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	

	state, done = env.reset(), False
	adv_generator.before_episode(env)
	episode_reward = 0
	episode_cost = 0
	episode_timesteps = 0
	episode_num = 0

	last_eval_step= 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		adv_generator.log_AV_history()

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		try:
			next_state, reward, done, info = env.step(action)
		except:
			done = True
			print('!!!!!!!!!!!!!Step Bug!!!!!!!!!!!!!!')
		done_bool = float(done)

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward
		episode_cost += info['cost']

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done:
			adv_generator.after_episode(update_AV_traj=False)

			print('#'*20)
			print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
			print(f"arrive destination: {info['arrive_dest']} , route_completion: {info['route_completion']}, out of road:{info['out_of_road']}  ")
			
			# Evaluate episode
			if t - last_eval_step > args.eval_freq:
				last_eval_step = t
				env.close()
				eval_env = WaymoEnv(config=config_test)
				evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv = eval_policy(policy, eval_env, adv_generator)
				eval_env.close()
				logger.update([evalRC_normal, evalCrash_normal, evalRC_adv, evalCrash_adv], total_steps=t + 1)
				
				env = WaymoEnv(config=config_train)

				if args.save_model: policy.save(f"./models/{file_name}")
			
			# Reset environment
			try:
				state, done = env.reset(), False
			except:
				state, done = env.reset(force_seed=0), False
				print('!!!!!!!!!!!!!Reset Bug!!!!!!!!!!!!!!')
			adv_generator.before_episode(env)

			if np.random.random() > max(1-(t/500000)*(1-0.1),0.1):
				print('ADVGEN')
				adv_generator.generate()
				env.engine.traffic_manager.set_adv_info(adv_generator.adv_agent,adv_generator.adv_traj)
			else:
				print('NORMAL')
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 
