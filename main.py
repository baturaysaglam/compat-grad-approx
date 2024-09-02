import argparse
import yaml

import gym

from algos import TD3, CPG, SAC
from utils.train import *


# Create a dictionary to map the parameter strings to the algo
agent_map = {
    "TD3": TD3.TD3,
    "SAC": SAC.SAC,
    "CPG": CPG.CPG,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compatible Gradient Approximations for Actor-Critic Algorithms')

    # Agent/Env setup
    parser.add_argument("--policy", default="CPG", help='Algorithm (default: CPG)')
    parser.add_argument("--env", default="Walker2d-v3", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=100, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym')

    # Imperfect environment conditions
    parser.add_argument("--reward_type", default=None, type=str, help='Reward type', choices=['sparse', 'delayed', 'noisy'])
    parser.add_argument("--delay_steps", default=None, type=int, help='Delay steps for the delayed rewards')
    parser.add_argument("--reward_prob", default=None, type=float, help='Probability of seeing sparse rewards rewards')

    # CPG parameters
    parser.add_argument("--mu", default=0.1, type=float, help='Std of Gaussian exploration noise and the CPG parameter')
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Actor learning rate')

    # Training setup
    parser.add_argument("--max_time_steps", default=1e6, type=float, help='Maximum number of steps')

    # CUDA
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers')

    # Model tracking
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    with open('params.yaml', 'r') as file:
        # Load the YAML content
        yaml_data = yaml.safe_load(file)
        args_dict.update(yaml_data)

    # Make sure to delete these since we don't treat them as a parameter
    del args_dict["env"]
    del args_dict["reward_type"]
    del args_dict["delay_steps"]
    del args_dict["reward_prob"]
    del args_dict["max_time_steps"]
    del args_dict["seed"]
    del args_dict["gpu"]
    del args_dict["save_model"]
    del args_dict["load_model"]

    gym_env_name = gym_eval_env_name = args.env

    # Set the reward-modifier function
    if args.reward_type is None:
        reward_func = identity_func
    elif args.reward_type == "sparse":
        args.env = "Sparse" + args.env
        if args.reward_prob is None:
            raise ValueError("Reward probability has to be specified")
        else:
            reward_func = RandomSparseRewardTracker(reward_probability=args.reward_prob)
    elif args.reward_type == "delayed":
        args.env = "Delayed" + args.env
        if args.delay_steps is None:
            raise ValueError("Delay steps has to be specified")
        else:
            reward_func = DelayedRewardTracker(delay_steps=args.delay_steps)
    elif args.reward_type == "noisy":
        args.env = "Noisy" + args.env
        reward_func = NoisyRewardTracker()
    else:
        raise NotImplementedError("Unknown reward type")

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    save_dir = "./results"

    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")

    save_dir = get_save_dir(args_dict, save_dir, args.env)
    model_dir = f"{save_dir}/models"

    if args.save_model and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(gym_env_name)
    eval_env = gym.make(gym_eval_env_name)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_space": env.action_space,
        "gamma": args_dict['gamma'],
        "tau": args_dict['tau'],
        "actor_lr": args_dict['actor_lr'],
        "critic_lr": args_dict['critic_lr'],
        "device": device
    }

    if any(policy in args.policy for policy in ["TD3", "CPG"]):
        kwargs["policy_noise"] = args_dict['policy_noise']
        kwargs["noise_clip"] = args_dict['noise_clip']
        kwargs["policy_freq"] = args_dict['policy_freq']

        reward_scale = 1.0
    elif "SAC" in args.policy:
        kwargs["alpha_lr"] = args_dict['alpha_lr']
        kwargs["automatic_entropy_tuning"] = args_dict['automatic_entropy_tuning']

        reward_scale = 20.0 if "Humanoid" in args.env else 5.0
    else:
        raise NotImplementedError("No other algorithm has been implemented yet")

    # Initialize the algorithm
    agent = agent_map[args.policy](**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, max_size=int(args_dict['buffer_size']), device=device)

    # Evaluate the untrained policy
    evaluations = [evaluate_policy(agent, eval_env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_time_steps = 0
    episode_num = 0

    for t in range(int(args.max_time_steps)):
        episode_time_steps += 1

        # Sample action from the action space or policy
        if t < args_dict['start_time_steps']:
            action = env.action_space.sample()
            noise, noise_std = np.ones_like(action), np.ones_like(action)
        else:
            action, noise, noise_std = select_action(agent, state, env.action_space, args_dict['mu'], stochastic=True)

        # Take the selected action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_time_steps <= env._max_episode_steps else 0

        episode_reward += reward

        # Modify the reward to mimic imperfect environment conditions
        reward = reward_func(reward, done) if args.reward_type == "noisy" else reward_func(reward)

        # Add the corresponding mu and noise as well
        scaled_reward = reward_scale * reward

        replay_buffer.add(state, action, next_state, scaled_reward, done_bool, noise, noise_std)

        state = next_state

        # Train the agent after collecting a sufficient number of samples
        if t >= args_dict['start_time_steps']:
            agent.update_parameters(replay_buffer, args_dict['batch_size'])

        if done:
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Reward: {episode_reward:.3f}")

            # Reset the environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_time_steps = 0
            episode_num += 1

        # Evaluate the agent over a number of episodes
        if (t + 1) % args_dict['eval_freq'] == 0:
            evaluations.append(evaluate_policy(agent, eval_env, args.seed))
            np.save(f"{save_dir}/{file_name}", evaluations)

            if args.save_model:
                agent.save(f"./{model_dir}/{file_name}")
