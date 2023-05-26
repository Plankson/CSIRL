import pickle

import hydra
import types
import torch
import os
import random
import gym
import torch.nn.functional as F
import utils.util as util
import itertools
import numpy as np
from tensorboardX import SummaryWriter
from itertools import count
from make_envs import make_env
from omegaconf import DictConfig, OmegaConf
from dataset.rs_memory import Memory
from dataset.load_data import Dataset
from torch.autograd import Variable
from model.sac_rs import SAC_RS
torch.set_num_threads(2)
cur_pth = os.getcwd()
def get_args(cfg: DictConfig):
    # cfg.device = "cpu"
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    args.agent.obs_dim = obs_dim
    args.agent.action_dim = action_dim
    agent = SAC_RS(obs_dim, action_dim, action_range, args.train.batch, args)
    return agent



def get_re_obs(obs):
    re_obs = np.array(obs)
    sz = re_obs.shape
    for i in range(1,sz[0]):
        re_obs[i]=re_obs[i]-re_obs[0]
    return re_obs

def save(agent,args,cnt):
    output_dir=f'{args.env.name}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/{args.agent.name}_{cnt}')
    print("saved successfully!")

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_args=args.env
    env = make_env(args)
    eval_env = make_env(args)
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)
    print(cur_pth)
    dataset_0=Dataset(cur_pth, args)
    g1 = int(env_args.g1)
    REPLAY_MEMORY = int(env_args.replay_mem)        # total buffer size
    INITIAL_MEMORY = int(env_args.initial_mem)      # buffer size that can start learning
    EPISODE_STEPS = int(env_args.eps_steps)         # maximum epoch_step number
    ROUND_LEARN_STEPS = int(env_args.round_steps)
    LEARN_STEPS = ROUND_LEARN_STEPS*dataset_0.expert_data["lengths"][0]  # maximum learning_step number
    agent = make_agent(env, args)
    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    learn_step = 0
    all_step = 0
    sg_count = 0
    writer = SummaryWriter(log_dir="./logs")
    output_dir=f'./data/{args.env.name}/CSIRL/{dataset_0.get_tra_num()}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = output_dir + f'/{args.seed}.pkl'
    test_reward = []
    test_step = []
    for _1 in count():
        sg_count += 1
        save(agent, args, sg_count)
        print("| subgoal count %d |" %(sg_count))
        online_memory_replay.clear()
        begin_learn = False
        goal_learn_step = 0
        for __ in count():
            if goal_learn_step > ROUND_LEARN_STEPS:
                break
            state = env.reset()
            episode_reward = 0
            done = False
            #print(_)
            train_reward = -999.9
            for episode_step in range(EPISODE_STEPS):
                # env.render()
                if learn_step % args.env.eval_interval == 1 and begin_learn == True:
                    eval_returns, eval_timesteps = util.evaluate(agent, eval_env, num_episodes=args.eval.eps)
                    returns = np.mean(eval_returns)
                    writer.add_scalar('eval/episode_reward', returns, learn_step)
                    test_step.append(learn_step)
                    test_reward.append(returns)
                    print("| test | steps: %2d | episode_reward: %.3f |" %(learn_step,returns))
                    record_data = {"steps": test_step, "rewards": test_reward}
                    torch.save(record_data, output_dir)
                if all_step < args.num_seed_steps:
                    # Seed replay buffer with random actions
                    action = env.action_space.sample()
                else:
                    with util.eval_mode(agent):
                        action = agent.choose_action(state, sample=True)
                next_state, reward, done, _ = env.step(action)
                train_reward = max(train_reward, -_["dis"])
                re_obs = get_re_obs(state)
                reward1= util.get_matching_reward(state, next_state, dataset_0, agent.get_reward(torch.tensor(re_obs)), g1, args)
                done_no_lim = done
                if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                    done_no_lim = 0
                online_memory_replay.add((state,next_state, action, re_obs, reward1, done_no_lim))
                if online_memory_replay.size() > INITIAL_MEMORY:
                    if begin_learn is False:
                        print('Learn begins!')
                        begin_learn = True

                    goal_learn_step += 1
                    learn_step += 1
                    agent.update(online_memory_replay, dataset_0, writer, learn_step)
                    if learn_step == LEARN_STEPS:
                        print('Finished!')
                        writer.close()
                        record_data = {"steps":test_step, "rewards": test_reward}
                        print(output_dir)
                        torch.save(record_data,output_dir)
                        return
                if done:
                    break
                state = next_state
            if begin_learn:
                writer.add_scalar('train/reward',train_reward,learn_step)
                print("\n| train | steps: %2d | episode_reward: %.3f |" %(learn_step,train_reward))
        eval_returns, eval_timesteps = util.evaluate(agent, eval_env, num_episodes=args.eval.eps)
        returns = np.mean(eval_returns)
        writer.add_scalar('eval/episode_reward', returns, learn_step)
        test_step.append(learn_step)
        test_reward.append(returns)
        print("| test | steps: %2d | episode_reward: %.3f |" %(learn_step,returns))
        dataset_0.select_subgoal(agent, args)

    writer.close()
if __name__ == "__main__":
    main()