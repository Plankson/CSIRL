import numpy as np
import glob
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

def write_to_file(dname, dmap, cmap, itr):
    fid = open(dname + '-results.log', 'a+')
    string_to_write = str(itr)
    for item in dmap:
        string_to_write += ' ' + '%.2f' % item
    string_to_write += ' ' + '%.2f' % cmap
    fid.write(string_to_write + '\n')
    fid.close()


def get_labels(seq_len, n_subgoals):
    # Equi-partition labels
    stops = np.array(range(1, n_subgoals + 1)).astype('float32') / n_subgoals
    labels = np.zeros((seq_len, len(stops)), dtype=float)
    prev_idx = 0
    for i, stop in enumerate(stops):
        idx = int(seq_len * stop)
        labels[prev_idx:idx, i] = 1.
        prev_idx = idx
    return labels


def dist(a, b):
    return np.sum(np.abs(a - b))
class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def get_matching_reward(s, next_s, tra_dataset, reward_w, g1, args):
    _s = np.expand_dims(s,axis=0)
    id, sg = tra_dataset.find_subgoal(_s)
    sg = np.squeeze(sg, axis=0)
    h1 = np.linalg.norm(s[args.env.l_pos:args.env.r_pos+1]-sg[args.env.l_pos:args.env.r_pos+1])
    h2 = np.linalg.norm(next_s[args.env.l_pos:args.env.r_pos+1]-sg[args.env.l_pos:args.env.r_pos+1])
    reward_m = ( h1 - h2 ) * g1
    return reward_m


def evaluate(actor, env, num_episodes=10, vis=True):
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False
        info={}
        ret = -999.9
        with eval_mode(actor):
            while not done:
                action = actor.choose_action(state, sample=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                ret = max(ret, -info['dis'])
        total_returns.append(ret)
    return total_returns, total_timesteps
