import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
from torch.autograd import Variable, grad


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def orthogonal_init_(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
def update_module(module, updates=None, memo=None):
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memo:
                module._parameters[param_key] = memo[p]
            else:
                updated = p + p.update
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memo:
                module._buffers[buffer_key] = memo[buff]
            else:
                updated = buff + buff.update
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module
def clone_module(module, memo=None):
    if memo is None:
        memo = {}
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone

def manual_update(model, lr, grads=None):
    if grads is not None:
        params = list(model.parameters())
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)
def update_module(module, updates=None, memo=None):
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memo:
                module._parameters[param_key] = memo[p]
            else:
                updated = p + p.update
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memo:
                module._buffers[buffer_key] = memo[buff]
            else:
                updated = buff + buff.update
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    # Finally, rebuild the flattened parameters for RNNs
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk
class SimpleAgent(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim, hidden_depth, action_range, device, output_mod=None):
        super(SimpleAgent, self).__init__()
        self.actor = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None)
        self.action_range = action_range
        self.device=device
    def forward(self, states):
         return self.actor(states)
    def choose_action(self,state,sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = self.actor(state)
        action = action.clamp(*self.action_range)
        return action.detach().cpu().numpy()[0]

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action, both=True)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class MultiQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, q_net_num, args):
        super(MultiQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.q_net_num=q_net_num
        self.args = args
        # Q architecture
        Q_list = [mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth) for x in range(self.q_net_num)]
        self.Q = nn.ModuleList(Q_list)
        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = [self.Q[i](obs_action) for i in range(self.q_net_num)]

        if self.args.method.tanh:
            q=[torch.tanh(q[i]) * 1/(1-self.args.gamma) for i in range(self.q_net_num)]
        if both:
            return q
        else:
            for i in range(1, self.q_net_num):
                q[0] = torch.min(q[0], q[i])
            return q[0]
    def get_UC(self,obs,action):

        obs_action = torch.cat([obs, action], dim=-1)
        q = [self.Q[i](obs_action) for i in range(self.q_net_num)]

        if self.args.method.tanh:
            q=[torch.tanh(q[i]) * 1/(1-self.args.gamma) for i in range(self.q_net_num)]
        q_v = [q[i].detach().tolist() for i in range(self.q_net_num)]
        q_v =np.array(q_v)         # Q * B * 1
        q_v = np.squeeze(q_v,axis=2).swapaxes(0,1)    # B * Q
        var = np.var(q_v,axis=1)
        mean = np.mean(q_v,axis=1)
        return np.absolute(np.true_divide(var,mean))
    # def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
    #     expert_data = torch.cat([obs1, action1], 1)
    #     policy_data = torch.cat([obs2, action2], 1)
    #
    #     alpha = torch.rand(expert_data.size()[0], 1)
    #     alpha = alpha.expand_as(expert_data).to(expert_data.device)
    #
    #     interpolated = alpha * expert_data + (1 - alpha) * policy_data
    #     interpolated = Variable(interpolated, requires_grad=True)
    #
    #     interpolated_state, interpolated_action = torch.split(
    #         interpolated, [self.obs_dim, self.action_dim], dim=1)
    #     q = self.forward(interpolated_state, interpolated_action, both=True)
    #     ones = torch.ones(q[0].size()).to(policy_data.device)
    #     gradient = grad(
    #         outputs=q,
    #         inputs=interpolated,
    #         grad_outputs=[ones, ones],
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #     )[0]
    #     grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
    #     return grad_pen

class DoubleQCriticMax(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCriticMax, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)
        
        if both:
            return q1, q2
        else:
            return torch.max(q1, q2)


class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(SingleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q architecture
        self.Q = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)

        if self.args.method.tanh:
            q = torch.tanh(q) * 1/(1-self.args.gamma)

        return q

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = torch.cat([obs1, action1], 1)
        policy_data = torch.cat([obs2, action2], 1)

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q.size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class DoubleQCriticState(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        # Q1 architecture
        self.Q1 = mlp(obs_dim, hidden_dim, 1, hidden_depth)

        # Q2 architecture
        self.Q2 = mlp(obs_dim, hidden_dim, 1, hidden_depth)

        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        assert obs.size(0) == action.size(0)

        q1 = self.Q1(obs)
        q2 = self.Q2(obs)

        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)

        if both:
            return q1, q2
        else:
            return torch.min(q1, q2)

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = obs1
        policy_data = obs2

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q[0].size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=[ones, ones],
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (gradient.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds):
        super().__init__()

        self.device="cpu"
        self.action_range=[-1.0,1.0]
        self.log_std_bounds = log_std_bounds
        self.trunk = mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(orthogonal_init_)

    def forward(self, obs):
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        if torch.isnan(mu).any():
            print("!")
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def log_prob(self, obs, action):
        dist = self.forward(obs)
        if torch.isnan(dist.log_prob(action)).any():
            print("!")
        return dist.log_prob(action).sum(-1, keepdim=True)

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.forward(state)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach().cpu().numpy()[0][0]

class Intrinsic_Reward_Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, args, output_dim=1):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim,
                         hidden_depth)
        self.apply(orthogonal_init_)

    def forward(self, obs):
        x = self.trunk(obs)
        x = torch.tanh(x)
        return x