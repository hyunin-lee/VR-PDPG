import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
import gym

import numpy as np

class Agent(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.state_layer1 = self._layer_init(nn.Linear(num_states, 32))
        # self.state_layer2 = self._layer_init(nn.Linear(64, 256))
        # self.state_layer3 = self._layer_init(nn.Linear(256, 64))
        self.actor = self._layer_init(nn.Linear(32, num_actions), std=0.01)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, 'bias'):
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_action(self, x, action=None, return_prob=False):
        state_latent = F.relu(self.state_layer1(x))
        # state_latent = F.relu(self.state_layer2(state_latent))
        # state_latent = F.relu(self.state_layer3(state_latent))
        logits = self.actor(state_latent)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        # log_probs = probs.log_prob(action)

        return action, torch.log(probs.probs)

def get_gradient_input_reward(reward,gamma,obs_buffer,action_buffer,agent_,optimizer_,episode,num_states,requires_grad=True):
    optimizer_.zero_grad()
    scores = calculate_scores(reward, gamma) # computes f(h) = \sum_{k=h}^{H-1} \gamma^k r(s_k,a_k)
    # scores is a $H$ length - vector with [f(0),f(1),...,f(H-1)]
    batch_input = F.one_hot(obs_buffer[episode, :].type(torch.long), num_classes=num_states).type(torch.float)
    if requires_grad :
        _, batch_logprob = agent_.get_action(Variable(batch_input))
        selected_logprob = batch_logprob.gather(1, Variable(action_buffer[episode, :].long().view(-1, 1))) # log \pi(a | s)
        # selected_logprob is $H$ length vector with [log pi(a0|s0), log pi (a1|s1), .., log pi (a_{h-1}|s_{h-1}}]
    else :
        _, batch_logprob = agent_.get_action(batch_input)
        selected_logprob = batch_logprob.gather(1, action_buffer[episode, :].long().view(-1, 1))
    loss = - torch.sum(selected_logprob * scores)
    loss.backward()
    d = get_flat_grads_from(agent_)

    return d
def set_flat_grads_to(model, flat_grads):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad = Variable(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grads_from(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.data.view(-1))
    flat_grads = torch.cat(grads)
    return flat_grads


def set_flat_grads_to(model, flat_grads):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.grad = Variable(flat_grads[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
def change_state_action_dim(state,action,num_actions) :
    return num_actions * state + action

def state_to_xy(state) :
    x = state // 8
    y = state % 8
    return y,x

def show_trajecgory(index_list,episode,save_foldername):
    fig, ax = plt.subplots()
    # Create an 8x8 grid
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.grid(True)

    # Fill specified cells with color
    for index_state in index_list:
        index = state_to_xy(index_state)
        print(index)
        rect = plt.Rectangle(index, 1, 1, color='blue', alpha=0.5)
        ax.add_patch(rect)
    plt.title(episode)
    plt.gca().invert_yaxis()  # Invert Y axis to match grid indexing
    plt.savefig(save_foldername + "/trajectory.png")
    plt.show()


def get_target_occupancy_measure_20x20(num_states,num_actions,gamma):
    # the (20,20) grid looks like
    # 0 1 2 ... 19
    # 20 9 10 ...27
    # ...
    # the action looks like
    target_sa = [(0,1),(20,1),(40,1),(60,1),(80,2),(81,2),(82,2),(83,2),(84,2),(85,2),(86,1),(106,1),(126,2),
                 (127,1),(147,2),(148,1),(168,1),(188,1),(208,2),(209,2),(210,1),(230,1),(250,1),(270,1),(290,2),
                 (291,2), (292,2),(293,2),(294,1),(314,2),(315,1),(335,1),(355,2),(356,1),(376,2),(377,2),(378,1),
                 (398,2)] #traj1
    target_sa = [(0,1),(20,2),(21,1),(41,2),(42,1),(62,2),(63,1),(83,2),(84,1),(104,2),(105,1),(125,2),(126,1),
                 (146,2),(147,1),(167,2),(168,1),(188,2),(189,1),(209,2),(210,1),(230,2),(231,1),(251,2),(252,1),
                 (272,2),(273,1),(293,2),(294,1),(314,2),(315,1),(335,2),(336,1),(356,2),(357,1),(377,2),(378,1),
                 (398,2)]
    with torch.no_grad():
        state_action_input = change_state_action_dim(torch.tensor(target_sa[-1][0]),
                                                     torch.tensor(target_sa[-1][1]), num_actions)
        occupancy_measure = F.one_hot(state_action_input.type(torch.long), num_classes=num_states * num_actions)

        for t in reversed(range(len(target_sa)-1)):

            state_action_input = change_state_action_dim(torch.tensor(target_sa[t][0]), torch.tensor(target_sa[t][1]), num_actions)
            occupancy_measure = F.one_hot(state_action_input.type(torch.long),
                                          num_classes=num_states * num_actions) + gamma * occupancy_measure  # compute line 4
    return occupancy_measure


def get_target_occupancy_measure(num_states,num_actions,gamma):
    # the (8,8) grid looks like
    # 0 1 2 ... 7
    # 8 9 10 ...15
    # ...
    # the action looks like
    # target_sa = [(0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,1),(15,1),(23,1),(31,1),(39,1),(47,1),(55,1)] #traj1
    target_sa = [(0,1),(8,2),(9,1),(17,2),(18,1),(26,2),(27,1),(35,2),(36,1),(44,2),(45,1),(53,2),(54,1),(62,2)] #traj2

    with torch.no_grad():
        state_action_input = change_state_action_dim(torch.tensor(target_sa[-1][0]),
                                                     torch.tensor(target_sa[-1][1]), num_actions)
        occupancy_measure = F.one_hot(state_action_input.type(torch.long), num_classes=num_states * num_actions)

        for t in reversed(range(len(target_sa)-1)):

            state_action_input = change_state_action_dim(torch.tensor(target_sa[t][0]), torch.tensor(target_sa[t][1]), num_actions)
            occupancy_measure = F.one_hot(state_action_input.type(torch.long),
                                          num_classes=num_states * num_actions) + gamma * occupancy_measure  # compute line 4
    return occupancy_measure

def get_target_occupancy_measure_mountaincar(num_states,num_actions,gamma, sa_oracle, states_low, states_high, n_discretize):
    # the (8,8) grid looks like
    # 0 1 2 ... 7
    # 8 9 10 ...14
    # ...
    # the action looks like
    target_sa = []
    for l in sa_oracle :
        s,a = l[0], l[1]
        discretize_s = discretization(s,states_low,states_high,n_discretize)
        discretize_a = a
        target_sa.append((discretize_s,discretize_a))
    # target_sa = [(0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,1),(15,1),(23,1),(31,1),(39,1),(47,1),(55,1)]
    with torch.no_grad():
        state_action_input = change_state_action_dim(torch.tensor(target_sa[-1][0]),
                                                     torch.tensor(target_sa[-1][1]), num_actions)
        occupancy_measure = F.one_hot(state_action_input.type(torch.long), num_classes=num_states * num_actions)

        for t in reversed(range(len(target_sa)-1)):

            state_action_input = change_state_action_dim(torch.tensor(target_sa[t][0]), torch.tensor(target_sa[t][1]), num_actions)
            occupancy_measure = F.one_hot(state_action_input.type(torch.long),
                                          num_classes=num_states * num_actions) + gamma * occupancy_measure  # compute line 4
    return occupancy_measure

def get_occupancy_measure(obs_buffer, action_buffer, episode,final_step, num_states,num_actions,gamma) :
    with torch.no_grad():
        state_action_input = change_state_action_dim(obs_buffer[episode, final_step],
                                                     action_buffer[episode, final_step], num_actions)
        occupancy_measure = F.one_hot(state_action_input.type(torch.long), num_classes=num_states * num_actions)
        for t in reversed(range(final_step)):
            state_action_input = change_state_action_dim(obs_buffer[episode, t], action_buffer[episode, t], num_actions)
            occupancy_measure = F.one_hot(state_action_input.type(torch.long),
                                          num_classes=num_states * num_actions) + gamma * occupancy_measure  # compute line 4

    return occupancy_measure

def compress_reward(obs_buffer,action_buffer,num_actions,occ_measure) :
    effective_reward = torch.zeros(obs_buffer.shape[0])
    assert obs_buffer.shape[0] == action_buffer.shape[0] #this is batch size
    for idx,(s,a) in enumerate(zip(obs_buffer,action_buffer)) :
        sa_idx = change_state_action_dim(s,a,num_actions)
        effective_reward[idx] = occ_measure[sa_idx.type(torch.long).item()]
    return effective_reward
def calculate_scores(rewards, gamma):
    len_r = rewards.shape[0]
    with torch.no_grad():
        scores = torch.zeros(len_r)
    cum_score = 0
    for idx in range(len_r):
        cum_score += gamma ** (len_r-1-idx) * rewards[len_r-1-idx]
        scores[len_r-1-idx] = cum_score
    return scores

def calculate_w(obs_buffer, action_buffer, episode,previous_policy, policy, num_states):
    batch_input = F.one_hot(obs_buffer[episode, :].type(torch.long), num_classes=num_states).type(torch.float)
    _, previous_logprob = previous_policy.get_action(batch_input)
    _, current_logprob = policy.get_action(batch_input)

    selected_previous_logprob = previous_logprob.gather(1, action_buffer[episode, :].long().view(-1, 1))
    selected_logprob = current_logprob.gather(1, action_buffer[episode, :].long().view(-1, 1))

    selected_previous_prob = torch.exp(selected_previous_logprob)
    selected_prob = torch.exp(selected_logprob)
    epsilon = 1e-8

    #donghao : nice catch! #
    ratio = selected_previous_prob / (selected_prob + epsilon)
    ratio = ratio.detach()
    return torch.prod(ratio)

## for mountain car ##

def discretization(x,low_x,high_x,n_discrete) :
    discrete_list = []
    for i in range(len(x)):
        if x[i] == high_x[i] :
            discrete_list.append(n_discrete -1)
        else :
            discrete_list.append(int( (x[i]-low_x[i]) * (n_discrete / (high_x[i]-low_x[i]))))

    return discrete_list[0] * n_discrete + discrete_list[1]
