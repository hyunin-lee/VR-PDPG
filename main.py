import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *


torch.autograd.set_detect_anomaly(True)

import gym


import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode", type=int, default=1000, help = "iteration number")
    parser.add_argument("--max_step", type=int, default = 1000, help = "trajectory length")
    parser.add_argument("--gamma", type=float, default=0.99, help="gamma")
    parser.add_argument("--init_lr_theta", type=float, default=0.01, help="initial learning rate for theta")
    parser.add_argument("--init_lr_mu", type=float, default=0.001, help="initial learning rate for mu")
    parser.add_argument("--alpha", type=float, default=0.01, help="alpha")
    parser.add_argument("--C0_mu", type=float, default=100, help="alpha")
    args = parser.parse_args()
    return args


def VR_PDPG(env,agent,previous_agent,agent_reference,args,num_states,num_actions) :
    max_episode = args.max_episode
    max_step = args.max_step
    gamma = args.gamma
    init_lr_theta = args.init_lr_theta
    init_lr_mu = args.init_lr_mu
    alpha = args.alpha

    # define optimizer.zero_grad()
    optimizer_agent = torch.optim.Adam(agent.parameters(), lr=init_lr_theta)
    optimizer_agent_reference = torch.optim.Adam(agent_reference.parameters(), lr=init_lr_theta)
    optimizer_previous_agent = torch.optim.Adam(previous_agent.parameters(), lr=init_lr_theta)

    # init mu_0
    with torch.no_grad():
        mu = torch.tensor([0.01])
    obs_buffer = torch.zeros(max_episode,max_step)
    action_buffer = torch.zeros(max_episode,max_step)
    reward_buffer = torch.zeros(max_episode,max_step)
    term_buffer = torch.zeros(max_episode,max_step)

    ## define variables
    target_occupancy_measure = get_target_occupancy_measure(num_states,num_actions,gamma)




    for episode in range(max_episode) :
        print("episode : " + str(episode))
        ## make agent refrence and agent same network before starting episode.
        set_flat_params_to(agent_reference, get_flat_params_from(agent))

        if episode == 0  :
            # line 2
            obs, infos = env.reset()
            final_step = 0
            for step in range(max_step) :
                with torch.no_grad():
                    obs_input = F.one_hot(torch.tensor(obs), num_classes=num_states).type(torch.float)
                    action, _ = agent.get_action(obs_input)
                    action_input = action.item()
                    next_obs, reward, termimate, _, infos = env.step(action_input)
                    ## save observation, rewards
                    obs_buffer[episode, step] = obs
                    action_buffer[episode,step] = action
                    reward_buffer[episode,step] = reward
                    term_buffer[episode,step] = termimate
                    obs = next_obs
                    final_step = step
                    if termimate :
                        break
            # line 3
            occupancy_measure = get_occupancy_measure(obs_buffer, action_buffer, episode, final_step, num_states, num_actions,gamma)

            # line 4
            r_f = reward_buffer[episode]
            occupancy_measure_gap = occupancy_measure - target_occupancy_measure
            r_g = compress_reward(obs_buffer[episode,:],action_buffer[episode,:],num_actions,occupancy_measure_gap)

            # line 5
            ## compute r_f ##
            d_f = get_gradient_input_reward(r_f,gamma,obs_buffer,action_buffer,agent_reference,optimizer_agent_reference,episode,num_states)

            ## compute r_g ##
            d_g = get_gradient_input_reward(r_g,gamma,obs_buffer,action_buffer,agent_reference,optimizer_agent_reference,episode,num_states)

            ## line 6
            d_L = d_f + mu * d_g
            lr_theta = init_lr_theta / torch.norm(d_L)

            optimizer_agent.zero_grad()
            set_flat_grads_to(agent,d_L)
            for g in optimizer_agent.param_groups:
                g['lr'] = lr_theta
            optimizer_agent.step()

            ## line 7
            mu = mu - init_lr_mu * 0.5 * torch.norm(occupancy_measure_gap)**2
            mu = torch.clamp(mu, min=0, max=args.C0_mu)

            ## define some vairbales
            previous_lambda = occupancy_measure
            previous_r_f = r_f
            previous_r_g = r_g
            previous_d_f = d_f
            previous_d_g = d_g
        else :
            # line 9
            obs, infos = env.reset()
            final_step = 0
            for step in range(max_step) :
                with torch.no_grad():
                    obs_input = F.one_hot(torch.tensor(obs), num_classes=num_states).type(torch.float)
                    action, _ = agent.get_action(obs_input)
                    action_input = action.item()
                    next_obs, reward, termimate, _, infos = env.step(action_input)
                    ## save observation, rewards
                    obs_buffer[episode, step] = obs
                    action_buffer[episode,step] = action
                    reward_buffer[episode,step] = reward
                    term_buffer[episode,step] = termimate
                    obs = next_obs
                    final_step = step
                    if termimate :
                        break

            # line 10
            occupancy_measure = get_occupancy_measure(obs_buffer, action_buffer, episode, final_step, num_states,
                                                      num_actions, gamma)
            #w = calculate_w(obs_buffer,action_buffer,episode,previous_agent,agent,num_states)
            w=0.9

            u = occupancy_measure * (1- w)
            lambda_ = alpha * occupancy_measure + (1-alpha) * (previous_lambda + u)

            # line 11
            r_f = reward_buffer[episode]
            occupancy_measure_gap = lambda_ - target_occupancy_measure
            r_g = compress_reward(obs_buffer[episode,:],action_buffer[episode,:],num_actions,occupancy_measure_gap)

            # line 12
            ## compute d_f ##
            d_tau_t_theta_t_r_f_t_1 = get_gradient_input_reward(previous_r_f,gamma,obs_buffer,action_buffer,agent_reference,optimizer_agent_reference,episode,num_states)
            # below code could be problemetic please check #
            v_f = d_tau_t_theta_t_r_f_t_1 - w * get_gradient_input_reward(previous_r_f,gamma,obs_buffer,action_buffer,previous_agent,optimizer_previous_agent,episode,num_states,requires_grad=False)
            ##################################################
            d_f = alpha * d_tau_t_theta_t_r_f_t_1 + (1- alpha) * (previous_d_f + v_f)

            ## compute d_g ##
            d_tau_t_theta_t_r_g_t_1 = get_gradient_input_reward(previous_r_g, gamma, obs_buffer, action_buffer,
                                                                agent_reference, optimizer_agent_reference, episode,
                                                                num_states)
            # below code could be problemetic please check #
            v_g = d_tau_t_theta_t_r_g_t_1 - w * get_gradient_input_reward(previous_r_g, gamma, obs_buffer,
                                                                          action_buffer, previous_agent,
                                                                          optimizer_previous_agent, episode, num_states,requires_grad=False)
            ##################################################
            d_g = alpha * d_tau_t_theta_t_r_g_t_1 + (1 - alpha) * (previous_d_g + v_g)

            # line 13
            d_L = d_f + mu * d_g
            lr_theta = init_lr_theta / torch.norm(d_L)

            optimizer_agent.zero_grad()
            set_flat_grads_to(agent, d_L)
            for g in optimizer_agent.param_groups:
                g['lr'] = lr_theta
            optimizer_agent.step()

            ## line 14
            mu = mu - init_lr_mu * 0.5 * torch.norm(occupancy_measure_gap) ** 2
            mu = torch.clamp(mu, min=0, max=args.C0_mu)
            ## define some vairbales
            previous_lambda = lambda_
            previous_r_f = r_f
            previous_r_g = r_g
            previous_d_f = d_f
            previous_d_g = d_g

        set_flat_params_to(previous_agent, get_flat_params_from(agent))
        print("reward : " + str(torch.sum(reward_buffer[episode]).item()))
        print("final step : " + str(final_step))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = parse_args()
    desc = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG",
    ]
    env = gym.make('FrozenLake-v1', desc=desc, map_name="8x8", is_slippery=False)
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    agent = Agent(num_states, num_actions)
    previous_agent = Agent(num_states,num_actions)
    agent_reference = Agent(num_states, num_actions)
    VR_PDPG(env,agent,previous_agent,agent_reference,args,num_states,num_actions)

