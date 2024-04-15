import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter


torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)

import gym
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode", type=int, default=4300, help = "iteration number")
    parser.add_argument("--max_step", type=int, default = 14, help = "trajectory length")
    parser.add_argument("--gamma", type=float, default=0.9, help="gamma")
    parser.add_argument("--init_lr_theta", type=float, default=1, help="initial learning rate for theta")
    parser.add_argument("--init_lr_mu", type=float, default=0.1, help="initial learning rate for mu")
    parser.add_argument("--alpha", type=float, default=0.1
                        , help="alpha")
    parser.add_argument("--init_mu", type=float, default=1, help="initial mu")
    parser.add_argument("--C0_mu", type=float, default=10, help="alpha")
    parser.add_argument("--d_0", type=float, default=0.0008, help="violance allowance")
    args = parser.parse_args()
    return args
    # 2000, 50, 0.9, 1, 0.1, 0.1, 1, 10, 2

def VR_PDPG(env,agent,previous_agent,agent_reference,args,num_states,num_actions,writer,save_foldername) :
    max_episode = args.max_episode
    max_step = args.max_step
    gamma = args.gamma
    init_lr_theta = args.init_lr_theta
    init_lr_mu = args.init_lr_mu
    alpha = args.alpha
    init_mu = args.init_mu
    C0_mu = args.C0_mu
    d_0 = args.d_0

    # define optimizer.zero_grad()
    optimizer_agent = torch.optim.Adam(agent.parameters(), lr=init_lr_theta)
    optimizer_agent_reference = torch.optim.Adam(agent_reference.parameters(), lr=init_lr_theta)
    optimizer_previous_agent = torch.optim.Adam(previous_agent.parameters(), lr=init_lr_theta)

    # init mu
    with torch.no_grad():
        mu = torch.tensor([init_mu])

    obs_buffer = torch.zeros(max_episode,max_step)
    action_buffer = torch.zeros(max_episode,max_step)
    reward_buffer = torch.zeros(max_episode,max_step)
    term_buffer = torch.zeros(max_episode,max_step)

    lastest_success_state_list, latest_success_episode = None, None
    ## define variables
    target_occupancy_measure = get_target_occupancy_measure(num_states,num_actions,gamma)

    for episode in tqdm(range(max_episode)) :
        # make alpha descrease as $1/t$
        # alpha = alpha * 1/(episode+1)
        ## make agent refrence and agent same network before starting episode.
        set_flat_params_to(agent_reference, get_flat_params_from(agent))
        current_state_list = []
        if episode == 0  :
            # line 2
            obs, infos = env.reset()
            current_state_list.append(obs)
            final_step = 0
            for step in range(max_step) :
                with torch.no_grad():
                    obs_input = F.one_hot(torch.tensor(obs), num_classes=num_states).type(torch.float)
                    action, _ = agent.get_action(obs_input)
                    action_input = action.item()
                    next_obs, reward, termimate, _, infos = env.step(action_input)
                    current_state_list.append(next_obs)
                    ## change reward ##
                    if termimate :
                        reward = 100
                    else :
                        reward = - 0.2
                    ####################
                    ## save observation, rewards
                    obs_buffer[episode, step] = obs
                    action_buffer[episode,step] = action
                    reward_buffer[episode,step] = reward
                    term_buffer[episode,step] = termimate
                    obs = next_obs
                    final_step = step
                    if termimate :
                        show_trajecgory(current_state_list,episode)
                        break
            # line 3
            occupancy_measure = get_occupancy_measure(obs_buffer, action_buffer, episode, final_step, num_states, num_actions,gamma)

            # line 4
            r_f = reward_buffer[episode]
            occupancy_measure_gap = occupancy_measure - target_occupancy_measure
            r_g = - compress_reward(obs_buffer[episode,:],action_buffer[episode,:],num_actions,occupancy_measure_gap) #donghao

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
            optimizer_agent.step() # gradient step

            ## line 7
            mu = mu + init_lr_mu * (0.5 * torch.norm(occupancy_measure_gap) ** 2 - d_0) #g(\lambda)
            mu = torch.clamp(mu, min=0, max=args.C0_mu)

            ## define some vairbales
            previous_lambda = occupancy_measure
            previous_previous_r_f = r_f
            previous_previous_r_g = r_g
            previous_r_f = r_f
            previous_r_g = r_g
            previous_d_f = d_f
            previous_d_g = d_g
        else :
            # line 9
            obs, infos = env.reset()
            current_state_list.append(obs)
            final_step = 0
            for step in range(max_step) :
                with torch.no_grad():
                    obs_input = F.one_hot(torch.tensor(obs), num_classes=num_states).type(torch.float)
                    action, _ = agent.get_action(obs_input)
                    action_input = action.item()
                    next_obs, reward, termimate, _, infos = env.step(action_input)
                    current_state_list.append(next_obs)
                    ## change reward ##
                    if termimate :
                        reward = 100
                    else :
                        reward = - 0.2
                    ####################
                    ## save observation, rewards
                    obs_buffer[episode, step] = obs
                    action_buffer[episode,step] = action
                    reward_buffer[episode,step] = reward
                    term_buffer[episode,step] = termimate
                    obs = next_obs
                    final_step = step
                    if termimate :
                        lastest_success_state_list = current_state_list
                        latest_success_episode = episode
                        # show_trajecgory(current_state_list,episode)
                        break

            # line 10
            occupancy_measure = get_occupancy_measure(obs_buffer, action_buffer, episode, final_step, num_states,
                                                      num_actions, gamma)
            w = calculate_w(obs_buffer,action_buffer,episode,previous_agent,agent,num_states)

            u = occupancy_measure * (1- w)
            lambda_ = alpha * occupancy_measure + (1-alpha) * (previous_lambda + u)

            # line 11
            r_f = reward_buffer[episode]
            occupancy_measure_gap = lambda_ - target_occupancy_measure
            r_g = - compress_reward(obs_buffer[episode,:],action_buffer[episode,:],num_actions,occupancy_measure_gap) #donghao

            # line 12
            ## compute d_f ##
            d_tau_t__theta_t__r_f_t_1 = get_gradient_input_reward(previous_r_f,gamma,obs_buffer,action_buffer,
                                                                agent_reference,optimizer_agent_reference,episode,
                                                                  num_states)
            d_tau_t__theta_t_1__r_f_t_2 = get_gradient_input_reward(previous_previous_r_f,gamma,obs_buffer,action_buffer,
                                                                  previous_agent,optimizer_previous_agent,episode,
                                                                    num_states,requires_grad=False)

            v_f = d_tau_t__theta_t__r_f_t_1 - w * d_tau_t__theta_t_1__r_f_t_2
            d_f = alpha * d_tau_t__theta_t__r_f_t_1 + (1- alpha) * (previous_d_f + v_f)

            ## compute d_g ##
            d_tau_t_theta_t__r_g_t_1 = get_gradient_input_reward(previous_r_g, gamma, obs_buffer, action_buffer,
                                                                agent_reference, optimizer_agent_reference, episode-1,
                                                                num_states)

            d_tau_t__theta_t_1__r_g_t_2 = get_gradient_input_reward(previous_previous_r_g, gamma, obs_buffer,
                                                                          action_buffer, previous_agent,
                                                                          optimizer_previous_agent, episode, num_states,requires_grad=False)
            # below code could be problemetic please check #
            v_g = d_tau_t_theta_t__r_g_t_1 - w * d_tau_t__theta_t_1__r_g_t_2
            ##################################################
            d_g = alpha * d_tau_t_theta_t__r_g_t_1 + (1 - alpha) * (previous_d_g + v_g)

            # line 13
            d_L = d_f + mu * d_g
            lr_theta = init_lr_theta / torch.norm(d_L)

            optimizer_agent.zero_grad()
            set_flat_grads_to(agent, d_L)
            for g in optimizer_agent.param_groups:
                g['lr'] = lr_theta
            optimizer_agent.step()

            ## line 14
            mu = mu + init_lr_mu * (0.5 * torch.norm(occupancy_measure_gap) ** 2 - d_0) #g(\lambda)
            mu = torch.clamp(mu, min=0, max=C0_mu)

            ## define some vairbales
            previous_lambda = lambda_
            previous_previous_r_f = previous_r_f
            previous_previous_r_g = previous_r_g
            previous_r_f = r_f
            previous_r_g = r_g
            previous_d_f = d_f
            previous_d_g = d_g


        set_flat_params_to(previous_agent, get_flat_params_from(agent))

        writer.add_scalar("return", torch.sum(reward_buffer[episode]).item(), episode)
        writer.add_scalar("final step", final_step, episode)
        writer.add_scalar("constraint violation", torch.sum(0.5 * torch.norm(occupancy_measure - target_occupancy_measure)**2).item(), episode)
        writer.add_scalar("learning rate",lr_theta,episode)
        # print(torch.sum(0.5 * torch.norm(occupancy_measure - target_occupancy_measure)**2).item())
        # print("reward : " + str(torch.sum(reward_buffer[episode]).item()))
        # print("final step : " + str(final_step))
        # print("constraint violation : " + str(torch.sum(occupancy_measure - target_occupancy_measure).item()))
        writer.flush()

    show_trajecgory(lastest_success_state_list,latest_success_episode,save_foldername)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = parse_args()
    env = gym.make("MountainCar-v0")
    states_high = env.observation_space.high
    states_low = env.observation_space.low

    n_discretize = 50
    num_states = n_discretize ** len(states_high)
    num_actions = env.action_space.n

    agent = Agent(num_states, num_actions)
    previous_agent = Agent(num_states,num_actions)
    agent_reference = Agent(num_states, num_actions)

    #folder_name = 'test-{date:%Y_%m_%d_%H:%M:%S}'.format(date=datetime.datetime.now())
    folder_name = "maxEp__" + str(args.max_episode) + '__maxS__'+str(args.max_step) + '__gm__' + str(args.gamma) + \
                   '__lrTh0__' + str(args.init_lr_theta) + '__lrMu0__' + str(args.init_lr_mu) + '__a__' + str(args.alpha) + \
                  '__mu0__' + str(args.init_mu) +"__d_0__" + str(args.d_0)

    writer = SummaryWriter('./runs_mc/' + folder_name)

    VR_PDPG(env,agent,previous_agent,agent_reference,args,num_states,num_actions,writer,'./runs2/' + folder_name)
    writer.close()

