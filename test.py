import logging
import sys
import os
import argparse
from datetime import datetime
import copy
import numpy as np

import gym
import minerl

import torch

import matplotlib.pyplot as plt

from dqnagent import DQNAgent

# import faulthandler; faulthandler.enable()

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="load trained model")
parser.add_argument("--pretrain", help="pretrain model before training")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # set to INFO if you want fewer messages

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = 'MineRLNavigateDense-v0'
env = gym.make(env_name)

# act_keys = list(env.action_space.sample().keys())
act_keys = ['attack','back','forward','jump','left','place','right','sneak','sprint','camera']
n_actions = len(act_keys) + 3 + 1 #act_keys + camera(the other direction) + no-op

video_height = 64
video_width = 64
    
def im_prep(data):
    data = torch.tensor(data)
    # data = data.view(video_height,video_width,-1)
    marg = 255/2.0
    data = (data - marg)/marg
    # data = data.permute(2,0,1).unsqueeze(0)
    return data.view(1,-1)

def get_state(obs):
    im = im_prep(obs['pov'])
    other = np.expand_dims(np.stack([obs['compassAngle'],obs['inventory']['dirt']]),axis=0)
    other = torch.Tensor(other)
    return torch.cat((im,other),dim=1).to(device)

def format_action(no_act,act_ind): 
    if act_ind >= n_actions:
        act_ind = n_actions - 1
    elif act_ind < 0:
        act_ind = 0

    action = no_act()

    if act_ind < 9:
        action[act_keys[act_ind]] = 1
    elif act_ind == 9 or act_ind == 10:
        action[act_keys[-1]][0] = 5.0*pow(-1,act_ind%9)
    elif act_ind == 11 or act_ind == 12:
        action[act_keys[-1]][1] = 5.0*pow(-1,act_ind%11)

    return action

num_eps = 300
rewards = []

dqn_agent = DQNAgent(video_height,video_width,n_actions,device)

if args.load:
    dqn_agent.load(os.path.join(".",args.load))
    print("Loaded trained model")

if args.pretrain:
    print("Start pretraining")
    # Sample some data from the dataset!
    data = minerl.data.make("MineRLNavigateDense-v0")

    # Iterate through a single epoch using sequences of at most 32 steps
    for obs, rew, done, act in data.seq_iter(num_epochs=10, batch_size=32):
    # Do something    
    print("Ended pretraining")

today = datetime.now().strftime("%Y%m%d_%H%M%S")

def save_plot(rewards):
    plt.clf()
    plt.plot([ind for ind in range(len(rewards))],rewards)
    plt.xlabel("Episode i")
    plt.ylabel("Cumulative reward during episode i")
    plt.title("Training in {}".format(env_name))
    plt.savefig('training_{}.png'.format(env_name))

for i_episode in range(num_eps):
    logger.info("Episode {} has started.".format(i_episode))  
    obs = env.reset()
    count = 0
    cum_rew = 0.0
    done = False
    while not done:
        count += 1
        state = get_state(obs)
        # Select and perform an action
        action = dqn_agent.act(state)
        action_command = format_action(env.action_space.noop,action.squeeze()[()])
        # action = env.action_space.sample()

        # Observe new state
        next_obs, reward, done, _ = env.step(action_command)
        cum_rew += reward
        next_state = get_state(obs)
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        dqn_agent.memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # state = copy.deepcopy(next_state)

        # Perform one step of the optimization (on the target network)
        dqn_agent.optimize()

        # Update the target network, copying all weights and biases in DQN
        if count % dqn_agent.TARGET_UPDATE == 0:
            dqn_agent.target_update()

        if cum_rew < -2.0 or (count > 100*dqn_agent.TARGET_UPDATE and cum_rew < 1.0):
            done = True
    rewards.append(cum_rew)
    logger.info("Episode {} has ended. Cumulative reward: {}".format(i_episode,cum_rew))
    save_plot(rewards)
    path_folder = os.path.join("./models",env_name,today)
    os.system("mkdir -p {}".format(path_folder))
    # dqn_agent.save(os.path.join(path_folder,"dqn_ep{}_rw{}.model".format(i_episode,int(cum_rew))))
logger.info("End of Training") 