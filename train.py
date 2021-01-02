import os
import random
import subprocess
import datetime
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from IPython.display import clear_output
from segment_tree import MinSegmentTree, SumSegmentTree

from PIL import Image

import minerl

video_height = 84
video_width = 84

resize = T.Compose([T.ToPILImage(), T.Resize((video_height,video_width), interpolation=Image.CUBIC), T.Grayscale(num_output_channels=1), T.ToTensor(), T.Normalize(mean=0,std=255)])

# video_height, video_width = 210, 160
# video_height, video_width = 64, 64

class ReplayBuffer:
    """A simple numpy replay buffer."""
    def __init__(self, state_dim, size: int, batch_size: int = 32):
        self.state_buf = torch.zeros([size]+[dim for dim in state_dim])
        self.next_state_buf = torch.zeros([size]+[dim for dim in state_dim])
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, state: torch.Tensor, act: torch.Tensor, rew: float, next_state: torch.Tensor, done: bool):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(state=self.state_buf[idxs],
                    next_state=self.next_state_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(self, state_dim, size: int, batch_size: int = 32, alpha: float = 0.6):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(state_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, state: torch.Tensor, act: int, rew: float, next_state: torch.Tensor, done: bool):
        """Store experience and priority."""
        super().store(state, act, rew, next_state, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        state = self.state_buf[indices]
        next_state = self.next_state_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(state=state,next_state=next_state,acts=acts,rews=rews,done=done,weights=weights,indices=indices)
        
    def update_priorities(self, indices: List[int], priorities: torch.Tensor):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

class Network(nn.Module):
    def __init__(self, in_dim, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        convw = in_dim[-1]
        convh = in_dim[-2]
        in_ch = in_dim[-3]
        kernels = [8,4,3]
        strides = [4,2,1]
        channels = [in_ch,32,64,64]
        layers = []
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        for ind in range(len(kernels)):
            convw = conv2d_size_out(size=convw, kernel_size=kernels[ind], stride=strides[ind])
            convh = conv2d_size_out(size=convh, kernel_size=kernels[ind], stride=strides[ind])
            layers.append(nn.Conv2d(channels[ind], channels[ind+1], kernel_size=kernels[ind], stride=strides[ind]))
            layers.append(nn.BatchNorm2d(channels[ind+1]))
            layers.append(nn.LeakyReLU())
        # linear_input_size = convw * convh * 32 + 2 # image features + compass + inventory
        linear_input_size = convw * convh * channels[-1]

        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if len(x.shape) == 1 or len(x.shape) == 3:
            x = x.unsqueeze(0)
        # im = x[:,:-2].view(-1,3,video_height,video_width)
        # im = x.view(-1,self.state_dim[-3],self.state_dim[-2],self.state_dim[-1])
        # im = x.view(-1,3,self.status_,video_width)
        x = self.net(x)
        # x = torch.cat([im_ft.view(im_ft.shape[0],-1),x[:,-2:]],dim=1)
        x = self.act1(self.fc1(x.view(x.shape[0],-1)))
        x = self.fc2(x)
        # x = self.net(x[:,-2:])
        return x

class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        beta (float): determines how much importance sampling is used
        prior_eps (float): guarantees every transition can be sampled
    """

    def __init__(self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        main_update: int,
        target_update: int,
        epsilon_decay: float,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        gamma: float = 0.98,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
        """
        self.env = env
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.main_update = main_update
        self.target_update = target_update
        self.gamma = gamma

        self.randaction = None #specifies if during random action sequence
        self.randperiod = 0 #random action sequence length
        self.count = 0

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: {}".format(self.device))

        # obs_dim = env.observation_space.shape[0]
        obs = self.env.reset()
        self.n_frames = 4
        self.frames = [self.get_screen(obs)]*self.n_frames
        self.state_dim = self.get_state(obs).shape

        self.act_keys = ['attack','back','forward','jump','left','place','right','sneak','sprint','camera']
        # self.action_dim = len(self.act_keys) + 3 + 1 #act_keys + camera(the other direction) + no-op
        self.action_dim = env.action_space.n
        # action_dim = env.action_space.shape[0]

        #Starting number of lives in Breakout
        self.lives = 5

        # Prioritized Experience Replay
        self.min_memory = 30000
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(self.state_dim, memory_size, self.batch_size, alpha)
        # self.memory = ReplayBuffer(self.state_dim, memory_size, self.batch_size)

        # networks: dqn, dqn_target
        self.dqn = Network(self.state_dim, self.action_dim).to(self.device)
        self.dqn_target = Network(self.state_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=0.00005)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        """Select an action from the input state."""
        if not self.randaction is None:
            selected_action = self.randaction
            self.count += 1
            if self.count > self.randperiod:
                self.randaction = None
        else:
            # epsilon greedy policy
            if self.epsilon > np.random.random():
                # selected_action = self.env.action_space.sample()
                selected_action = torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
                self.count = 0
                # self.randperiod = random.randrange(start=1,stop=10)
                self.randperiod = 0
                self.randaction = selected_action
            else:
                with torch.no_grad():
                    selected_action = self.dqn(state).to(self.device).argmax()
                    selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def format_action(self,no_act,act_ind): 
        if act_ind >= self.action_dim:
            act_ind = self.action_dim - 1
        elif act_ind < 0:
            act_ind = 0

        action = no_act()

        if act_ind == 5:
            action[self.act_keys[act_ind]] = "dirt"
        elif act_ind < 9:
            action[self.act_keys[act_ind]] = 1
        elif act_ind == 9 or act_ind == 10:
            action[self.act_keys[-1]][0] = 3.0*pow(-1,act_ind%9)
        elif act_ind == 11 or act_ind == 12:
            action[self.act_keys[-1]][1] = 3.0*pow(-1,act_ind%11)

        return action

    def step(self, action: torch.Tensor, time, score) -> Tuple[torch.Tensor, np.float64, bool]:
        """Take an action and return the response of the env."""
        # next_obs, reward, done, _ = self.env.step(self.format_action(noop,action))
        next_obs, reward, done, info = self.env.step(action)

        # time_limit_idx = 2000
        # rew_limit = -5
        # if score < rew_limit or (time > time_limit_idx and score < -rew_limit) or time>time_limit_idx*5:
        #     done = True

        next_state = self.get_state(next_obs)

        end_ep = done
        if info["ale.lives"] != self.lives:
            self.lives = info["ale.lives"]
            done = True

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, end_ep

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]
        # samples = self.memory.sample_batch()

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)
        # loss = torch.mean(elementwise_loss)
        # loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def im_prep(self,data):
        data = torch.Tensor(data)
        
        # data = data.view(video_height,video_width,-1)
        # marg = 255/2.0
        # data = (data - marg)/marg
        # data = data.permute(2,0,1).unsqueeze(0)
        # return data.view(-1,3*video_height*video_width)
        if len(data.shape) == 3:
            data = data.permute(2,0,1)
            # data = resize(data).unsqueeze(0)
            # data = torch.mean(data,dim=0)
            # return torch.flatten(data)
        # else:
            # data = torch.mean(data,dim=1)
            # return data.view(data.shape[0],-1)
        
        return data

    def get_state(self,obs=None):
        # self.frames = self.frames[1:] + [self.get_screen(obs)]
        self.frames = self.frames[1:] + [self.get_screen(obs)]
        state = torch.cat(self.frames, dim=-3).to(self.device)
        # # state = torch.Tensor(obs).to(self.device)
        # return state
        # im = self.im_prep(obs['pov'])
        return state
        # other = np.stack([obs['compassAngle'],obs['inventory']['dirt']],axis=len(obs['compassAngle'].shape)-1)
        # other = np.expand_dims(other, axis=0)
        # other = torch.Tensor(other)
        # if len(other.shape) == 1:
        #     return torch.cat((im,other),dim=0).to(self.device)
        # else:
        #     return torch.cat((im,other.view(other.shape[0],-1)),dim=1).to(self.device)
        

    def get_screen(self,obs=None):
        # def get_cart_location(screen_width):
        #     world_width = self.env.x_threshold * 2
        #     scale = screen_width / world_width
        #     return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        # screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))/255.0
        # Cart is in the lower half, so strip off the top and bottom of the screen
        # _, screen_height, screen_width = screen.shape
        # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        # view_width = int(screen_width * 0.6)
        # cart_location = get_cart_location(screen_width)
        # if cart_location < view_width // 2:
        #     slice_range = slice(view_width)
        # elif cart_location > (screen_width - view_width // 2):
        #     slice_range = slice(-view_width, None)
        # else:
        #     slice_range = slice(cart_location - view_width // 2,
        #                         cart_location + view_width // 2)
        # # Strip off the edges, so that we have a square image centered on a cart
        # screen = screen[:, :, slice_range]
        # # Convert to float, rescale, convert to torch tensor
        # # (this doesn't require a copy)
        # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # screen = torch.Tensor(screen)
        # Resize, and add a batch dimension (BCHW)
        # return resize(screen).unsqueeze(0)
        screen = self.im_prep(obs)
        return resize(screen)

    def act_to_ind(self,action,batch_size):
        indeces = torch.zeros(size=(batch_size,1),dtype=torch.long)
        noop = self.env.action_space.noop()
        for b in range(batch_size):
            inds = []
            for ind,key in enumerate(self.act_keys,0):
                if key == "camera":
                    if action[key][b][0][0] > 0:
                        inds.append(9)
                    elif action[key][b][0][0] < 0:
                        inds.append(10)
                    if action[key][b][0][1] > 0:
                        inds.append(11)
                    elif action[key][b][0][1] < 0:
                        inds.append(12)
                elif noop[key] != action[key][b].item():
                    inds.append(ind)

            if len(inds) == 0:
                indeces[b][0] = self.action_dim - 1
            elif len(inds) == 1:
                indeces[b][0] = inds[0]
            else:
                indeces[b][0] = inds[random.randrange(start=0,stop=len(inds)-1)]
        return indeces.cpu()

    def pretrain(self,num_frames):
        # Iterate through a single epoch using sequences of at most 32 steps
        batch_size = self.batch_size
        losses = []
        upt_cnt = 0

        # Sample some data from the dataset!
        data = minerl.data.make("MineRLNavigateDense-v0",data_dir="data")
        for batch in data.batch_iter(batch_size=self.batch_size, seq_len=1, num_epochs=-1):
            upt_cnt += 1
            samples = {}
            samples["state"] = self.get_state(batch[0]).cpu()
            samples["next_state"] = self.get_state(batch[3]).cpu()
            samples["acts"] = self.act_to_ind(batch[1],self.batch_size) 
            samples["rews"] = batch[2]
            samples["done"] = batch[4]
            elementwise_loss = self._compute_dqn_loss(samples)
            loss = torch.mean(elementwise_loss)
            losses.append(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if upt_cnt % self.target_update == 0:
                self._target_hard_update()
                plt.clf()
                plt.plot(losses)
                plt.savefig('pretraining_{}.png'.format(self.env.spec.id))
                plt.close()

            if upt_cnt > num_frames:
                break
        

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        obs = self.env.reset()
        state = self.get_state(obs)
        for _ in range(random.randint(0, 10)):
            obs, _, _, _= self.env.step(1)
            state = self.get_state(obs)
        # self.frames = [self.get_screen(obs)]*self.n_frames
        

        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        avg_scores = []
        score = 0
        score_avg = 0
        tau_avg = 100

        begin_idx = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action,frame_idx-begin_idx,score)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                obs = self.env.reset()
                self.frames = [self.get_screen(obs)]*self.n_frames
                for _ in range(random.randint(0, 10)):
                    obs, _, _, _= self.env.step(1)
                    state = self.get_state(obs)
                scores.append(score)
                score_avg += (-score_avg + score)/tau_avg
                avg_scores.append(score_avg)
                score = 0
                begin_idx = frame_idx

            # if training is ready
            if len(self.memory) >= self.min_memory:
                if update_cnt % self.main_update == 0:
                    loss = self.update_model()
                losses.append(loss)
                
                # linearly decrease epsilon
                self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()
                    # self.target_update = min(int(self.target_update*1.2),3000)
                    # update_cnt = 0
                    # print("target_update period: {}".format(self.target_update))
                update_cnt += 1

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, avg_scores, losses, epsilons)
                
        self.env.close()
                
    def test(self) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames

    def _compute_dqn_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return dqn loss."""
        state = torch.FloatTensor(samples["state"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_state"]).to(self.device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        # next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        next_q_value = self.dqn_target(next_state).gather(1, self.dqn(next_state).argmax(dim=1, keepdim=True)).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        # elementwise_loss = F.smooth_l1_loss(curr_q_value, target)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(self, frame_idx: int, scores: List[float], avg_scores: List[float], losses: List[float], epsilons: List[float]):
        """Plot the training progresses."""
        # clear_output(True)
        plt.clf()
        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(142)
        plt.title('avg score')
        plt.plot(avg_scores)
        plt.subplot(143)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(144)
        plt.title('epsilons')
        plt.plot(epsilons)
        # plt.show()
        plt.savefig('training_{}_withlargermemory_lifetrick_ddqn_per.png'.format(self.env.spec.id))
        plt.close()

# environment
env_id = "CartPole-v1"
# env_id = "CarRacing-v0"
# env_id = 'MineRLNavigateDense-v0'
# env_id = "BreakoutDeterministic-v4"
env = gym.make(env_id)
# env.reset()

# parameters
pre_num_frames = 500000
num_frames = 5000000
memory_size = 10000
batch_size = 32
main_update = 4
target_update = 10000
epsilon_decay = 1 / 500000

now = datetime.datetime.now()
save_path = os.path.join("./models",env_id)
subprocess.run(["mkdir", "-p", save_path])
save_path = os.path.join(save_path,now.strftime('ddqn_per_%Y%m%d%H%M%S.ml'))

agent = DQNAgent(env=env, memory_size=memory_size, batch_size=batch_size, main_update=main_update, target_update=target_update, epsilon_decay=epsilon_decay)

# agent.pretrain(pre_num_frames)

agent.train(num_frames)
