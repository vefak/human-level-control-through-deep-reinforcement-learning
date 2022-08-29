import os
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from baselines_wrappers import DummyVecEnv, SubprocVecEnv, Monitor
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

GAMMA=0.99  # discount rate for computing temporal difference target
BATCH_SIZE=32 # sample size (transitions) to send gradient computing
BUFFER_SIZE=1000000 # max number of transitions before overwriting old ones
MIN_REPLAY_SIZE=5000 # how many transitions in replay buffer before training 
EPSILON_START=1.0 # starting value of epsilon
EPSILON_END=0.1  # ending value of epsilon
EPSILON_DECAY=1000000 # decay rate of epsilon
#TARGET_UPDATE_FREQ = 1000 # number of steps where target params = online params
NUM_ENVS = 4 # paper specified as four. 
TARGET_UPDATE_FREQ=10000 // NUM_ENVS # it is measured in the number of env steps
LR = 4.5e-5 # learning rate
SAVE_PATH = './atari_model.pack'.format(LR) 
SAVE_INTERVAL = 10000
LOG_DIR = './logs/atari' + str(LR)
LOG_INTERVAL = 1000


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512): 
    # It is CNN networks that mentioned inside the paper

    n_input_channels = observation_space.shape[0] # CHW format first one is channel

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())

    # compute shape of output tensor by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


class Network(nn.Module):
    
    def __init__(self,env, device):
        # inherit from nn module
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device
        # how many features that env has. UP DOWN RIGHT etc..
        in_features = int(np.prod(env.observation_space.shape))
        
        conv_net = nature_cnn(env.observation_space)

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    # required for nn implementation
    def forward(self,x):
        return self.net(x)

    def act(self,obses, epsilon):
        # calculate Q value and return highest one as integer
        # this is the how action selected in Q learning
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device) # torch tensor
        q_values = self(obses_t) 
        max_q_indices = torch.argmax(q_values, dim=1)# get highest q value
        actions = max_q_indices.detach().tolist() # tensor to integer
        for i in range(len(actions)): # epsilon greedy policy
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions -1 ) # replace action with completely random action
                #actions[i] = np.integers(0, self.num_actions -1 ) # replace action with completely random action
        return actions

    def compute_loss(self,transitions, target_net):
        
        # transitions has tuple format. Need to get all values individually 
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]
 
        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)
        
        # torch tensor faster than numpy array
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # computing targets for loss functions
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t) # get target q values from network
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0] # get max value at dimension 1. 
        
        # calculates y_i
        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # compute loss
        q_values = self(obses_t) # have q values for each obs
        # this time we dont get max values
        # get q value for actual action in transition(actions_t)
        # give predicted q value for the action
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) 

        # calculate loss function, huber loss (smooth l1 loss in torch)
        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()} # convert torch model to numpy array to save model
        #cpu stands for move tensor cores to cpu before convert them to numpy array
        params_data = msgpack.dumps(params) # like pickle format

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()} #

        self.load_state_dict(params)

        # Note:  we dont need to save target network params. Since, online net make all decisions making

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # creating atari env
    # create batch env for running multiple instances parallel
    make_env = lambda: Monitor(make_atari_deepmind('BreakoutNoFrameskip-v4', scale_values=True), allow_early_resets=True)
    # vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)]) # run each of env in sequence. actually dont gain parallelism 
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)]) # step each env in parallel. We can use multiple cores

    env = BatchedPytorchFrameStack(vec_env, k=4) # stack frames for each observation without duplicating frames. Useful memory usage   


    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100) #reward buffer, rewards earned by agent 
    episode_count = 0

    summary_writer = SummaryWriter(LOG_DIR)
    # Set networks
    online_net = Network(env, device=device)
    target_net = Network(env, device=device)

    online_net.apply(init_weights)

    # models run on device, gpu or cpu 
    online_net = online_net.to(device)
    target_net = target_net.to(device)

    # Set target net params equal to online net params 
    # Algorithm specified this in paper
    target_net.load_state_dict(online_net.state_dict()) 
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)


    # replay buffer init
    obses = env.reset() # reset env when get first observation
    for _ in range(MIN_REPLAY_SIZE):

        actions = [env.action_space.sample() for _ in range(NUM_ENVS)] # select random action for each env
        
        new_obses, rews, dones, _ = env.step(actions) # move to next observation, get reward, check action wheter done or not

        for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
            transition = (obs, action, rew, done, new_obs) # data
            replay_buffer.append(transition) # hold data on buffer
            

        obses = new_obses # assign new obs to old obs

    # Algorithm 1: deep Q-learning with experience replay
    obses = env.reset() 
    for step in itertools.count():
        # need to compute epsilon first. Epsilon greedy policy
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]) # go from 1.0 to 0.02

        rnd_sample = random.random()
        if isinstance(obses[0], PytorchLazyFrames): #checking first batch envs is pytorchlazyframe or not
            act_obses = np.stack([o.get_frames() for o in obses]) # if we have, we need to convert the pytorchlazyframe to numpy array
            actions = online_net.act(act_obses, epsilon)  # online net predicts action
        else:
            actions = online_net.act(obses, epsilon)


        new_obses, rews, dones, infos = env.step(actions) 

        for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, done, new_obs) 
            replay_buffer.append(transition) 
            
            if done:
                    epinfos_buffer.append(info['episode']) # if episode done, add infos to buffer
                    episode_count += 1

        obses = new_obses 

    

        # start gradient descent computing
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.compute_loss(transitions, target_net) # compute loss

        # gradient descent
        optimizer.zero_grad() # Sets the gradients of all optimized torch.Tensors to zero.
        loss.backward() # computes loss
        optimizer.step() # parameter update

        # update targetnet
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        if step % LOG_INTERVAL == 0:
            rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0  # r is dict key for reward
            len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0  # l is dict key for length 
            # both l and r added by Monitor

            print()
            print('Step', step)
            print('Avg Rew', rew_mean)
            print('Avg Ep Len', len_mean)
            print('Episodes', episode_count)

            summary_writer.add_scalar('AvgRew', rew_mean, global_step=step)
            summary_writer.add_scalar('AvgEpLen', len_mean, global_step=step)
            summary_writer.add_scalar('Episodes', episode_count, global_step=step)

        # Save
        if step % SAVE_INTERVAL == 0 and step !=0:
            print('Saving...')
            online_net.save(SAVE_PATH)