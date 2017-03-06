import gym
from gym import wrappers
import tensorflow as tf
import os

from net import Net
from pcl import PCL
from replay_buffer import ReplayBuffer

log_dir = './log'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
env = gym.make('CartPole-v0')
env_spec = dict(
        action_space=env.action_space,
        observation_space=env.observation_space)
env = wrappers.Monitor(env, log_dir, force=True)

net = Net(env_spec)

replay_buffer = ReplayBuffer()

sess = tf.Session()

agent = PCL(100000, env, env_spec, replay_buffer, sess, net)


agent.train()



