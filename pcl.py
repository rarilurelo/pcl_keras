import tensorflow as tf
from keras import backend as K
from keras.layers import Input
import numpy as np

class PCL(object):
    def __init__(self, epoch, env, env_spec, replay_buffer, sess=None, net=None,
            pi_optimizer=None, v_optimizer=None, off_policy_rate=20,
            pi_lr=7e-4, v_rate=0.5, entropy_tau=0.5, rollout_d=20, gamma=1):
        self.epoch = epoch
        self.env = env
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.sess = sess
        self.net = net
        if pi_optimizer is None:
            self.pi_optimizer = tf.train.AdamOptimizer(pi_lr)
        else:
            self.pi_optimizer = pi_optimizer
        if v_optimizer is None:
            self.v_optimizer = tf.train.AdamOptimizer(pi_lr*v_rate)
        else:
            self.v_optimizer = v_optimizer
        self.off_policy_rate = off_policy_rate
        self.entropy_tau = entropy_tau
        self.rollout_d = rollout_d
        self.gamma = gamma

        self.state_shape = env_spec.get('observation_space').shape
        self.action_shape = [env_spec.get('action_space').n]
        print('observation space: ', env_spec.get('observation_space'))
        print('action space: ', env_spec.get('action_space'))
        self.built = False

    def build(self):
        pi_model = self.net.pi_model
        v_model = self.net.v_model
        self.state = tf.placeholder(tf.float32, shape=[None, None, *self.state_shape], name='state')
        self.R = tf.placeholder(tf.float32, shape=[None, None], name='R')
        self.action = tf.placeholder(tf.float32, shape=[None, None, *self.action_shape], name='action')
        self.discount = tf.placeholder(tf.float32, shape=[None], name='discount')

        v_s_t = v_model(self.state[:, 0, :])
        v_s_t_d = v_model(self.state[:, -1, :])
        self.pi = pi_model(self.state)
        C = K.sum(-v_s_t + self.gamma ** self.rollout_d * v_s_t_d + \
                K.sum(self.R, axis=1) - self.entropy_tau * K.sum(self.discount * \
                K.sum(K.log(self.pi+K.epsilon()) * self.action, axis=2), axis=1), axis=0)
        self.loss = C ** 2

        self.updater = [self.pi_optimizer.minimize(self.loss, var_list=pi_model.trainable_weights),
                self.v_optimizer.minimize(self.loss, var_list=v_model.trainable_weights)]
        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def optimize(self, episode):
        if not self.built:
            self.build()
        if len(episode['states']) < self.rollout_d:
            rollout_d = len(episode['states'])
        else:
            rollout_d = self.rollout_d
        discount = np.array([self.gamma**i for i in range(rollout_d)], dtype=np.float32)
        state = []
        action = []
        R = []
        for i in range(len(episode['states'])-rollout_d+1):
            state.append(episode['states'][i:i+rollout_d])
            a = episode['actions'][i:i+rollout_d]
            action.append(np.eye(*self.action_shape, dtype=np.int32)[a])
            R.append(episode['rewards'][i:i+rollout_d])
        feed_in = {self.state: state, self.action: action, self.R: R, self.discount: discount}
        self.sess.run(self.updater, feed_in)

    def rollout(self, max_path_length=None):
        if max_path_length is None:
            max_path_length = self.env.spec.tags.get(
                    'wrapper_config.TimeLimit.max_episode_steps')
        states = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        s = self.env.reset()
        path_length = 0
        while path_length < max_path_length:
            a, agent_info = self.get_action(s)
            next_s, r, d, env_info = self.env.step(a)
            states.append(s)
            rewards.append(r)
            actions.append(a)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            s = next_s
        return dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=np.array(agent_infos),
            env_infos=np.array(env_infos),
        )

    def get_action(self, state):
        if not self.built:
            self.build()
        pi = self.sess.run(self.pi, {self.state: [[state]]})[0][0]
        a = np.random.choice(np.arange(self.action_shape[0]), p=pi)
        return a, dict(prob=pi)

    def train(self):
        for i in range(self.epoch):
            episode = self.rollout()
            self.optimize(episode)
            r = episode['rewards']
            print(r.sum())
            p = np.array([agent_info['prob'] for agent_info in episode['agent_infos']])
            ent = -np.sum(p * np.log(p+K.epsilon()), axis=1)
            mean_ent = ent.mean()
            print(mean_ent)
            self.replay_buffer.add(episode)
            if self.replay_buffer.trainable:
                for _ in range(self.off_policy_rate):
                    episode = self.replay_buffer.sample()
                    self.optimize(episode)


