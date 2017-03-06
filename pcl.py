import tensorflow as tf
from keras import backend as K
from keras.layers import Input
import numpy as np

class PCL(object):
    def __init__(self, epoch, env, env_spec, replay_buffer, sess=None, net=None,
            pi_optimizer=None, v_optimizer=None, off_policy_rate=10,
            pi_lr=7e-4, v_rate=0.5, entropy_tau=0.01, rollout_d=20, gamma=0.99):
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
        self.built = False

    def build(self):
        pi_model = self.net.pi_model
        v_model = self.net.v_model
        self.s_t = tf.placeholder(tf.float32, shape=[None, *self.state_shape], name='s_t')
        self.s_t_d = tf.placeholder(tf.float32, shape=[None, *self.state_shape], name='s_t_d')
        self.R = tf.placeholder(tf.float32, shape=[None], name='R')
        self.a = tf.placeholder(tf.float32, shape=[None, *self.action_shape], name='a')
        self.s = tf.placeholder(tf.float32, shape=[None, *self.state_shape], name='s')
        self.discount = tf.placeholder(tf.float32, shape=[None], name='discount')

        v_s_t = v_model(self.s_t)
        v_s_t_d = v_model(self.s_t_d)
        self.pi = pi_model(self.s)
        C = K.sum(-v_s_t + self.gamma ** self.rollout_d * v_s_t_d + \
                K.sum(self.R) - self.entropy_tau * K.sum(self.discount * \
                K.sum(K.log(self.pi+K.epsilon()) * self.a, axis=-1, keepdims=True)))
        self.loss = C ** 2
        self._grad_pi = K.gradients(self.loss, pi_model.trainable_weights)
        self._grad_v = K.gradients(self.loss, v_model.trainable_weights)

        self.grad_pi = [tf.placeholder(tf.float32, shape=g.get_shape(), name="grad_pi_{}".format(i)) for i, g in enumerate(self._grad_pi)]
        self.grad_v = [tf.placeholder(tf.float32, shape=g.get_shape(), name="grad_v_{}".format(i)) for i, g in enumerate(self._grad_v)]

        self.pi_applier = self.pi_optimizer.apply_gradients(
                [(g, w) for g, w in zip(self.grad_pi, pi_model.trainable_weights)])
        self.v_applier = self.v_optimizer.apply_gradients(
                [(g, w) for g, w in zip(self.grad_v, v_model.trainable_weights)])
        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def optimize(self, episode):
        if not self.built:
            self.build()
        grad_pi = [np.zeros(shape=w.get_shape()) for w in self.net.pi_model.trainable_weights]
        grad_v = [np.zeros(shape=w.get_shape()) for w in self.net.v_model.trainable_weights]
        if len(episode['states']) < self.rollout_d:
            rollout_d = len(episode['states'])
        else:
            rollout_d = self.rollout_d
        discount = np.array([self.gamma**i for i in range(rollout_d)], dtype=np.float32)
        for i in range(len(episode['states'])-rollout_d+1):
            s_t = [episode['states'][i]]
            s_t_d = [episode['states'][i+rollout_d-1]]
            R = episode['rewards'][i:i+rollout_d]
            a = episode['actions'][i:i+rollout_d]
            a = np.eye(*self.action_shape, dtype=np.int32)[a]
            s = episode['states'][i:i+rollout_d]
            feed_in = {self.s_t: s_t, self.s_t_d: s_t_d, self.R: R, self.a: a, self.s: s, self.discount: discount}
            grads = self.sess.run([self._grad_pi, self._grad_v], feed_in)
            grad_pi = [g_pi+g for g_pi, g in zip(grad_pi, grads[0])]
            grad_v = [g_v+g for g_v, g in zip(grad_v, grads[1])]
        feed_grad = {g_pi: g_pi_feed for g_pi, g_pi_feed in zip(self.grad_pi, grad_pi)}
        feed_grad.update({g_v: g_v_feed for g_v, g_v_feed in zip(self.grad_v, grad_v)})
        self.sess.run([self.pi_applier, self.v_applier], feed_grad)

    def rollout(self, max_path_length=None):
        if max_path_length is None:
            max_path_length = self.env._spec.tags.get(
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
        pi = self.sess.run(self.pi, {self.s: [state]})[0]
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


