from keras.layers import Dense
from keras.models import Sequential, Model

class Net(object):

    def __init__(self, env_spec):
        self.env_spec = env_spec
        in_dim = env_spec.get('observation_space').shape[0]
        action_dim = env_spec.get('action_space').n
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=in_dim))
        model.add(Dense(50, activation='relu'))
        self.pi_model = Sequential([model])
        self.pi_model.add(Dense(50, activation='relu'))
        self.pi_model.add(Dense(action_dim, activation='softmax'))
        self.v_model = Sequential([model])
        self.v_model.add(Dense(50, activation='relu'))
        self.v_model.add(Dense(1))

