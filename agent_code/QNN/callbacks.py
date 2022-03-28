import os.path

import numpy as np
import settings as s

import tensorflow as tf

from keras.optimizers import adam_v2

import random


from collections import deque

from collections import Counter


# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


OUT_OF_BOUND = -100
EXPLOSION = -3
WALL = -1
FREE = 0
CRATE = 1
COIN = 2
BOMB = -2

s.ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT','BOMB']

# find 4 perspectives from agent itself
s.DEFAULT_PERSPECTIVE_DISTANCE=3

# neural_netowrk parameter
state_shape = [1, 4, s.DEFAULT_PERSPECTIVE_DISTANCE] # 4 directions


def setup(self):
    '''
    Called once in the beginning of the episode to setup bombi
    '''
    self.logger.info('Bombi awakes.')
    np.random.seed(123)

    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.005

    #self.states = np.zeros((0, 5))

    # build models
    self.eval_model = build_model(self)


    if self.train:
        self.logger.info('Training mode started.')
        if (os.path.exists("time-cartpole-dqn.h5" )):
            weight_load(self,name="time-cartpole-dqn.h5")
            self.logger.info("Loading model from saved state.")

        self.logger.info("Loading model from saved state.")
    else:
        # 读取预训练的模型
        if (os.path.exists("time-cartpole-dqn.h5" )):
            weight_load(self,name="time-cartpole-dqn.h5")
            self.logger.info("Loading model from saved state.")

        self.logger.info("Loading model from saved state.")


def weight_load(self, name):
    self.eval_model.load_weights(name)



def act(self, game_state):
    '''
    Called in every steps to determine the `next_action`
    '''
    self.logger.info('MY_Agent active.')

    self.last_pos = game_state['self'][3]
    self.states = perspective(game_state)

    if self.train:

        #if 3 <= self.epsilon:
        if np.random.uniform() <= self.epsilon:
            self.next_action=random.choice(s.ACTIONS)
            action_index=s.ACTIONS.index(self.next_action)
            self.last_actions = np.vstack((self.last_actions, action_index))
            print('Random Action:' + str(self.next_action))

            return self.next_action
        else:
            new_array=np.zeros((1, 1, 4, s.DEFAULT_PERSPECTIVE_DISTANCE))
            new_array[0][0]=self.states
            es_act_values = self.eval_model.predict(new_array)
            act_values = np.argmax(es_act_values[0][0], 0)
            collection_words = Counter(act_values)
            most_counterNum = collection_words.most_common(1)[0][0]
            self.next_action = s.ACTIONS[most_counterNum]


        self.last_actions = np.vstack((self.last_actions, s.ACTIONS.index(self.next_action)))


        print('Action:' + str(self.next_action))

        return self.next_action
        #return np.argmax(act_values[0])  # returns action

    else:
        new_array = np.zeros((1, 1, 4, s.DEFAULT_PERSPECTIVE_DISTANCE))
        new_array[0][0] = self.states
        es_act_values = self.eval_model.predict(new_array)
        act_values = np.argmax(es_act_values[0][0], 0)
        collection_words = Counter(act_values)
        most_counterNum = collection_words.most_common(1)[0][0]
        self.next_action = s.ACTIONS[most_counterNum]

        print('Action:' + str(self.next_action))

        return self.next_action




def build_model(self):
        # ---------Neural Net for evaluation model

        eval_model = tf.keras.models.Sequential()

        eval_model.add(tf.keras.layers.Dense(
            12, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
        eval_model.add(tf.keras.layers.Dense(
            12, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
        eval_model.add(tf.keras.layers.Dense(
            12, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
        eval_model.add(tf.keras.layers.Dense(
            12, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))


        # output
        eval_model.add(tf.keras.layers.Dense(len(s.ACTIONS), activation='linear'))


        #eval_model.summary()

        # loss function is square difference, Adam to optimize
        eval_model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=self.learning_rate, epsilon=1e-08, decay=0.0))


        return eval_model



def perspective(game_state, distance=s.DEFAULT_PERSPECTIVE_DISTANCE):
    '''
    Returns a 4 x distance matrix with the agent's view into each direction
    '''
    result = np.zeros((4, distance), dtype=np.int8)
    x, y = game_state['self'][3][0], game_state['self'][3][1]
    arena = game_state['arena']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosions = game_state['explosions']

    dire_list=['LEFT','RIGHT','UP','DOWN']

    bound_x=arena.shape[0]
    bound_y=arena.shape[1]

    k = 0
    # left, right, up, down
    for it_x, it_y in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        for i in range(0, distance):
            new_x= x + it_x * (i+1)
            new_y= y + it_y * (i+1)
            if new_x > bound_x or new_y > bound_y:
                #print('Out of arena, the direction: '+dire_list[i]+' has out of bound.')
                result[k, i] = OUT_OF_BOUND

            else:
                if arena[x + it_x * (i), y + it_y * (i)] == WALL:
                    # walls
                    result[k, i] = WALL
                else:
                    # bombs
                    for b in bombs:
                        if (new_x,new_y) is b[0]:
                            result[k, i] = BOMB
                    # coins
                    for c in coins:
                        pos=(new_x,new_y)
                        if pos is c:
                            result[k, i] = COIN



        k = k + 1

    # store if bomb is available
    result[0, 0] = game_state['self'][2]

    return result.astype('float32')
