

from typing import List
import numpy as np
import os

import events as e

import settings as s

import pandas

#from keras.utils import plot_model


"""
Remaining proble:
1. Stuck in oscillation
 solution: if position remains the same, action is useless, write a function to walk out
 
2. Kill itself
 solution: write a function to walk out explosions sacle with A* algorithm


"""

s.ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT','BOMB']

# find 4 perspectives from agent itself
s.DEFAULT_PERSPECTIVE_DISTANCE=3

n_actions = len(s.ACTIONS)


# hyperparameters
alpha = 0.75 # learning rate
gama = 0.95 # discount


n_features = s.DEFAULT_PERSPECTIVE_DISTANCE

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    self.states = np.zeros((0, n_features))
    self.last_actions = np.zeros((0, 1), dtype=np.int8)  # empty at present
    self.rewards = np.zeros((0, n_actions))
    self.loss=0


    self.max_length = 500
    self.state_memory = np.zeros((self.max_length, 1, 4, s.DEFAULT_PERSPECTIVE_DISTANCE))
    self.qvalue_memory = np.zeros((self.max_length, 1, 1, n_actions))

    self.memory_size = 25


    self.batch_size= 16

    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug(f'EVENTS: {events}')
    if len(events) != 0:
        # update the rewards list
        current_reward = np.zeros((1, n_actions))
        if old_game_state is None:
            print("new start, without last game state.")
        else:
            self.last_pos=old_game_state['self'][3]

        current_reward[0][self.last_actions[-1]] = reward_from_events(self, events, new_game_state['self'][3], new_game_state)  # r
        self.rewards = np.vstack((self.rewards, current_reward))

        memorize(self, self.states, current_reward)
    else:
        print('Error, no events')


def memorize(self, state, q_value):
    if not hasattr(self, 'memory_counter'):
        self.memory_counter = 0
    if self.memory_counter==self.memory_size:
        self.memory_counter=0
        self.state_memory = np.zeros((self.max_length, 1, 4, s.DEFAULT_PERSPECTIVE_DISTANCE))
        self.qvalue_memory = np.zeros((self.max_length, 1, 1, n_actions))
    self.state_memory[self.memory_counter][0] = state
    self.qvalue_memory[self.memory_counter][0] = q_value

    self.memory_counter+=1



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    """
    self.inputs = np.zeros((last_game_state['step'], len(np.concatenate(last_game_state['arena'], axis=0))))  # 32, 80, 80, 4
    self.targets = np.zeros((self.inputs.shape[0], len(s.ACTIONS))) # 32, 2

    # update the rewards list
    current_reward = np.zeros((1, n_actions))
    current_reward[0][self.last_actions[-1]] = reward_from_events(self, events, last_game_state['self'][3],last_game_state)  # r
    self.rewards = np.vstack((self.rewards, current_reward))



    # update rule
    self.Q_values = np.zeros((0, n_actions))


    if os.path.isfile("states.csv"):
        file=pandas.read_csv('states.csv')
        self.inputs = np.array(file, dtype=float)

        new_whold_states = np.row_stack((self.inputs, self.states))

        file_q=pandas.read_csv('Q_values.csv')
        old_Q_values = np.array(file_q, dtype=float)

        now_qvs = np.amax(self.model.predict(self.states), axis=1)

        self.Q_values = old_Q_values

        for i in range(len(now_qvs)-1):
            now_action = self.last_actions[i][0]
            now_reward = self.rewards[i][now_action]  # current reward
            target = now_reward + self.gamma * now_qvs[i + 1]
            Q_s_a = np.zeros((1, n_actions))  # new Q(s,a)
            Q_s_a[0][now_action] = now_qvs[i] + alpha * target
            self.Q_values = np.vstack((self.Q_values, Q_s_a))

        self.Q_values = np.vstack((self.Q_values, self.rewards[i + 1]))



    else:
        new_whold_states = self.states
        self.Q_values = self.rewards
    """
    print('Round: %s end!' % last_game_state['round'])


    # 随机取出记忆
    if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    batch_memory_state = self.state_memory[sample_index, :]
    batch_memory_qreward=self.qvalue_memory[sample_index, :]

    # 这里需要得到估计值加上奖励 成为训练中损失函数的期望值
    # q_next是目标神经网络的q值，q_eval是估计神经网络的q值
    # q_next是用现在状态得到的q值 q_eval是用这一步之前状态得到的q值
    # print(batch_memory[:, -self.n_features:])
    q_eval = self.eval_model.predict(batch_memory_state, batch_size=self.batch_size)

    # change q_target w.r.t q_eval's action
    q_target = q_eval.copy()

    batch_index = np.arange(self.batch_size, dtype=np.int32)
    reward = batch_memory_qreward

    for i in batch_index:
            q_target[i] += (reward[i] + self.gamma * np.max(q_eval[i], axis=1))[0]

    #self.loss += self.model.fit(new_whold_states, self.Q_values, batch_size=32)
    self.cost = self.eval_model.train_on_batch(batch_memory_state, q_target)



    # Store the model
    if self.memory_counter% 10 == 0:
        print("Round ended! Save model!")
        self.eval_model.save_weights("time-cartpole-dqn.h5")



def reward_from_events(self, events: List[str], last_game_state_pos, game_state) -> int:
    """
    *This is to modify the rewards your agent get so as to en/discourage
    certain behavior.
    """

    game_rewards = {
        e.COIN_COLLECTED: 200,
        #e.INVALID_ACTION: -10,
        e.KILLED_SELF: -1000,
        e.COIN_FOUND: 1000,
        e.KILLED_OPPONENT:2000,
        e.GOT_KILLED:-200,
        e.OPPONENT_ELIMINATED:1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

        elif self.last_actions[0][-1]==5 and self.next_action=='BOMB':
            print('Repeating bomb!')
            reward_sum-=100

        # drop bomb, check if it can escape
        elif event == e.BOMB_DROPPED:
            state_for_explo = check_explostion_escape(self, last_game_state_pos, game_state)
            if state_for_explo:
                reward_sum += 2000
            else:
                # join the coin finding if explostion can find coins
                # state=find_coin?
                reward_sum -= 1000
        # valid action for a walk
        elif self.last_pos != last_game_state_pos:
            reward_sum += 100
        elif self.last_pos == last_game_state_pos:
            reward_sum += -10 # invalid action
            print('Invalid action for a walk!')





        else:
            reward_sum += 0
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    print(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum



# check if it can escape
def check_explostion_escape(self, now_pos, game_state):
    from items import Coin, Explosion, Bomb

    pos_x = now_pos[0]
    pos_y = now_pos[1]
    import settings as s
    blast = Bomb((pos_x, pos_y), game_state['self'][0], s.BOMB_TIMER, s.BOMB_POWER, 0)

    explosion_scale = blast.get_blast_coords(game_state['arena'])

    available_step=s.BOMB_TIMER

    left = (pos_x, pos_y-1)
    right = (pos_x, pos_y+1)
    up = (pos_x-1, pos_y)
    down = (pos_x+1, pos_y)

    state=[left, right, up, down]

    #find_target(self, state, available_step)

    # use A* to find path (theoretically best choice)
    #state_find_escape_path = a_star_algorithm(self,game_state, (pos_x, pos_y), target)


def find_target(self, state, distance):
    left = state[0]
    right = state[1]
    up = state[2]
    down = state[3]
    for i in range(distance):
        # if


        left[1]-=1
        right[1] += 1
        up[0] += 1
        down[0] -= 1


def a_star_algorithm(self, game_state, start_point, end_point):
    pass





