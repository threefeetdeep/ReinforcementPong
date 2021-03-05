# ACTIONS (what can the agent do in the environment, ANN output layer)
# [1,0] move paddle up
# [0,1] move paddle down
# [0,0,1] don't move paddle

# REWARDS (what do we want it to achieve: goals and wins!)
# +1 hit the ball
# +5 score goal     -5 let in goal
# +10 win game       -10 lose game

# STATES (what information can the agent use to decide on the action?, ANN input layer)
# ATTEMPT #1: numeric state variables
# ball-to-paddle x 
# ball-to-paddle y  
# e.g. [30, 14], ball-to-paddle_x = 30, ball-to-paddle_y = 14

# ATTEMPT #2: binary state variables:
# 


import torch
import random
import numpy as np
from collections import deque
from pong_rl import PongRL, Direction, Vector, Point
from model import Linear_QNet, QTrainer
import itertools

MAX_MEMORY = 20000
BATCH_SIZE = 1000  # training batch size taken from MEMORY (covers typically a few games)
LR = 0.001
NUM_GAMES_NO_MORE_EPS = 100 # no more random actions beyond this game count
DISCOUNT_RATE = 0.996 # 0 is myopic -immediate rwd only; 1 uses all past rwds equally
NUM_ACTIONS = 3     # number of actions the agent can take, one-hot encoded.

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0        # eps-Greedy value
        self.gamma = DISCOUNT_RATE   # discount rate (<1, usually 0.8-0.9~ish)
        self.memory = deque(maxlen= MAX_MEMORY)   # if exceeded, loses left (FIFO)
        self.model = Linear_QNet(2,32,3) # input, hidden, out layer sizes 
        self.trainer = QTrainer(model=self.model, lr=LR, gamma=self.gamma)
        
    # there are 3 state values
    def get_state(self, game):
        bpx = (game.ball.bv.x2 - game.P1.x)  # range 0 to 1
        bpy = (game.ball.bv.y2 - game.P1.y)    # range -1 to + 1

        state = [
            # ball-to-paddle x
            bpx,
            # ball-to-paddle y
            bpy
            ]


        return np.array(state, dtype=int)

    def remember(self,state, action, reward, next_state, done):
        # FIFO popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))  # store a tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            m = len(self.memory)
            # NOTE: below, both random and sequential batch training work OK!
        
            # RANDOM:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return list of tuples

            # SEQUENTIAL:
            #range_start = random.randint(0,m-BATCH_SIZE-1)
            #mini_sample = list(itertools.islice(self.memory, range_start, range_start+1000))
            
        else: # nto many samples yet available
            mini_sample = self.memory
        
        # use zip function
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # train over may random samples from previous games
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self,state, action, reward, next_state, done):
        # train for current single game step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # eps-Greedy exploration/exploitation trade off random moves
        # (make fewer random moves the better the agent gets)
        # NOTE: epsilon can go negative (ignored, as if it was 0)
        # starting at 0.5 and reduces to zero by NUM_GAMES_NO_MORE_EPS
        self.epsilon = NUM_GAMES_NO_MORE_EPS - self.n_games
        final_move = [0,0,0]
        
        # RANDOM MOVE:
        if random.randint(0,NUM_GAMES_NO_MORE_EPS*2) < self.epsilon: 
            move = random.randint(0,NUM_ACTIONS-1)
            final_move[move] = 1
        # GREEDY MOVE:
        else: # not random, get best action (from prediction)
            state0 = torch.tensor(state, dtype=torch.float)
             # pytorch will call Forward function with 'state0' as 'x'
            prediction = self.model(state0) 
            move = torch.argmax(prediction).item()  # move index is returned from ANN
            final_move[move] = 1

        return final_move


# global train function
def train():
    total_score = 0
    record = 0
    agent = Agent()
    game = PongRL()
    
    # Agent training loop
    while True:
        # get old state
         state_old = agent.get_state(game)

         # get move
         final_move = agent.get_action(state_old)

         # perform move and get new state
         reward, done, score = game.play_step(final_move)
         state_new = agent.get_state(game)

         # train short memory
         agent.train_short_memory(state_old, final_move, reward, state_new, done)

         # remember in deque FIFO
         agent.remember(state_old, final_move, reward, state_new, done)

        # after each game
         if done:
             # train long memory (experience replay, train again on old moves)
             game.reset(agent.n_games)
             agent.n_games += 1
             agent.train_long_memory()

             if score[0] > record:
                 record = score[0]
                 agent.model.save()
    
            # summarise game results
             total_score += score[0]
             mean_score = total_score / agent.n_games

             print('Game,' + str(agent.n_games) +  '; Score:' +  str(score[0]) +
               '; Record:' + str(record) + '; Mean Score:' + str(mean_score))


# run with 'python3 agent.py'
if __name__ == '__main__':
    # no arguments, train from scratch
    train()

    # TBD: if argument = "trained", run trained model, no learning.
