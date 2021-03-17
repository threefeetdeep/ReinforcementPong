# ACTIONS (what can the agent do in the environment, ANN output layer)
# -------
# [1,0] move paddle up
# [0,1] move paddle down
# [0,0,1] don't move paddle

# REWARDS (what do we want it to achieve: goals and wins!)
# -------
# +1 hit the ball
# +5 score goal     -5 let in goal
# +10 win game       -10 lose game

# STATES (what information can the agent use to decide on the action?, ANN input layer)
# ------
# ATTEMPT #1: numeric state variables
# ball-to-paddle x 
# ball-to-paddle y  
# e.g. [30, 14], ball-to-paddle_x = 30, ball-to-paddle_y = 14

# ATTEMPT #2: 'binary' state variables:
# [x,y] where x=-1 if ball y above paddle, 0 if inline, +1 if ball y below paddle
# and y = 0 if ball moving away, +1 if ball moving towards agent paddle.

# ATTEMPT #3: more binary state variables:
# [x,y] where x=-1 if ball y above paddle, 0 if inline, +1 if ball y below paddle
# and z = 0 if ball moving away, +1 if ball moving towards agent paddle.
# and u = 1 if ball angle is positive, -1 if ball angle is negative
# e.g. [ 0,1,0,-1] ball below agents paddle, ball moving away, ball angle negative

# We want the agent to stop training when the MSE loss for the state-action neural network
# drops below a certain limit (e.g. ~0.15), otherwise if we keep training it, we start to overfit the
# network, and the agent can start to act "funnily" i.e. sitting in the corners!

import torch
import random
import numpy as np
from collections import deque
from pong_rl import PongRL, Direction, Vector, Point
from model import Linear_QNet, QTrainer
import itertools

MAX_MEMORY = 200000
BATCH_SIZE = 4000  # training batch size taken from MEMORY (covers typically a few games)
LOSS_THRESHOLD = 0.15  # when NN loss reaches this target, stop training and just play!
LR = 0.001  # learning rate (regularization) to reduce overfitting
NUM_GAMES_NO_MORE_EPS = 100 # no more random actions beyond this game count
DISCOUNT_RATE = 0.99 # 0 is myopic -immediate rwd only; 1 uses all past rwds equally
NUM_ACTIONS = 2     # number of actions the agent can take, one-hot encoded.
STATE_SIZE = 4

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0        # eps-Greedy value
        self.gamma = DISCOUNT_RATE   # discount rate (<1, usually 0.8-0.9~ish)
        self.memory = deque(maxlen= MAX_MEMORY)   # if exceeded, loses left (FIFO)
        self.model = Linear_QNet(STATE_SIZE,16,NUM_ACTIONS) # input, hidden, out layer sizes 
        self.trainer = QTrainer(model=self.model, lr=LR, gamma=self.gamma)
        
    # what is a good set of "states" for the game? Agent actions should change the state...
    def get_state(self, game):
        # is ball above or below the agent's paddle?
        above = below = 0
        if game.ball.bv.y2 < (game.P1.y - game.paddle_length/2):
            above = 1
        elif game.ball.bv.y2 >(game.P1.y + game.paddle_length/2):
            below = 1

        # is the ball coming towards or going away from the agent's paddle?
        if game.ball.ball_direction == Direction.LEFT:
            towards = 1
        else:
            towards = 0
        
        # what is the sign of the ball angle?
        bounce_dir = game.ball.get_angle_sign()      
   
        # build state array
        state = [towards, 
                 above,
                 below,
                 bounce_dir
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
            #mini_sample = random.sample(self.memory, BATCH_SIZE) # return list of tuples

            # SEQUENTIAL:
            range_start = random.randint(0,m-BATCH_SIZE-1)
            mini_sample = list(itertools.islice(self.memory, range_start, range_start+1000))
            
        else: # not many samples yet available
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
        final_move = [0,0]
        
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
def train(training_phase=True):
    total_score = 0
    record = 0
    agent = Agent()
    
    mse_loss = 999 # default crazy high value

    if (training_phase == False):
        # load model, skip learning phase
        agent.model.load()
        # game running at human-viewable 30FPS.
        game = PongRL(frame_rate=30)
    else:
        # game running at max frame rate for faster trainin
        game = PongRL()

    # Agent training loop
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move
        reward, done, score = game.play_step(final_move)

        # train short memory
        if (mse_loss > LOSS_THRESHOLD and training_phase == True):
            # get new game state
            state_new = agent.get_state(game)
            # train agent with this new state, the old state, the reward etc.
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            # remember in deque FIFO
            agent.remember(state_old, final_move, reward, state_new, done)
        elif (mse_loss <= LOSS_THRESHOLD and training_phase == True):
            training_phase = False
            print("Training Completed: agent switching to Play Mode!")
      

        # after each game
        if done:
             # train long memory (experience replay, train again on old moves)
            game.reset(agent.n_games)
            agent.n_games += 1
            if (training_phase == True):
                agent.train_long_memory()
                agent.model.save()

            if score[0] > record:
                record = score[0]
                

        # summarise game results
            total_score += score[0]
            mean_score = total_score / agent.n_games
            

            print('Game,' + str(agent.n_games) +  '; Score:' +  str(score[0]) +
               '; Record:' + str(record) + '; Mean Score:' + str(mean_score))
            
            if (training_phase == True):
                mse_loss = agent.trainer.loss
                print('MSE Loss:' + str(mse_loss.item()))
                print()


# run with 'python3 agent.py'
if __name__ == '__main__':
    # no arguments, train from scratch
    train()

    # TBD: if argument = "trained", run trained model, no learning.
