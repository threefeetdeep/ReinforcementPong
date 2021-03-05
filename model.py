import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle

#  The Model: inherit from nn.Module class
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # initialise feedforward ANN
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):  # overrides nn.Module forward(self,x)
        # forward propagation
        #x = F.relu(self.linear1(x))  # input layer with RELU
        x = F.sigmoid(self.linear1(x))  # input layer with sigmoid activation

        x = self.linear2(x)     # raw linear output; we'll take max as move action
        #x = F.sigmoid(self.linear2(x))     # output layer with sigmoid activation
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        # save the model state dictionary 
        torch.save(self.state_dict(), file_name)
        

# The Trainer: is passed the model 
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # inputs can be single values (train_short) or list of tuples (train_long)
        # so convert to handle them in same way
        state = torch.tensor(state,dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)

        if len(state.shape) == 1:
            # force tuples of lists from memory to 1-D lists:
            state = torch.unsqueeze(state, 0) # will make x into (1,x) 
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            done = (done,) # as tuple, binary sequence of game progress/end


        # BELLMAN UPDATE EQUATION:
        # 1. Q = model.predict(state_n)

        # 2. Q_new = R = gamma*max(Q(state_n+1))  << next predicted Q value
        # we break this down into two steps:
        # pred.clone()  << gives e.g. [0,1,0] for "turn right"
        # preds[argmax(action)] = Q_new  <<< e.g. preds[2] = Q_new = turn right

        # give the ANN the state(s), and get its prediction(s) of action to take
        pred = self.model(state)
        target = pred.clone()

        # for each game step, estimate the action-value (estimated reward)
        for idx in range(len(done)):
            # current estimate at this step
            Q_new = reward[idx]
            # not including game over steps, compute:
            if not done[idx]:
                # updated estimate for this step
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state[idx]))

            # NOTE:
            # Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            #   |           |             |           |                     |
            # new Q     prev Q     discount rate                   predicted action
            #                                                      for next state
            #
            # see Sutton p131 eqn(6.8) "Bellman Equation" Q-Learning algorithm
            # New Q(s,a) = Q(s,a) + alpha[R(s,a) + lr.maxQ'(s',a') - Q(s,a)]
            #    |           |         |                  |             |
            #  Q_new       Q_new     slef.gamma*       ANN prediction on next state

            # Set the optimal action in the clone of action predictions
            # based on the reward estimate update above
            target[idx][torch.argmax(action).item()] = Q_new

        
        # reset to zero the gradient optimizer before backprop
        self.optimizer.zero_grad() 
        
        # the MSE loss between target and predicted, which we want to minimize
        # by tweaking the NN weights via backprop
        loss = self.criterion(target, pred)

        # do the back propogation step
        loss.backward() 

        # command to actually update the parameters (weights)
        self.optimizer.step()





