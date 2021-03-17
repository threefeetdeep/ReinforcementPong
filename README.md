# ReinforcementPong
My introduction to Reinforcememt Learning: a simple pong game with RL agent. Written in Python3, you wil need the following Python modules:

* torch
* pygame

![alt pong!](https://github.com/threefeetdeep/ReinforcementPong/blob/master/images/pong1.png?raw=true)

## The Game 
Player1 has the red paddle, and is on the left side of the Pong 'court'
Player2 has the blue paddle, and is on the right side.
A goal is scored if the ball makes it past a paddle.
A game is ended when a total of ten games have been scored. (e.g. final scores 5-5, 2-8, 9-1 etc.)

## Instructions
Pong can be run in three ways:

1) Player1 as a human (you!) versus a simple CPU controlled Player2
2) Player1 as the RL agent, learning to play from scratch against a CPU controlled Player2
3) Player1 as the RL agent, used the learned model against a CPU controlled Player 2

To play in these modes, uses:
1) `python3 PlayPong.py --human`
2) `python3 PlayPong.py --agent`
3) `python3 PlayPong.py --model`

## Settings
The game itself is highly configurable, using the constants at the top of pong_rl.py

The constants given here produce a game that a human player has a fair chance of winning.

## Agent Learning Mode
When training the agent, you can set the threshold at which learning will stop using the `LOSS_THRESHOLD` variable in agent.py.
This loss is currently the MSE loss between the neural network predicted action and the actual action taken by the agent at each
game step. The value of 0.15 will be met after about 15-30 training games (or 'episodes'). At this point, the agent has the skill 
of a 'poor' human player, but it will score a few points, but maybe struggle to win a game!

If you lower the `LOSS_THRESHOLD` dramatically, e.g. below 0.1, the agent's model will start to overfit the training data is has 
experienced, and do odd things like sitting at the top or bottom of the screen. It will occassionally win points by chance doing
this, but it is not in general the best behaviour. 

## Future Improvements
Some of thie thing that would be good to add to the game:

* Abililty to change the state-action model (e.g. from neural network to SVM or decision tree) before training the agent
* Use regularizatio to prevent overfitting
* Add an automatic stop to training once the agent has won X games against the CPU Player2.

![alt pong!](https://github.com/threefeetdeep/ReinforcementPong/blob/master/images/pong2.png?raw=true)

