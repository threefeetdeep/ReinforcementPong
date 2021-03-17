import argparse
from pong_rl import PongRL
from agent import Agent, train

GOALS_AT_GAME_END = 10  # the game finishes after this many total goals

if __name__ =='__main__':    
    parser = argparse.ArgumentParser(description='Reinforcement Pong Game')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--human', action='store_true', 
                        help='You play Pong as Player #1, CPU as Player #2')
    group.add_argument('--agent', action='store_true', 
                        help='RL Agent learns to play Pong as Player #1 from scratch')
    group.add_argument('--model', action='store_true', 
                        help='Agent is Player #1 using trained model vs. CPU as Player #2')
    arg_in = vars(parser.parse_args())

    if arg_in['model'] == True:
        train(training_phase=False)
    elif arg_in['human'] == True:
        game = PongRL(frame_rate = 30)
        # game loop
        while True:
            reward, game_over, score = game.play_step()
            if game_over == True:
                break   
        print('Final Score', score)       
        pygame.quit() 
    elif arg_in['agent'] == True:
        train(training_phase=True)
    
