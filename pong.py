# This version is human vs CPU only

import pygame
from enum import Enum
import random
from math import pi, sin, cos, copysign

pygame.init()
font_large = pygame.font.Font('Seven Segment.ttf', 48)
font_small = pygame.font.Font('Seven Segment.ttf', 16)

# rgb colors
WHITE = (255, 255, 255)     # ball
YELLOW = (200,200,0)        # score
RED = (200,0,0)             # P1 paddle (human/agent control) on left
BLUE = (0, 0, 200)          # P2 paddle (traditional cpu opponent) on right
BLACK = (0,0,0)             # background

# other game constants
FRAME_RATE = 30 # frame rate FRAME_RATE - higher is faster, recommend 40 or higher for training, 5 for human game!
GOALS_AT_GAME_END = 10  # the game finishes after this many total goals
NO_GOAL_FRAME_COUNT_TIMEOUT = 1000  # if no goals have been scored within this frame count, reset the ball with no goal being award to either side
BALL_RESTART_POSITION = 0.85  # how many screen widths away from the receiving player does the ball restart after a game (fair values 0.7-0.9)
PADDLE_X_MARGIN = 8 # how for paddle centre is set in from the edge
BALL_SIZE = 10
MAX_BALL_START_ANGLE = 30 # degrees from horizontal
MAX_SPIN = 50 # maximum spin when ball hits bat at shallow angle. Reduces with ball angle.
MAX_BALL_ANGLE = 60 # otherwise ball takes ages to bounce across pitch many times!
MIN_BALL_ANGLE = 15 # otherwise players don't have to move paddle very much!
PADDLE_INITIAL_SIZE = 60 # how long is the paddle at the start of the game
PADDLE_END_SIZE = 20  # how short will the paddle shrink to as the game progresses?
PADDLE_THICKNESS = 8 # paddle thickness
PADDLE_SPEED = 22 # paddle y pixel speed step of paddle motion up/down per frame
P2_SPEED_PENALTY = 8 # paddle y pixel speed penalty for traditional cpu player 
P2_DEAD_ZONE = 15 # if the ball y is +/- this many pixel of P2's paddle, P2 won't move
PADDLE_SHRINKAGE = 0 # paddle shrinkage in pixels each time a goal is scored
INITIAL_BALL_SPEED = 15 # pixels per game step. 
BALL_SPEED_INC_PCT = 1 # percentage ball FRAME_RATE increase each goal (sensible range 1-3)
SCREEN_HEIGHT = 480
SCREEN_WIDTH = 640

# direction of ball and paddles
class Direction(Enum):
    STILL = 0
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# ball vector - old pos, new pos, speed, angle from horizontal
class Vector():
    def __init__(self, x1, y1, x2, y2, speed, angle):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.speed = speed
        self.angle = angle

# paddle location
class Point():
    def __init__(self, x,y):
        self.x = x
        self.y = y

# a class to handle ball location, velocity and mechanics
class Ball:
    def __init__(self,game):

        self.game = game
        # set intial ball speed
        self.bv = Vector(0, 0,0,0, INITIAL_BALL_SPEED, 0) 
        # set location,angle and direction
        self.reset()    
    
    # reset ball location after goal (speed same, angle randomly set, direction random)
    def reset(self):
        # initial direction of ball is left or right
        if random.randint(0,1):
            self.direction = Direction.RIGHT
            mult = 1
        else:
            self.direction = Direction.LEFT
            mult = 0

        self.bv.x1 = 0
        self.bv.x2 = SCREEN_WIDTH * abs(mult - BALL_RESTART_POSITION)
        self.bv.y1 = 0
        self.bv.y2 = SCREEN_HEIGHT/2
        initial_angle = 0
        # don't want zero-ish angle (horizontal)
        while initial_angle < 5 and initial_angle > -5:
            initial_angle = random.randint(-MAX_BALL_START_ANGLE, MAX_BALL_START_ANGLE)
        self.bv.angle = initial_angle

        
        
    # move the ball according to its angle and speed
    def move(self):
        # compute speed from score
        goal_total = self.game.score[0] + self.game.score[1]
        self.bv.speed = INITIAL_BALL_SPEED ** ( ((100+BALL_SPEED_INC_PCT)/100) ** goal_total)
        # store previous location
        self.bv.x1 = self.bv.x2
        self.bv.y1 = self.bv.y2

        # compute new ball pos x
        if (self.direction == Direction.RIGHT):
            self.bv.x2  += int(self.bv.speed*cos(self.bv.angle*2*pi/360))
        else: # must be moving left!
            self.bv.x2  -= int(self.bv.speed*cos(self.bv.angle*2*pi/360))
        # compute ball pos y
        self.bv.y2 += int(self.bv.speed*sin(self.bv.angle*2*pi/360))

        # handle wall bounce
        self.check_wall_bounce()

        # useful sign and clamp functions for handling ball angle
        sign = lambda x: copysign(1, x)

        def clamp(x, minx,maxx):
            # clamp x: 
            # if x<0, -maxx <= x <= -minx
            # if x>=0, minx <= x <= maxx
            if sign(x) == 1:
                # positive
                return max(min(maxx, x), minx)
            else:
                # negative
                return max(min(-minx,x),-maxx)


        # handle paddle collision & add some "spin" ie. angle change with paddle motion
        if self.check_paddle_collision(self.game.P1) is True:      
            # add spin
            if self.game.direction == Direction.DOWN:
                spin = -MAX_SPIN/(abs(self.bv.angle)**0.5 + 1) * sign(self.bv.angle)
            elif self.game.direction == Direction.UP:
                spin = MAX_SPIN/(abs(self.bv.angle)**0.5 + 1) * sign(self.bv.angle)
            else:
                spin = 0
            #P1 is on left - change sign of ball angle, and direction
            self.direction = Direction.RIGHT
            self.bv.x2 = 2*PADDLE_X_MARGIN - self.bv.x2

            print("angle0:", self.bv.angle)
            self.bv.angle =  clamp(self.bv.angle + spin, MIN_BALL_ANGLE, MAX_BALL_ANGLE)
            print("spin:  ",spin)
            print("angle1:",(self.bv.angle))
            print()

        if self.check_paddle_collision(self.game.P2) is True:
            # add spin
            if self.game.direction == Direction.DOWN:
                spin = MAX_SPIN/(abs(self.bv.angle)**0.5 + 1) * sign(self.bv.angle)
            elif self.game.direction == Direction.UP:
                spin = -MAX_SPIN/(abs(self.bv.angle)**0.5 + 1) * sign(self.bv.angle)
            else:
                spin = 0
            
            #P2 is on right - change sign of ball angle, and direction
            if self.direction == Direction.RIGHT:
                self.direction = Direction.LEFT
                self.bv.x2 = 2*SCREEN_WIDTH - PADDLE_X_MARGIN - self.bv.x2
            self.bv.angle =  clamp(self.bv.angle + spin, MIN_BALL_ANGLE, MAX_BALL_ANGLE)
           

        # and check for goal (if ball made it *past* paddle!)
        if self.bv.x2 > SCREEN_WIDTH:
            # P1 has a goal
            self.game.score[0] += 1
            self.game.goal = True
        if self.bv.x1 < 0:
            # P2 has a goal
            self.game.score[1] += 1
            self.game.goal = True

        
    # automatically handle wall bounce
    def check_wall_bounce(self):
        # check for top/bottom wall bounce
        if self.bv.y2 < 0:
            # change sign of angle
            self.bv.angle = -1 * self.bv.angle
            # for shallow angles, add a bit more to stop "stationary play"
            if abs(self.bv.angle) < 15:
                self.bv.angle *= 1.3
            # change y by double the overshoot (i.e. change the sign)
            self.bv.y2 = -1 * self.bv.y2
        if self.bv.y2 > SCREEN_HEIGHT:
            # change sign of angle
            self.bv.angle = -1 * self.bv.angle
            # for shallow angles, add a bit more to stop "stationary play"
            if abs(self.bv.angle) < 15:
                self.bv.angle *= 1.3
            # change y by double the overshoot (i.e. the amount we've exceeded the boundary)
            self.bv.y2 = 2 * SCREEN_HEIGHT - self.bv.y2 

    # check for ball-to-paddle collision
    def check_paddle_collision(self,player):
        # see https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
        def ccw(A,B,C):
            return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
        def intersect(A,B,C,D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        # create points for ball start/end trajectory
        bs = Point(self.bv.x1, self.bv.y1)
        be = Point(self.bv.x2, self.bv.y2)
        # create points for paddle top/bottom ends
        pt = Point(player.x, player.y + self.game.paddle_length/2)
        pb = Point(player.x, player.y - self.game.paddle_length/2)

        # return true if intersection of ball trajectory and paddle
        return intersect(bs,be,pt,pb)


class PongRL:    
    def __init__(self, w=SCREEN_WIDTH, h=SCREEN_HEIGHT):
        # screen size
        self.w = w
        self.h = h

        # current user paddle direction
        self.direction = Direction.STILL

        # current CPU paddle direction
        self.cpu_direction = Direction.STILL

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Pong')
        self.goal = False
        self.clock = pygame.time.Clock()

        #reset can also be under agent control between each training game
        self.reset()

 

    # reset the game to starting state  
    def reset(self, game_number=0):
        self.game_number = game_number

        # create new ball
        self.ball = Ball(self)

        # reset goal flag
        self.goal = False
  
        # initialise P1 and P3 paddle centres and lengths
        self.P1 = Point(PADDLE_X_MARGIN, self.h/2)
        self.P2 = Point(self.w-PADDLE_X_MARGIN, self.h/2)
        self.paddle_length = PADDLE_INITIAL_SIZE

        self.score = [0,0]
        self.frame_iteration = 0

    # play one frame of the game, accepting action input from the agent if provided
    def play_step(self):
     
        # increment frame count
        self.frame_iteration += 1
        game_over = False

        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
          
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
            else:
                self.direction = Direction.STILL


        # move the paddles (before moving the ball to give the players a chance!)
        self.move_P1_paddle()
        self.move_P2_paddle()

        # move the ball & check for wall bounce
        self.ball.move()


        # update UI and clock
        self._update_ui()
        self.clock.tick(FRAME_RATE)

        # if a goal was scored, reset
        if self.goal == True:
            self.ball.reset()
            # shrink paddle sizes as more goals are scored
            self.paddle_length -= PADDLE_SHRINKAGE
            # clear goal flag
            self.goal = False

        return game_over, self.score

    # CPU-controlled paddle
    def move_P2_paddle(self):
        # track the ball but occassionally make a wrong move
        r = random.randint(0,100)
        if (r > 40) and (self.ball.bv.y2 > (self.P2.y + 5)):
            self.P2.y += PADDLE_SPEED
            if self.P2.y > SCREEN_HEIGHT-self.paddle_length/2:
                self.P2.y = SCREEN_HEIGHT-self.paddle_length/2
        elif (r > 40) and (self.ball.bv.y2 < (self.P2.y - 5)):
            self.P2.y -= PADDLE_SPEED
            if self.P2.y < self.paddle_length/2:
                self.P2.y = self.paddle_length/2
        elif r > 30 and r <= 40:
            # move up
            self.P2.y += PADDLE_SPEED
            if self.P2.y > SCREEN_HEIGHT-self.paddle_length/2:
                self.P2.y = SCREEN_HEIGHT-self.paddle_length/2
        elif r > 10 and r <= 30:
            # move down
            self.P2.y -= PADDLE_SPEED
            if self.P2.y < self.paddle_length/2:
                self.P2.y = self.paddle_length/2
        else:
            # no move
            pass

    
    # human-controlled paddle
    def move_P1_paddle(self):
        if self.direction == Direction.DOWN:
            self.P1.y += PADDLE_SPEED
            if self.P1.y > SCREEN_HEIGHT-self.paddle_length/2:
                self.P1.y = SCREEN_HEIGHT-self.paddle_length/2
        elif self.direction == Direction.UP:
            self.P1.y -= PADDLE_SPEED
            if self.P1.y < self.paddle_length/2:
                self.P1.y = self.paddle_length/2

    def _update_ui(self):
        self.display.fill(BLACK)
        
        # draw paddles
        pygame.draw.rect(self.display, RED, pygame.Rect(self.P1.x-PADDLE_THICKNESS/2, 
        self.P1.y-self.paddle_length/2, PADDLE_THICKNESS, self.paddle_length))
        pygame.draw.rect(self.display, BLUE, pygame.Rect(self.P2.x-PADDLE_THICKNESS/2, 
        self.P2.y-self.paddle_length/2, PADDLE_THICKNESS, self.paddle_length))

        # draw ball   
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.ball.bv.x2, self.ball.bv.y2, BALL_SIZE, BALL_SIZE))
        
        # show score in the middle
        text = font_large.render(str(self.score[0]) + ":" + str(self.score[1]), True, YELLOW)
        self.display.blit(text, [SCREEN_WIDTH/2 - 20, 5])

        # show frame number bottom left
        text = font_small.render("Frame: " + str(self.frame_iteration), True, WHITE)
        self.display.blit(text, [10, SCREEN_HEIGHT - 30])

        # show game count bottom right
        text = font_small.render("Game: " + str(self.game_number), True, WHITE)
        self.display.blit(text, [SCREEN_WIDTH - 80, SCREEN_HEIGHT - 30])

        pygame.display.flip()

if __name__ == '__main__':
    game = PongRL()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break
        if  (score[0]+score[1]) >= GOALS_AT_GAME_END:
            break
        
    print('Final Score', score)       
    pygame.quit() 