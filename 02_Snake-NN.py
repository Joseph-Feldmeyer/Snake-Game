''' 
02_Snake-NN.py
~~~
I have already created a basic snake game. 

Basic snake game contains following: 
- class Snake 
- class Snack 
- def drawWindow 
- def playGame

Need to add following: 
- class Network 
    - def feedforward: returns the output of the network 
- class Generation
    - def interpopulate: returns new nn_list to be used in next gen 


main(): 
  - initialize backgrounds: 
  - while GEN < NN_MAX_GEN: 
    - if GEN == 1: 
      - create 100 random NN's 
    
    - for nn in gen.nn_list: 
      - gen.fitness(nn) = nn.run_game()

    - next_nn_list = gen.interpopulate()
    - GEN += 1
    - gen = Generation(next_nn_list, GEN)

    - print(GEN: best fitness score) 

run_game(nn): 
  - create snake 
  - create snack 
  - score = 0 
  - for event: 
    - inputs = (length to walls) (length to snack) 
    - output = nn.feedforward(inputs) 
    - move based off of output 
    - moveCounter -= 1
    - score += 1

    - if snack: score += 200 
    - if death: return score


'''

# Import libraries
import pygame
import random
import numpy as np
import itertools
import math

# Global Variables
SCREEN_WIDTH = 300
CUBE_WIDTH = 20

# Neural Network variables:
NN_SHAPE = [ 8, 20, 20, 4 ]
NN_POPULATION = 1000             # Number of populations in each generation
NN_BEST_NUM = 5                 # Number of best fitness NNs to carry to next generation 
NN_MAX_GEN = 200                # Number of generations
NN_MAX_MOVES = 50              # Maximum number of available moves assigned at the beginning of the game
NN_MAX_SCORE = SCREEN_WIDTH**2 / CUBE_WIDTH**2  # Maximum possible score
GEN = 1                         # Start at 1st generation


# Classes

## Class : Snake
class Snake():

    def __init__(self, body, direction=(0,1)):
        self.body = body
        self.direction = direction

    def drawSnake(self, surface):
        for i in range(len(self.body)):
            if i == 0:
                pygame.draw.rect(surface,
                                 (255,255,255),
                                 ( self.body[i][0], self.body[i][1],
                                   CUBE_WIDTH, CUBE_WIDTH )
                                 )
            else:
                pygame.draw.rect(surface,
                                 (0,255,0),
                                 ( self.body[i][0], self.body[i][1],
                                   CUBE_WIDTH, CUBE_WIDTH )
                                 )

    def moveSnake(self, direction, snack = False):
        next_block = [ self.body[0][0]+direction[0]*CUBE_WIDTH,
                       self.body[0][1]+direction[1]*CUBE_WIDTH ]
        self.body.insert(0, next_block)
        
        if not snack:
            self.body.pop()

    def area_around(self):
        area = np.zeros(25)
        area_counter = 0

        # Start from top_left
        row_co = self.body[0][0]-2*CUBE_WIDTH
        for x in range(5):
            col_co = self.body[0][1]-2*CUBE_WIDTH
            for y in range(5):
                current_pos = [row_co + x*CUBE_WIDTH, col_co + y*CUBE_WIDTH]
                if current_pos not in [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                              for x in range(SCREEN_WIDTH//CUBE_WIDTH)
                              for y in range(SCREEN_WIDTH//CUBE_WIDTH)  ]:
                    area[area_counter] = 1
                elif current_pos in self.body:
                    area[area_counter] = 1

                area_counter += 1
        return(area) 


## Class : Snack
class Snack():

    def __init__(self, body):
        self.body = body

    def drawSnack(self, surface):
        pygame.draw.rect(surface,
                         (255,0,0),
                         ( self.body[0], self.body[1],
                           CUBE_WIDTH, CUBE_WIDTH )
                         )


## Class : Network
class Network():

    def __init__(self, shape):
        self.num_layers = len(shape)
        self.shape = shape
        self.biases = [np.random.randn(y) for y in shape[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(shape[:-1], shape[1:])]

    def feedforward(self, a):
        """ Return the output of the network if 'a' is the input. """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a))
        return a


## Class : Generation
class Generation():

    def __init__(self, nn_list, gen):
        """ 
        Create the generation. 
        Each generation has a list of Networks, a list 
        of the fitness of each network, and a generation number
        """
        self.nn_list = nn_list
        self.fit_list = np.zeros(len(nn_list))  
        self.gen = gen

    def evaluate(self):
        """ Find and return the top 5 nns in terms of fitness """
        top_5_index = np.argpartition(self.fit_list, -25)[-25:]  
        return np.take(self.nn_list, top_5_index)

    def create_child(self, perm):
        perm1, perm2 = perm[0], perm[1]

        new_NN = Network(perm1.shape) 
        
        # for i in range(len(perm1.biases)):
        #     for j in range(len(perm1.biases[i])):
        #         if j <= len(perm1.biases)//4:
        #             new_NN.biases[i][j] = perm1.biases[i][j]
        #         else:
        #             new_NN.biases[i][j] = perm2.biases[i][j]

        for i in range(len(perm1.weights)):
            for j in range(len(perm1.weights[i])):
                if j <= len(perm1.weights[i])//2:
                    new_NN.biases[i][j] = perm1.biases[i][j]
                else:
                    new_NN.biases[i][j] = perm2.biases[i][j]

        return new_NN


    def interpopulate(self):
        """ 
        Return a new nn_list with the following: 
        - top 5 NN
        - top 5 x 4 = 20 interpopulated NN
        - 3 variations each of the above 
        """

        # Top 5
        next_gen_NN = top_5_nn = self.evaluate()
        
        # 20 children
        # for i in itertools.permutations(top_5_nn, 2):
        #     next_gen_NN = np.append(next_gen_NN, self.create_child(i))

        # 75 slight mutations
        for i in range(40):
            for j in range(25):
                new_NN = Network(next_gen_NN[j].shape)
                new_NN.biases = next_gen_NN[j].biases
                new_NN.weights = next_gen_NN[j].weights

                for k in range(len(new_NN.biases)):
                    for l in range(len(new_NN.biases[k])):
                        if np.random.randn() > 3:
                            new_NN.biases[k][l] = np.random.randn()

                for k in range(len(new_NN.weights)):
                    for l in range(len(new_NN.weights[k])):
                        if np.random.randn() > 3:
                            new_NN.weights[k][l] = np.random.randn()

                next_gen_NN = np.append(next_gen_NN, new_NN) 
            
        return(next_gen_NN) 

    
        
        

    


# Basic Functions

## Func : sigmoid
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

## Func : drawWindow
def drawWindow(surface):
    # global s
    surface.fill( (0,0,0) )
    s.drawSnake(surface)
    snack.drawSnack(surface) 
    pygame.display.update()

## Func : main
def main():
    global SCREEN_WIDTH, GEN

    # Initialize the pygame module
    pygame.init()
    background = pygame.display.set_mode( (SCREEN_WIDTH,SCREEN_WIDTH) )
    
    if GEN == 1:
        nn_list = []
        for _ in range(100):
            nn_list.append(Network( [27, 20, 20, 4] ))
        gen = Generation(nn_list, GEN)

    while GEN <= NN_MAX_GEN: 
        for i, nn in enumerate(gen.nn_list):
            gen.fit_list[i] = run_game(nn, background)

        next_nn_list = gen.interpopulate()
        print("                                                 ", GEN, ": Best finess score was ", max(gen.fit_list))
        GEN += 1
        gen = Generation(next_nn_list, GEN)

def run_game(nn, background):
    global s, snack 
    # Clock
    clock = pygame.time.Clock()
    # Create snake
    s = Snake( [ [60,60] ] )
    # Create first snack
    snack = Snack( [200,200] )

    # Score
    score = 0
    # Move counter
    moves = NN_MAX_MOVES

    # Main loop:
    while True:

        # Slow down the game
        # clock.tick(300)

        inputs = s.area_around()
        inputs = np.append(inputs, distance_to_snack(s, snack))
        inputs = np.append(inputs, angle_to_snack(s, snack))

        output = nn.feedforward(inputs)

        # Find direction
        direction = np.argmax(output)

        # Lose method: (0) Face itself
        if (direction == 0 and s.direction == ( 1, 0)):
            return score
        elif (direction == 1 and s.direction == (-1, 0)):
            return score
        elif (direction == 2 and s.direction == ( 0, 1)):
            return score
        elif (direction == 3 and s.direction == ( 0,-1)):
            return score

        if direction == 0:      # LEFT
            s.direction = (-1, 0) 
        elif direction == 1:    # RIGHT 
            s.direction = ( 1, 0) 
        elif direction == 2:    # UP
                s.direction = ( 0,-1) 
        elif direction == 3:    # DOWN
            s.direction = ( 0, 1)

        # Logic for snack
        if (s.body[0][0]+s.direction[0]*CUBE_WIDTH, s.body[0][1]+s.direction[1]*CUBE_WIDTH) == (snack.body[0], snack.body[1]):
            s.moveSnake( s.direction, snack = True )
            score += 100
            moves += NN_MAX_MOVES

            # Find all empty cells:
            empty_cells = [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                            for x in range(SCREEN_WIDTH//CUBE_WIDTH)
                                for y in range(SCREEN_WIDTH//CUBE_WIDTH)  ]
            for body_part in s.body:
                index = empty_cells.index(body_part)
                empty_cells.pop(index)

            # Choose a random empty cell
            snack_x, snack_y = random.choice(empty_cells) 
            # New snack
            snack = Snack( [snack_x, snack_y] )

        else:
            s.moveSnake( s.direction )
            score += 0.1
            moves -= 1

        # Logic for losing the game
        ## (1) Running into body part
        if s.body[0] in s.body[1:]:
            return score
        ## (2) Run into screen edge
        if s.body[0] not in [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                              for x in range(SCREEN_WIDTH//CUBE_WIDTH)
                              for y in range(SCREEN_WIDTH//CUBE_WIDTH)  ]:
            return score
        ## (3) moves < 0
        if moves < 0:
            return score

        drawWindow(background)


# Other useful functions: 
def distance_to_snack(snake, snack):
    x = snake.body[0][0] - snack.body[0]
    y = snake.body[0][1] - snack.body[1]
    distance = math.sqrt(x**2 + y**2) / (SCREEN_WIDTH*1.5)
    return distance

def angle_to_snack(snake, snack):
    x = snake.body[0][0] - snack.body[0]
    y = snake.body[0][1] - snack.body[1]
    return math.atan2(y, x)


## Run the main function
if __name__ == "__main__":
    main()


