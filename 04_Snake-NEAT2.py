''' 
~~ 03_Snake-NEAT.py ~~
Create a Snake game that utilizes the neat-python package 
Draft: 
Basic snake game contains the following: 
- class Snake 
- class Snack 
- def drawWindow
- def playGame
Need to add the following: 
- def eval_genomes() <- formerly: playGame
- def run(config) 
            ~~~
Change input implementation to inlude the whole board. 
- Value  1 for snack 
- Value -1 for snake 
- Value  0 for empty
            ~~~ 
Change screen space so that __ number of snakes can play at once. 
This means that every snake within a generation will play at the same time. 
Time will be saved since program will not have to iterate over all individuals within a generation. 

'''


# Global Variables
SCREEN_HEIGHT = 720             # Height of screen (in pixels) 
SCREEN_WIDTH = 1200             # Width of screen (in pixels) 
SCREEN_WIDTH_i = 60             # With if the individual screens

CUBE_WIDTH = 6                  # Width of cube (in pixels)
NUM_CUBES = SCREEN_WIDTH_i // CUBE_WIDTH  # Number of cubes in each 
NN_MAX_GEN = 5000               # Number of generations to run
NN_MAX_MOVES = 30              # Restriction on number of moves
GEN = 0                         # Start at generation 1 



'''
TODO: config-feedforward.txt
1. adjust fitness_threshold
2. change num_inputs 
'''

# Import libraries
import pygame
import random
import numpy as np
import os
import neat
import visualize


pygame.init()
background = pygame.display.set_mode( (SCREEN_WIDTH,SCREEN_HEIGHT) )


# Create classes

## Class: Snake
class Snake():

    def __init__(self, start_x, start_y, direction = (0,1)):
        '''
        Initialize the Snake class with a body parameter. 
        Default direction is DOWN 
        '''
        self.direction = direction
        self.moves = NN_MAX_MOVES
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = self.start_x + SCREEN_WIDTH_i
        self.end_y = self.start_y + SCREEN_WIDTH_i
        self.body = [ [self.start_x + random.randrange(NUM_CUBES)*CUBE_WIDTH,
                       self.start_y + random.randrange(NUM_CUBES)*CUBE_WIDTH] ]
        self.snack_x, self.snack_y = self.makeSnack()

    def makeSnack(self):
        ''' 
        Return the snack x and y coordinates 
        such that they don't overlap with the snake body. 
        '''
        empty_cells = [ [self.start_x + x*CUBE_WIDTH, self.start_y + y*CUBE_WIDTH]
                        for x in range(NUM_CUBES)
                        for y in range(NUM_CUBES) ]
        for part in self.body:
            index = empty_cells.index(part)
            empty_cells.pop(index)
        return(random.choice(empty_cells))

    def drawSnake(self, surface):
        ''' 
        Draw the Snake. 
        Give the head a different color from the body. 
        '''
        for i in range(len(self.body)):
            # If head
            if i == 0:
                pygame.draw.rect(surface,
                                 (255,255,255),  # head is white
                                 ( self.body[i][0], self.body[i][1],
                                   CUBE_WIDTH, CUBE_WIDTH )
                                 )
            else:
                pygame.draw.rect(surface,
                                 (0,255,0),  # head is green
                                 ( self.body[i][0], self.body[i][1],
                                   CUBE_WIDTH, CUBE_WIDTH )
                                 )

    def drawSnack(self, surface):
        ''' Draw the snack '''
        pygame.draw.rect(surface,
                         (255,0,0),  # Red
                         ( self.snack_x, self.snack_y,
                           CUBE_WIDTH, CUBE_WIDTH )
                         )

    def moveSnake(self, direction):
        ''' 
        Move the snake. 
        Define where the next block (head) should be depending on the direction. 
        Logic for removing the last cube of body will be done in the main loop after moving the snake.
        '''
        next_block = [ self.body[0][0] + direction[0]*CUBE_WIDTH,
                       self.body[0][1] + direction[1]*CUBE_WIDTH ]
        self.body.insert(0, next_block)  # Add new head


    def area_around(self):
        ''' 
        Return an array describing the entire area around snake
        Array values will be as follows: 
         - Value  1: Snack position
         - Value -1: Snake position 
         - Value  0: Empty Space
        Also, append 2 values to describe the direction of snake
        '''
        area_around = np.zeros( (NUM_CUBES)**2 )
        for i in range(len(area_around)):
            x_pos = self.start_x + (i % NUM_CUBES) * CUBE_WIDTH
            y_pos = self.start_y + (i // NUM_CUBES) * CUBE_WIDTH
            if ([x_pos, y_pos]) in self.body:
                area_around[i] = -1
            if (x_pos == self.snack_x) and (y_pos == self.snack_y) :
                area_around[i] = 1
        area_around = np.concatenate((area_around, self.direction))
        return area_around

    def dToSnack(self):
        '''
        Find the distance from the head to the snack 
         * No need to take the sqrt after summing squares, since this will only be used for comparison purposes
        '''
        return ( (self.body[0][0] - self.snack_x)**2 + (self.body[0][1] - self.snack_y)**2 ) 

# Functions

## drawWindow
def drawWindow(surface, snakes):
    surface.fill( (0,0,0) )     # Black
    for snake in snakes: 
        snake.drawSnake(surface)
        snake.drawSnack(surface)
    drawGrid(SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH_i, surface) 
    pygame.display.update()

## drawGrid(SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH_i, surface)
def drawGrid(SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_WIDTH_i, surface):
    num_cols = SCREEN_WIDTH // SCREEN_WIDTH_i
    num_rows = SCREEN_HEIGHT // SCREEN_WIDTH_i

    x = 0
    for _ in range(num_cols):
        x = x + SCREEN_WIDTH_i
        pygame.draw.line(surface,
                         (255,255,255),
                         (x,0), (x,SCREEN_HEIGHT) )
    y = 0
    for _ in range(num_rows):
        y = y + SCREEN_WIDTH_i
        pygame.draw.line(surface,
                         (255,255,255),
                         (0,y), (SCREEN_WIDTH,y) ) 

## run(config_file):
def run(config_file):
    ''' 
    runs the NEAT algorithm 
    '''
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)

    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(200))

    # Run for the number of generations
    winner = p.run(eval_genomes, NN_MAX_GEN)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)
    

## eval_genomes(genomes, config)
def eval_genomes(genomes, config):
    ''' 
    Main run function for the game
    '''
    global snakes, GEN
    GEN += 1
    # Clock
    clock = pygame.time.Clock()

    # Containers
    ge = []
    nets = []
    snakes = []
    pos_i = 0


    # Initial variables
    for genome_id, genome in genomes:
        genome.fitness = 0  # Start with a fitness of 0 
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        ge.append(genome)
        snakes.append(Snake( (pos_i*SCREEN_WIDTH_i) % SCREEN_WIDTH ,
                             ( (pos_i*SCREEN_WIDTH_i) // SCREEN_WIDTH ) * SCREEN_WIDTH_i ) )
        pos_i += 1

    # main loop:
    while len(snakes) > 0: 
        # Slow down the game
        # clock.tick()

        for i, snake in enumerate(snakes):
            # inputs: what the snake sees
            input = snake.area_around()
            # output: output of NN
            output = nets[i].activate(input)
            # direction: max of outputs
            direction = np.argmax(output)

            if direction == 0:      # LEFT
                snake.direction = (-1, 0) 
            elif direction == 1:    # RIGHT
                snake.direction = ( 1, 0) 
            elif direction == 2:    # UP
                snake.direction = ( 0,-1) 
            elif direction == 3:    # DOWN
                snake.direction = ( 0, 1)

            # Move snake (and find distances)
            d_pre = snake.dToSnack()
            snake.moveSnake(snake.direction)
            d_post = snake.dToSnack()

            # Logic: snack
            if snake.body[0] == [snake.snack_x, snake.snack_y]:
                ge[i].fitness += NN_MAX_MOVES*3
                snake.moves += NN_MAX_MOVES
                snake.snack_x, snake.snack_y = snake.makeSnack()

            else:
                if d_pre > d_post:
                    ge[i].fitness -= 0.3
                elif d_post > d_pre: 
                    ge[i].fitness += 0.1
                snake.moves -= 1
                snake.body.pop()

            # Logic for losing the game

            ## (1): Running into body part
            if snake.body[0] in snake.body[1:]:
                snakes.pop(i)
                nets.pop(i)
                ge.pop(i)
            ## (2): Running into screen edge
            if snake.body[0] not in [ [snake.start_x + x*CUBE_WIDTH, snake.start_y + y*CUBE_WIDTH]
                        for x in range(NUM_CUBES)
                        for y in range(NUM_CUBES) ]:
                snakes.pop(i)
                nets.pop(i)
                ge.pop(i)
            ## (3): out of moves
            if snake.moves <= 0:
                snakes.pop(i)
                nets.pop(i)
                ge.pop(i)

        drawWindow(background, snakes)


if __name__ == "__main__":
    # Set path to the configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
