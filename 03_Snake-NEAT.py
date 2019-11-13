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

'''


# Global Variables
SCREEN_WIDTH = 500              # Width of screen (in pixels) 
CUBE_WIDTH = 20                 # Width of cube (in pixels) 
NN_MAX_GEN = 50000                # Number of generations to run
NN_MAX_MOVES = 100              # Restriction on number of moves
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
background = pygame.display.set_mode( (SCREEN_WIDTH,SCREEN_WIDTH) )


# Create classes

## Class: Snake
class Snake():

    def __init__(self, body, direction = (0,1)):
        '''
        Initialize the Snake class with a body parameter. 
        Default direction is DOWN 
        '''
        self.body = body
        self.direction = direction

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

    def moveSnake(self, direction):
        ''' 
        Move the snake. 
        Define where the next block (head) should be depending on the direction. 
        Logic for removing the last cube of body will be done in the main loop after moving the snake.
        '''
        next_block = [ self.body[0][0] + direction[0]*CUBE_WIDTH,
                       self.body[0][1] + direction[1]*CUBE_WIDTH ]
        self.body.insert(0, next_block)  # Add new head


    def area_around(self, snack):
        ''' 
        Return an array describing the area of "sight" around the snake
        Array will consist of three parts: 
        1. Distances to walls   - Array of 8 for every direction
        2. Distances to self    - Array of 8 for every direction
        3. Distance to snack    - Array of two for x and y 
        For (2.), a distance of 0 will mean that there is nothing
        All distances will be scaled to (roughly) fit between 0, 1
        '''
        a_walls = np.zeros(4)
        a_self  = np.zeros(4)
        a_snack = np.zeros(2)

        # Define local variables
        SCALE = SCREEN_WIDTH // CUBE_WIDTH
        array_counter = 0
        head_x = self.body[0][0]
        head_y = self.body[0][1]

        # Look N   - array_counter = 0 
        ## Distance to wall
        a_walls[array_counter] = abs(head_y - 0) / SCREEN_WIDTH
        ## Distance to self
        for i in range(SCALE):
            if [head_x, head_y - (i+1)*CUBE_WIDTH] in self.body:
                a_self[array_counter] = i / SCALE
                break

        # # Look NE  - array_counter = 1 
        # array_counter += 1
        # ## Distance to wall
        # ## -- The number of cubes, diagonally from the snake, to the nearest wall is the minimum number of cubes looking N or looking W
        # a_walls[array_counter] = min(abs(head_y - 0), abs(head_x - SCREEN_WIDTH)) / SCREEN_WIDTH
        # ## Distance to self
        # for i in range(SCALE):
        #     if [head_x + (i+1)*CUBE_WIDTH, head_y - (i+1)*CUBE_WIDTH] in self.body:
        #         a_self[array_counter] = i / SCALE
        #         break

        # Look E   - array_counter = 2
        array_counter += 1
        ## Distance to wall
        a_walls[array_counter] = abs(head_x - SCREEN_WIDTH) / SCREEN_WIDTH
        ## Distance to self
        for i in range(SCALE):
            if [head_x + (i+1)*CUBE_WIDTH, head_y] in self.body:
                a_self[array_counter] = i / SCALE
                break

        # # Look ES  - array_counter = 3
        # array_counter += 1
        # ## Distance to wall
        # a_walls[array_counter] = min(abs(head_x - SCREEN_WIDTH), abs(head_y - SCREEN_WIDTH)) / SCREEN_WIDTH
        # ## Distance to self
        # for i in range(SCALE):
        #     if [head_x + (i+1)*CUBE_WIDTH, head_y + (i+1)*CUBE_WIDTH] in self.body:
        #         a_self[array_counter] = i / SCALE
        #         break

        # Look S   - array_counter = 4
        array_counter += 1
        ## Distance to wall
        a_walls[array_counter] = abs(head_y - SCREEN_WIDTH) / SCREEN_WIDTH
        ## Distance to self
        for i in range(SCALE):
            if [head_x, head_y + (i+1)*CUBE_WIDTH] in self.body:
                a_self[array_counter] = i / SCALE
                break

        # # Look SW  - array_counter = 5
        # array_counter += 1
        # ## Distance to wall
        # a_walls[array_counter] = min(abs(head_y - SCREEN_WIDTH), abs(head_x - 0)) / SCREEN_WIDTH
        # ## Distance to self
        # for i in range(SCALE):
        #     if [head_x - (i+1)*CUBE_WIDTH, head_y + (i+1)*CUBE_WIDTH] in self.body:
        #         a_self[array_counter] = i / SCALE
        #         break

        # Look W   - array_counter = 6
        array_counter += 1
        ## Distance to wall
        a_walls[array_counter] = abs(head_x - 0) / SCREEN_WIDTH
        ## Distance to self
        for i in range(SCALE):
            if [head_x - (i+1)*CUBE_WIDTH, head_y] in self.body:
                a_self[array_counter] = i / SCALE
                break

        # # Look WN  - array_counter = 7
        # array_counter += 1
        # ## Distance to wall
        # a_walls[array_counter] = min(abs(head_x - 0), abs(head_y - 0)) / SCREEN_WIDTH
        # ## Distance to self
        # for i in range(SCALE):
        #     if [head_x - (i+1)*CUBE_WIDTH, head_y - (i+1)*CUBE_WIDTH] in self.body:
        #         a_self[array_counter] = i / SCALE
        #         break


        # Distance to snack
        a_snack[0] = (head_x - snack.body[0]) / SCREEN_WIDTH
        a_snack[1] = (head_y - snack.body[1]) / SCREEN_WIDTH

        return(np.concatenate((a_walls, a_self, a_snack)))


## Class: Snack
class Snack():

    def __init__(self, body):
        self.body = body

    def drawSnack(self, surface):
        pygame.draw.rect(surface,
                         (255,0,0),  # Red
                         ( self.body[0], self.body[1],
                           CUBE_WIDTH, CUBE_WIDTH )
                         )



# Functions

## drawWindow
def drawWindow(surface):
    surface.fill( (0,0,0) )     # Black
    snake.drawSnake(surface)
    snack.drawSnack(surface)
    pygame.display.update()

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
    global snake, snack, GEN
    GEN += 1
    # Clock
    clock = pygame.time.Clock()

    # Initial variables
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
       
        # Create first snake and snack
        empty_cells = [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                        for x in range(SCREEN_WIDTH // CUBE_WIDTH)
                            for y in range(SCREEN_WIDTH // CUBE_WIDTH) ]
        ## snake 
        snake = Snake( [random.choice(empty_cells)] )
        for cube in snake.body:
            index = empty_cells.index(cube)
            empty_cells.pop(index)
        ## snack
        snack = Snack( random.choice(empty_cells) ) 
        ## moves 
        moves = NN_MAX_MOVES

        # main loop:
        while True: 

            # Slow down the game
            if GEN > 2000: 
                clock.tick(100)

            # inputs: what the snake sees
            inputs = snake.area_around(snack)
            # outputs: which direction to go 
            output = net.activate(inputs)
            # direction: max of the inputs
            direction = np.argmax(output)

            if direction == 0:      # LEFT
                snake.direction = (-1, 0) 
            elif direction == 1:    # RIGHT 
                snake.direction = ( 1, 0) 
            elif direction == 2:    # UP
                snake.direction = ( 0,-1) 
            elif direction == 3:    # DOWN
                snake.direction = ( 0, 1)

            # Move snake
            snake.moveSnake(snake.direction)

            # Logic: snack
            if snake.body[0] == snack.body:
                genome.fitness += NN_MAX_MOVES
                moves += NN_MAX_MOVES

                # Find all empty cells
                empty_cells = [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                                for x in range(SCREEN_WIDTH//CUBE_WIDTH)
                                    for y in range(SCREEN_WIDTH//CUBE_WIDTH)  ]
                for body_part in snake.body:
                    index = empty_cells.index(body_part)
                    empty_cells.pop(index)

                # Make new snack
                snack = Snack( random.choice(empty_cells) )

            else:
                genome.fitness += 0.1
                moves -= 1
                snake.body.pop()

            # Logic for losing the game

            ## (1): Running into body part
            if snake.body[0] in snake.body[1:]:
                break
            ## (2): Running into screen edge
            if snake.body[0] not in [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                                for x in range(SCREEN_WIDTH//CUBE_WIDTH)
                                    for y in range(SCREEN_WIDTH//CUBE_WIDTH)  ]:
                break
            ## (3): out of moves
            if moves < 0:
                break

            drawWindow(background)


if __name__ == "__main__":
    # Set path to the configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)




