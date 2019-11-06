''' 
01_Snake.py
~~~
I have already created a basic snake game. 

Let's try to clean up the Snake game code. 
Then, I will also add a Neural Network to control the snake 

Draft: 
- class Snake 
- class Snack 
- def drawWindow 
- def playGame

'''

# Import libraries
import pygame
import random

# Global Variables
SCREEN_WIDTH = 500
CUBE_WIDTH = 20

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


# Basic Functions

## Func : drawWindow
def drawWindow(surface):
    # global s
    surface.fill( (0,0,0) )
    s.drawSnake(surface)
    snack.drawSnack(surface) 
    pygame.display.update()

## Func: Main
def main():
    global SCREEN_WIDTH, s, snack

    # Initialize the pygame module
    pygame.init()
    background = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
    clock = pygame.time.Clock()

    # Create the snake
    s = Snake( [ [60,60] ] )
    # Create snack
    snack = Snack( [200,200] )

    # Main loop:
    running = True
    while( running ):

        # Slow down the game
        clock.tick(15)

        for event in pygame.event.get():

            # Quit
            if event.type == pygame.QUIT:
                running = False

            # Logic for key presses, and direction change
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                s.direction = (-1, 0) 
            elif keys[pygame.K_RIGHT]:
                s.direction = ( 1, 0) 
            elif keys[pygame.K_UP]:
                s.direction = ( 0,-1) 
            elif keys[pygame.K_DOWN]:
                s.direction = ( 0, 1) 


        # Logic for snack collision
        if (s.body[0][0]+s.direction[0]*CUBE_WIDTH, s.body[0][1]+s.direction[1]*CUBE_WIDTH) == (snack.body[0], snack.body[1]):
            s.moveSnake( s.direction, snack = True )

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


        # Logic for losing the game
        ## (1) Running into body part
        if s.body in s.body[1:]:
            running = False
        ## Run into screen edge
        if s.body[0] not in [ [x*CUBE_WIDTH, y*CUBE_WIDTH]
                              for x in range(SCREEN_WIDTH//CUBE_WIDTH)
                              for y in range(SCREEN_WIDTH//CUBE_WIDTH)  ]:
            running = False

        drawWindow(background)


    print("Your score is ", len(s.body))



        
## Run the main function
if __name__ == "__main__":
    main()                      # Call the funciton
    


