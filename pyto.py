import random
import pygame
pygame.init()

class player(object):

    def __init__(self,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.x_vel = 0
        self.y_vel = 0
        self.y_walkCount = 0
        self.x_walkCount = 0
        self.cube_num = [1,2,3,4]
        
    def draw(self, win):

        pygame.draw.rect(win, (0,255,0), (self.x,self.y,self.width,self.height))
"""
        pygame.draw.rect(win, (0,255,0), (self.x,self.y,self.width,self.height))
        for cube in self.cube_num:
            if self.y_vel == -15 :
                pygame.draw.rect(win, (0,255,0), (self.x - (cube * 16),self.y + (cube * 16),self.width,self.height))
            elif self.y_vel == -15:
                pygame.draw.rect(win, (0,255,0), (self.x,self.y + (cube * 16),self.width,self.height))
            elif self.y_vel == 15 :
                pygame.draw.rect(win, (0,255,0), (self.x,self.y - (cube * 16),self.width,self.height))
            elif self.x_vel == 15 :
                pygame.draw.rect(win, (0,255,0), (self.x- (cube * 16),self.y,self.width,self.height))
            elif self.x_vel == -15 :
                pygame.draw.rect(win, (0,255,0), (self.x+ (cube * 16),self.y,self.width,self.height))
"""
class Body_Cubes(object):

    def __init__(self,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cube_num = [1,2,3,4]
    def draw(self, win):
        pygame.draw.rect(win, (0,255,0), (self.x,self.y,self.width,self.height))

def redrawGameWindow():
    win.fill((0,0,0))
    snake.draw(win)
    body.draw(win)
    pygame.display.update()

x_walkCount = 0
y_walkCount = 0
snake = player(50, 50, 15,15)

win = pygame.display.set_mode((1000,600))
run = True
while run: 
    pygame.time.delay(300)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        if snake.x_vel == 15:
            pass
        else:
            snake.x_vel = -15
            snake.y_vel = 0

    elif keys[pygame.K_RIGHT]:
        if snake.x_vel == -15:
            pass
        else:
            snake.x_vel = 15
            snake.y_vel = 0

    elif keys[pygame.K_UP]:
        if snake.y_vel == 15 :
            pass
        else:
            snake.y_vel = -15
            snake.x_vel = 0

    elif keys[pygame.K_DOWN]:        
        if snake.y_vel == -15 :
            pass
        else:
            snake.y_vel = 15
            snake.x_vel = 0

    snake.x = snake.x + snake.x_vel
    snake.y = snake.y + snake.y_vel
    redrawGameWindow()

pygame.quit()            