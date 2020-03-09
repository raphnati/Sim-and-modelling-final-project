import pygame, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
import math

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

partitions = 20


def get_coord(maincoord, screen_height, screen_width):
    modx = screen_width/partitions
    mody = screen_height/partitions
    return (math.floor(maincoord[0]/modx),math.floor(maincoord[1]/mody))


class Boid(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, screen_height, screen_width, imgfile):
        pygame.sprite.Sprite.__init__(self)

        self.image_og = pygame.image.load(imgfile)
        self.image_og = pygame.transform.scale(self.image_og, (30, 50))
        self.image = self.image_og
        self.rect = self.image_og.get_rect()
        self.pos = (x,y)
        self.rect.centerx = self.pos[0]
        self.rect.centery = self.pos[1]
        self.angle = angle
        self.vx = 0
        self.vy = 0
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.speedThreshold = 300

    def rotate(self, angle):
        angle = angle % 360
        self.image = pygame.transform.rotate(self.image_og, angle)

    def rotateBy(self, dthet):
        self.angle = (self.angle + dthet) % 360
        self.image = pygame.transform.rotate(self.image_og, self.angle)

    def move(self):
        dx = 0.0333 * self.vx
        dy = 0.0333 * self.vy
        newx = (self.pos[0] + dx) % self.screen_width
        newy = (self.pos[1] + dy) % self.screen_height
        self.pos = (newx, newy)

    def setaccs(self, acarr):
        if acarr[0] != 0 or acarr[1] != 0:
            acarr = acarr/np.linalg.norm(acarr)
        self.vx += 0.0333 * acarr[0]
        self.vy += 0.0333 * acarr[1]
        totalSpeed = np.linalg.norm(np.array([self.vx,self.vy]))

        if (totalSpeed > self.speedThreshold):
            self.vx = self.vx / totalSpeed * self.speedThreshold
            self.vy = self.vy/totalSpeed * self.speedThreshold



    def update(self):
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        #self.rect.centery = self.screen_height - self.rect.centery




class Simulation:
    def __init__(self, boids, screen_height, screen_width):
        self.boids = boids
        self.boidGroup = pygame.sprite.Group(boids)
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.board = [0] * partitions
        for i in range(0, partitions):
            self.board[i] = [0] * partitions

    def addBoid(self, boid):
        self.boids.append(boid)

    def draw(self, screen):
        self.boidGroup.draw(screen)


    def getposdiff(self, pos1,pos2):
        # get distance to a boid
        pos1np = np.array(pos1)
        pos2np = np.array(pos2)
        # x distance
        dx = np.linalg.norm(pos2np[0] - pos1np[0])
        # y distance
        dy = np.linalg.norm(pos2np[1] - pos1np[1])

        # check if object is on other side of board wrap around
        if (3 / partitions * self.screen_width <= dx):
            # subtract from x value to move into correct measuring distance
            if pos2np[0] > pos1np[0]:
                pos2np[0] -= self.screen_width
            else:
                pos1np[0] -= self.screen_width
        # check if object is on other side of board wrap around
        if (3 / partitions * self.screen_height <= dy):
            # subtract from x value to move into correct measuring distance
            if pos2np[1] > pos1np[1]:
                pos2np[1] -= self.screen_height
            else:
                pos1np[1] -= self.screen_height
        # subtract points
        diff = pos2np - pos1np
        return diff

    def update(self):
        cordboid = (0,0)
        for i in range (0, partitions):
            for j in range (0,partitions):
                self.board[i][j] = []
        for i in range (0,len(self.boids)):
            bpos = get_coord(self.boids[i].pos, self.screen_height, self.screen_width)
            self.board[bpos[0]][bpos[1]].append(i)

        #array of shared space boids
        #print("\n\nnew step")

        for x in range (0, len(self.boids)):
            cohesion = np.array([0,0])
            alignment = np.array([0,0])
            seperation = np.array([0,0])
            sharedw = []
            cordboid = get_coord(self.boids[x].pos, self.screen_height, self.screen_width)
            for i in range (-1,2):
                 for j in range (-1,2):
                    sharedw += self.board[(cordboid[0]+i) % partitions][(cordboid[1] + j) % partitions]


            total = 0
            for i in range(0, len(sharedw)):
                #distance to boid
                if (x != sharedw[i]):
                    posdif = self.getposdiff(self.boids[x].pos,self.boids[sharedw[i]].pos)
                    lentob = np.linalg.norm(posdif)
                    distanceThresh = self.screen_width/partitions*1.5
                    if (lentob <= distanceThresh):
                        cohesion[0] += posdif[0]
                        cohesion[1] += posdif[1]
                        seperation[0] += (distanceThresh - lentob) * -posdif[0] * 0.5
                        seperation[1] += (distanceThresh - lentob) * -posdif[1] * 0.5
                        alignment[0] += self.boids[sharedw[i]].vx
                        alignment[1] += self.boids[sharedw[i]].vy



                        total += 1


                        #print("Boid ",x, " and ", sharedw[i], " is ", lentob, " Y ", posdif)
            if total > 0:
                cohesion = cohesion / total
                seperation = seperation / total
                alignment = alignment / total
                #print("avg ",cohesion)
                self.boids[x].setaccs((cohesion + seperation + alignment)/3)
                #print("Ok ", self.xxx, self.yyy)

        for i in range(0,len(self.boids)):
            self.boids[i].rotateBy(5)
            self.boids[i].move()

        self.boidGroup.update()



def main():

   # initializing pygame
    #pygame.mixer.init()
    pygame.init()
    clock = pygame.time.Clock()

    # some music
    #pygame.mixer.music.load('madame-butterfly.wav')

    # top left corner is (0,0)
    win_width = 640
    win_height = 640
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('2D projectile motion')
    numofboids = 35

    boids = []

    ang = np.random.uniform(0, 360)
    boids.append(Boid(np.random.uniform(0, 360), np.random.uniform(0, 360), ang, win_height, win_width, 'x3.png'))
    for i in range(1, numofboids):
        ang = np.random.uniform(0, 360)
        boids.append(Boid(np.random.uniform(0, 360), np.random.uniform(0, 360), ang, win_height, win_width, 'x2.png'))
    sim = Simulation(boids, win_height, win_width)
    #mygroup = pygame.sprite.Group(boids)

    paused = False
    while True:
        # 30 fps
        clock.tick(30)

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)
        else:
            pass

        if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
            paused = True
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            paused = False

        if paused == False:
            # clear the background, and draw the sprites
            screen.fill(WHITE)


            #print(pygame.mouse.get_pos())

            for i in range (1, partitions):
                pygame.draw.line(screen, BLACK, (win_width/partitions * i, 0), (win_width/partitions * i, win_height), 3)
                pygame.draw.line(screen, BLACK, (0, win_height / partitions * i), (win_width, win_height / partitions * i),3)


            sim.update()
            sim.draw(screen)

            #print(box.rect.center)
            pygame.display.flip()
            pygame.display.update()

if __name__ == '__main__':
    main()
