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

partitions = 15


def get_coord(maincoord, screen_height, screen_width):
    modx = screen_width/partitions
    mody = screen_height/partitions
    return (math.floor(maincoord[0]/modx),math.floor(maincoord[1]/mody))

def vecangle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi

'''
class Circle(pygame.sprite.Sprite):
    def __init__(self, x, y, radius):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([radius*2, radius*2])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        cx = self.rect.centerx
        cy = self.rect.centery
        pygame.draw.circle(self.image, BLUE, (x, y), x, y)
        self.rect = self.image.get_rect()
        self.pos = [x,y]
'''

class Circle():
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

class Boid(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, screen_height, screen_width, imgfile):
        pygame.sprite.Sprite.__init__(self)

        self.image_og = pygame.image.load(imgfile)
        self.image_og = pygame.transform.scale(self.image_og, (15, 25))
        self.image = self.image_og
        self.rect = self.image_og.get_rect()
        self.pos = (x,y)
        self.rect.centerx = self.pos[0]
        self.rect.centery = self.pos[1]
        self.angle = angle
        self.vx = -1
        self.vy = -1
        self.dirx = 0
        self.diry = 0
        self.midx = 0
        self.midy = 0
        self.temp = np.array([0,0])
        self.drawx = 0
        self.drawy = 0
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.speedThreshold = 150

    def rotate(self, angle):
        angle = angle % 360
        self.image = pygame.transform.rotate(self.image_og, angle)

    def rotateBy(self, dthet):
        self.angle = (self.angle + dthet) % 360
        self.image = pygame.transform.rotate(self.image_og, self.angle)


    def velLimit(self):
        if self.vx != 0 or self.vx != 0:
            totalSpeed = np.linalg.norm(np.array([self.vx, self.vy]))
            if (totalSpeed > self.speedThreshold):
                self.vx = self.vx / totalSpeed * self.speedThreshold
                self.vy = self.vy / totalSpeed * self.speedThreshold


    def move(self):
        accelX = self.dirx * 12
        accelY = self.diry * 12
        self.vx += 0.0333 * accelX
        self.vy += 0.0333 * accelY
        self.velLimit()
        dx = 0.0333 * self.vx
        dy = 0.0333 * self.vy
        self.rotate(np.arctan2(self.vx,self.vy)*180/np.pi + 180)
        newx = (self.pos[0] + dx) % self.screen_width
        newy = (self.pos[1] + dy) % self.screen_height
        self.pos = (newx, newy)

    def setdir(self, dirarr):
        '''
        if acarr[0] != 0 or acarr[1] != 0:
            acarr = acarr/np.linalg.norm(acarr)
        '''
        #self.dirx = self.pos[0]
        #self.diry = self.pos[1]
        curdir = np.array([0,0])
        if self.vx != 0 or self.vx != 0:
            totalSpeed = np.linalg.norm(np.array([self.vx, self.vy]))
            curdir = np.array([self.vx/totalSpeed, self.vy/totalSpeed])
            curdir = curdir * totalSpeed * 0.5
        if dirarr[0] != 0 or dirarr[1] != 0:
            dirmag = np.linalg.norm(dirarr)
            dirunit = dirarr / dirmag
            self.midx = curdir[0] + self.pos[0]
            self.midy = curdir[1] + self.pos[1]
            curdir += dirarr / 2
            #curdir += dirunit * 75
            self.dirx = curdir[0]
            self.diry = curdir[1]
            self.drawx = curdir[0] + self.pos[0]
            self.drawy = curdir[1] + self.pos[1]





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
        self.circ =  Circle(320,320,50)
        self.circles = [self.circ]

    def addBoid(self, boid):
        self.boids.append(boid)

    def draw(self, screen):
        self.boidGroup.draw(screen)
        pygame.draw.circle(screen, BLUE, (self.circles[0].x, self.circles[0].y), self.circles[0].r)
        pygame.draw.line(screen, BLACK, (self.boids[0].pos[0],self.boids[0].pos[1]), (self.boids[0].midx,self.boids[0].midy), 3)
        pygame.draw.line(screen, BLACK, (self.boids[0].midx,self.boids[0].midy),(self.boids[0].drawx, self.boids[0].drawy), 3)
        #print("X ",self.boids[0].dirx, " Y ",self.boids[0].diry)


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
            totalDir = np.array([0,0])
            circleseperation = np.array([0,0])
            sharedw = []
            cordboid = get_coord(self.boids[x].pos, self.screen_height, self.screen_width)
            '''
            for i in range (-1,2):
                 for j in range (-1,2):
                    sharedw += self.board[(cordboid[0]+i) % partitions][(cordboid[1] + j) % partitions]
            '''
            for i in range (-1,2):
                 for j in range (-1,2):
                    #if ((cordboid[0]+i) >= 0 and (cordboid[1] + j) >= 0 and (cordboid[0]+i) < partitions and (cordboid[1] + j) < partitions):
                    sharedw += self.board[(cordboid[0]+i) % partitions][(cordboid[1] + j) % partitions]


            total = 0
            totalcirc = 0
            for i in range(0, len(sharedw)):
                #distance to boid
                if (x != sharedw[i]):
                    posdif = self.getposdiff(self.boids[x].pos,self.boids[sharedw[i]].pos)
                    lentob = np.linalg.norm(posdif)
                    distanceThresh = self.screen_width/partitions*1.5
                    if (lentob <= distanceThresh):
                        cohesion[0] += posdif[0]
                        cohesion[1] += posdif[1]
                        seperation[0] += max((distanceThresh*0.9 - lentob),0) * -posdif[0]
                        seperation[1] += max((distanceThresh*0.9 - lentob),0) * -posdif[1]
                        alignment[0] += self.boids[sharedw[i]].vx
                        alignment[1] += self.boids[sharedw[i]].vy



                        total += 1


                        #print("Boid ",x, " and ", sharedw[i], " is ", lentob, " Y ", posdif)
            for circ in self.circles:
                distanceThresh = self.screen_width / partitions * 1.5
                diff = self.getposdiff(self.boids[x].pos, np.array([circ.x,circ.y]))
                dist = np.linalg.norm(diff)
                circleseperation[0] += (max((circ.r*2 - dist), 0))/2**2 * -diff[0]
                circleseperation[1] += (max((circ.r*2 - dist), 0))/2**2 * -diff[1]
                totalcirc += 1.0
            if totalcirc > 0:
                circleseperation = circleseperation / totalcirc
                totalDir = totalDir + circleseperation

            if total > 0:
                cohesion = cohesion / total
                seperation = seperation / total
                alignment = alignment / total
                #print("avg ",cohesion)
                #Pass variables into
                totalDir += cohesion * 3 + seperation / 2 + alignment
                #self.boids[x].setdir(cohesion)
            self.boids[x].setdir(totalDir)

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
    numofboids = 55

    boids = []

    ang = np.random.uniform(0, 360)
    boids.append(Boid(np.random.uniform(0, 360), np.random.uniform(0, 360), ang, win_height, win_width, 'x3.png'))
    for i in range(1, numofboids):
        ang = np.random.uniform(0, 360)
        boids.append(Boid(np.random.uniform(200, 500), np.random.uniform(200, 500), ang, win_height, win_width, 'x2.png'))
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
