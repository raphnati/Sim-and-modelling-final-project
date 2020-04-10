#audio stuff
import pyaudio, wave
from scipy.fftpack import fft




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

#number of partitions
partitions = 17





#audio stuff

CHUNK = 1024*4 #lower chunk size = more samples per frame, increases refresh rate
FORMAT = pyaudio.paInt16 #bytes per sample (audio format)
CHANNELS = 1

RATE = 44100

audio_path = "test.wav"
wf = wave.open(audio_path, 'rb')

# create an audio object
p = pyaudio.PyAudio()


stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),
              channels=wf.getnchannels(), rate=wf.getframerate(), input=True,
             output=True, frames_per_buffer=1024)

files_seconds = wf.getnframes()/RATE #length of wav file
files_seconds

def y_range(filename):
    wf = wave.open(filename, 'rb')
    x = []
    count = 0
    while count <= (int(wf.getnframes() / CHUNK)):
        data = wf.readframes(CHUNK)  # read 1 chunk
        data_int = np.frombuffer(data, dtype=np.int16)
        x.append(data_int)
        count = count + 1

    result = []
    for list in x:
        result.append(min(list))
        result.append(max(list))

    y_max = np.amax(result)
    y_min = np.amin(result)
    return y_max, y_min

'''
used from vivian chen's audio visualizer code
https://github.com/vivianschen/Audio_Visualizer
'''




















#boid stuff

def get_coord(maincoord, screen_height, screen_width):
    modx = screen_width/partitions
    mody = screen_height/partitions
    return (math.floor(maincoord[0]/modx),math.floor(maincoord[1]/mody))

def vecangle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))*180/np.pi





class Circle():
    def __init__(self, x, y, r,col, win_width, win_height):
        self.x = x
        self.y = y
        self.r = r
        self.regr = r
        self.win_width = win_width
        self. win_height = win_height
        self.col = col
    def update(self, levelRadius):
        self.r = self.regr + levelRadius






class Boid(pygame.sprite.Sprite):
    def __init__(self, x, y, angle, screen_height, screen_width, imgfile,freqlev):
        pygame.sprite.Sprite.__init__(self)

        self.image_og = pygame.image.load(imgfile)
        self.image_og = pygame.transform.scale(self.image_og, (15, 25))
        self.image = self.image_og
        self.rect = self.image_og.get_rect()
        self.pos = (x,y)
        self.rect.centerx = self.pos[0]
        self.rect.centery = self.pos[1]
        self.angle = angle
        #initial velocity
        self.vx = -1
        self.vy = -1
        #initial steering direction
        self.dirx = 0
        self.diry = 0
        #steering direction offset related to current velocity
        self.midx = 0
        self.midy = 0
        self.temp = np.array([0,0])
        self.drawx = 0
        self.drawy = 0
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.speedThreshold = 150
        self.initSpeedThreshold = 150
        self.freqlev = freqlev

    #rotates to an angle
    def rotate(self, angle):
        angle = angle % 360
        self.image = pygame.transform.rotate(self.image_og, angle)

    #rotates by an angle
    def rotateBy(self, dthet):
        self.angle = (self.angle + dthet) % 360
        self.image = pygame.transform.rotate(self.image_og, self.angle)

    #call max velocity limit
    def velLimit(self):
        if self.vx != 0 or self.vx != 0:
            totalSpeed = np.linalg.norm(np.array([self.vx, self.vy]))
            if (totalSpeed > self.speedThreshold):
                self.vx = self.vx / totalSpeed * self.speedThreshold
                self.vy = self.vy / totalSpeed * self.speedThreshold

    #move by step
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


    def setdir(self, dirarr, addto):
        self.speedThreshold = self.initSpeedThreshold + addto**1.7
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

            #non limited
            #curdir += dirarr / 2

            #limited
            curdir += dirunit * self.speedThreshold * 0.5


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
        self.circ1 =  Circle(100,160,5, GREEN, self.screen_width, self.screen_height)
        self.circ2 = Circle(380, 160, 5, RED, self.screen_width, self.screen_height)
        self.circ3 = Circle(320, 480, 5, BLUE, self.screen_width, self.screen_height)
        self.circ4 = Circle(530, 480, 5, BLACK, self.screen_width, self.screen_height)
        self.circles = [self.circ1,self.circ2,self.circ3,self.circ4]

    def addBoid(self, boid):
        self.boids.append(boid)

    def draw(self, screen):
        self.boidGroup.draw(screen)
        for i in range(0, 4):
            pygame.draw.circle(screen, self.circles[i].col, (self.circles[i].x, self.circles[i].y), self.circles[i].r)

        #pygame.draw.line(screen, BLACK, (self.boids[0].pos[0],self.boids[0].pos[1]), (self.boids[0].midx,self.boids[0].midy), 3)
        #pygame.draw.line(screen, BLACK, (self.boids[0].midx,self.boids[0].midy),(self.boids[0].drawx, self.boids[0].drawy), 3)
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
            # subtract from y value to move into correct measuring distance
            if pos2np[1] > pos1np[1]:
                pos2np[1] -= self.screen_height
            else:
                pos1np[1] -= self.screen_height
        # subtract points
        diff = pos2np - pos1np
        return diff


    def update(self, levels):
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

            #for other boids in list
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


            #for circle
            for circ in self.circles:
                distanceThresh = self.screen_width / partitions * 1.5
                diff = self.getposdiff(self.boids[x].pos, np.array([circ.x,circ.y]))
                dist = np.linalg.norm(diff)
                circleseperation[0] += (max((circ.r*2 - dist), 0))**1.5 * -diff[0]
                circleseperation[1] += (max((circ.r*2 - dist), 0))**1.5 * -diff[1]
                totalcirc += 1.0
            #if multiple circles
            if totalcirc > 0:
                circleseperation = circleseperation / 1
                totalDir = totalDir + circleseperation

            #
            if total > 0:
                cohesion = cohesion / total
                seperation = seperation / total
                alignment = alignment / total
                #print("avg ",cohesion)
                #Pass variables into

                #weighted vectors
                totalDir += cohesion * 3 + seperation / 2 + alignment
                #self.boids[x].setdir(cohesion)
            self.boids[x].setdir(totalDir,levels[self.boids[x].freqlev])

        for i in range(0,len(self.boids)):
            self.boids[i].rotateBy(5)
            self.boids[i].move()

        self.boidGroup.update()
        for i in range (0,4):
                self.circles[i].update(levels[i])



def main():

   # initializing pygame
    pygame.init()
    clock = pygame.time.Clock()


    # top left corner is (0,0)
    win_width = 640
    win_height = 640
    screen = pygame.display.set_mode((win_width, win_height))
    pygame.display.set_caption('2D projectile motion')
    numofboids = 100
    voladjust = 0.5
    boids = []

    ang = np.random.uniform(0, 360)
    boids.append(Boid(np.random.uniform(0, 360), np.random.uniform(0, 360), ang, win_height, win_width, 'x4.png',3))
    for i in range(1, int(numofboids/4)):
        ang = np.random.uniform(0, 360)
        boids.append(Boid(np.random.uniform(200, 500), np.random.uniform(200, 500), ang, win_height, win_width, 'x2.png',0))
        ang = np.random.uniform(0, 360)
        boids.append(Boid(np.random.uniform(200, 500), np.random.uniform(200, 500), ang, win_height, win_width, 'x3.png', 1))
        ang = np.random.uniform(0, 360)
        boids.append(Boid(np.random.uniform(200, 500), np.random.uniform(200, 500), ang, win_height, win_width, 'x5.png', 2))
        ang = np.random.uniform(0, 360)
        boids.append(Boid(np.random.uniform(200, 500), np.random.uniform(200, 500), ang, win_height, win_width, 'x6.png', 3))
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


            '''
            for i in range (1, partitions):
                pygame.draw.line(screen, BLACK, (win_width/partitions * i, 0), (win_width/partitions * i, win_height), 3)
                pygame.draw.line(screen, BLACK, (0, win_height / partitions * i), (win_width, win_height / partitions * i),3)
            '''


            #audio stuff
            data = wf.readframes(CHUNK)  # read 1 chunk
            data_int = np.fromstring(data, dtype=np.int16)
            y_fft = fft(data_int)
            # slice and rescale
            h = np.abs(y_fft[0:CHUNK]) * 2 / (10000 * CHUNK)
            h1sum = np.sum(h[0:400])
            h2sum = np.sum(h[400:800])
            h3sum = np.sum(h[800:1200])
            h4sum = np.sum(h[1200:4096])
            stream.write(data)





            sim.update([h1sum*3*voladjust, h2sum* 4*voladjust,h3sum * 6*voladjust,h4sum*4*voladjust])
            sim.draw(screen)

            pygame.display.flip()
            pygame.display.update()

if __name__ == '__main__':
    main()
