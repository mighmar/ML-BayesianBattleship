import numpy as np
from numpy import random as rnd
import game_config as cfg
from numpy import random 

X_SIZE = cfg.X_SIZE
Y_SIZE = cfg.Y_SIZE
SHIPS  = cfg.SHIPS

class Game: 
    def __init__(self):
        self.matrix = np.zeros((X_SIZE, Y_SIZE)) 
        self.hit = np.zeros((X_SIZE, Y_SIZE)) 

    def _out(self, pos):
        if pos[0] < 0 or pos[0] >= X_SIZE:
            return True
        if pos[1] < 0 or pos[1] >= Y_SIZE:
            return True
        if not self.matrix[pos]:
            return True 
    def _hatched(self, pos):
        pos = tuple(pos)
        return not out(pos) and self.matrix[pos] 
    def _hit(self, pos):
        return self.hit[tuple(pos)] 
    def _destroyed(self, pos):
        pos = tuple(pos)
        return not out(pos) and self.matrix[pos] and self.hit[pos]

   
    def reset(self):
        hatched = self._hatched 

        self.matrix = np.zeros((X_SIZE, Y_SIZE))                              
        self.hit = np.zeros((X_SIZE, Y_SIZE)) 
 
        for ship in SHIPS:
            length = ship[0] 
            num = ship[1]
            for _ in range(num):
                while True:
                    if rnd.choice([True, False]):
                        #Vertical
                        pos = np.array([rnd.randint(X_SIZE), 
                                       rnd.randint(Y_SIZE-length)])
                        if self.matrix[pos[0],pos[1] : pos[1]+length].any():
                            continue
                        else:
                            self.matrix[pos[0],pos[1] : pos[1]+length] = 1
                            break
                    else:
                        #Horizontal
                        pos = np.array([rnd.randint(X_SIZE-length), 
                                       rnd.randint(Y_SIZE)])
                        if self.matrix[pos[0] : pos[0]+length, pos[1]].any():
                            continue
                        else:
                            self.matrix[pos[0] : pos[0]+length, pos[1]] = 1
                            break 
    


    def step(self, pos):
        hatched = self._hatched
        hit = self._hit
        destroyed = self._destroyed

        self.hit[pos] = 1
        if not self.matrix[pos]:
            return (0, (0, 0, 0))
        #else: 
        for step in list(map(np.array, [[-1,0], [0,-1]])):
            #if destroyed(pos + step):
            if hatched(pos + step):
                if hit(pos + step):
                    i = 2
                    while destroyed(pos + step*i):
                        i += 1
                    if hatched(pos + step*i):
                       return (1, (0,0,0))
                    j = 1
                    while destroyed(pos - step*j):
                        j += 1
                    if hatched(pos - step*j):
                       return (1, (0,0,0))
                    break
                else:
                    return (1, (0,0,0)) 
        else: 
            for step in list(map(np.array, [[1,0], [0,1]])):
                if hatched(pos + step):
                    if hit(pos + step):
                        i = 2
                        while destroyed(pos + step*i):
                            i += 1 
                        if hatched(pos - step*j):
                            return (1, (0,0,0))
                        break
                    else:
                        return (1, (0,0,0)) 
            else:
                return (1, (pos,1,0))

        if (step == [1,0]).all() or (step == [-1,0]).all():
            orientation = cfg.HORIZONTAL
        else:
            orientation = cfg.VERTICAL

        if (step == [-1,0]).all() or (step == [0,-1]).all():
            return (1, (pos + step*(i-1), i + j - 2, orientation))
        else:
            return (1, (pos, i - 1, orientation)) 

    def print(self):
        output = np.zeros((X_SIZE, Y_SIZE), 'U1')
        output[:] = '.'
        output[self.hit == 1] = 'x'
        output[(self.hit == 1) & (self.matrix == 1)] = 'o'
        for i in output.T:
            print('|', end='')
            for j in i:
               print(j,end='')
            print('|')
