import numpy as np
import game_config as cfg 


X_SIZE = cfg.X_SIZE
Y_SIZE = cfg.Y_SIZE
SHIPS  = cfg.SHIPS


ships = []

# Class for pickling a tensor
class Calculation:
    def collision(self, ship, solution):
        pass 
    
    def enter(self, solution):
        pass
    
    def find(self, solution, rest):
        if not rest:
            enter(solution)
            return 
    
        length = rest[0][0] 
        # orientation = cfg.HORIZONTAL
        for x, y in product(range(X_SIZE-length+1), range(Y_SIZE)):
            ship = (x, y, cfg.HORIZONTAL, length)
            if not collision(ship, solution): 
                solution.append(ship)
                if rest[0][1] == 1:
                    rest.pop()
                    find(solution, rest)
                    rest.append((length,1))
                else:
                    rest [0][1] -= 1 
                    find(solution + [ship], rest)
                    rest[0][1] += 1
                solution.pop()
    
        # orientation = cfg.VERTICAL 
        for x, y in product(range(X_SIZE), range(Y_SIZE-length+1)):
            ship = (x, y, cfg.VERTICAL, length)
            if not collision(ship, solution): 
                solution.append(ship)
                if rest[0][1] == 1:
                    rest.pop()
                    find(solution, rest)
                    rest.append((length,1))
                else:
                    rest [0][1] -= 1 
                    find(solution + [ship], rest)
                    rest[0][1] += 1
                solution.pop()
    
    def __init__(self, pos):
        self.pos = pos

        num_ships = sum([ship[2] for ship in SHIPS])
        tensor_dimensions = [(X_SIZE-length)*Y_SIZE + X_SIZE*(Y_SIZE-length)
                             for length, _ in SHIPS if length != 1]
        tensor_dimensions += [X_SIZE*Y_SIZE for length, _ in SHIPS
                              if length == 1]
        tensor_dimensions = tuple([num_ships] + tensor_dimensions) 

        self.tensor = np.zeros(tensor_dimensions) 

    def calculate(self): 
        find([], SHIPS) 
        return self.tensor
