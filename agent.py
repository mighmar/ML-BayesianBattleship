from node import RootNode, BranchNode
from battleship import Game
import game_config as cfg
from tensor_manager import get_tensor

class Agent:
    X_SIZE = cfg.X_SIZE
    Y_SIZE = cfg.Y_SIZE
    SHIPS  = cfg.SHIPS

    def __init__(self):
        self.ship_layer = []
        for ship in SHIPS: 
            length = ship[1]
            if length != 1:
                n_states = (X_SIZE-length+1)*Y_SIZE + X_SIZE(Y_SIZE-length+1),
                priors = np.ones(n_states)/n_states
                state_names = []
                for i in range((X_SIZE-length+1)*Y_SIZE):
                   state_names[i] = (str(i % (X_SIZE-length+1)) + ','
                                     + str(i // (X_SIZE-length+1)) + 'h')
                for i in range(X_SIZE*(Y_SIZE-length+1)):
                   pos = (X_SIZE-length+1)*Y_SIZE + i
                   state_names[pos] = (str(i % (X_SIZE-length+1)) + ',' +
                                       str(i // (X_SIZE-length+1)) + 'v')
                for i in range(ship[0]):
                    self.ship_layer.append(RootNode(
                                               str(length) + '.' + str(i),
                                               n_states, priors, state_names))
            else:
                n_states = X_SIZE*Y_SIZE
                priors = np.ones(n_states)/n_states
                state_names = []
                for i in range(X_SIZE*Y_SIZE):
                   state_names[i] = (str(i % X_SIZE) + ',' + str(i // X_SIZE))
                for i in range(ship[0]):
                    self.ship_layer.append(RootNode('1.' + str(i), n_states,
                                               priors, state_names))

        state_names = [str(length) + '.' + str(i) for length, i in SHIPS]   
        state_names += ['none']
        self.pos_layer = []
        for position in range(X_SIZE*Y_SIZE): 
            self.pos_layer.append(BranchNode(
                                      str(position%X_SIZE) + ',' +
                                      str(position//X_SIZE),
                                      len(state_names),
                                      self.ship_layer, get_tensor(position),
                                      state_names)) 
        self.hit_layer = []
        transition_matrix = np.zeros((2, len(state_names)))
        transition_matrix[0, :-1] = 1
        transition_matrix[1,-1] = 1
        for position in range(X_SIZE*Y_SIZE): 
            self.hit_layer.append(BranchNode(
                                      str(position%X_SIZE) + ',' +
                                      str(position//X_SIZE), 2,
                                      [self.pos_layer[position]],
                                      transition_matrix,
                                      ['hit', 'miss'])) 
    def play(game):
       game.reset()
       while [node for node in self.hit_layer if not node.is_set]:
           for node in self.ship_layer + self.pos_layer + self.hit_layer :
               node.calculate_belief()
           likeliest_hit = np.argmax([node.pos[0] for node in self.hit_layer
                                      if not node.is_set])
           hit, position = game.step(likeliest_hit.reshape(X_SIZE, Y_SIZE))
           self.hit_layer[likeliest_hit].set_state(state)
