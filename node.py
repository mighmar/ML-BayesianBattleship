import numpy as np 
from functools import reduce, partial

class Node:

    def __init__(self, name, state_n, state_names=None): 
        self.name = name
        self.state_n = state_n
        self.state_names = state_names 
        self.is_set = False

        self.children = []
        self.child_index = {}

        self.child_ls = []
        self.ls = []
        self.l = np.ones(state_n) 

        self.ps = []
        self.p = None 

    def set_state(self, state):
        self.is_set = True
        self.l = np.zeros(self.state_n)
        self.l[state] = 1 
        self.ps = [self.l for _ in self.ps]

    def get_l(self, parent_name):
        return self.ls[self.parent_index[parent_name]]

    def get_p(self, child_name):
        return self.ps[self.child_index[child_name]]
    
    def add_child(self, child):
        child_l = child.get_l(self.name)
        self.child_index[child.name] = len(self.children)
        self.children.append(child) 
        self.child_ls.append(child_l) 
        self.l *= child_l 

    def calculate_belief(self): 
        # Calculate belief function BEL = α*λπ, λ = Π_k λ_k
        self.bel = self.l * self.p
        sum_ = np.sum(self.bel)
        if sum_ != 0:
            alpha = 1/sum_
        else:
            alpha = 0
        self.bel *= alpha

        # Calculate priors as BEL/λ_k to be propagated to children
        ps = [reduce(np.multiply, 
              [x for j,x in enumerate(self.child_ls) if i!=j])
              for j,_ in enumerate(self.child_ls)] 
        self.ps = map((lambda x: alpha * self.p * x), ps)

    def calculate_evidence(self, child_changed=None, child_l=None):
        if child_changed:
            self.child_ls[child_index[child_changed]] = child_l
        else:
            self.child_ls = [child.get_l(self.name) for child in self.children]
        self.l = np.product(self.child_ls)
         
 

class RootNode(Node):
    def __init__(self, name, state_n, priors, state_names=None):
        Node.__init__(self, name, state_n, state_names)
        self.p = priors


class BranchNode(Node): 
    def __init__(self, name, state_n, parents, M, state_names=None):
        Node.__init__(self, name, state_n, state_names) 
        self.parents = parents
        self.parent_index = ({parent.name : i 
                              for i, parent in enumerate(parents)}) 
        self.M = M
        # Marginalize probability tensor to matrices for calculating likelihoods
        self.Ms = [np.sum(M, axis=
                       tuple([j for j in range(1, len(M.shape)) if j!=i])) 
                       for i in range(1, len(M.shape))] 
        for u in parents:
            u.add_child(self) 

    def add_child(self, child):
        Node.add_child(self, child) 
        self.ls = [self.l @ self.Ms[i] for i, _ in enumerate(self.parents)]

    def set_state(self, state):
        Node.set_state(self, state) 
        self.ls = map(partial(np.dot,self.l), self.Ms)
        for i, u in enumerate(self.parents):
            u.calculate_evidence(self.name, self.ls) 

    def calculate_belief(self):
        parent_ps = (u.get_p(self.name) for u in self.parents)
        # Apply probability tensor on received priors
        self.p = reduce(np.dot, [M] + parent_ps)
        Node.calculate_belief(self)

    def calculate_evidence(self, child_changed=None, child_l=None):
        Node.calculate_evidence(self, child_changed, child_l)
        for i, parent in enumerate(self.parents):
            self.ls[i] = self.l @ self.Ms[i] 
            parent.calculate_evidence(self.name, self.ls[i]) 
