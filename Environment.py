'''
Copyright (c) 2018 Toru Ito
Released under the MIT license
http://opensource.org/licenses/mit-license.php
'''

'''
Environment
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
'''

'''
Action
0: Left
1: Down
2: Right
3: Up
'''

import numpy as np

class EnvironmentClass:
    def __init__( self ):
        
        self.action_string = ['←', '↓', '→', '↑']

        self.environment = []
        self.environment.append( [ 'S', 'F', 'F', 'F' ] )         
        self.environment.append( [ 'F', 'H', 'F', 'H' ] )         
        self.environment.append( [ 'F', 'F', 'F', 'H' ] )         
        self.environment.append( [ 'H', 'F', 'F', 'G' ] )         
        
        self.s = [0, 0]
        self.observation_space_n_i = 4
        self.observation_space_n_j = 4
        self.observation_space_n   = 16
        self.action_space_n        = 4

    def get_environment_index( self ):
        return self.s[0] * self.observation_space_n_i + self.s[1]

    def reset( self ):
        self.s = [0, 0]
        return self.get_environment_index()

    def step( self, action ):
        
        if action == 0:
            if self.s[1] > 0:
                self.s[1] = self.s[0] - 1 

        elif action == 1:
            if self.s[0] < self.observation_space_n_i - 1:
                self.s[0] = self.s[0] + 1 
            
        elif action == 2:
            if self.s[1] < self.observation_space_n_j - 1:
                self.s[1] = self.s[1] + 1             

        elif action == 3:
            if self.s[0] > 0:
               self.s[0] = self.s[0] - 1 
            
        d  = False
        r  = 0.0
        s1 = self.get_environment_index() 
        e = self.environment[self.s[0]][self.s[1]]

        if e == 'H':
            d = True
            r = 0.0
        elif e == 'G':
            d = True
            r = 1.0

        return s1, r, d

    def print_Q_Action( self, Q ):
        
        for i in range( self.observation_space_n_i ):
            text = ''

            for j in range( self.observation_space_n_j ):
                e = self.environment[i][j]

                if e == 'G' or e == 'H': 
                    text += e
                else:
                    s = i * self.observation_space_n_i + j
                    a = np.argmax(Q[s,:]) 
                    text += self.action_string[a]
    
            print(text)
