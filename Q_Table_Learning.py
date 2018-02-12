#  Based on Simple Reinforcement Learning with Tensorflow Part 0: Q-Learning with Tables and Neural Networks
#  https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import numpy as np
from Environment import EnvironmentClass

'''
Main Program
'''
if __name__ == '__main__':
    
    env = EnvironmentClass()

    #Initialize table with all zero
    Q = np.zeros([ env.observation_space_n, env.action_space_n ] )

    # Set learning parameters
    lr = 0.2
    y  = 0.95
    num_episodes = 10000

    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        d = False
    
        #The Q-Table learning algorithm
        for j in range(100):
        
            #Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space_n)*(1.0/(i+1)))

            #Get new state and reward from environment
            s1, reward, d = env.step( a )

            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + lr * ( reward + y * np.max( Q[s1,:]) - Q[s,a] )
            s = s1

            if d == True:
                break

    #print( Q )
    env.print_Q_Action( Q )
