"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epislon <= 1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """
        ################### Your code here #######################
        #raise NotImplementedError('[EpislonGreedy] update function NOT IMPLEMENTED')
        self._Q[action]= self._Q[action] + 1/(self._action_N[action])*(immi_reward-self._Q[action])

        ##########################################################

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore.
        Return: action selection
        """
        ################### Your code here #######################
        s=np.random.random(1) 
        if s>self._epislon:  
            #greedy
           action=np.argmax(self._Q) #finds action that maximizes Q
           a=np.where(self._Q==self._Q[action]) #finds if there are multiple values that maximizes Q
           if len(a)>1:
               action=np.random.randint(len(a))
               #action=np.argmax(np.take(self._action_N,a)) #takes most frequently chosen value
        else: 
          
             explore=np.arange(0,self._nb) # creates a list with all actions
             action=np.random.choice(np.delete(explore,np.argmax(self._Q))) #removes the greedy option and randomly chooses from the others
             
           
        self._action_N[action]=self._action_N[action]+1  
        return action
        
    
        #raise NotImplementedError('[EpislonGreedy] act function NOT IMPLEMENTED')
        ##########################################################

class UCB(object):
    """
    Implementation of upper confidence bound.
    """
    def __init__(self, NumofBandits=10, c=2):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        
        ################### Your code here #######################
        
        self._Q[action]= self._Q[action] + (immi_reward-self._Q[action])/(self._action_N[action])
        
        
        
        #raise NotImplementedError('[UCB] update function NOT IMPLEMENTED')
        ##########################################################

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        if t<self._nb:
            action=np.copy(t) #action = t to make sure every all arms are tried
        else:
            action=np.argmax(self._Q + self._c*np.sqrt(np.log(t)/self._action_N))#finds action using argmax
        
        self._action_N[action]+=1
        return action
       
class Gradient(object):
    """
    Implementation of your gradient-based method
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._action_N = np.zeros(self._nb, dtype=int)
        self._epislon = epislon
        self._Q = np.zeros(self._nb, dtype=float)
        
        
    
    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
       
        self._Q[action]= self._Q[action] + (immi_reward-self._Q[action])/(self._action_N[action])
       

    def act(self, t):
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """
       
        action=np.argmax(np.exp(self._Q)/np.sum(np.exp(self._Q)))
        
               
        a=np.where(self._Q==self._Q[action])
        if len(a)>1: 
           #action=(a[np.argmax(np.take(self._action_N,a))]) #takes most frequently chosen value
           action=np.random.randint(len(a))
        self._action_N[action]+=1
        return action
    
    

