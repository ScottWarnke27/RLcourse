import gym
from gym import spaces
from gym.utils import seeding

##https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/blackjack.py
#https://github.com/dalmia/David-Silver-Reinforcement-learning/blob/master/Week%205%20-%20Model%20Free%20Control/SARSA.ipynb
#https://github.com/hartikainen/easy21
#help from these websites and DATA643 classmates debugging the code


deck_firstcard = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

deck_nextcard = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,   #2/3 black cards (positive)
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                -1, -2,-3,-4,-5,-6,-7,-8,-9,-10]   #1/3 red cards (negative)

def draw_firstcard(np_random):
    return np_random.choice(deck_firstcard)

def draw_nextcard(np_random):
    return np_random.choice(deck_nextcard)

def sum_hand(hand):  # Return current hand total
    return sum(hand)

class Simple21(gym.Env):
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(21),    #state space for the player
            spaces.Discrete(10) ))   #dealer's showing card
        self.seed()

        self.reset()
        self.nA = 2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_nextcard(self.np_random))
            if (sum_hand(self.player) > 21) or (sum_hand(self.player) < 1):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17: # & sum_hand(self.dealer) >0:
                self.dealer.append(draw_nextcard(self.np_random))
            if (sum_hand(self.dealer) > 21) or (sum_hand(self.dealer) < 1):
                reward = 1
            elif sum_hand(self.dealer) == sum_hand(self.player):
                reward = 0
            elif sum_hand(self.dealer) > sum_hand(self.player):
                reward = -1
            else:
                reward = 1
            
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0])


    def reset(self):
        self.dealer = [draw_firstcard(self.np_random)]
        self.player = [draw_firstcard(self.np_random)]
        while sum_hand(self.player) < 12:
            self.player.append(draw_nextcard(self.np_random))

        return self._get_obs()