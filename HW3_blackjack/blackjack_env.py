import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
#deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
deck_firstcard = [2, 3, 4, 5, 6, 7, 8, 9, 10]

deck_nextcard = [2, 3, 4, 5, 6, 7, 8, 9, 10,   #2/3 black cards (positive)
                 2, 3, 4, 5, 6, 7, 8, 9, 10,
                -2,-3,-4,-5,-6,-7,-8,-9,-10]   #1/3 red cards (negative)

def draw_firstcard(np_random):
    return np_random.choice(deck_firstcard)

def draw_nextcard(np_random):
    return np_random.choice(deck_nextcard)

def draw_hand(np_random):
    return [draw_firstcard(np_random), draw_nextcard(np_random)]


#def usable_ace(hand):  # Does this hand have a usable ace?
#    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    #if usable_ace(hand):
    #    return sum(hand) + 10
    return sum(hand)

def is_bust(hand):  # Is this hand a bust?
    if sum_hand(hand)>21:
        return True
    elif sum_hand(hand) < 1:
        return True
    else:
        return False

    #return sum_hand(hand) > 21

def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand)==True  else sum_hand(hand)

#def is_natural(hand):  # Is this hand a natural blackjack?
#    return sorted(hand) == [1, 10]

class Simple21(gym.Env):

    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(22),    #The nine possible options for the players hand 0-21...0 includes a bust either over 21 or under 0
            spaces.Discrete(9) ))   #The nine possible options for the dealer's showing card
            # spaces.Discrete(32),  #self's state-space?  not sure how they got to 32?!?
            # spaces.Discrete(11),  #state space of the dealer's hand ?...10 card values plus 1 AND 11 for the ACE
            # spaces.Discrete(2)))
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rule # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        #self.natural = natural
        # Start the first game
        self._reset()
        print('this is after the reset')
        self.nA = 2

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_nextcard(self.np_random))
            #print("I am stuck here player draw")
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_nextcard(self.np_random))
                #print('dealer drawing a card')
                reward = cmp(score(self.player), score(self.dealer))
            #print('I am stuck here')
            #if self.natural and is_natural(self.player) and reward == 1:
            #    reward = 1.5
            #if reward == 1:
            #    reward = 1
            
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        #return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))
        return (sum_hand(self.player), self.dealer[0])


    def _reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        
        #auto draw another card if the score is less than 12
        while sum_hand(self.player) < 12:
            self.player.append(draw_nextcard(self.np_random))

        return self._get_obs()