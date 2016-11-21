# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.history = {}
    for state in mdp.getStates():
      self.history[state] = []

    for i in range(iterations + 1):
      for state in mdp.getStates():
        if i == 0:
          self.history[state].append(0)
          continue

        if mdp.isTerminal(state):
          self.history[state].append(0)
          continue

        actions = mdp.getPossibleActions(state)
        if 'exit' in actions:
          self.history[state].append(mdp.getReward(state, 'exit',
                              mdp.getTransitionStatesAndProbs(state, 'exit')[0][0]))

          continue

        max = -99999999
        for action in actions:
          statesAndProbs = mdp.getTransitionStatesAndProbs(state, action)
          tempMax = 0
          for (s, p) in statesAndProbs:
            tempMax += p * (mdp.getReward(state, action, s)
                            + self.discount * self.history[s][i-1])

          if tempMax > max:
            max = tempMax

        self.history[state].append(max)

    #print history
    for key, value in self.history.iteritems():
      #print (key, value)
      self.values[key] = value[iterations]

    print self.values

    #print self.values
    "*** YOUR CODE HERE ***"
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
    ret = 0
    for (s, p) in statesAndProbs:
      ret += p * (self.mdp.getReward(state, action, s)
                      + self.discount * self.value[s])
    return ret
    #util.raiseNotDefined()

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    if self.mdp.isTerminal(state):
      return None

    actions = self.mdp.getPossibleActions(state)
    if 'exit' in actions:
      return 'exit'

    max = -99999999
    policy = None
    for action in actions:
      statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
      tempMax = 0
      for (s, p) in statesAndProbs:
        tempMax += p * (self.mdp.getReward(state, action, s)
                        + self.discount * self.history[s][self.iterations - 1])

      if tempMax >= max:
        #print (state, tempMax, max, action)
        max = tempMax
        policy = action

    return policy
    #util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
