# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent


class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.
    getAction chooses among the best options according to the evaluation function.
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    ghostPos = currentGameState.getGhostPositions()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newScaredTime = (max(newScaredTimes) + min(newScaredTimes))/2
    score = successorGameState.getScore()
    "*** YOUR CODE HERE ***"
    #isWin?
    if successorGameState.isWin():
        return 999999
    if successorGameState.isLose():
        return -999999
    ghostposition = currentGameState.getGhostPosition(1)
    
   
    distScore = min(util.manhattanDistance(gost, newPos) for gost in ghostPos)
    score += distScore  

    
    foodlist = newFood.asList()
    closestfood = 100
    for foodpos in foodlist:
        thisdist = util.manhattanDistance(foodpos, newPos)
        if (thisdist < closestfood):
            closestfood = thisdist
    if (currentGameState.getNumFood() > successorGameState.getNumFood()):
        score += 200

    score -= 3 * closestfood
    capsuleplaces = currentGameState.getCapsules()
    if successorGameState.getPacmanPosition() in capsuleplaces:
        score += 100
    
        
    for ghost in newGhostStates:
          ghostdist = manhattanDistance(newPos, ghost.getPosition())
          if ghost.scaredTimer > ghostdist:
            score += max((ghost.scaredTimer - ghostdist),3)
    return score





def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def maxScore(self, gameState, depth, numGhost):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        maxNum = -99999
        actions = gameState.getLegalActions(0)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxNum = max(maxNum, self.minScore(successor, depth + 1, 1, numGhost))
        return maxNum
    
  def minScore(self, gameState, depth, currentIndex, numGhost):
   
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        minNum = 99999
        actions = gameState.getLegalActions(currentIndex)
        if currentIndex == numGhost:
            for action in actions:
                successor = gameState.generateSuccessor(currentIndex, action)
                minNum = min(minNum, self.maxScore(successor, depth +1, numGhost))
        else:
            for action in actions:
                successor = gameState.generateSuccessor(currentIndex, action)
                minNum = min(minNum, self.minScore(successor, depth, currentIndex + 1, numGhost))
        return minNum
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.
      Here are some method calls that might be useful when implementing minimax.
      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
      Directions.STOP:
        The stop direction, which is always legal
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    actions = gameState.getLegalActions()
    numAgents = gameState.getNumAgents() 
    maxAction = Directions.STOP
    maxNum = -99999
    for action in actions:
        successor = gameState.generateSuccessor(0, action)
        flag = maxNum
        maxNum = max(maxNum, self.minScore(successor, 0, 1, numAgents-1))
        maxAction = maxAction if flag == maxNum else action
    return maxAction
        
  
    

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def maxScore(self,gameState, alpha, beta, depth):
        numAgent = gameState.getNumAgents() 
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        maxNum = -99999
        actions = gameState.getLegalActions(0)
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxNum = max(maxNum, self.minScore(successor, alpha,beta, 1, depth))
            if maxNum > beta :
                return maxNum
            alpha = max(alpha,maxNum)                
        return maxNum
    
  def minScore(self, gameState, alpha, beta, currentIndex, depth):
        numAgent = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        minNum = 99999
        actions = gameState.getLegalActions(currentIndex)
        if currentIndex == numAgent-1:
            for action in actions:
                successor = gameState.generateSuccessor(currentIndex, action)
                minNum = min(minNum, self.maxScore(successor,alpha, beta,depth+1))
                if minNum <= alpha:
                    return minNum
                beta = min(beta, minNum)
        else:
            for action in actions:
                successor = gameState.generateSuccessor(currentIndex, action)
                minNum = min(minNum, self.minScore(successor, alpha, beta, currentIndex + 1, depth))
                if minNum <= alpha:
                    return minNum
                beta = min(beta, minNum)
        return minNum
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.
      Here are some method calls that might be useful when implementing minimax.
      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
      Directions.STOP:
        The stop direction, which is always legal
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    actions = gameState.getLegalActions(0)
    maxAction = Directions.STOP
    result = -999999
    alpha = -999999
    beta = 999999
    for action in actions:
        successor = gameState.generateSuccessor(0, action)
        flag = result
        result = max(result, self.minScore(successor, alpha, beta, 1, 0))
        if result > flag:
            maxAction = action
        if result >= beta:
            return maxAction
        alpha = max(alpha, result)
    return maxAction


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def expectedScore(self,gameState,currentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(currentIndex)
        numAgent = gameState.getNumAgents() 
        
        num = len(actions)
        sumScore = 0
        for action in actions:
            successor = gameState.generateSuccessor(currentIndex, action)
            if (currentIndex == numAgent-1):
                sumScore += self.maxScore(successor, depth +1)
            else:
                sumScore += self.expectedScore(successor, currentIndex + 1, depth)
        return sumScore / num
  def maxScore(self,gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        maxAction = Directions.STOP
        maxScore = -99999
        for action in actions:
            flag =  maxScore
            successor = gameState.generateSuccessor(0, action)
            maxScore = max(maxScore, self.expectedvalue(successor, 1, depth))
        return maxScore
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    actions = gameState.getLegalActions(0)
    maxAction = Directions.STOP
    maxScore = -999999
    for action in actions:
        succeccor = gameState.generateSuccessor(0, action)
        flag = maxScore
        maxScore = max(maxScore, self.expectedScore(succeccor, 1, self.depth))
        if maxScore > flag:
            maxAction = action
    return maxAction
    

def betterEvaluationFunction(currentGameState):
    from util import manhattanDistance as dis
    
    agentPos= currentGameState.getPacmanPosition()  #new_pos
    foods = currentGameState.getFood()    #new_food
    ghostStates = currentGameState.getGhostStates()
    initialScore = currentGameState.getScore()
    scaredTimes = [gs.scaredTimer for gs in ghostStates]
    foodNum = currentGameState.getNumFood()
    ghostDis = [dis(agentPos, gh.getPosition()) for gh in ghostStates]
    if foodNum == 0:
        return 9999
    if ghostDis.__contains__(0):
        return -999


    nearestFood = 100
    for i, item in enumerate(foods):
        for j, foodItem in enumerate(item):
            nearestFood = min(nearestFood, dis(agentPos, (i, j)) if foodItem else 100)
    ghostA = lambda d: -30 + d**4 if d < 3 else -1.0/d
    ghostB = sum([ghostA(ghostDis[i]) if scaredTimes[i] < 1 else 0 for i in range(len(ghostDis))])
    foodScore= 1.0 / nearestFood   
    if all((t > 0 for t in scaredTimes)):
        ghostB *= (-1)
    
    
    foodPara = -1.0
    capsulePara = -10 if all((t == 0 for t in scaredTimes)) else 0   
    capsules = currentGameState.getCapsules()
    capsuleNearest = 150
    capsuleNow = len(capsules)
    if capsuleNow > 0:
        capsuleNearest = min(capsuleNearest, min([dis(agentPos, capsule) for capsule in capsules]))
    capsuleScore = 1.0/capsuleNearest    
  
    
    result = initialScore + 8 *capsuleScore +  capsuleNow * capsulePara + 2* ghostB+ foodScore + 2 *foodPara * foodNum
    return result
            
# Abbreviation
better = betterEvaluationFunction
