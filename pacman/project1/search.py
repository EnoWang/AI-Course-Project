 # search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from game import Directions

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    result = util.Stack()
    result.push( (problem.getStartState(), [], []) )
    while not result.isEmpty():
        node, actions, visited = result.pop()

        for coord, direction, steps in problem.getSuccessors(node):
            if  coord not in visited:
                if problem.isGoalState(coord):
                    return actions + [direction]
                result.push((coord, actions + [direction], visited + [node] ))

    return []

def breadthFirstSearch(problem):
    result = util.Queue()
    result.push( (problem.getStartState(), []) )

    visited = []
    while not result.isEmpty():
        node, actions = result.pop()

        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in visited:
                if problem.isGoalState(coord):
                    return actions + [direction]
                result.push((coord, actions + [direction]))
                visited.append(coord)

    return []

def uniformCostSearch(problem):
    result = util.PriorityQueue()
    result.push( (problem.getStartState(), []), 0)
    explored = []

    while not result.isEmpty():
        node, actions = result.pop()

        if problem.isGoalState(node):
            return actions

        explored.append(node)

        for coord, direction, steps in problem.getSuccessors(node):
            if not coord in explored:
                new_actions = actions + [direction]
                result.push((coord, new_actions), problem.getCostOfActions(new_actions))

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    close = []
    result = util.PriorityQueue()
    start = problem.getStartState()
    result.push( (start, []), heuristic(start, problem))

    while not result.isEmpty():
        node, actions = result.pop()

        if problem.isGoalState(node):
            return actions

        close.append(node)

        for coord, direction, cost in problem.getSuccessors(node):
            if coord not in close:
                new_actions = actions + [direction]
                score = problem.getCostOfActions(new_actions) + heuristic(coord, problem)
                result.push( (coord, new_actions), score)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
