# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        """ print("new food", newFood)
        print("newPos",newPos)
        print(newScaredTimes) """

        "*** YOUR CODE HERE ***"
        #foodList = successorGameState.getFood().asList()
        minFood = float('inf')
        for food in newFood.asList():
            minFood = min(float('inf'), util.manhattanDistance(newPos, food))

        
        #unVisitedPos = []
        #for pos in successorGameState:
        #    unVisitedPos.append(pos)

        if action == Directions.STOP:
            return -float('inf')
        scaredGhost = [time for time in newScaredTimes if time < 100]

        score = successorGameState.getScore()
        score += 1/ (minFood + 1)
        #score += len(foodList)
        score += sum(scaredGhost)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    def maxValue (self,depth, gameState:GameState):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        bestScore = -float('inf')
        legalAction = gameState.getLegalActions(0)
        for action in legalAction:
            sucessor = gameState.generateSuccessor(0, action)
            score = self.minValue(depth, sucessor, 1)
            bestScore = max(bestScore, score)
        return bestScore
    
    def minValue(self, depth, gameState:GameState, index):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        legalAction = gameState.getLegalActions(index)
        bestMin = float('inf')

        for action in legalAction:
            sucessor = gameState.generateSuccessor(index, action)

            nextAgent = index + 1 if index < gameState.getNumAgents() - 1 else 0
            score = self.maxValue(depth + 1,sucessor) if nextAgent == 0 else self.minValue(depth,sucessor, nextAgent)
            bestMin = min(bestMin, score)
        return bestMin
            

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = -float('inf')

        for action in legalActions:
            sucessor = gameState.generateSuccessor(0, action)
            score = self.minValue(0, sucessor, 1)

            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
        
        
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
   def maxValue (self,depth, gameState:GameState, alpha, beta):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        
        bestScore = -float('inf')
        legalAction = gameState.getLegalActions(0)
        for action in legalAction:
            sucessor = gameState.generateSuccessor(0, action)
            score = self.minValue(depth, sucessor, 1, alpha, beta)
            bestScore = max(bestScore, score)

            if bestScore > beta:
                return bestScore
            alpha = max(alpha, bestScore)

        return bestScore
    
    def minValue(self, depth, gameState:GameState, index, alpha, beta):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        legalAction = gameState.getLegalActions(index)
        bestMin = float('inf')

        for action in legalAction:
            sucessor = gameState.generateSuccessor(index, action)

            nextAgent = index + 1 if index < gameState.getNumAgents() - 1 else 0
            score = self.maxValue(depth + 1,sucessor, alpha, beta) if nextAgent == 0 else self.minValue(depth,sucessor, nextAgent, alpha, beta)
            bestMin = min(bestMin, score)

            if bestMin < alpha:
                return bestMin
            beta = min(beta, bestMin)
        return bestMin

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legalAction = gameState.getLegalActions(0)
        alpha = -float('inf')
        beta = float('inf')
        bestScore = -float('inf')
        bestAction = None

        for action in legalAction:
            sucessor = gameState.generateSuccessor(0, action)
            score = self.minValue(0, sucessor, 1, alpha, beta)
            alpha = max(alpha, score)

            if score >= bestScore:
                bestScore = score
                bestAction = action
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, depth, gameState: GameState):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        bestScore = -float('inf')
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.minValue(depth, successor, 1)
            bestScore = max(bestScore, score)
        return bestScore

    def minValue(self, depth, gameState: GameState, index):
        if depth == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(index)

        if index == gameState.getNumAgents() - 1:
            expectedValue = 0.0
            for action in legalActions:
                successor = gameState.generateSuccessor(index, action)
                expectedValue += self.maxValue(depth + 1, successor)
            return expectedValue / len(legalActions)
        else:

            bestMin = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(index, action)
                score = self.minValue(depth, successor, index + 1)
                bestMin = min(bestMin, score)
            return bestMin
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = -float('inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.minValue(0, successor, 1)

            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
   if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostPositions()
    newScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    newCapsules = currentGameState.getCapsules()

    
    ghostDistance = [util.manhattanDistance(newPos, ghostPosition) for ghostPosition in newGhostStates]
    minGhostDis = min(ghostDistance) if ghostDistance else 0

    foodDistance = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
    minFoodDis = min(foodDistance) if foodDistance else 0
    capsuleDistance = [util.manhattanDistance(newPos, capsule) for capsule in newCapsules]
    minCapsule = min(capsuleDistance) if capsuleDistance else 0

    score = currentGameState.getScore() + ((1/minFoodDis + 1)) - (5 *minGhostDis) - (500 * len(newFood.asList())) - (100 * len(newCapsules)) - (3 * minCapsule) 

    for scareTime in newScaredTimes:
        score += 1/ (scareTime +10) if scareTime > 0 else 0

    return score

# Abbreviation
better = betterEvaluationFunction
