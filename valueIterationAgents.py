# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        Vk = util.Counter()
        for iteration in range(self.iterations):
          for state in self.mdp.getStates():
            Vk_plus1 = list()
            for action in self.mdp.getPossibleActions(state):
              Vk_plus1.append(self.getQValue(state, action))
            Vk[state] = max(Vk_plus1) if len(Vk_plus1) > 0 else 0
          self.values = Vk.copy()   
          

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for i in range(len(transitions)):
          nextState = transitions[i][0]
          transProb = transitions[i][1]
          reward    = self.mdp.getReward(state, action, nextState)
          discValue = self.discount * self.getValue(nextState)
          q_value  += transProb * (reward + discValue)
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None
        bestAction = "STOP"
        bestValue  = -9999999
        actions    = self.mdp.getPossibleActions(state)
        for action in actions:
          value = self.computeQValueFromValues(state, action)
          if value > bestValue:
            bestValue  = value
            bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        Vk = util.Counter()
        counter = 0 
        states  = self.mdp.getStates()
        length  = len(states)
        for i in range(self.iterations):
          Vk_plus1 = list()
          state = states[counter]
          for action in self.mdp.getPossibleActions(state):
            Vk_plus1.append(self.getQValue(state, action))
          Vk[state] = max(Vk_plus1) if len(Vk_plus1) > 0 else 0
          counter = counter + 1
          if counter == length:
            counter = 0   
          self.values = Vk.copy()

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
          for action in self.mdp.getPossibleActions(state):
            for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
              if prob > 0:
                if nextState not in predecessors.keys():
                  predecessors.update({nextState: [state]})
                else:
                  currentPredecessors = predecessors.get(nextState)
                  currentPredecessors.append(state)
                  predecessors.update({nextState: currentPredecessors})
        for key in predecessors:
          predecessors.update({key: set(predecessors.get(key))})
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            currVal  = self.values[state]
            q_value  = -999999
            for action in self.mdp.getPossibleActions(state):
              q = self.computeQValueFromValues(state, action)
              if q > q_value:
                q_value = q
            diff = abs(currVal - q_value)
            pq.push(state, -diff)
        for iteration in range(0, self.iterations):
          if pq.isEmpty():
            return
          state = pq.pop()
          if not self.mdp.isTerminal(state):
            bestValue  = -999999
            for action in self.mdp.getPossibleActions(state):
              currentValue = 0
              for (nextState, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                if prob > 0:
                  reward = self.mdp.getReward(state, action, nextState)
                  currentValue += prob*(reward + self.discount*self.values[nextState])
              if currentValue > bestValue:
                bestValue  = currentValue
            self.values[state] = bestValue
          for predecessor in predecessors.get(state):
            currentValue = self.values[predecessor]
            q_value = -999999
            for action in self.mdp.getPossibleActions(predecessor):
              q = self.computeQValueFromValues(predecessor, action)
              if q > q_value:
                q_value = q
            diff = abs(currentValue - q_value)
            if diff > self.theta:
              pq.update(predecessor, -diff)



