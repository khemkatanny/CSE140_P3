from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
import random
from pacai.util import probability

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.states = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        if (state, action) in self.states:
            return self.states[(state, action)]
        else:
            return 0.0

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        best_score = -100
        best_action = None
        for action in self.getLegalActions(state):
            score = self.getQValue(state, action)
            if score > best_score:
                best_score = score
                best_action = action
        if not best_action:
            return 0.0
        return best_score

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        best_score = -100
        best_actions = []
        legalActions = self.getLegalActions(state)
        if legalActions:
            for action in legalActions:
                score = self.getQValue(state, action)
                if score > best_score:
                    best_score = score
                    best_actions = [action]
                if score is best_score:
                    best_actions.append(action)
        else:
            return None
        return random.choice(best_actions)

    def getAction(self, state):
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if legalActions:
            if probability.flipCoin(self.getEpsilon()):
                return random.choice(legalActions)
            else:
                return self.getPolicy(state)
        # if no legal actions, choose none
        else:
            return action

    def update(self, state, action, nextState, reward):
        if not (state, action) in self.states:
            self.states[(state, action)] = 0.0
        # print("self.states[(state,action)]: ", self.states[(state,action)])
        # print("state: ", state)
        # print("action: ", action)
        # print("reward: ", reward)
        # print("Value: ", self.getValue(nextState))
        self.states[(state, action)] = (1 - self.getAlpha()) * self.getQValue(state, action) + \
            self.getAlpha() * (reward + self.getDiscountRate() * self.getValue(nextState))

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    getQValue: looping though to get weights of all features and then using formula from notes
    as we need to find the features that are functions of state action pair.
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = {}

    def getQValue(self, state, action):
        feature = self.featExtractor.getFeatures(self, state, action)
        Q_sa = 0.0
        for f in feature:
            if f in self.weights:
                Q_sa += self.weights[f] * feature[f]
        return Q_sa

    # using the formula given in question to update feature values
    def update(self, state, action, nextState, reward):
        feature = self.featExtractor.getFeatures(self, state, action)
        for f in feature:
            correction = (reward + self.getDiscountRate() * self.getValue(nextState)) \
                - self.getQValue(state, action)
            self.weights[f] = self.weights[f] + self.getAlpha() * [correction] * feature[f]

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            raise NotImplementedError()
