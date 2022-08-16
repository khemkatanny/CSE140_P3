from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.
        # self.next_value = []

        # Compute the values here.
        # for each iteration, compute the new value and check if best
        for iteration in range(self.iters):
            next_value = {}
            # get states
            states = self.mdp.getStates()
            for state in states:
                best_action = None
                # check if any actions are present for that state, if yes- get best q value
                if self.mdp.getPossibleActions(state):
                    for action in self.mdp.getPossibleActions(state):
                        if best_action is None:
                            best_action = action
                        if self.getQValue(state, action) > self.getQValue(state, best_action):
                            best_action = action
                    next_value[state] = self.getQValue(state, best_action)
                # if not action present
                else:
                    next_value[state] = self.getValue(state)
            # update each value for each iteration
            self.values = next_value

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values.get(state, 0.0)

    # added - need to compute the next action the agent would take
    # if initial action, set first action as best
    # if multiple actions, check which value is best
    def getPolicy(self, state):
        best_action = None
        for action in self.mdp.getPossibleActions(state):
            if best_action is None:
                best_action = action
            if self.getQValue(state, action) > self.getQValue(state, best_action):
                best_action = action
        return best_action

    # added - calculate q value using formula
    def getQValue(self, state, action):
        q_value = 0.0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += transition[1] * (self.mdp.getReward(state, action, transition[0])
                + self.discountRate * self.getValue(transition[0]))
        return q_value

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.getPolicy(state)
