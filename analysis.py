"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    [Enter a description of what you did here.]
    We know that noise refers to the randomness of the agent for going into
    a successor state. So changing the noise, helps the agent get the highest reward
    it needs to cross the bridge.
    """

    answerDiscount = 0.9
    answerNoise = 0.01

    return answerDiscount, answerNoise

def question3a():
    """
    [Enter a description of what you did here.]
    Since we prefer the close exit and risk the cliff, I am guessing the
    living reward would be negative
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    [Enter a description of what you did here.]
    Since we prefer the close exit and want to avoid the cliff, the agent
    would care about the rewards in the distant fututre relative to the one
    in immediate future. So, we can alter the discount factor
    """

    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    [Enter a description of what you did here.]
    Since the agent prefers the distant exit, and risks the cliff, again,
    living reward would be negative.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    [Enter a description of what you did here.]
    Here again since agent prefers the distant exit, and avoids the cliff,
    we can say that living reward would not change. At the same time doscount wouldn't
    change either as we are performing the action that will give us the best value.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    [Enter a description of what you did here.]
    Here the agent avoids both exits and also avoids the cliff. Hence, our learning
    rate would increase.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.2

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Enter a description of what you did here.]
    """

    answerEpsilon = 0.0
    answerLearningRate = 0.0
    return NOT_POSSIBLE

    return answerEpsilon, answerLearningRate

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
