from typing_extensions import TypedDict, NotRequired

class AgentConfig(TypedDict):
  """ TypedDict for a TrafficAgent hyperparameters configuration. """
  alpha: NotRequired[float]
  """ Learning rate. """
  gamma: NotRequired[float]
  """ Discount rate. """
  init_eps: NotRequired[float]
  """ Initial value for epsilon, the exploration chance. """
  min_eps: NotRequired[float]
  """ Final value for epsilo, the exploration chance. """
  decay: NotRequired[float]
  """
  If a QLearningTrafficAgent: the constant by which epsilon is multiplied to decrease it at each action taken.
  If a DeepQLearningTrafficAgent: the fraction of the training time in which to bring epsilon from init_eps to min_eps.
  """
