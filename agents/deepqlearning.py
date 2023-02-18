from trafficagent import TrafficAgent
from typing import Callable
from sumo_rl import SumoEnvironment
from stable_baselines3.dqn.dqn import DQN

class DeepQLearningTrafficAgent(TrafficAgent):
  def __init__(
    self,
    name: str,
    net_file: str,
    route_file: str,
    num_seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1,
    min_epsilon: float = 0.005
  ):
    super().__init__(name, False, False, net_file, route_file, num_seconds, delta_time, yellow_time, min_green, max_green)
    self.alpha = alpha
    self.gamma = gamma
    self.initial_epsilon = initial_epsilon
    self.min_epsilon = min_epsilon

  def _get_agent(self, env: SumoEnvironment) -> DQN:
    return DQN(
      policy = "MlpPolicy",
      env = env,
      learning_rate = self.alpha,
      learning_starts = 0,
      gamma = self.gamma,
      train_freq = 1,
      gradient_steps = 1,
      target_update_interval = 500,
      exploration_fraction = 0.1,
      exploration_initial_eps = self.initial_epsilon,
      exploration_final_eps = self.min_epsilon,
      verbose = 1
    )

  def _learn(self, env: SumoEnvironment, agent: DQN, updateMetrics: Callable[[str, dict[str, float]], None]):
    agent.learn(total_timesteps = self.num_seconds, callback = lambda locals, _: updateMetrics(self.name, locals['infos'][0]))

  def _save_model(self):
    pass
