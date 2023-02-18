from trafficagent import TrafficAgent
from typing import Callable
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

class QLearningTrafficAgent(TrafficAgent):
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

  def _get_agent(self, env: SumoEnvironment) -> dict[str, QLAgent]:
    initial_states = env.reset()
    ql_agents = {
      ts: QLAgent(
        starting_state = env.encode(initial_states[ts], ts),
        state_space = env.observation_space,
        action_space = env.action_space,
        alpha = self.alpha,
        gamma = self.gamma,
        exploration_strategy = EpsilonGreedy(
          initial_epsilon = self.initial_epsilon,
          min_epsilon = self.min_epsilon,
          decay = 0.9
        )
      ) for ts in env.ts_ids
    }
    return ql_agents

  def _learn(self, env: SumoEnvironment, agent: dict[str, QLAgent], updateMetrics: Callable[[str, dict[str, float]], None]):
    done: dict[str, bool] = {'__all__': False}
    while not done['__all__']:
      actions = {ts: agent[ts].act() for ts in agent.keys()}
      state, reward, done, _ = env.step(action = actions) # type: ignore
      updateMetrics(self.name, env.metrics[-1])
      for agent_id in agent.keys():
        agent[agent_id].learn(next_state = env.encode(state[agent_id], agent_id), reward = reward[agent_id]) # type: ignore

  def _save_model(self):
    pass
