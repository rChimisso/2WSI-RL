from trafficagent import TrafficAgent
from typing import Callable
from sumo_rl import SumoEnvironment

class FixedCycleTrafficAgent(TrafficAgent):
  def __init__(self, name: str, net_file: str, route_file: str, num_seconds: int, delta_time: int, yellow_time: int, min_green: int, max_green: int):
    super().__init__(name, True, False, net_file, route_file, num_seconds, delta_time, yellow_time, min_green, max_green)

  def _get_agent(self, _: SumoEnvironment) -> None:
    return None

  def _learn(self, env: SumoEnvironment, _: None, updateMetrics: Callable[[str, dict[str, float]], None]):
    done: dict[str, bool] = {'__all__': False}
    while not done['__all__']:
      _, _, done, _ = env.step({}) # type: ignore
      updateMetrics(self.name, env.metrics[-1])

  def _save_model(self):
    pass

