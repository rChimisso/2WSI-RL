from typing import Union
from typing_extensions import TypedDict
from traffic.environment import TrafficEnvironment
from traffic.agent import TrafficAgent
from utils.configs import TrafficAgentConfig, LearningAgentConfig, CanvasConfig
from utils.plotter import MultiPlotter

class RunsConfig(TypedDict):
  """ TypedDict for one or more training runs of a TrafficAgent. """
  cls: type[TrafficAgent]
  """ Class of the TrafficAgent to use for all training runs. """
  configs: list[Union[TrafficAgentConfig, LearningAgentConfig]]
  """ List of configs to use to configure the TrafficAgent at each training run. """

class RunnerAgent(TypedDict):
  """ TypedDict for a TrafficAgent and its hyperparameters configurations. """
  agent: TrafficAgent
  """ TrafficAgent to run """
  configs: list[Union[TrafficAgentConfig, LearningAgentConfig]]
  """ Hyperparameters configurations for each run. """

class Runner():
  """ Runner for several TrafficAgents with possibly different hyperparameters configuration and  """

  def __init__(self, canvas_config: CanvasConfig, traffic_env: TrafficEnvironment, runs_configs: list[RunsConfig]) -> None:
    """
    :param canvas_config: PlotData to instantiate Plotters.
    :type canvas_config: CanvasConfig
    :param traffic_env: TrafficEnvironment to perform each run in.
    :type traffic_env: TrafficEnvironment
    :param agents: Dictionary of agents names along with their runs configurations.
    :type agents: list[RunsConfig]
    """
    self._traffic_env: TrafficEnvironment = traffic_env
    self._agents: dict[str, TrafficAgent] = {config['name']: agent['cls'](config, traffic_env, canvas_config) for agent in runs_configs for config in agent['configs']}
    self._multi_plotter: MultiPlotter = MultiPlotter([agent.config for agent in self._agents.values()], canvas_config)

  def learn(self) -> dict[str, list[str]]:
    """
    Trains all TrafficAgents, each run for each agent with a different hyperparameters configuration.

    :return: List of all saved models for each agent.
    :rtype: dict[str, list[str]]
    """
    models: dict[str, list[str]] = {}
    for agent in self._agents.values():
      models[agent.name] = []
      while agent.config['repeat']:
        models[agent.name].append(agent.run())
      self._multi_plotter.add_run(agent.means, agent.name)
    self._multi_plotter.save()
    return models

  def run(self, models: dict[str, list[str]], seconds: Union[int, None] = None, use_gui: bool = True) -> None:
    """
    Resets the hyperparameters configuration for all TrafficAgents, then runs all specified TrafficAgents loading each specified model for each run.
    
    :param models: Dictionary of agent names paired with a list of models to load.
    :type models: dict[str, list[str]]
    :param seconds: Amount of simulation seconds to run, if None the same amount of simulation seconds used during learning will be used.
    :type seconds: Union[int, None]
    :param use_gui: Whether to show SUMO GUI.
    :type use_gui: bool
    """
    self._multi_plotter.clear()
    if seconds is not None:
      self._traffic_env.set_seconds(seconds)
    for model in models:
      if model in self._agents:
        agent = self._agents[model]
        agent.reset()
        while agent.config['repeat']:
          agent.run(use_gui, models[model][agent._runs])
        self._multi_plotter.add_run(agent.means, agent.name)
    self._multi_plotter.save()
