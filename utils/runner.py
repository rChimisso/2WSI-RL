from typing_extensions import Type, TypedDict
from traffic.environment import TrafficEnvironment
from traffic.agent import TrafficAgent
from utils.configs import AgentConfig
from utils.plotter import PlotData, MultiPlotter

class AgentRuns(TypedDict):
  """ TypedDict for one or more training runs of a TrafficAgent. """
  cls: Type[TrafficAgent]
  """ Class of the TrafficAgent to use for all training runs. """
  color: str
  """ Color for the plotted lines of all training runs. """
  configs: list[AgentConfig]
  """ List of AgentConfigs to use to configure the TrafficAgent at each training run. """

class RunnerAgent(TypedDict):
  """ TypedDict for a TrafficAgent and its hyperparameters configurations. """
  agent: TrafficAgent
  """ TrafficAgent to run """
  configs: list[AgentConfig]
  """ Hyperparameters configurations for each run. """

class Runner():
  """ Runner for several TrafficAgents with possibly different hyperparameters configuration and  """

  def __init__(self, plot_data: PlotData, traffic_env: TrafficEnvironment, agents: dict[str, AgentRuns]) -> None:
    """
    :param plot_data: (PlotData) PlotData to instantiate Plotters.
    :param traffic_env: (TrafficEnvironment) TrafficEnvironment to perform each run in.
    :param agents: (dict[str, AgentRuns]) Dictionary of agents names along with their runs configurations.
    """
    self._traffic_env: TrafficEnvironment = traffic_env
    self._agents: dict[str, RunnerAgent] = {
      agent: {
        'agent': agents[agent]['cls'](agent, agents[agent]['color'], traffic_env, plot_data),
        'configs': agents[agent]['configs']
      } for agent in agents
    }
    self._multi_plotter: MultiPlotter = MultiPlotter(list(map(lambda agent: {'name': agent, 'color': agents[agent]['color']}, list(agents))), plot_data)

  def learn(self) -> dict[str, list[str]]:
    """
    Trains all TrafficAgents, each run for each agent with a different hyperparameters configuration.
    Returns a dictionary with a list of all saved models for each TrafficAgent.
    """
    models: dict[str, list[str]] = {agent: [] for agent in self._agents}
    for agent in self._agents:
      for config in self._agents[agent]['configs']:
        self._agents[agent]['agent'].set_config(config)
        models[agent].append(self._agents[agent]['agent'].run(self._multi_plotter.append))
    self._multi_plotter.save()
    return models

  def run(self, models: dict[str, list[str]], use_gui: bool = True):
    """ 
    Resets the hyperparameters configuration for all TrafficAgents, then runs all specified TrafficAgents loading each specified model for each run.
    
    :param models: (dict[str, list[str]]) 
    :param use_gui: (bool) 
    """
    for agent in self._agents:
      self._agents[agent]['agent'].set_config({})
      if agent in models:
        for load_path in models[agent]:
          self._agents[agent]['agent'].run(self._multi_plotter.append, use_gui, load_path)
    self._multi_plotter.save()
