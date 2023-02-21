import json
import numpy
from pathlib import Path
from datetime import datetime
from warnings import warn
from typing import Union, Literal, Callable, Generic, TypeVar
from abc import ABC, abstractmethod
from utils.plotter import Plotter, PlotData, Metric
from utils.configs import AgentConfig
from traffic.environment import TrafficEnvironment
from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

default_metrics: list[Metric] = [
  'system_total_stopped',
  'system_total_waiting_time',
  'system_mean_waiting_time',
  'system_mean_speed'
]

class QLAgentEncoder(json.JSONEncoder):
  """ JSONEncoder for QLAgent instances. """
  def default(self, o):
    if isinstance(o, QLAgent):
      return {
        'alpha': o.alpha,
        'gamma': o.gamma,
        'qtable': [{'key': tuple(str(sa) for sa in key), 'value': [str(reward) for reward in value]} for key, value in o.q_table.items()],
        'eps': o.exploration.epsilon,
        'min_eps': o.exploration.min_epsilon,
        'decay': o.exploration.decay
      }
    return super().default(o)

class QLAgentDecoder(json.JSONDecoder):
  """ JSONDecoder for QLAgent instances. """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(object_hook = self.object_hook, *args, **kwargs)

  def object_hook(self, json):
    if 'qtable' in json:
      return {
        'alpha': json['alpha'],
        'gamma': json['gamma'],
        'qtable': { tuple(numpy.float32(key) for key in pair['key']): [numpy.float32(value) for value in pair['value']] for pair in json['qtable']},
        'exploration_strategy': EpsilonGreedy(initial_epsilon = json['eps'], min_epsilon = json['min_eps'], decay = json['decay'])
      }
    return json

A = TypeVar('A', None, QLAgent, DQN)
""" Generic TypeVar for the kind of agent used in a instatiable subclass of TrafficAgent. """

class TrafficAgent(ABC, Generic[A]):
  """ TrafficAgent to control traffic lights in a TrafficEnvironment. """

  def __init__(self, name: str, color: str, traffic_env: TrafficEnvironment, plot_data: PlotData = PlotData(default_metrics), fixed: bool = False) -> None:
    """
    TrafficAgent to control traffic lights in a TrafficEnvironment.

    :param name: (str) Name used to plot and save models.
    :param color: (str) Color used to plot.
    :param traffic_env: (TrafficEnvironment) TrafficEnvironment to perform in.
    :param plot_data: (PlotData) PlotData to instantiate a Plotter.
    :param fixed: (bool) Whether a fixed cycle or a reinforcement learning schema is used.
    """
    self._time: str = '-'
    self._config: AgentConfig = {}
    self._name: str = name
    self._color: str = color
    self._traffic_env: TrafficEnvironment = traffic_env
    self._plotter: Plotter = Plotter(color, plot_data)
    self._fixed: bool = fixed

  @property
  def time(self) -> str:
    """ Timestamp of the start of the latest run. """
    return self._time

  @property
  def config(self) -> AgentConfig:
    """ Current AgentConfig hyperparameters configuration. """
    return self._config

  @property
  def name(self) -> str:
    """ Name used to plot and save models. """
    return self._name

  @property
  def color(self) -> str:
    """ Color used to plot the lines of each run. """
    return self._color

  @property
  def fixed(self) -> bool:
    """
    False: a fixed cycle schema is used to control the traffic lights of the environment, learning and saving models are noops.
    True: a reinforcement learing agent is used to control the traffic lights, learning and saving models can be done.
    """
    return self._fixed

  def set_config(self, config: AgentConfig) -> None:
    """
    Sets the current AgentConfig hyperparameters configuration.
    
    :param config: (AgentConfig) hyperparameters configuration to use.
    """
    self._config = config

  def get_config_item(self, key: Literal['alpha', 'gamma', 'init_eps', 'min_eps', 'decay']) -> float:
    """
    Returns the given value from the hyperparameters configuration, if present.
    Returns 0 otherwise (to avoid hard crashes).
    """
    value = self.config.get(key)
    if value is None:
      warn(f'Hyperparameters configuration has no property {key}, this indicates a missing value in the configuration for the run, make sure to have everything setup properly.')
      return 0
    return value

  def run(
    self,
    update_metrics: Callable[[float, Metric, str], None],
    use_gui: bool = False,
    load_path: Union[str, None] = None
  ) -> str:
    """
    Runs the agent model on a new SumoEnvironment, saves the csv data, the plots and, if load_path is None, saves the agent model to a file.
    
    :param update_metrics: (Callable[[float, Metric, str], None])
    :param use_gui: (bool) Whether to show SUMO GUI while running (if True, will slow down the run).
    :param load_path: (Union[str, None]) The path from which to load the agent model. Must be None to train a new agent model, if set the pre-trained agent model will be loaded and run without further training.
    
    :return: The path of the saved agent model, '' if no model was saved.
    """
    learn = load_path is None
    self._time = str(datetime.now()).split('.')[0].replace(':', '-')
    env = self._traffic_env.get_sumo_env(self._fixed, self._get_filename('csv'), use_gui)
    agent = self._get_agent(env) if learn else self._load_model(env, load_path)
    self._run(env, agent, lambda info: self._update_metrics(info, update_metrics), learn)
    self._plotter.save(self._name)
    env.close()
    return self._save_model(agent) if learn else ''

  def _get_filename(self, type: Literal['csv', 'save']) -> str:
    """ Returns the file path and name to use to save either csv data or a model. """
    return f'outputs/{self._name}/{type}s/{self._time},seconds={self._traffic_env.seconds},delta_time={self._traffic_env.delta_time},yellow_time={self._traffic_env.yellow_time},min_green={self._traffic_env.min_green},max_green={self._traffic_env.max_green}'

  def _update_metrics(self, info: dict[Metric, float], callback: Callable[[float, Metric, str], None]) -> None:
    for metric in info:
      self._plotter.append(info[metric], metric)
      callback(info[metric], metric, self._name)

  @abstractmethod
  def _get_agent(self, env: SumoEnvironment) -> A:
    """
    Returns the agent to run on the given SumoEnvironment.

    :param env: (SumoEnvironment) SumoEnvironment to run the agent on.
    """
    raise NotImplementedError('Method _get_agent() must be implemented in a subclass.')

  @abstractmethod
  def _load_model(self, env: SumoEnvironment, path: str) -> A:
    """
    Loads and returns the agent model from the given path.

    :param env: (SumoEnvironment) SumoEnvironment to initialize the agent.
    :param path: (str) Path to the agent model file.
    """
    raise NotImplementedError('Method _load_model() must be implemented in a subclass.')

  @abstractmethod
  def _run(self, env: SumoEnvironment, agent: A, update_metrics: Callable[[dict[Metric, float]], None], learn: bool) -> None:
    """
    Actually runs the given agent on the given environment.

    :param env: (SumoEnvironment) SumoEnvironment to run the agent in.
    :param agent: (A) Agent to run.
    :param 
    :param learn: (bool) Whether the agent should learn or run.
    """
    raise NotImplementedError('Method _run() must be implemented in a subclass.')
  
  @abstractmethod
  def _save_model(self, agent: A) -> str:
    """
    Saves the agent model to a file and then returns its relative path.

    :param agent: (A) agent model to save.

    :return: The relative path of the save file.
    """
    raise NotImplementedError('Method _save_model() must be implemented in a subclass.')

class FixedCycleTrafficAgent(TrafficAgent[None]):
  """ FixedCycleTrafficAgent to control traffic lights with a fixed cycle schema in a TrafficEnvironment. """

  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    """
    FixedCycleTrafficAgent to control traffic lights with a fixed cycle schema in a TrafficEnvironment.

    :param name: (str) Name used to plot and save models.
    :param color: (str) Color used to plot.
    :param traffic_env: (TrafficEnvironment) TrafficEnvironment to perform in.
    :param plot_data: (PlotData) PlotData to instantiate a Plotter.
    """
    super().__init__(name, color, traffic_env, plot_data, True)

  def _get_agent(self, env: SumoEnvironment) -> None:
    return None

  def _load_model(self, env: SumoEnvironment, path: str) -> None:
    return None

  def _run(self, env: SumoEnvironment, agent: None, update_metrics: Callable[[dict[Metric, float]], None], learn: bool) -> None:
    env.reset()
    if not learn:
      done = False
      while not done:
        done = self._step(env)
        update_metrics(env.metrics[-1])
      env.reset()

  def _step(self, env: SumoEnvironment) -> bool:
    """
    Step the given SumoEnvironment, used instead of env.step(action) because a FixedCycleTrafficAgent has no actions and needs less operations to be done.

    :param env: (SumoEnvironment) SumoEnvironment to step.

    :return: Whether the given SumoEnvironment simulation has terminated.
    """
    for _ in range(self._traffic_env.delta_time):
      env._sumo_step()
    env._compute_observations()
    env._compute_rewards()
    env._compute_info()
    return env._compute_dones()['__all__']

  def _save_model(self, agent: None) -> str:
    return ''

class LearningTrafficAgent(TrafficAgent[A], ABC, Generic[A]):
  """ TrafficAgent that can learn. """

  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    """
    TrafficAgent that can learn.

    :param name: (str) Name used to plot and save models.
    :param color: (str) Color used to plot.
    :param traffic_env: (TrafficEnvironment) TrafficEnvironment to perform in.
    :param plot_data: (PlotData) PlotData to instantiate a Plotter.
    """
    super().__init__(name, color, traffic_env, plot_data)
    self._config = {}
    # self.alpha: float = 0.1
    # self.gamma: float = 0.75
    # self.init_eps: float = 1
    # self.min_eps: float = 0.1
    # self.decay: float = 1
  
  def _get_filename(self, type: Literal['csv', 'save']) -> str:
    alpha = self.get_config_item('alpha')
    gamma = self.get_config_item('gamma')
    init_eps = self.get_config_item('init_eps')
    min_eps = self.get_config_item('min_eps')
    decay = self.get_config_item('decay')
    return f'{super()._get_filename(type)},{alpha=},{gamma=},{init_eps=},{min_eps=},{decay=}'

class QLTrafficAgent(LearningTrafficAgent[QLAgent]):
  """ LearningTrafficAgent using a Q-Learning model. """

  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    """
    LearningTrafficAgent using a Q-Learning model.

    :param name: (str) Name used to plot and save models.
    :param color: (str) Color used to plot.
    :param traffic_env: (TrafficEnvironment) TrafficEnvironment to perform in.
    :param plot_data: (PlotData) PlotData to instantiate a Plotter.
    """
    super().__init__(name, color, traffic_env, plot_data)

  def _get_agent(self, env: SumoEnvironment) -> QLAgent:
    return QLAgent(
      starting_state = env.encode(env.reset()[0], env.ts_ids[0]),
      state_space = env.observation_space,
      action_space = env.action_space,
      alpha = self.get_config_item('alpha'),
      gamma = self.get_config_item('gamma'),
      exploration_strategy = EpsilonGreedy(self.get_config_item('init_eps'), self.get_config_item('min_eps'), self.get_config_item('decay'))
    )

  def _load_model(self, env: SumoEnvironment, path: str) -> QLAgent:
    with open(path, 'r') as file:
      agent_data = json.load(file, cls = QLAgentDecoder)
      agent = QLAgent(
        starting_state = env.encode(env.reset()[0], env.ts_ids[0]),
        state_space = env.observation_space,
        action_space = env.action_space,
        alpha = agent_data['alpha'],
        gamma = agent_data['gamma'],
        exploration_strategy = agent_data['exploration_strategy']
      )
      agent.q_table = agent_data['qtable']
      return agent

  def _run(self, env: SumoEnvironment, agent: QLAgent, update_metrics: Callable[[dict[Metric, float]], None], learn: bool) -> None:
    done = False
    while not done:
      state, reward, _, done, _ = env.step(agent.act()) # type: ignore
      update_metrics(env.metrics[-1])
      if learn:
        agent.learn(env.encode(state, env.ts_ids[0]), reward)
    env.reset()

  def _save_model(self, agent: QLAgent) -> str:
    path = '{}.json'.format(self._get_filename('save'))
    Path(Path(path).parent).mkdir(parents = True, exist_ok = True)
    with open(path, 'w+') as file:
      json.dump(agent, file, indent = 2, cls = QLAgentEncoder)
    return path

class DQLTrafficAgent(LearningTrafficAgent[DQN]):
  """ LearningTrafficAgent using a Deep Q-Learning model. """

  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    """
    LearningTrafficAgent using a Deep Q-Learning model.

    :param name: (str) Name used to plot and save models.
    :param color: (str) Color used to plot.
    :param traffic_env: (TrafficEnvironment) TrafficEnvironment to perform in.
    :param plot_data: (PlotData) PlotData to instantiate a Plotter.
    """
    super().__init__(name, color, traffic_env, plot_data)

  def _get_agent(self, env: SumoEnvironment) -> DQN:
    return DQN(
      policy = "MlpPolicy",
      env = env,
      learning_rate = self.get_config_item('alpha'),
      learning_starts = 0,
      gamma = self.get_config_item('gamma'),
      train_freq = (1, 'step'),
      gradient_steps = -1,
      target_update_interval = max(1, self._traffic_env.seconds // 100),
      exploration_fraction = self.get_config_item('decay'),
      exploration_initial_eps = self.get_config_item('init_eps'),
      exploration_final_eps = self.get_config_item('min_eps'),
      verbose = 1
    )

  def _load_model(self, env: SumoEnvironment, path: str) -> DQN:
    return DQN.load(env = env, path = path)

  def _run(self, env: SumoEnvironment, agent: DQN, update_metrics: Callable[[dict[Metric, float]], None], learn: bool) -> None:
    if learn:
      # total_timesteps needs to be divided by delta_time because DQN counts the actual seconds rather than the agent steps.
      agent.learn(total_timesteps = self._traffic_env.seconds // self._traffic_env.delta_time, log_interval = 1, callback = lambda locals, globals: update_metrics(locals['infos'][0]))
    else:
      done = False
      state = env.reset()[0]
      while not done:
        state, _, _, done, _ = env.step(agent.predict(state)[0]) # type: ignore
        update_metrics(env.metrics[-1])
      env.reset()

  def _save_model(self, agent: DQN) -> str:
    path = '{}.zip'.format(self._get_filename('save'))
    agent.save(path)
    return path
