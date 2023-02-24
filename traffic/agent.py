import json
import numpy
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod
from typing import Union, Literal, Generic, TypeVar
from utils.plotter import Plotter
from utils.configs import TrafficAgentConfig, LearningAgentConfig, CanvasConfig, Metric
from traffic.environment import TrafficEnvironment
from stable_baselines3 import DQN
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

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

C = TypeVar('C', bound = TrafficAgentConfig)
""" Generic TypeVar for a TrafficAgent hyperparameters and plots configuration. """

class TrafficAgent(ABC, Generic[A, C]):
  """ TrafficAgent to control traffic lights in a TrafficEnvironment. """

  def __init__(self, config: C, traffic_env: TrafficEnvironment, canvas_config: CanvasConfig = CanvasConfig(), fixed: bool = False) -> None:
    """
    TrafficAgent to control traffic lights in a TrafficEnvironment.

    :param config: TrafficAgentConfig.
    :type config: C
    :param traffic_env: TrafficEnvironment to perform in.
    :type traffic_env: TrafficEnvironment
    :param canvas_config: Canvas configuration.
    :type canvas_config: CanvasConfig
    :param fixed: Whether a fixed cycle or a reinforcement learning schema is used.
    :type fixed: bool
    """
    self._config: C = config
    self._traffic_env: TrafficEnvironment = traffic_env
    self._plotter: Plotter = Plotter(config['color'], canvas_config)
    self._fixed: bool = fixed
    self._runs: int = 0

  @property
  def config(self) -> C:
    """ Current AgentConfig hyperparameters configuration. """
    return self._config

  @property
  def name(self) -> str:
    """ Name used to plot and save models. """
    return self.config['name']

  @property
  def color(self) -> str:
    """ Color used to plot the lines of each run. """
    return self.config['color']

  @property
  def fixed(self) -> bool:
    """
    False: a fixed cycle schema is used to control the traffic lights of the environment, learning and saving models are noops.
    True: a reinforcement learing agent is used to control the traffic lights, learning and saving models can be done.
    """
    return self._fixed

  @property
  def current_run(self) -> int:
    """ Number of the current run. """
    return self._runs

  @property
  def means(self) -> dict[Metric, list[float]]:
    """ Means of each metric of each run since the last reset. """
    return self._plotter.means

  @property
  def _folder(self) -> str:
    """ Folder path to use to save files. """
    return 'outputs/{}/'.format(self.config['name'])

  def _get_subfolder(self, kind: Literal['csv', 'save', 'plot']) -> str:
    """
    Returns the subfolder path to use to save either csv data, a model or plots.
    
    :param kind: Kind of the file to save.
    :type kind: Literal['csv', 'save', 'plot']

    :return: Subfolder path to use.
    :rtype: str
    """
    return f'{self._folder}{kind}s/'

  def _get_filename(self, kind: Literal['csv', 'save', 'plot'], learn: bool) -> str:
    """
    Returns the file path and name to use to save either csv data, a model or plots.

    :param kind: Kind of the file to save.
    :type kind: Literal['csv', 'save', 'plot']
    :param learn: Whether the Agent is learning or running.
    :type learn: bool

    :return: File path and name.
    :rtype: str
    """
    return '{}{}{}'.format(self._get_subfolder(kind), 'lrn' if learn else 'run', self._runs)

  def reset(self) -> None:
    """
    Resets the TrafficAgent, clearing the plots, resetting the config and the counter of the runs done.
    """
    self._config['repeat'] = self._runs
    self._plotter.clear()
    self._runs = 0

  def run(self, use_gui: bool = False, load_path: Union[str, None] = None) -> str:
    """
    Runs the agent model on a new SumoEnvironment, saves the csv data, the plots and, if load_path is None, saves the agent model to a file.
    
    :param use_gui: Whether to show SUMO GUI while running (if True, will slow down the run).
    :type use_gui: bool
    :param load_path: The path from which to load the agent model. Must be None to train a new agent model, if set the pre-trained agent model will be loaded and run without further training.
    :type load_path: Union[str, None]
    
    :return: The path of the saved agent model, '' if no model was saved.
    :rtype: str
    """
    self._runs += 1
    self._config['repeat'] -= 1
    learn = load_path is None
    env = self._traffic_env.get_sumo_env(self._fixed, self._get_filename('csv', learn), use_gui)
    agent = self._get_agent(env) if learn else self._load_model(env, load_path)
    self._plotter.add_run(self._run(env, agent, learn))
    if not self._config['repeat']:
      self._plotter.save(learn, self.config['name'])
    env.close()
    return self._save_model(agent) if learn else ''

  @abstractmethod
  def _get_agent(self, env: SumoEnvironment) -> A:
    """
    Returns the agent to run on the given SumoEnvironment.

    :param env: SumoEnvironment to run the agent on.
    :type env: SumoEnvironment

    :return: The actual agent to run.
    :rtype: A
    """
    raise NotImplementedError('Method _get_agent() must be implemented in a subclass.')

  @abstractmethod
  def _load_model(self, env: SumoEnvironment, path: str) -> A:
    """
    Loads and returns the agent model from the given path.

    :param env: SumoEnvironment to initialize the agent.
    :type env: SumoEnvironment
    :param path: Path to the agent model file.
    :type path: str

    :return: The agent model loaded.
    :rtype: A
    """
    raise NotImplementedError('Method _load_model() must be implemented in a subclass.')

  @abstractmethod
  def _run(self, env: SumoEnvironment, agent: A, learn: bool) -> dict[Metric, list[float]]:
    """
    Actually runs the given agent on the given environment.

    :param env: SumoEnvironment to run the agent in.
    :type env: SumoEnvironment
    :param agent: Agent to run.
    :type agent: A
    :param learn: Whether the agent should learn or run.
    :type learn: bool

    :return: Data of each metric of the run.
    :rtype: dict[Metric, list[float]]
    """
    raise NotImplementedError('Method _run() must be implemented in a subclass.')
  
  @abstractmethod
  def _save_model(self, agent: A) -> str:
    """
    Saves the agent model to a file and then returns its relative path.

    :param agent: Agent model to save.
    :type agent: A

    :return: The relative path of the save file.
    :rtype: str
    """
    raise NotImplementedError('Method _save_model() must be implemented in a subclass.')

class FixedCycleTrafficAgent(TrafficAgent[None, TrafficAgentConfig]):
  """ FixedCycleTrafficAgent to control traffic lights with a fixed cycle schema in a TrafficEnvironment. """

  def __init__(self, config: TrafficAgentConfig, traffic_env: TrafficEnvironment, canvas_config: CanvasConfig = CanvasConfig()) -> None:
    """
    FixedCycleTrafficAgent to control traffic lights with a fixed cycle schema in a TrafficEnvironment.

    :param config: TrafficAgentConfig.
    :type config: TrafficAgentConfig
    :param traffic_env: TrafficEnvironment to perform in.
    :type traffic_env: TrafficEnvironment
    :param canvas_config: Canvas configuration.
    :type canvas_config: CanvasConfig
    """
    super().__init__(config, traffic_env, canvas_config, True)

  def _get_agent(self, env: SumoEnvironment) -> None:
    return None

  def _load_model(self, env: SumoEnvironment, path: str) -> None:
    return None

  def _run(self, env: SumoEnvironment, agent: None, learn: bool) -> dict[Metric, list[float]]:
    metrics: dict[Metric, list[float]] = {}
    env.reset()
    if not learn:
      done = False
      while not done:
        done = self._step(env)
      metrics = {metric: [info[metric] for info in env.metrics] for metric in self._plotter.metrics}
      env.reset()
    return metrics

  def _step(self, env: SumoEnvironment) -> bool:
    """
    Step the given SumoEnvironment, used instead of env.step(action) because a FixedCycleTrafficAgent has no actions and needs less operations to be done.

    :param env: SumoEnvironment to step.
    :param env: SumoEnvironment

    :return: Whether the given SumoEnvironment simulation has terminated.
    :rtype: bool
    """
    for _ in range(self._traffic_env.delta_time):
      env._sumo_step()
    env._compute_observations()
    env._compute_rewards()
    env._compute_info()
    return env._compute_dones()['__all__']

  def _save_model(self, agent: None) -> str:
    return ''

class QLTrafficAgent(TrafficAgent[QLAgent, LearningAgentConfig]):
  """ TrafficAgent using a Q-Learning model. """

  def __init__(self, config: LearningAgentConfig, traffic_env: TrafficEnvironment, canvas_config: CanvasConfig = CanvasConfig()) -> None:
    """
    TrafficAgent using a Q-Learning model.

    :param config: TrafficAgentConfig.
    :type config: LearningAgentConfig
    :param traffic_env: TrafficEnvironment to perform in.
    :type traffic_env: TrafficEnvironment
    :param canvas_config: Canvas configuration.
    :type canvas_config: CanvasConfig
    """
    super().__init__(config, traffic_env, canvas_config)

  def _get_agent(self, env: SumoEnvironment) -> QLAgent:
    return QLAgent(
      starting_state = env.encode(env.reset()[0], env.ts_ids[0]),
      state_space = env.observation_space,
      action_space = env.action_space,
      alpha = self.config['alpha'],
      gamma = self.config['gamma'],
      exploration_strategy = EpsilonGreedy(self.config['init_eps'], self.config['min_eps'], self.config['decay'])
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

  def _run(self, env: SumoEnvironment, agent: QLAgent, learn: bool) -> dict[Metric, list[float]]:
    metrics: dict[Metric, list[float]] = {}
    done = False
    while not done:
      state, reward, _, done, _ = env.step(agent.act()) # type: ignore
      if learn:
        agent.learn(env.encode(state, env.ts_ids[0]), reward)
    metrics = {metric: [info[metric] for info in env.metrics] for metric in self._plotter.metrics}
    env.reset()
    return metrics

  def _save_model(self, agent: QLAgent) -> str:
    path = '{}.json'.format(self._get_filename('save', True))
    Path(Path(path).parent).mkdir(parents = True, exist_ok = True)
    with open(path, 'w+') as file:
      json.dump(agent, file, indent = 2, cls = QLAgentEncoder)
    return path

class DQLTrafficAgent(TrafficAgent[DQN, LearningAgentConfig]):
  """ TrafficAgent using a Deep Q-Learning model. """

  def __init__(self, config: LearningAgentConfig, traffic_env: TrafficEnvironment, canvas_config: CanvasConfig = CanvasConfig()) -> None:
    """
    TrafficAgent using a Deep Q-Learning model.

    :param config: TrafficAgentConfig.
    :type config: LearningAgentConfig
    :param traffic_env: TrafficEnvironment to perform in.
    :type traffic_env: TrafficEnvironment
    :param canvas_config: Canvas configuration.
    :type canvas_config: CanvasConfig
    """
    super().__init__(config, traffic_env, canvas_config)

  def _get_agent(self, env: SumoEnvironment) -> DQN:
    return DQN(
      policy = "MlpPolicy",
      env = env,
      learning_rate = self.config['alpha'],
      learning_starts = 0,
      gamma = self.config['gamma'],
      train_freq = (1, 'step'),
      gradient_steps = -1,
      target_update_interval = max(1, self._traffic_env.seconds // 100),
      exploration_fraction = self.config['decay'],
      exploration_initial_eps = self.config['init_eps'],
      exploration_final_eps = self.config['min_eps'],
      verbose = 0
    )

  def _load_model(self, env: SumoEnvironment, path: str) -> DQN:
    return DQN.load(env = env, path = path)

  def _run(self, env: SumoEnvironment, agent: DQN, learn: bool) -> dict[Metric, list[float]]:
    metrics: dict[Metric, list[float]] = {}
    if learn:
      metrics = {metric: [] for metric in self._plotter.metrics}
      agent.learn(self._traffic_env.seconds // self._traffic_env.delta_time, lambda locals, globals: {metrics[metric].append(locals['infos'][0][metric]) for metric in metrics}, 1)
    else:
      done = False
      state = env.reset()[0]
      while not done:
        state, _, _, done, _ = env.step(agent.predict(state)[0]) # type: ignore
      metrics = {metric: [info[metric] for info in env.metrics] for metric in self._plotter.metrics}
      env.reset()
    return metrics

  def _save_model(self, agent: DQN) -> str:
    path = '{}.zip'.format(self._get_filename('save', True))
    agent.save(path)
    return path
