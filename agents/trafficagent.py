import os
import sys
import json
import numpy
from pathlib import Path
from datetime import datetime
from typing import Union, Literal, Callable, Generic, TypeVar
from abc import ABC, abstractmethod
from utils.plotter import Plotter, PlotData, Metric
from agents.environments import TrafficEnvironment
from stable_baselines3.dqn.dqn import DQN
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

default_metrics: list[Metric] = [
  'system_total_stopped',
  'system_total_waiting_time',
  'system_mean_waiting_time',
  'system_mean_speed',
  't_stopped',
  't_accumulated_waiting_time',
  't_average_speed',
  'agents_total_stopped',
  'agents_total_accumulated_waiting_time'
]

class QLAgentEncoder(json.JSONEncoder):
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

A = TypeVar('A')

class TrafficAgent(ABC, Generic[A]):
  def __init__(
    self,
    name: str,
    color: str,
    fixed: bool,
    traffic_env: TrafficEnvironment,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    self.time = '-'
    self.name = name
    self.color = color
    self.fixed = fixed
    self.traffic_env = traffic_env
    self.plotter = Plotter(color, plot_data)

  def _get_file_name(self, type: Literal['csv', 'save']) -> str:
    return f'outputs/{self.name}/{type}s/{self.time},seconds={self.traffic_env.seconds},delta_time={self.traffic_env.delta_time},yellow_time={self.traffic_env.yellow_time},min_green={self.traffic_env.min_green},max_green={self.traffic_env.max_green}'
  
  @abstractmethod
  def _get_agent(self, env: SumoEnvironment):
    raise NotImplementedError()

  def _update_metrics(self, info: dict[Metric, float], callback: Callable[[float, Metric, str], None]) -> None:
    for metric in info:
      self.plotter.append(info[metric], metric)
      callback(info[metric], metric, self.name)

  @abstractmethod
  def _run(self, env: SumoEnvironment, agent: A, update_metrics: Callable[[dict[Metric, float]], None], learn: bool) -> None:
    raise NotImplementedError()
  
  @abstractmethod
  def _save_model(self, agent: A):
    raise NotImplementedError()

  def _save_csv(self, env: SumoEnvironment) -> None:
    env.save_csv(env.out_csv_name, 0)

  def _save_plots(self) -> None:
    self.plotter.save(self.name)

  @abstractmethod
  def _load_model(self, env: SumoEnvironment, path: str) -> A:
    raise NotImplementedError()

  def run(
    self,
    update_metrics: Callable[[float, Metric, str], None],
    use_gui: bool = False,
    load_path: Union[str, None] = None
  ) -> None:
    self.time = str(datetime.now()).split('.')[0].replace(':', '-')
    env = self.traffic_env.get_sumo_env(self.fixed, self._get_file_name('csv'), use_gui)
    agent = self._get_agent(env) if load_path is None else self._load_model(env, load_path)
    self._run(env, agent, lambda info: self._update_metrics(info, update_metrics), load_path is None)
    if load_path is None:
      self._save_model(agent)
    self._save_csv(env)
    self._save_plots()
    env.close()

class FixedCycleTrafficAgent(TrafficAgent[None]):
  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    super().__init__(name, color, True, traffic_env, plot_data)

  def _get_agent(self, env: SumoEnvironment) -> None:
    return None

  def _run(self, env: SumoEnvironment, agent: None, update_metrics: Callable[[dict[Metric, float]], None], learn: bool):
    env.reset()
    done = False
    while not done:
      done = self._step(env)
      update_metrics(env.metrics[-1])

  def _step(self, env: SumoEnvironment):
    for _ in range(self.traffic_env.delta_time):
      env._sumo_step()
    env._compute_observations()
    env._compute_rewards()
    env._compute_info()
    return env._compute_dones()['__all__']

  def _save_model(self, agent: None):
    pass

  def _load_model(self, env: SumoEnvironment, path: str) -> None:
    return None

class LearningTrafficAgent(TrafficAgent[A], ABC, Generic[A]):
  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    alpha: float,
    gamma: float,
    init_eps: float,
    min_eps: float,
    decay: float,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    super().__init__(name, color, False, traffic_env, plot_data)
    self.alpha = alpha
    self.gamma = gamma
    self.init_eps = init_eps
    self.min_eps = min_eps
    self.decay = decay
  
  def _get_file_name(self, type: Literal['csv', 'save']) -> str:
    return f'{super()._get_file_name(type)},alpha={self.alpha},gamma={self.gamma},init_eps={self.init_eps},min_eps={self.min_eps},decay={self.decay}'

class QLearningTrafficAgent(LearningTrafficAgent[QLAgent]):
  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    alpha: float,
    gamma: float,
    init_eps: float,
    min_eps: float,
    decay: float,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    super().__init__(name, color, traffic_env, alpha, gamma, init_eps, min_eps, decay, plot_data)

  def _get_agent(self, env: SumoEnvironment) -> QLAgent:
    return QLAgent(
      starting_state = env.encode(env.reset()[0], env.ts_ids[0]),
      state_space = env.observation_space,
      action_space = env.action_space,
      alpha = self.alpha,
      gamma = self.gamma,
      exploration_strategy = EpsilonGreedy(self.init_eps, self.min_eps, self.decay)
    )

  def _run(self, env: SumoEnvironment, agent: QLAgent, update_metrics: Callable[[dict[Metric, float]], None], learn: bool):
    done = False
    while not done:
      action = agent.act()
      state, reward, _, done, _ = env.step(action) # type: ignore
      update_metrics(env.metrics[-1])
      if learn:
        agent.learn(next_state = env.encode(state, env.ts_ids[0]), reward = reward)

  def _save_model(self, agent: QLAgent):
    path = '{}.json'.format(self._get_file_name('save'))
    Path(Path(path).parent).mkdir(parents = True, exist_ok = True)
    with open(path, 'w+') as file:
      json.dump(agent, file, indent = 2, cls = QLAgentEncoder)

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

class DeepQLearningTrafficAgent(LearningTrafficAgent[DQN]):
  def __init__(
    self,
    name: str,
    color: str,
    traffic_env: TrafficEnvironment,
    alpha: float,
    gamma: float,
    init_eps: float,
    min_eps: float,
    decay_time: float,
    plot_data: PlotData = PlotData(default_metrics)
  ) -> None:
    super().__init__(name, color, traffic_env, alpha, gamma, init_eps, min_eps, decay_time, plot_data)

  def _get_agent(self, env: SumoEnvironment) -> DQN:
    return DQN(
      policy = "MlpPolicy",
      env = env,
      learning_rate = self.alpha,
      learning_starts = 0,
      # batch_size = self.traffic_env.seconds,
      gamma = self.gamma,
      # train_freq = 1,
      # train_freq = (1, 'episode'),
      train_freq = self.traffic_env.delta_time,
      gradient_steps = -1,
      target_update_interval = self.traffic_env.seconds // 100, # Cambiato di recente
      exploration_fraction = self.decay,
      exploration_initial_eps = self.init_eps,
      exploration_final_eps = self.min_eps,
      verbose = 1
    )

  def _run(self, env: SumoEnvironment, agent: DQN, update_metrics: Callable[[dict[Metric, float]], None], learn: bool):
    if learn:
      # reset_num_timesteps default a True, False sembra non avere effetti
      agent.learn(total_timesteps = self.traffic_env.seconds, reset_num_timesteps = True, callback = lambda locals, globals: update_metrics(locals['infos'][0]))
    else:
      done = False
      state = env.reset()[0]
      while not done:
        action= agent.predict(state)[0]
        state, _, _, done, _ = env.step(action) # type: ignore
        update_metrics(env.metrics[-1])

  def _save_model(self, agent: DQN):
    agent.save('{}.zip'.format(self._get_file_name('save')))

  def _load_model(self, env: SumoEnvironment, path: str) -> DQN:
    return DQN.load(env = env, path = path)
