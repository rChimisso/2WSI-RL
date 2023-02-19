import os
import sys
from datetime import datetime
from typing import Union, Callable, Generic, TypeVar
from abc import ABC, abstractmethod
from utils.plotter import Plotter, Metric
from stable_baselines3.dqn.dqn import DQN
if 'SUMO_HOME' in os.environ:
  tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

A = TypeVar('A')

class TrafficAgent(ABC, Generic[A]):
  def __init__(
    self,
    name: str,
    color: str,
    fixed: bool,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int
  ) -> None:
    self.name = name
    self.color = color
    self.fixed = fixed
    self.net = net
    self.rou = rou
    self.seconds = seconds
    self.delta_time = delta_time
    self.yellow_time = yellow_time
    self.min_green = min_green
    self.max_green = max_green
    self.plotter = Plotter(color, ['system_total_stopped', 'system_total_waiting_time', 'system_mean_waiting_time', 'system_mean_speed', 't_stopped', 't_accumulated_waiting_time', 't_average_speed', 'agents_total_stopped', 'agents_total_accumulated_waiting_time'])

  def _get_csv_name(self, *args: tuple[str, Union[int, float]]) -> str:
    experiment_time = str(datetime.now()).split('.')[0].replace(':', '-')
    experiment_parameters = ','.join([f'{arg[0]}={arg[1]}' for arg in args])
    return f'outputs/{self.name}/{experiment_time},{experiment_parameters}'

  def _get_env(
    self,
    out_csv_name: str,
    use_gui: bool,
    add_system_info: bool,
    add_per_agent_info: bool
  ) -> SumoEnvironment:
    return SumoEnvironment(
      net_file = self.net,
      route_file = self.rou,
      out_csv_name = out_csv_name,
      use_gui = use_gui,
      num_seconds = self.seconds,
      delta_time = self.delta_time,
      yellow_time = self.yellow_time,
      min_green = self.min_green,
      max_green = self.max_green,
      add_system_info = add_system_info,
      add_per_agent_info = add_per_agent_info,
      fixed_ts = self.fixed,
      single_agent = True
    )
  
  @abstractmethod
  def _get_agent(self, env: SumoEnvironment):
    raise NotImplementedError()

  def _update_metrics(self, info: dict[Metric, float], callback: Callable[[float, Metric, str], None]) -> None:
    for metric in info:
      self.plotter.append(info[metric], metric)
      callback(info[metric], metric, self.name)

  @abstractmethod
  def _learn(self, env: SumoEnvironment, agent: A, update_metrics: Callable[[dict[Metric, float]], None]) -> None:
    raise NotImplementedError()
  
  @abstractmethod
  def _save_model(self):
    raise NotImplementedError()

  def _save_csv(self, env: SumoEnvironment) -> None:
    env.save_csv(env.out_csv_name, 0)

  def _save_plots(self) -> None:
    self.plotter.save(self.name)

  def learn(
    self,
    update_metrics: Callable[[float, Metric, str], None],
    add_system_info: bool = True,
    add_per_agent_info: bool = True,
    use_gui: bool = False
  ) -> None:
    env = self._get_env(
      self._get_csv_name(
        ('seconds', self.seconds),
        ('delta_time', self.delta_time),
        ('yellow_time', self.yellow_time),
        ('min_green', self.min_green),
        ('max_green', self.max_green)
      ),
      use_gui,
      add_system_info,
      add_per_agent_info
    )
    self._learn(env, self._get_agent(env), lambda info: self._update_metrics(info, update_metrics))
    self._save_model()
    self._save_csv(env)
    self._save_plots()
    env.close()

class FixedCycleTrafficAgent(TrafficAgent[None]):
  def __init__(
    self,
    name: str,
    color: str,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int
  ) -> None:
    super().__init__(name, color, True, net, rou, seconds, delta_time, yellow_time, min_green, max_green)

  def _get_agent(self, _: SumoEnvironment) -> None:
    return None

  def _learn(self, env: SumoEnvironment, _, update_metrics: Callable[[dict[Metric, float]], None]):
    env.reset()
    done = False
    while not done:
      done = self._step(env)
      update_metrics(env.metrics[-1])

  def _step(self, env: SumoEnvironment):
    for _ in range(self.delta_time):
      env._sumo_step()
    env._compute_observations()
    env._compute_rewards()
    env._compute_info()
    return env._compute_dones()['__all__']

  def _save_model(self):
    pass

class LearningTrafficAgent(TrafficAgent[A], ABC, Generic[A]):
  def __init__(
    self,
    name: str,
    color: str,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int,
    alpha: float,
    gamma: float,
    init_eps: float,
    min_eps: float,
    decay: float
  ) -> None:
    super().__init__(name, color, False, net, rou, seconds, delta_time, yellow_time, min_green, max_green)
    self.alpha = alpha
    self.gamma = gamma
    self.init_eps = init_eps
    self.min_eps = min_eps
    self.decay = decay
  
  def _get_csv_name(self, *args: tuple[str, Union[int, float]]) -> str:
    return f'{super()._get_csv_name(*args)},alpha={self.alpha},gamma={self.gamma},init_eps={self.init_eps},min_eps={self.min_eps},decay={self.decay}'

class QLearningTrafficAgent(LearningTrafficAgent[QLAgent]):
  def __init__(
    self,
    name: str,
    color: str,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int,
    alpha: float,
    gamma: float,
    init_eps: float,
    min_eps: float,
    decay: float
  ) -> None:
    super().__init__(name, color, net, rou, seconds, delta_time, yellow_time, min_green, max_green, alpha, gamma, init_eps, min_eps, decay)

  def _get_agent(self, env: SumoEnvironment) -> QLAgent:
    return QLAgent(
      starting_state = env.encode(env.reset()[0], env.ts_ids[0]),
      state_space = env.observation_space,
      action_space = env.action_space,
      alpha = self.alpha,
      gamma = self.gamma,
      exploration_strategy = EpsilonGreedy(
        initial_epsilon = self.init_eps,
        min_epsilon = self.min_eps,
        decay = self.decay
      )
    )

  def _learn(self, env: SumoEnvironment, agent: QLAgent, update_metrics: Callable[[dict[Metric, float]], None]):
    done = False
    while not done:
      action = agent.act()
      state, reward, _, done, _ = env.step(action = action) # type: ignore
      update_metrics(env.metrics[-1])
      agent.learn(next_state = env.encode(state, env.ts_ids[0]), reward = reward)

  def _save_model(self):
    pass

class DeepQLearningTrafficAgent(LearningTrafficAgent[DQN]):
  def __init__(
    self,
    name: str,
    color: str,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int,
    alpha: float,
    gamma: float,
    init_eps: float,
    min_eps: float,
    decay_time: float
  ) -> None:
    super().__init__(name, color, net, rou, (seconds + delta_time) // delta_time, delta_time, yellow_time, min_green, max_green, alpha, gamma, init_eps, min_eps, decay_time)

  def _get_agent(self, env: SumoEnvironment) -> DQN:
    return DQN(
      policy = "MlpPolicy",
      env = env,
      learning_rate = self.alpha,
      learning_starts = 0,
      gamma = self.gamma,
      train_freq = 1,
      gradient_steps = -1,
      target_update_interval = 500,
      exploration_fraction = self.decay,
      exploration_initial_eps = self.init_eps,
      exploration_final_eps = self.min_eps,
      verbose = 1
    )

  def _learn(self, _: SumoEnvironment, agent: DQN, update_metrics: Callable[[dict[Metric, float]], None]):
    agent.learn(total_timesteps = self.seconds, callback = lambda locals, _: update_metrics(locals['infos'][0]))

  def _save_model(self):
    pass
