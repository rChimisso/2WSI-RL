from pathlib import Path
from pandas import DataFrame
from sumo_rl import SumoEnvironment

class FixedSumoEnvironment(SumoEnvironment):
  """ Same as SumoEnvironment, but overrides _compute_dones to fix https://github.com/LucasAlegre/sumo-rl/issues/132 and save_csv to change the filename standard. """

  def __init__(
    self,
    net_file: str,
    route_file: str,
    out_csv_name: str,
    use_gui: bool,
    num_seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int,
    fixed_ts: bool
  ) -> None:
    """
    Same as SumoEnvironment, but overrides _compute_dones to fix https://github.com/LucasAlegre/sumo-rl/issues/132 and save_csv to change the filename standard.

    :param net_file: (str) Path to the net_file.
    :param route_file: (str) Path to the route_file.
    :param out_csv_name: (str) Path for the csv file to save.
    :param use_gui: (bool) Whether to use SUMO GUI when running.
    :param num_seconds: (int) Number of simulation seconds.
    :param delta_time: (int) Simulation seconds between actions. Must be at least greater than yellow_time.
    :param yellow_time: (int) Fixed yellow time between actions.
    :param min_green: (int) Minimum green time in a phase.
    :param max_green: (int) Max green time in a phase.
    :param fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions.
    """
    super().__init__(
      net_file = net_file,
      route_file = route_file,
      out_csv_name = out_csv_name,
      use_gui = use_gui,
      num_seconds = num_seconds,
      max_depart_delay = 10000,
      delta_time = delta_time,
      yellow_time = yellow_time,
      min_green = min_green,
      max_green = max_green,
      fixed_ts = fixed_ts,
      single_agent = True,
      add_per_agent_info = True
    )

  def _compute_dones(self):
    """ Returns the dones for each agent, overridden to fix https://github.com/LucasAlegre/sumo-rl/issues/132 """
    dones = {ts_id: False for ts_id in self.ts_ids}
    dones['__all__'] = self.sim_step >= self.sim_max_time # type: ignore
    return dones

  def save_csv(self, out_csv_name: str, run: int):
    """
    Saves the csv data to the given filepath.

    :param out_csv_name: (str) filepath to save to.
    :param run: (int) Number of the run.
    """
    Path(Path(out_csv_name).parent).mkdir(parents = True, exist_ok = True)
    DataFrame(self.metrics).to_csv(out_csv_name + '.csv', index = False)

class TrafficEnvironment():
  """ Wrapper to get new SumoEnvironments. """
  def __init__(
    self,
    net: str,
    rou: str,
    seconds: int,
    delta_time: int,
    yellow_time: int,
    min_green: int,
    max_green: int
  ) -> None:
    """
    Wrapper to get new SumoEnvironments.

    :param net: (str) Path to the net.xml file.
    :param rou: (str) Path to the rou.xml file.
    :param seconds: (int) Number of simulation seconds.
    :param delta_time: (int) Simulation seconds between actions. Must be at least greater than yellow_time.
    :param yellow_time: (int) Fixed yellow time between actions.
    :param min_green: (int) Minimum green time in a phase.
    :param max_green: (int) Max green time in a phase.
    """
    self._net: str = net
    self._rou: str = rou
    self._seconds: int = seconds + (delta_time - seconds % delta_time) % delta_time
    self._delta_time: int = delta_time
    self._yellow_time: int = yellow_time
    self._min_green: int = min_green
    self._max_green: int = max_green

  @property
  def seconds(self) -> int:
    """ Number of simulation seconds. """
    return self._seconds

  @property
  def delta_time(self) -> int:
    """ Simulation seconds between actions. """
    return self._delta_time
  
  @property
  def yellow_time(self) -> int:
    """ Fixed yellow time between actions. """
    return self._yellow_time

  @property
  def min_green(self) -> int:
    """ Minimum green time in a phase. """
    return self._min_green

  @property
  def max_green(self) -> int:
    """ Max green time in a phase. """
    return self._max_green

  def get_sumo_env(self, fixed: bool, out_csv_name: str, use_gui: bool) -> SumoEnvironment:
    """
    Returns a new SumoEnvironment.

    :param fixed: (bool) Whether a fixed cycle or a reinforcement learning schema is used.
    :param out_csv_name: (str) filepath to save csv data.
    :param use_gui: (bool) Whether to show SUMO GUI while running (if True, will slow down the run).
    """
    return FixedSumoEnvironment(
      net_file = self._net,
      route_file = self._rou,
      out_csv_name = out_csv_name,
      use_gui = use_gui,
      num_seconds = self.seconds,
      delta_time = self.delta_time,
      yellow_time = self.yellow_time,
      min_green = self.min_green,
      max_green = self.max_green,
      fixed_ts = fixed
    )
