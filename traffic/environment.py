from pathlib import Path
from pandas import DataFrame
from sumo_rl import SumoEnvironment

class SumoEnvironmentWrapper(SumoEnvironment):
  """ Wrapper for a SumoEnvironment, overrides save_csv to change the filename standard. """

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
    Wrapper for a SumoEnvironment, overrides save_csv to change the filename standard.

    :param net_file: Path to the net_file.
    :type net_file: str
    :param route_file: Path to the route_file.
    :type route_file: str
    :param out_csv_name: Path for the csv file to save.
    :type out_csv_name: str
    :param use_gui: Whether to use SUMO GUI when running.
    :type use_gui: bool
    :param num_seconds: Number of simulation seconds.
    :type num_seconds: int
    :param delta_time: Simulation seconds between actions. Must be at least greater than yellow_time.
    :type delta_time: int
    :param yellow_time: Fixed yellow time between actions.
    :type yellow_time: int
    :param min_green: Minimum green time in a phase.
    :type min_green: int
    :param max_green: Max green time in a phase.
    :type max_green: int
    :param fixed_ts: If true, it will follow the phase configuration in the route_file and ignore the actions.
    :type fixed_ts: bool
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
      add_per_agent_info = False
    )

  def save_csv(self, out_csv_name: str, run: int) -> None:
    """
    Saves the csv data to the given filepath.

    :param out_csv_name: Filepath to save to.
    :type out_csv_name: str
    :param run: Number of the run.
    :type run: int
    """
    Path(Path(out_csv_name).parent).mkdir(parents = True, exist_ok = True)
    DataFrame(self.metrics).to_csv(out_csv_name + '.csv', index = False)

class TrafficEnvironment():
  """ Wrapper to get new SumoEnvironments. """
  def __init__(self, net: str, rou: str, seconds: int, delta_time: int, yellow_time: int, min_green: int, max_green: int) -> None:
    """
    Wrapper to get new SumoEnvironments.

    :param net: Path to the net.xml file.
    :type net: str
    :param rou: Path to the rou.xml file.
    :type rou: str
    :param seconds: Number of simulation seconds.
    :type seconds: int
    :param delta_time: Simulation seconds between actions. Must be at least greater than yellow_time.
    :type delta_time: int
    :param yellow_time: Fixed yellow time between actions.
    :type yellow_time: int
    :param min_green: Minimum green time in a phase.
    :type min_green: int
    :param max_green: Max green time in a phase.
    :type max_green: int
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

  def set_seconds(self, seconds: int) -> None:
    """
    Sets the simulation seconds.

    :param seconds: New amount of simulation seconds.
    :type seconds: int
    """
    self._seconds = seconds

  def get_sumo_env(self, fixed: bool, out_csv_name: str, use_gui: bool) -> SumoEnvironment:
    """
    Returns a new SumoEnvironment.

    :param fixed: Whether a fixed cycle or a reinforcement learning schema is used.
    :type fixed: bool
    :param out_csv_name: Filepath to save csv data.
    :type out_csv_name: str
    :param use_gui: Whether to show SUMO GUI while running (if True, will slow down the run).
    :type use_gui: bool

    :return: A new SumoEnvironment.
    :rtype: SumoEnvironment.
    """
    return SumoEnvironmentWrapper(
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
