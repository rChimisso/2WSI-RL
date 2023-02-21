from sumo_rl import SumoEnvironment

class FixedSumoEnvironment(SumoEnvironment):
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
    dones = {ts_id: False for ts_id in self.ts_ids}
    dones['__all__'] = self.sim_step >= self.sim_max_time # type: ignore
    return dones

class TrafficEnvironment():
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
    self.net = net
    self.rou = rou
    self.seconds = seconds + (delta_time - seconds % delta_time) % delta_time
    self.delta_time = delta_time
    self.yellow_time = yellow_time
    self.min_green = min_green
    self.max_green = max_green

  def get_sumo_env(self, fixed: bool, out_csv_name: str, use_gui: bool) -> SumoEnvironment:
    return FixedSumoEnvironment(
      net_file = self.net,
      route_file = self.rou,
      out_csv_name = out_csv_name,
      use_gui = use_gui,
      num_seconds = self.seconds,
      delta_time = self.delta_time,
      yellow_time = self.yellow_time,
      min_green = self.min_green,
      max_green = self.max_green,
      fixed_ts = fixed
    )
