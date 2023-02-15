import os
import sys
from datetime import datetime
from typing import Callable

if 'SUMO_HOME' in os.environ:
  tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
  sys.path.append(tools)
else:
  sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

def execution(updateMetrics: Callable[[str, dict[str, int | float]], None], name: str, seconds: int, fixed: bool):
  route='nets/single-intersection/single-intersection.rou.xml'
  alpha=0.1
  gamma=0.99
  epsilon=1
  min_epsilon=0.005
  decay=0.9
  min_green=10
  max_green=50
  gui=False
  experiment_time = str(datetime.now()).split('.')[0]
  out_csv = f'outputs/single-intersection/{experiment_time}_alpha{alpha}_gamma{gamma}_eps{epsilon}_decay{decay}_fixed{fixed}'.replace(':', '-')

  env = SumoEnvironment(
    net_file='nets/single-intersection/single-intersection.net.xml',
    delta_time=5,
    yellow_time=3,
    route_file=route,
    out_csv_name=out_csv,
    use_gui=gui,
    num_seconds=seconds,
    min_green=min_green,
    max_green=max_green,
    fixed_ts=fixed
  )

  initial_states = env.reset()
  ql_agents = {
    ts: QLAgent(
      starting_state=env.encode(initial_states[ts], ts),
      state_space=env.observation_space,
      action_space=env.action_space,
      alpha=alpha,
      gamma=gamma,
      exploration_strategy=EpsilonGreedy(initial_epsilon=epsilon, min_epsilon=min_epsilon, decay=decay)
    ) for ts in env.ts_ids
  }

  done = {'__all__': False}
  if fixed:
    while not done['__all__']:
      _, _, done, _ = env.step({})
      updateMetrics(name, env.metrics[-1])
  else:
    while not done['__all__']:
      actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

      s, r, done, _ = env.step(action=actions)

      updateMetrics(name, env.metrics[-1])

      for agent_id in ql_agents.keys():
        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

  # env.save_csv(out_csv, 1)
  env.close()
