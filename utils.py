import os
from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


_WORKER_ID_BLOCK_SIZE = 32
_WORKER_ID_OFFSET = int(
    os.environ.get(
        "SOCCERTWOS_WORKER_ID_OFFSET",
        (int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 16)
        * _WORKER_ID_BLOCK_SIZE
        + 1,
    )
)
FIELD_X = 14.0
GOALIE_X = 10.5


def _clamp(value, low, high):
    return max(low, min(high, value))


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _position(info_dict, key):
    value = info_dict.get(key)
    if not value or "position" not in value:
        return None
    pos = value["position"]
    return pos[0], pos[1]


def _team_sign(agent_id):
    agent_id = int(agent_id)
    return 1.0 if agent_id in (0, 1) else -1.0


def _teammate_id(agent_id):
    agent_id = int(agent_id)
    return {0: 1, 1: 0, 2: 3, 3: 2}.get(agent_id)


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    def set_curriculum_task(self, task_index):
        if hasattr(self.env, "set_curriculum_task"):
            self.env.set_curriculum_task(task_index)


class ShapedRewardWrapper(gym.Wrapper):
    """Team-aware shaping for progress, pressure, spacing, and defensive blocking.
    Team 1 (agents 0,1) attacks +x; team 2 (agents 2,3) attacks -x."""

    def reset(self, **kw):
        self._prev_bx = 0.0
        return self.env.reset(**kw)

    def set_curriculum_task(self, task_index):
        if hasattr(self.env, "set_curriculum_task"):
            self.env.set_curriculum_task(task_index)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        bx_now = self._prev_bx
        ball_pos = None
        for aid in rew:
            ball_pos = _position(info.get(aid, {}), "ball_info")
            if ball_pos is not None:
                bx_now = ball_pos[0]
                break

        for aid in rew:
            agent_id = int(aid)
            sign = _team_sign(aid)
            player_pos = _position(info.get(aid, {}), "player_info")

            rew[aid] -= 0.001
            if ball_pos is not None:
                rew[aid] += 0.010 * sign * (ball_pos[0] - self._prev_bx)

            if player_pos is None or ball_pos is None:
                continue

            ball_dist = _distance(player_pos, ball_pos)
            rew[aid] += 0.002 * max(0.0, 1.0 - ball_dist / 8.0)

            teammate = _teammate_id(aid)
            teammate_info = info.get(teammate, info.get(str(teammate), {}))
            teammate_pos = _position(teammate_info, "player_info")
            if teammate_pos is not None:
                spacing = _distance(player_pos, teammate_pos)
                if 3.0 <= spacing <= 10.0:
                    rew[aid] += 0.0015
                elif spacing < 2.0:
                    rew[aid] -= 0.0010

            if agent_id in (1, 3):
                own_goal_x = -FIELD_X if sign > 0 else FIELD_X
                target = (
                    -GOALIE_X if sign > 0 else GOALIE_X,
                    _clamp(ball_pos[1], -3.5, 3.5),
                )
                rew[aid] += 0.003 * max(0.0, 1.0 - _distance(player_pos, target) / 8.0)
                between_ball_and_goal = (
                    own_goal_x <= player_pos[0] <= ball_pos[0]
                    if sign > 0
                    else ball_pos[0] <= player_pos[0] <= own_goal_x
                )
                if between_ball_and_goal:
                    rew[aid] += 0.002 * max(
                        0.0, 1.0 - abs(player_pos[1] - ball_pos[1]) / 5.0
                    )
            else:
                support_target = (ball_pos[0] - (2.0 * sign), ball_pos[1])
                rew[aid] += 0.002 * max(
                    0.0, 1.0 - _distance(player_pos, support_target) / 8.0
                )

        self._prev_bx = bx_now
        return obs, rew, done, info


class CurriculumResetWrapper(gym.Wrapper):
    """Applies staged initial-state randomization at episode reset."""

    def __init__(self, env, tasks, task_index=0):
        super().__init__(env)
        self.tasks = tasks
        self.task_index = task_index

    def set_curriculum_task(self, task_index):
        self.task_index = max(0, min(int(task_index), len(self.tasks) - 1))

    def reset(self, **kw):
        obs = self.env.reset(**kw)
        self._apply_curriculum_task()
        return obs

    def _apply_curriculum_task(self):
        task = self.tasks[self.task_index]
        ranges = task["ranges"]
        self.env.env_channel.set_parameters(
            ball_state=sample_pos_vel(ranges["ball"]),
            players_states={
                player: sample_player(player_ranges)
                for player, player_ranges in ranges["players"].items()
            },
        )


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = _WORKER_ID_OFFSET + (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    elif "worker_id" not in env_config:
        env_config = {**env_config, "worker_id": _WORKER_ID_OFFSET}
    internal_keys = {
        "curriculum_task_index",
        "curriculum_tasks",
        "num_envs_per_worker",
        "shaped_reward",
    }
    make_config = {
        key: value for key, value in env_config.items() if key not in internal_keys
    }
    env = soccer_twos.make(**make_config)
    if env_config.get("curriculum_tasks"):
        env = CurriculumResetWrapper(
            env,
            env_config["curriculum_tasks"],
            env_config.get("curriculum_task_index", 0),
        )
    if env_config.get("shaped_reward"):
        env = ShapedRewardWrapper(env)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
