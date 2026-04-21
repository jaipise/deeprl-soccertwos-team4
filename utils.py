import os
from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos


_WORKER_ID_OFFSET = int(os.environ.get("SLURM_JOB_ID", os.getpid())) % 5000 + 1000


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class ShapedRewardWrapper(gym.Wrapper):
    """Team-aware shaping: ball progress toward each team's goal + step penalty.
    Team 1 (agents 0,1) attacks +x; team 2 (agents 2,3) attacks -x."""

    def reset(self, **kw):
        self._prev_bx = 0.0
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        bx_now = self._prev_bx
        for aid in rew:
            rew[aid] -= 0.001
            bi = info.get(aid, {}).get("ball_info")
            if bi is not None:
                bx_now = bi["position"][0]
                sign = 1.0 if aid in (0, 1) else -1.0
                rew[aid] += 0.01 * sign * (bx_now - self._prev_bx)
        self._prev_bx = bx_now
        return obs, rew, done, info


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
    env = soccer_twos.make(**env_config)
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
