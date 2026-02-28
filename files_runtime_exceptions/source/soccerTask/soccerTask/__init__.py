import gymnasium as gym

from . import train

##
# Register Gym environments.
##

gym.register(
    id="Loco-G1-Dribbling",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{train.__name__}.dribbling.dribbling_env_cfg:DribblingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{train.__name__}.dribbling.agents.rsl_rl_ppo_cfg:DribblingPPORunnerCfg",
    },
)
