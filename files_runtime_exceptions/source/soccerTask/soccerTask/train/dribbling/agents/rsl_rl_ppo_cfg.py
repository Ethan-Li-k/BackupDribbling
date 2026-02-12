from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class DribblingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 # Reverted to 24 to restore training speed
    max_iterations = 30001
    save_interval = 100
    experiment_name = "dribbling_g1"
    empirical_normalization = False
    logger = "tensorboard"
    
    # Resume configuration
    resume = False
    load_run = "/root/logs/rsl_rl/dribbling_g1/2025-12-11_18-39-30"
    load_checkpoint = "model_2900.pt"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4, # Reduced from 1e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
