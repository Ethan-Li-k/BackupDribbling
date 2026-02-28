from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

import soccerTask.train.dribbling.mdp as mdp

##
# Pre-defined configs
##
# from soccerTask.assets.t1_humanoid import T1_HUMANOID_CFG  # Removed: Not found
# from isaaclab_assets.robots.t1_humanoid import T1_HUMANOID_CFG as T1_REF # Removed: Not found

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

# Define T1 Config locally since it's not in the python path
T1_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/root/data/assets/assetslib/T1new/T1.urdf",
        fix_base=False,
        make_instanceable=False, # Disable instanceable to avoid path issues with Fabric
        joint_drive=None,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0, # Increased to help with penetration
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65), # Adjust height to 0.65m
        joint_pos={
            ".*": 0.0, # Default to 0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips_knees": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_.*_joint", "right_hip_.*_joint", "left_knee_joint", "right_knee_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=20.0,
            stiffness=250.0,
            damping=10.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_.*_joint", "right_ankle_.*_joint"],
            effort_limit_sim=200.0,
            velocity_limit_sim=20.0,
            stiffness=100.0,
            damping=2.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["Head_yaw", "Head_pitch"],
            effort_limit_sim=20.0,
            velocity_limit_sim=10.0,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)

@configclass
class DribblingSceneCfg(InteractiveSceneCfg):
    """Configuration for the dribbling scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5, restitution=0.0), # Increased friction
        ),
    )

    # robot
    robot: ArticulationCfg = T1_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ball
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.10,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.33),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=1.5,
                angular_damping=0.0,
                max_depenetration_velocity=100.0, # Increased significantly to force separation
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.01), # Increased offset and added rest offset to prevent clipping
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.8, static_friction=1.0, dynamic_friction=1.0), # High bounce and friction
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.10)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

@configclass
class CommandsCfg:
    """Command specifications for the environment."""
    # We might not need commands if the task is just "dribble forward" or "chase ball"
    # But usually we have a velocity command or target position.
    # For this specific "dribbling" task, the target is the ball.
    pass

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["left_hip_.*_joint", "right_hip_.*_joint", "left_knee_joint", "right_knee_joint", "left_ankle_.*_joint", "right_ankle_.*_joint", "Head_yaw", "Head_pitch"], 
        scale=0.3,  # Increased from 0.25
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Base observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )

        # Joint observations
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=AdditiveUniformNoiseCfg(n_min=-1.5, n_max=1.5))
        
        # Actions
        actions = ObsTerm(func=mdp.last_action)

        # Task observations (Ball)
        ball_pos = ObsTerm(
            func=mdp.ball_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
        )
        ball_vel = ObsTerm(
            func=mdp.ball_velocity_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "ball_cfg": SceneEntityCfg("ball")},
        )

        # Gait Phase (New)
        gait_phase = ObsTerm(func=mdp.gait_phase_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # Startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0), # Fixed friction
            "dynamic_friction_range": (1.0, 1.0), # Fixed friction
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}, # Reduced range
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0), # Fixed scale
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (1.0, 2.0), "y": (-0.5, 0.5)}, # Spawn ball in front (Narrowed from -1.0, 1.0)
            "velocity_range": {},
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the environment."""

    # -- Stage 1 Reference Rewards --
    
    # Joint Position (Reference Motion)
    joint_pos = RewTerm(
        func=mdp.joint_pos_reward_stage1,
        weight=0.5, # Increased from 0.2 to force following the motion
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_joint_pos_scale": 0.8, # Increased from 0.5 to 0.8 (Aggressive Stepping)
            "ref_pos_dir": [1, -1, 1, -1, 1, -1],
            "cycle_time": 0.8,
            "double_stand_phase": 0.5,
        },
    )

    # Feet Orientation
    feet_orientation = RewTerm(
        func=mdp.feet_orientation_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")},
    )

    # Feet Distance
    feet_distance = RewTerm(
        func=mdp.feet_distance_reward,
        weight=0.2,
        params={
            "min_dist": 0.24,
            "max_dist": 0.28,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # Feet Contact Forces
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces_reward,
        weight=-0.0003,
        params={
            "max_contact_force": 300.0,
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

        # Feet Clearance (Stage 1: 1.0)
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_reward,
        weight=0.5, # Reduced from 2.0 to avoid high-stepping in place
        params={
            "target_feet_height": 0.08, 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # Feet Air Time (Aggressive Stepping)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0, # Increased from 1.0
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.5,
        },
    )

    # Ball Distance (Approach ball)
    ball_distance = RewTerm(
        func=mdp.ball_distance_exp,
        weight=2.5, # Increased from 2.0 to force movement towards ball
        params={
            "std": 2.0, # Larger std for wider attraction basin
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # Base Height (New)
    base_height = RewTerm(
        func=mdp.base_height_reward,
        weight=4.0, 
        params={
            "target_height": 0.60, # Reduced from 0.65 to lower COG
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Joint Position (Reference Motion)
    joint_pos = RewTerm(
        func=mdp.joint_pos_reward_stage1,
        weight=2.0, # Increased back to 1.5 to guide motion
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_joint_pos_scale": 1.2,
            "ref_pos_dir": [1, -1, 1, -1, 1, -1],
            "cycle_time": 0.8,
            "double_stand_phase": 0.5,
        },
    )

    # Feet Distance
    feet_distance = RewTerm(
        func=mdp.feet_distance_reward,
        weight=0.8, 
        params={
            "min_dist": 0.20, # Reduced min to allow crossing if needed
            "max_dist": 0.60, # Increased max from 0.28 to allow large steps
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # Feet Stride (New: Encourage large steps)
    feet_stride = RewTerm(
        func=mdp.feet_stride_reward,
        weight=3.0, # Increased from 2.0
        params={
            "min_stride": 0.35, # Increased from 0.3
            "max_stride": 0.80,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
        },
    )

    # Head Position (New: Keep head up)
    head_pos = RewTerm(
        func=mdp.head_position_reward,
        weight=1.0, # Restored to 1.0 to keep head stable
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Track Ball Velocity (Restored: Encourage speed)
    track_ball_vel = RewTerm(
        func=mdp.track_ball_velocity,
        weight=1.5, # Increased from 1.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "target_speed": 1.0, # Target 1.0 m/s approach speed
        },
    )

    # Collision (Undesired Contacts)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip_.*|.*_shank_link|Trunk|.*_knee_.*|AL.*|AR.*|.*_hand_.*"), "threshold": 1.0}, # Added arms and hands
    )

    # Tracking Ball View (Reduced)
    tracking_ball_view = RewTerm(
        func=mdp.tracking_ball_view,
        weight=0.1, # Reduced from 0.2
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # Tracking Ball Target Velocity (Reduced)
    tracking_ball_target_vel = RewTerm(
        func=mdp.tracking_ball_target_vel_reward_fixed,
        weight=1.5, # Increased back to 1.5 to encourage chasing
        params={
            "target_vel_x": 1.0,
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # Tracking Ball View (Restored)
    tracking_ball_view = RewTerm(
        func=mdp.tracking_ball_view,
        weight=0.5, # Increased to encourage looking at ball
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # Tracking Ball Target Velocity (Disabled: Ball is static)
    # tracking_ball_target_vel = RewTerm(
    #     func=mdp.tracking_ball_target_vel_reward_fixed,
    #     weight=1.5,
    #     params={
    #         "target_vel_x": 1.0, # Target 1.0 m/s forward
    #         "ball_cfg": SceneEntityCfg("ball"),
    #     },
    # )

    # Tracking Angular Velocity
    tracking_ang_vel = RewTerm(
        func=mdp.tracking_ang_vel_reward,
        weight=0.2,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # Base Orientation
    orientation = RewTerm(
        func=mdp.orientation_reward,
        weight=1.0, # Set to 1.0 as requested
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Action Smoothness
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_reward,
        weight=-0.002, # Reduced from -0.005
    )

    # Torques
    torques = RewTerm(
        func=mdp.torques_reward,
        weight=-5e-6, # Reduced from -1e-5
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # DOF Velocity
    dof_vel = RewTerm(
        func=mdp.dof_vel_reward,
        weight=-1e-6, # Reduced from -1e-5
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # DOF Acceleration
    dof_acc = RewTerm(
        func=mdp.dof_acc_reward,
        weight=-1e-9,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Collision (Undesired Contacts)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip_.*|.*_shank_link|Trunk"), "threshold": 1.0},
    )

    # Survival reward (Optional, not in Stage 1 list but good for stability)
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)

@configclass
class TerminationsCfg:
    """Termination terms for the environment."""
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk|.*_hip_.*|.*_knee_.*|AL.*|AR.*|.*_hand_.*"), "threshold": 1.0},
    )

@configclass
class DribblingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the dribbling environment."""

    # Scene settings
    scene: DribblingSceneCfg = DribblingSceneCfg(num_envs=8192, env_spacing=2.5) # Increased to 8192 to utilize GPU
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post-initialization settings
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.ground.spawn.physics_material
        
        # PhysX settings from reference
        if self.sim.physx is not None:
            self.sim.physx.solver_type = 1 # TGS
            self.sim.physx.bounce_threshold_velocity = 0.1
            self.sim.physx.friction_offset_threshold = 0.01 # contact_offset? No, friction_offset_threshold is different.
            # Isaac Lab PhysxCfg has:
            # solver_type, min_position_iteration_count, min_velocity_iteration_count, 
            # bounce_threshold_velocity, friction_offset_threshold, friction_correlation_distance, 
            # enable_enhanced_determinism, enable_stabilization, max_depenetration_velocity, 
            # gpu_max_rigid_contact_count, gpu_max_rigid_patch_count, gpu_found_lost_pairs_capacity, 
            # gpu_found_lost_aggregate_pairs_capacity, gpu_total_aggregate_pairs_capacity, 
            # gpu_max_soft_body_contacts, gpu_max_particle_contacts, gpu_heap_capacity, 
            # gpu_temp_buffer_capacity, gpu_max_num_partitions, gpu_max_soft_body_particles, 
            # gpu_max_particle_fluids, gpu_max_particle_cloths, gpu_max_particle_inflateds, 
            # gpu_max_particle_diffuses, gpu_max_hair_contacts
            
            # contact_offset in Isaac Gym is usually handled by simulation mesh accuracy or collision offset.
            # In Isaac Lab, it might be `friction_offset_threshold` or similar, but `contact_offset` is a property of the scene or shapes.
            # Let's stick to what we can map easily.
            self.sim.physx.bounce_threshold_velocity = 0.1
            # max_depenetration_velocity is usually on rigid bodies, but can be global.
            # self.sim.physx.max_depenetration_velocity = 1.0 # Not always available in global config depending on version.
            
        # update sensor update periods
        # we want all sensors to update on every step
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

@configclass
class LeggedRobotCfgPPO: # Removed inheritance from ManagerBasedRLEnvCfg as it's a config class for the runner
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01 # Increased from 0.001 for exploration
        learning_rate = 1e-4 # Increased from 1e-5 for speed
        num_learning_epochs = 2
        schedule = 'adaptive' 
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 30001  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'dribbling_g1' # Updated to match log folder
        run_name = ''
        # load and resume
        resume = True
        load_run = "/root/logs/rsl_rl/dribbling_g1/2025-12-11_14-12-08" # -1 = last run
        checkpoint = "model_950.pt" # -1 = last saved model
        resume_path = None  # updated from load_run

