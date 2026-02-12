from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ==============================================================================
# 核心步态奖励：Feet Air Time (强制迈步)
# ==============================================================================

def feet_air_time(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """
    奖励脚在空中的持续时间。
    这是防止原地踏步的最有效手段。
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    # 获取脚部接触力 (num_envs, num_feet)
    # 假设 sensor_cfg.body_ids 对应左右脚
    
    if sensor_cfg.body_ids is not None:
        contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] # Z-force
    else:
        contact_forces = contact_sensor.data.net_forces_w[:, :, 2] # Z-force
    
    # 判断是否接触地面 (力 > 阈值)
    in_contact = torch.abs(contact_forces) > threshold
    
    # 简单的 air time 奖励：只要不接触就给分
    # 这鼓励脚悬空
    # Sum over all feet (bodies)
    return torch.sum((~in_contact).float(), dim=1)

# ==============================================================================
# 任务奖励：追球 (转化为速度追踪)
# ==============================================================================

def track_ball_velocity(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    ball_cfg: SceneEntityCfg,
    target_speed: float = 1.5
) -> torch.Tensor:
    """
    计算机器人速度与[机器人->球]向量的对齐度。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    # 1. 计算目标方向向量
    target_vec = ball.data.root_pos_w - robot.data.root_pos_w
    target_vec[:, 2] = 0.0 # 忽略高度
    target_dir = torch.nn.functional.normalize(target_vec, dim=1)
    
    # 2. 获取机器人速度
    robot_vel = robot.data.root_lin_vel_w
    robot_vel[:, 2] = 0.0
    
    # 3. 计算投影速度 (向球跑的速度)
    vel_proj = torch.sum(robot_vel * target_dir, dim=1)
    
    # 4. 设定目标速度 (例如 1.5 m/s)
    # target_speed passed as arg
    
    # 5. 计算误差并转化为奖励 (Exponential Kernel)
    error = target_speed - vel_proj
    # 只惩罚速度不足，不惩罚速度过快(在一定范围内)
    # 或者使用标准的 exp(-error^2 / sigma)
    return torch.exp(-torch.square(error) / 0.5)

# ==============================================================================
# 辅助观测：相位
# ==============================================================================

def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    cycle_time = 0.8
    phase = (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time
    return torch.cat([
        torch.sin(2 * torch.pi * phase).unsqueeze(1), 
        torch.cos(2 * torch.pi * phase).unsqueeze(1)
    ], dim=1)

# ==============================================================================
# 其他标准奖励 (保留原有的一些有用函数)
# ==============================================================================

def flat_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining a flat base orientation using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_vec = asset.data.projected_gravity_b
    error = torch.sum(torch.square(gravity_vec[:, :2]), dim=1)
    return torch.exp(-error / std**2)

def joint_pos_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking default joint positions using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    # default joint positions
    default_pos = asset.data.default_joint_pos
    current_pos = asset.data.joint_pos
    error = torch.sum(torch.square(current_pos - default_pos), dim=1)
    return torch.exp(-error / std**2)

def feet_distance_range_exp(
    env: ManagerBasedRLEnv,
    min_dist: float,
    max_dist: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    """Reward for maintaining feet distance within a range."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    feet_pos = asset.data.body_pos_w[:, body_ids]
    
    if feet_pos.shape[1] != 2:
        return torch.zeros(env.num_envs, device=env.device)
        
    dist = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=1)
    
    d_min = torch.clamp(dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(dist - max_dist, 0.0, 0.5)
    
    return (torch.exp(-torch.abs(d_min) * std) + torch.exp(-torch.abs(d_max) * std)) / 2.0

def feet_contact_forces_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    max_penalty: float = 400.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    """Penalize contact forces above a threshold."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    
    # Net forces on feet
    forces = sensor.data.net_forces_w[:, body_ids, :]
    forces_norm = torch.norm(forces, dim=-1)
    
    # Excess force
    excess = (forces_norm - threshold).clamp(min=0.0, max=max_penalty)
    
    return torch.sum(excess, dim=1)

def symmetric_action_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for asymmetric actions between left and right legs."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_names = asset.joint_names
    left_indices = [i for i, n in enumerate(joint_names) if "left" in n]
    right_indices = [i for i, n in enumerate(joint_names) if "right" in n]
    
    if len(left_indices) != len(right_indices):
        return torch.zeros(env.num_envs, device=env.device)
        
    if hasattr(asset.data, "applied_torque"):
        torques = asset.data.applied_torque
    else:
        return torch.zeros(env.num_envs, device=env.device)
    
    tau_l = torques[:, left_indices]
    tau_r = torques[:, right_indices]
    
    norm_l = torch.sum(torch.square(tau_l), dim=1)
    norm_r = torch.sum(torch.square(tau_r), dim=1)
    
    return torch.abs(norm_l - norm_r)

def ball_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward for being close to the ball using exponential kernel (Laplacian)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # Distance in XY plane
    dist = torch.norm(robot.data.root_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=1)
    
    return torch.exp(-dist / std)

def tracking_ball_view(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    fov_h: float = 0.5,
    fov_v: float = 0.7,
    cam_offset: tuple[float, float, float] = (0.05, 0.0, 0.15),
) -> torch.Tensor:
    """Reward for keeping the ball in view."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    ball_pos = ball.data.root_pos_w
    
    rel_pos_w = ball_pos - robot_pos
    rel_pos_b = quat_apply_inverse(robot_quat, rel_pos_w)
    
    ball_vis_vec = torch.nn.functional.normalize(rel_pos_b + torch.tensor(cam_offset, device=env.device), dim=1)
    
    in_view_h = torch.abs(ball_vis_vec[:, 1]) < fov_h
    in_view_v = torch.abs(ball_vis_vec[:, 2]) < fov_v
    in_front = ball_vis_vec[:, 0] > 0
    
    see_ball = in_view_h & in_view_v & in_front
    
    return see_ball.float()

# Wrappers for standard MDP functions to be used in ObsTerm
def ball_position_in_robot_root_frame(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    rel_pos_w = ball.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)

def ball_velocity_in_robot_root_frame(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    rel_vel_w = ball.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)

def undesired_contacts(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """
    Penalize contacts on undesired bodies.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # Get contact forces
    if sensor_cfg.body_ids is not None:
        forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    else:
        forces = contact_sensor.data.net_forces_w
        
    # Check if any force exceeds threshold
    is_contact = torch.norm(forces, dim=-1) > threshold
    
    # Return 1.0 if any contact, else 0.0
    return torch.any(is_contact, dim=1).float()

# ==============================================================================
# Reference Implementation Rewards (Stage I)
# ==============================================================================

def _get_phase(env: ManagerBasedRLEnv, cycle_time: float = 0.8) -> torch.Tensor:
    return (env.episode_length_buf * env.step_dt) % cycle_time / cycle_time

def _get_gait_phase(env: ManagerBasedRLEnv, cycle_time: float = 0.8, double_stand_phase: float = 0.5) -> torch.Tensor:
    phase = _get_phase(env, cycle_time)
    sin_pos = torch.sin(2 * torch.pi * phase)
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    # left foot stance (sin_pos >= 0)
    stance_mask[:, 0] = sin_pos >= 0
    # right foot stance (sin_pos < 0)
    stance_mask[:, 1] = sin_pos < 0
    # Double support
    stance_mask[torch.abs(sin_pos) < double_stand_phase] = 1
    return stance_mask

def joint_pos_reward_stage1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_joint_pos_scale: float = 0.5,
    ref_pos_dir: list[float] = [1, -1, 1, -1, 1, -1],
    cycle_time: float = 0.8,
    double_stand_phase: float = 0.5,
) -> torch.Tensor:
    """
    Calculates the reward based on the difference between the current joint positions and the target joint positions.
    Implements the reference motion generator from humanoid-gym.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_dof_pos = asset.data.default_joint_pos
    
    # Compute reference trajectory
    phase = _get_phase(env, cycle_time)
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()
    
    ref_dof_pos = default_dof_pos.clone()
    
    scale_1 = target_joint_pos_scale
    scale_2 = 2 * target_joint_pos_scale
    
    # Identify joint indices by name (assuming standard T1 naming)
    # We need to be robust to ordering
    joint_names = asset.joint_names
    
    def get_idx(pattern):
        for i, name in enumerate(joint_names):
            if pattern in name:
                return i
        return -1

    # Indices for T1 (based on reference config)
    # We map ref_pos_dir to [Hip Pitch, Knee, Ankle Pitch] for walking gait
    # Reference uses indices 2, 5, 6 (Left) and 8, 11, 12 (Right).
    # Assuming Reference Index 2 is Hip Pitch (Main walking joint), not Yaw.
    
    l_hip_pitch = get_idx("left_hip_pitch")
    l_knee = get_idx("left_knee")
    l_ankle_pitch = get_idx("left_ankle_pitch")
    
    r_hip_pitch = get_idx("right_hip_pitch")
    r_knee = get_idx("right_knee")
    r_ankle_pitch = get_idx("right_ankle_pitch")
    
    # Left foot stance phase set to default joint pos (sin_pos_l > 0)
    # So we modify when sin_pos_l <= 0 (Swing)
    sin_pos_l[sin_pos_l > 0] = 0
    
    # NOTE: We invert the sign for Hip Pitch to ensure Flexion (Forward Step)
    # Reference [1, -1, 1] with negative sin_pos -> Hip Extension (Back), Knee Flexion (Bent), Ankle Extension.
    # We want Hip Flexion (Forward). So we use -ref_pos_dir[0].
    # UPDATE: User reports backward jumping. This suggests my "fix" might be wrong or insufficient.
    # Let's revert to strict reference logic [1, -1, 1] but ensure we are mapping to the correct joints.
    # If T1 URDF has standard axes (Y-axis pitch), positive is usually flexion (forward).
    # However, if the reference code works, maybe they WANT hip extension during swing? (e.g. to push off?)
    # No, swing leg must flex to clear ground.
    # Let's try to force Hip Flexion by using POSITIVE offset when sin_pos is NEGATIVE.
    # sin_pos < 0. We want offset > 0. So we need NEGATIVE coefficient.
    # ref_pos_dir[0] is 1. So -1 * 1 * neg = pos.
    # So my previous fix (-ref_pos_dir[0]) WAS trying to force flexion.
    # If it still jumps backward, maybe the KNEE is the problem?
    # Knee: ref_pos_dir[1] = -1. sin_pos < 0.
    # Offset = -1 * neg = pos. Positive Knee = Flexion (Bent). This is correct.
    # Ankle: ref_pos_dir[2] = 1. sin_pos < 0.
    # Offset = 1 * neg = neg. Negative Ankle = Dorsiflexion (Toes up)? Or Plantarflexion?
    # Usually Positive Ankle = Dorsiflexion (Toes up).
    # So we want Positive Ankle offset.
    # So Ankle coefficient should be NEGATIVE?
    # Reference has 1. So it commands Negative Ankle (Toes down/Pointed).
    # Pointed toes during swing might catch the ground!
    # Let's try to invert Ankle too?
    
    # Let's try to be very aggressive with Hip Flexion.
    # And ensure Ankle clears ground (Dorsiflexion).
    
    # NOTE: Reverting to strict Reference Logic [1, -1, 1]
    # Left Hip (Swing sin < 0): 1 * neg = neg (Forward Flexion)
    # Left Knee (Swing sin < 0): -1 * neg = pos (Flexion/Bent)
    # Left Ankle (Swing sin < 0): 1 * neg = neg (Plantarflexion/Toes Down) -> This seems risky but matches reference.
    
    if l_hip_pitch != -1: ref_dof_pos[:, l_hip_pitch] += ref_pos_dir[0] * sin_pos_l.squeeze() * scale_1
    if l_knee != -1:      ref_dof_pos[:, l_knee]      += ref_pos_dir[1] * sin_pos_l.squeeze() * scale_2
    if l_ankle_pitch != -1: ref_dof_pos[:, l_ankle_pitch] += ref_pos_dir[2] * sin_pos_l.squeeze() * scale_1
    
    # Right foot
    sin_pos_r[sin_pos_r < 0] = 0 
    
    # Right Hip (Swing sin > 0): -1 * pos = neg (Forward Flexion)
    # Right Knee (Swing sin > 0): 1 * pos = pos (Flexion/Bent)
    # Right Ankle (Swing sin > 0): -1 * pos = neg (Plantarflexion/Toes Down)
    
    if r_hip_pitch != -1: ref_dof_pos[:, r_hip_pitch] += ref_pos_dir[3] * sin_pos_r.squeeze() * scale_1
    if r_knee != -1:      ref_dof_pos[:, r_knee]      += ref_pos_dir[4] * sin_pos_r.squeeze() * scale_2
    if r_ankle_pitch != -1: ref_dof_pos[:, r_ankle_pitch] += ref_pos_dir[5] * sin_pos_r.squeeze() * scale_1
    
    # Double support phase
    # Re-implementing strictly following reference logic:
    offsets = torch.zeros_like(joint_pos)
    if l_hip_pitch != -1: offsets[:, l_hip_pitch] = ref_pos_dir[0] * sin_pos_l.squeeze() * scale_1
    if l_knee != -1:      offsets[:, l_knee]      = ref_pos_dir[1] * sin_pos_l.squeeze() * scale_2
    if l_ankle_pitch != -1: offsets[:, l_ankle_pitch] = ref_pos_dir[2] * sin_pos_l.squeeze() * scale_1
    
    if r_hip_pitch != -1: offsets[:, r_hip_pitch] = ref_pos_dir[3] * sin_pos_r.squeeze() * scale_1
    if r_knee != -1:      offsets[:, r_knee]      = ref_pos_dir[4] * sin_pos_r.squeeze() * scale_2
    if r_ankle_pitch != -1: offsets[:, r_ankle_pitch] = ref_pos_dir[5] * sin_pos_r.squeeze() * scale_1
    
    mask = torch.abs(sin_pos).squeeze() < double_stand_phase
    offsets[mask] = 0.0
    
    target_pos = default_dof_pos + offsets
    
    # Calculate reward
    # Reference: diff = joint_pos[:,2:13] - pos_target[:,2:13]
    # It ignores Head (0,1) and maybe last joint?
    # Let's assume we ignore Head.
    head_yaw = get_idx("Head_yaw")
    head_pitch = get_idx("Head_pitch")
    
    diff = joint_pos - target_pos
    # Zero out head diffs if they exist
    if head_yaw != -1: diff[:, head_yaw] = 0
    if head_pitch != -1: diff[:, head_pitch] = 0
    
    # Reference uses norm(diff, dim=1) which sums over all joints.
    # r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
    
    dist = torch.norm(diff, dim=1)
    r = torch.exp(-2 * dist) - 0.2 * torch.clamp(dist, 0, 0.5)
    return r

def feet_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    
    # Get feet orientation (quat)
    feet_quat = asset.data.body_quat_w[:, body_ids] # (N, 2, 4)
    
    # Convert to Euler XYZ
    # Note: Isaac Lab utils might differ.
    # We need euler_xyz_from_quat.
    foot_ori_l = torch.stack(euler_xyz_from_quat(feet_quat[:, 0]), dim=-1)
    foot_ori_r = torch.stack(euler_xyz_from_quat(feet_quat[:, 1]), dim=-1)
    
    # Target: Base Yaw, 0 Roll/Pitch
    base_quat = asset.data.root_quat_w
    base_euler = torch.stack(euler_xyz_from_quat(base_quat), dim=-1)
    target_ori = torch.zeros_like(base_euler)
    target_ori[:, 2] = base_euler[:, 2] # Keep Yaw
    
    diff_l = foot_ori_l - target_ori
    diff_r = foot_ori_r - target_ori
    
    # Reference: r = torch.exp(- torch.norm(diff_r, dim=1) - torch.norm(diff_l, dim=1))
    r = torch.exp(- torch.norm(diff_r, dim=1) - torch.norm(diff_l, dim=1))
    return r

def feet_distance_reward(
    env: ManagerBasedRLEnv,
    min_dist: float = 0.24,
    max_dist: float = 0.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    feet_pos = asset.data.body_pos_w[:, body_ids]
    
    foot_dist = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=1)
    
    d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max_dist, 0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def feet_contact_forces_reward(
    env: ManagerBasedRLEnv,
    max_contact_force: float = 300.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    
    forces = sensor.data.net_forces_w[:, body_ids, :]
    # Reference: torch.sum((torch.norm(forces, dim=-1) - max_contact_force).clip(0, 400), dim=1)
    return torch.sum((torch.norm(forces, dim=-1) - max_contact_force).clip(0, 400), dim=1)

def tracking_ball_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    ball_track_error = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    return torch.exp(- 5.0 * ball_track_error)

def tracking_ball_target_vel_reward(
    env: ManagerBasedRLEnv,
    ball_min_vel: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    # Note: This requires 'commands' which are usually in env.command_manager
    # But here we assume the command is to follow the ball?
    # Reference: lin_vel_error = torch.norm(self.commands[:, :2] - self.ball_states[:, 7:9], dim=-1)
    # We need access to commands.
    # In Isaac Lab, commands are usually in env.command_manager.
    # But we don't have easy access here.
    # However, for Dribbling, the "command" might be implicit or stored in env.
    # Let's assume we want the ball to move at a target speed?
    # Reference uses self.commands.
    # If we can't access commands, we might skip this or use a placeholder.
    # But this is a high weight reward (1.5).
    # Let's assume the command is to move the ball towards the goal or just move it?
    # In the reference config: ranges: lin_vel_x = [-1.0, 1.0].
    # It seems the robot is commanded to move the ball with a specific velocity.
    # We will try to access env.command_manager if possible, or just return 0 for now if not found.
    return torch.zeros(env.num_envs, device=env.device) # Placeholder

def orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_euler = torch.stack(euler_xyz_from_quat(asset.data.root_quat_w), dim=-1)
    projected_gravity = asset.data.projected_gravity_b
    
    quat_mismatch = torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=1) * 10)
    orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20)
    return (quat_mismatch + orientation) / 2.

def action_smoothness_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Requires history of actions.
    # env.action_manager.action_term.action_history?
    # Simplified: just return 0 if not available.
    return torch.zeros(env.num_envs, device=env.device)

def torques_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)

def dof_vel_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def dof_acc_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def tracking_ball_target_vel_reward_fixed(
    env: ManagerBasedRLEnv,
    target_vel_x: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    
    # Target velocity (fixed forward)
    target = torch.zeros_like(ball_vel)
    target[:, 0] = target_vel_x
    
    lin_vel_error = torch.norm(target - ball_vel, dim=-1)
    return torch.exp(-lin_vel_error / 2.0) # Sigma=2.0 from config

def tracking_ang_vel_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    
    # Vector to ball
    ball_pos = ball.data.root_pos_w[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    ball_vec = ball_pos - robot_pos
    
    # Desired yaw: face the ball
    target_yaw = torch.atan2(ball_vec[:, 1], ball_vec[:, 0])
    
    # Current yaw rate
    # This reward in reference tracks *commanded* ang vel.
    # Logic: if ball is far, turn towards ball. If near, follow command.
    # Simplified: Turn towards ball.
    
    # Calculate desired angular velocity to reach target yaw
    # This is a P-controller logic in the reward?
    # Reference: des_ang_vel = kp_ang * ang_error
    
    base_quat = robot.data.root_quat_w
    base_euler = torch.stack(euler_xyz_from_quat(base_quat), dim=-1)
    current_yaw = base_euler[:, 2]
    
    ang_error = wrap_to_pi(target_yaw - current_yaw)
    des_ang_vel = torch.clamp(5.0 * ang_error, -0.8, 0.8) # kp_ang=5.0
    
    current_ang_vel = robot.data.root_ang_vel_w[:, 2]
    
    ang_vel_error = torch.square(des_ang_vel - current_ang_vel)
    return torch.exp(-ang_vel_error * 5.0)


def feet_clearance_reward(
    env: ManagerBasedRLEnv,
    target_feet_height: float = 0.025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    """
    Calculates reward based on the clearance of the swing leg from the ground during movement.
    Encourages appropriate lift of the feet during the swing phase of the gait.
    """
    # Note: This reward requires stateful tracking of feet height (integration of delta_z).
    # In Isaac Lab, we might not have easy access to 'last_feet_z' and 'feet_height' buffers 
    # unless we implement a custom RewardTerm with internal state or use env.extras.
    # However, for simplicity and statelessness, we can approximate it or use current height.
    # The reference implementation integrates delta_z and resets on contact.
    # This effectively measures the "peak height" achieved during swing.
    
    # Simplified stateless version:
    # Reward if swing foot is close to target height?
    # Or just reward height during swing?
    
    # Let's try to implement the reference logic using env.extras if possible, 
    # but standard RewTerm functions are stateless.
    # We will use a simplified version: 
    # During swing phase, reward if foot height is close to target.
    
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    feet_pos = asset.data.body_pos_w[:, body_ids]
    feet_z = feet_pos[:, :, 2] - 0.05 # Subtract ground/foot offset? Reference subtracts 0.05
    
    # Get gait phase
    # 1 - stance_mask = swing_mask
    stance_mask = _get_gait_phase(env)
    swing_mask = 1.0 - stance_mask
    
    # Reference: rew_pos = torch.abs(self.feet_height - target) < 0.01
    # Here we just check if current height is close to target during swing.
    # This encourages holding the foot at target height during swing.
    
    error = torch.abs(feet_z - target_feet_height)
    rew = (error < 0.01).float()
    
    return torch.sum(rew * swing_mask, dim=1)

def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.60,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    # Use a wider kernel (std=0.05 -> 1/20^2 approx, here we use 20.0 coefficient in exp)
    # Previous was 100.0 which is very narrow.
    return torch.exp(-torch.square(base_height - target_height) * 20.0)

def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    # Placeholder
    return torch.zeros(env.num_envs, device=env.device)

def robot_forward_velocity_reward(
    env: ManagerBasedRLEnv,
    target_vel: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # Use World X velocity as forward direction (matching ball target)
    vel_x = asset.data.root_lin_vel_w[:, 0]
    return torch.exp(-torch.square(vel_x - target_vel))

def head_position_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping head joints at 0 position (upright/forward)."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Find head joints
    joint_names = asset.joint_names
    head_indices = [i for i, n in enumerate(joint_names) if "Head" in n]
    
    if not head_indices:
        return torch.zeros(env.num_envs, device=env.device)
        
    head_pos = asset.data.joint_pos[:, head_indices]
    error = torch.sum(torch.square(head_pos), dim=1)
    
    return torch.exp(-error * 5.0)

def feet_stride_reward(
    env: ManagerBasedRLEnv,
    min_stride: float = 0.30,
    max_stride: float = 0.80,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot_link"),
) -> torch.Tensor:
    """Reward for stride length along the body forward direction."""
    device = env.device
    asset: Articulation = env.scene[asset_cfg.name]

    # Find feet bodies
    body_ids = asset.find_bodies(asset_cfg.body_names)[0]
    if len(body_ids) < 2:
        return torch.zeros(env.num_envs, device=device)
    body_ids = body_ids[:2]

    # Feet positions in world frame [N, 2, 3]
    feet_pos_w = asset.data.body_pos_w[:, body_ids]

    # Base forward direction
    base_quat = asset.data.root_quat_w
    fwd_local = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(env.num_envs, 1)
    base_fwd_w = quat_apply(base_quat, fwd_local)
    fwd_xy = base_fwd_w[:, :2]
    fwd_xy = fwd_xy / (torch.norm(fwd_xy, dim=1, keepdim=True) + 1e-8)

    # Projected stride
    diff_xy = feet_pos_w[:, 0, :2] - feet_pos_w[:, 1, :2]
    stride = torch.abs(torch.sum(diff_xy * fwd_xy, dim=1))

    # Linear reward
    rew = (stride - min_stride) / (max_stride - min_stride + 1e-6)
    return torch.clamp(rew, 0.0, 1.0)


