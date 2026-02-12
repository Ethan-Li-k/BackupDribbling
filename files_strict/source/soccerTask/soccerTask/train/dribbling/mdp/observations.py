from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ball_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """The position of the ball relative to the robot's root frame.
    
    Returns:
        The relative position (x, y, z) in the robot's root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    ball_pos_w = ball.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # Calculate relative position in world frame
    rel_pos_w = ball_pos_w - robot_pos_w
    
    # Rotate to robot base frame
    rel_pos_b = quat_apply_inverse(robot_quat_w, rel_pos_w)
    
    return rel_pos_b

def ball_velocity_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """The linear velocity of the ball relative to the robot's root frame.
    
    Returns:
        The relative linear velocity (vx, vy, vz) in the robot's root frame.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    ball_vel_w = ball.data.root_lin_vel_w
    robot_vel_w = robot.data.root_lin_vel_w
    robot_quat_w = robot.data.root_quat_w

    # Calculate relative velocity in world frame
    rel_vel_w = ball_vel_w - robot_vel_w

    # Rotate to robot base frame
    rel_vel_b = quat_apply_inverse(robot_quat_w, rel_vel_w)

    return rel_vel_b
