"""Dribbling task entrypoint.

The known-good dribbling configuration is maintained under
`soccerTask.train.dribbling.mdp.ref.dribbling_env_cfg`.
This module intentionally re-exports that snapshot to keep the task import
path stable while avoiding config drift.
"""

from soccerTask.train.dribbling.mdp.ref.dribbling_env_cfg import *  # noqa: F401,F403

