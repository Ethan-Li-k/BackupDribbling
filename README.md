# BackupDribbling (master)

本仓库的 `master` 分支按 `soccerLab` 原生目录组织（`scripts/`、`source/`、`logs/`），用于跨主机复现 Dribbling 训练。

## 必须先满足

在目标主机上，必须先安装并验证 **IsaacLab** 可用（含对应 Python 环境）。

> 未安装好 IsaacLab 时，本仓库内容不能直接运行。

## 目录说明

- `scripts/`：训练入口与扩展脚本（含 `scripts/rsl_rl/base/train.py`）
- `source/soccerTask/`：任务代码（含 dribbling）
- `source/third_party/beyondAMP/`：依赖的第三方训练模块
- `logs/rsl_rl/dribbling_g1/2025-12-11_18-01-51/`：resume 所需快照

## 新主机最短步骤

```bash
git clone https://github.com/Ethan-Li-k/BackupDribbling.git /root/BackupDribbling
rsync -a /root/BackupDribbling/scripts/ /root/soccerLab/scripts/
rsync -a /root/BackupDribbling/source/ /root/soccerLab/source/
rsync -a /root/BackupDribbling/logs/ /root/logs/
python -m pip install -e /root/soccerLab/source/soccerTask
python /root/soccerLab/scripts/rsl_rl/base/train.py --task Loco-G1-Dribbling --headless --resume True --load_run 2025-12-11_18-01-51 --checkpoint model_1300.pt
```

## 分支约定

- `backup`：保留原先 master 的旧目录版本
- `master`：当前按 `soccerLab` 目录组织的新版本
