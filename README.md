# BackupDribbling

用于跨主机复现 `Loco-G1-Dribbling` 训练运行环境（不包含视频）。

## 前置条件

目标主机已具备：

- IsaacLab 可运行环境
- `/root/soccerLab` 工程目录（与当前脚本路径一致）

## 最短上手（新主机）

```bash
git clone https://github.com/Ethan-Li-k/BackupDribbling.git /root/BackupDribbling
bash /root/BackupDribbling/scripts/restore_on_target.sh --bundle-dir /root/BackupDribbling --workspace-root /root --run-id 2025-12-11_18-01-51 --checkpoint model_1300.pt --use-runtime-exceptions true
python /root/soccerLab/scripts/rsl_rl/base/train.py --task Loco-G1-Dribbling --headless --resume True --load_run 2025-12-11_18-01-51 --checkpoint model_1300.pt
```

## 说明

- `restore_on_target.sh` 会把本仓库中的 `soccerTask` 与定制 `train.py` 同步到目标路径。
- 默认使用 `files_runtime_exceptions`（推荐，保证可运行）。
- 若你只想恢复严格时间截断文件，可把 `--use-runtime-exceptions` 改为 `false`（可能无法完整运行）。

## 相关文档

- 迁移说明（中文）：`README_MIGRATION_CN.md`
