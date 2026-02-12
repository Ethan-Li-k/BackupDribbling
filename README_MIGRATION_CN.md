# Dribbling 迁移仓库（跨机复现）

该仓库分两部分：

- `files_strict/`：满足时间约束（<= 2025-12-11 18:39:30）的文件
- `files_runtime_exceptions/`：超时但在当前工程中通常“运行必需”的文件（如任务入口配置、T1.urdf）

## 强烈建议（避免渲染问题）

- 全程 `--headless`
- **不要**传 `--video`
- **不要**启用 camera/recording

## 新机快速落地（仅有 IsaacLab 环境）

1. 将 `files_strict/source/soccerTask` 拷贝到新机：`/root/soccerLab/source/soccerTask`
2. 若你允许运行例外文件，再将 `files_runtime_exceptions/source/soccerTask` 合并覆盖到 `/root/soccerLab/source/soccerTask`
3. 日志快照拷贝到新机：`/root/logs/rsl_rl/dribbling_g1/2025-12-11_18-01-51`
4. 若目标机缺 T1 资产，把 `files_runtime_exceptions/data/assets/assetslib/T1/T1.urdf` 放到 `/root/data/assets/assetslib/T1/T1.urdf`
5. 安装任务包：`pip install -e /root/soccerLab/source/soccerTask`
6. 启动（无视频）：
   - `python /root/soccerLab/scripts/rsl_rl/base/train.py --task Loco-G1-Dribbling --headless --resume True --load_run 2025-12-11_18-01-51 --checkpoint model_1300.pt`

## 注意

`files_strict` 在“严格时间约束”下可能不足以独立启动任务（取决于你是否允许使用运行例外文件）。

详见：`manifests/manifest_timecheck.txt`
