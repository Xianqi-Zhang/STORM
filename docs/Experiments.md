# STORM Baseline Kickoff (InterAct + InterMimic)

## 0) 目录约定
- 根目录: `/media/zhangxq/Data/STORM`
- baseline 代码: `/media/zhangxq/Data/STORM/repos`
- 数据集: `/media/zhangxq/Data/STORM/DATASETS`
- 你的主工程: `/media/zhangxq/Data/STORM/storm`

## 1) 建议先做的三件事
1. 先跑数据就绪检查，明确缺口（数据、模型、软链接）。
2. 先打通 InterAct 的数据处理最小链路（至少 OMOMO）。
3. 再接 InterMimic 做 replay/inference，确认可执行性链路。

## 2) 一键检查与链接
在 `storm` 目录执行：

```bash
cd /media/zhangxq/Data/STORM/storm
./scripts/baseline_bootstrap.sh check
```

如果检查结果提示 OMOMO 未解压：

```bash
tar -xzf /media/zhangxq/Data/STORM/DATASETS/OMOMO/data.tar.gz -C /media/zhangxq/Data/STORM/DATASETS/OMOMO
unzip -o /media/zhangxq/Data/STORM/DATASETS/OMOMO/omomo_text_anno.zip -d /media/zhangxq/Data/STORM/DATASETS/OMOMO
```

然后建立 InterAct/InterMimic 需要的软链接：

```bash
./scripts/baseline_bootstrap.sh link
```

## 3) 环境策略（必须分开）
- `interact` 环境: 跑 InterAct 数据处理/训练。
- `intermimic-gym` 环境: 跑 InterMimic + Isaac Gym。
- `storm` 环境: 你自己的方法实现与评测。

不要把三个环境混在一起。

## 4) InterAct 启动顺序（最小可跑）

```bash
cd /media/zhangxq/Data/STORM/repos/InterAct

# 先处理现有数据（你当前最现实的是 OMOMO，InterCap 有数据后再加）
python process/process_omomo.py
# 如 InterCap 原始数据已放好，再执行
python process/process_intercap.py
```

处理完成后，先做一次可视化抽查：

```bash
python visualization/visualize.py omomo
# 有 intercap 再加
python visualization/visualize.py intercap
```

## 5) 连接到 InterMimic
InterAct 转 InterMimic 的脚本：

```bash
cd /media/zhangxq/Data/STORM/repos/InterAct/simulation
python interact2mimic.py --dataset_name omomo
# 有 intercap 数据后再跑
python interact2mimic.py --dataset_name intercap
```

转换后，`baseline_bootstrap.sh link` 会把 OMOMO 转换结果链接到 InterMimic 默认读取目录 `InterAct/OMOMO_new`。

## 6) InterMimic 冒烟测试

```bash
cd /media/zhangxq/Data/STORM/repos/InterMimic
sh isaacgym/scripts/data_replay.sh
```

如果你用 IsaacLab：

```bash
./isaaclab/scripts/run_data_replay.sh --num-envs 8 --motion-dir InterAct/OMOMO_new
```

## 7) 里程碑定义（建议）
- M1: `process_omomo.py` 跑通 + 可视化正常。
- M2: `interact2mimic.py --dataset_name omomo` 产出 `.pt`。
- M3: InterMimic `data_replay` 正常播放。
- M4: 开始在 `storm/src` 写统一评测与对比脚本。

## 8) 当前已知阻塞项
- 你本地 `InterCap` 目录目前未检测到原始 `res.pkl` 序列。
- InterAct 需要 SMPL/SMPLH/SMPLX 模型文件放在 `repos/InterAct/models`。
- InterMimic 依赖 Isaac Gym/IsaacLab，环境配置成本高，建议等 M2 后再投入训练。
