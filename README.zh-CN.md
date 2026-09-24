# ProgDraft: Acoustic Progress Propagation for Speculative ASR

论文：**Acoustic Progress Propagation for Long-Horizon Speculative Decoding in ASR**

作者：**Yuanyuan Jia, Qianqian Yang**

单位：**Zhejiang University**

仓库：https://github.com/yuanyuanjia71-spec/ProgDraft

论文链接与许可证：**TBD**，确定后补充；当前没有正式 `LICENSE`。

该目录整理的是已经完成的 Ours：Joint Progress-Aware + Random-K[3,8]，包含 0.6B 和 1.7B 两套配置。原实验目录与权重保持不变。

## 目录内容

- `src/progress_asr/`：drafter、共享声学位移预测器、Random-K 训练目标、训练入口、真实缓存投机推理、端到端测速及文本评分。
- `configs/`：两个模型规模的实际配置，包括 epoch 末尾 batch 的已有差异。
- `scripts/export_research_assets.py`：将原始可信研究资产导出为独立缓存和 safetensors 权重。
- `manifests/`：固定训练、验证和测试样本 ID、数据版本与校验信息；不包含音频和转录。
- `results/`：Ours 已有 Final Test K=8 分数据集结果和数据来源。
- `tests/`：词元错位、KV 递归、loss reduction、CE 到 predictor 的梯度、首拒和 EOS 等契约测试。
- `docs/`：实际实现、复现步骤、数据与权重状态、验证记录。

## 使用方式

安装适配当前 CUDA 的 PyTorch 后执行：

```bash
python -m pip install -e '.[asr]'
python -m unittest discover -s tests -v
```

训练、推理和测速命令见 [英文 README](README.md)。所有公开脚本从参数读取数据、输出和权重路径，不依赖作者原实验目录；只有显式指定 `--source-root` 的资产转换工具读取历史目录布局。

训练使用 target greedy trajectory 的 teacher-forced token；声学位置、隐藏状态与 self-KV 在模型内递归。推理使用真实预测 token 和实时 target cache，L21 初始化来自当前已提交前缀，不按固定 K 索引旧缓存。保留独立 correction/bonus forward，全部计入端到端耗时。

## 发布状态

代码包已整理并进行单元测试与小规模原实现等价检查。没有重新训练，也没有将公开包的小规模检查冒充完整 Final Test 测速。

现有最终权重和 step-0 初始化以不含 target 权重的 safetensors 格式暂存在本地、被 Git 忽略的 `artifacts/`。正式权重下载地址、完整训练缓存下载地址、许可证和论文链接仍待补充。原始数据需按各数据集的官方条款获取。

具体检查范围和限制见 [validation.md](docs/validation.md)。
