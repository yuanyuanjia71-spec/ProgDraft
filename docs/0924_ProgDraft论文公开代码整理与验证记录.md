# ProgDraft 论文公开代码整理与验证记录

- 仓库名：ProgDraft；仓库地址 `github.com/yuanyuanjia71-spec/ProgDraft`。
- README 标题：ProgDraft: Acoustic Progress Propagation for Speculative ASR。
- 论文标题：Acoustic Progress Propagation for Long-Horizon Speculative Decoding in ASR。
- 作者：Yuanyuan Jia, Qianqian Yang；单位：Zhejiang University。
- 许可证与论文链接：TBD；未添加正式 LICENSE，未填写虚构出版信息。

## 整理内容

独立包保留 Ours：Joint Progress-Aware + Random-K[3,8] 的 drafter、共享 A2 声学位移预测器、可变深度训练损失、训练入口、真实 target KV 缓存推理和端到端基准测试。加入两个规模的固定配置、数据标识与校验、历史 Final Test K=8 表格、初始化/最终权重转换工具、复现文档和自动测试。

原实验代码与训练权重没有修改，没有启动新训练。训练目标沿用实际 Random-K 代码，包括 0.7^(k−1) depth weights、经验展开概率修正和 source-batch FA 有效数归一化。保留两个模型规模在 epoch 尾部 batch 上的差异。音频时间网格如实标为按时长均分的中点近似。

## 验证结果

- 包安装成功，四个命令入口的帮助信息可用，Python 编译检查通过。
- 6 个契约测试通过：teacher forcing、位置递归、self-KV、loss reduction、仅 CE 梯度、FA 映射、EOS 和验证分支等。
- 两个规模在真实训练缓存、真实 target embedding/LM head 上，对原始实现的逐步 state、position、loss 与 observation 逐值相等。
- 仅 token CE 在原 step-0 权重下对 predictor 的 grad norm：0.6B 约 0.09826；1.7B 约 0.54441。
- 两个规模各选五个数据集的一条真实音频，K=8 共 10/10 条与原实现及 target-only 的 token IDs 一致；轮数与 accepted-draft count 一致。
- 已有论文表格结果仅以历史数据形式保留，未用此次小规模检查替代完整 Final Test。

详细数值与检查范围见 `source_equivalence_audit.json`、`runtime_equivalence_audit.json` 和 `validation.md`。后者同时记录：1.7B 最终权重在小型梯度 fixture 上的局部 CE 梯度为零；初始化下的梯度连通性检查通过。没有据此推断全数据集行为。

## 外部资产与待填写项

公开 Git 内容不包含原始音频、转录、本机绝对路径、上游 target 权重或训练 optimizer 状态。两个规模的最终/初始 safetensors 权重共约 727 MB，暂存本地忽略目录，不进入源码提交。

正式许可证、论文链接、权重/完整缓存公开下载地址仍待填写。源码发布到上述 GitHub 仓库；正式权重和完整训练缓存不随源码提交。
