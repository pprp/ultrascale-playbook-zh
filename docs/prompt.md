# 超规模AI实战手册

### 关于本书

《超规模AI实战手册》是一本全面介绍超规模AI技术的实用指南，涵盖了从基础理论到实际应用的各个方面。无论您是AI领域的初学者还是有经验的专业人士，本书都能为您提供宝贵的见解和实用技巧。英文原版在 https://huggingface.co/spaces/nanotron/ultrascale-playbook 翻译源码在： https://github.com/pprp/ultrascale-playbook-zh 欢迎提交PR进行勘误。
### 目录概览

# 目录

## 高级概览与并行化技术

### 并行化术语
- Batch Size术语
- 内存术语
- 实用公式

### 实用步骤
- 单个GPU训练
- 数据并行（Data Parallelism）
- 张量并行（Tensor Parallel）
- Transformer块中的张量并行

### 参考文献
- Sequence Parallel
- Context Parallel
- Ring Attention
- Zig-zag Ring Attention
- Pipeline Parallel
- AFAB
- Zero Bubble & Dual Pipe
- Expert Parallel
- 5D Parallelism

## 寻找最佳训练配置
- 步骤1：将模型放入内存中
- 步骤2：实现目标全局批处理大小
- 步骤3：优化训练吞吐量
- 成千上万个配置的基准测试
- 基准测试中的经验教训

## GPU深度挖掘与性能优化

### GPU入门与性能提升
- 如何用kernel提升性能
- 内存合并
- 分块处理（Tiling）
- 线程粗化（Thread Coarsening）
- 最小化控制分歧
- 融合内核（Fused Kernels）

### 混合精度训练
- Flash Attention 1-3
- 混合精度训练（Mixed Precision Training）
- FP16和BF16训练
- FP8预训练

### 并行编程速成
- 归约 & 全局归约（Reduce & AllReduce）
- Gather & AllGather
- Scatter & ReduceScatter
- 快速关注Ring AllReduce
- Barrier 屏障
- NCCL：NVIDIA 集体通信库

### 分布式训练性能分析
- 内核
- CPP扩展
- 计算LLM训练中的规模
- 计算/通信重叠需要的数学
- 数据并行DP通信分析
- Zero-3（FSDP）通信分析
- TP通信分析

### 获取方式

您可以通过以下方式获取本书的PDF版本：

1. 关注微信公众号 **GiantPandaLLM**
2. 在公众号后台发送关键词 **"Ultrascale"**
3. 系统将自动回复PDF下载链接


*注：最后更新于2025年3月12日