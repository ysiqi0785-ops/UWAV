# UWAV: Uncertainty-weighted Weakly-supervised Audio-Visual Video Parsing - 主实验复现分析报告

## 0. 主实验复现结论总览

| Experiment ID | 场景/数据集 | 任务 | 论文主指标与数值 | 代码入口 | 复现难度 | 可复现性判断 | 主要风险点 |
|---------------|-------------|------|------------------|----------|----------|--------------|------------|
| E1 | LLP数据集 | 弱监督音视频视频解析 | Segment-level Type@AV: 65.9, Event-level Type@AV: 60.9 (Table 1) | scripts/train_transformer.sh → scripts/pseudo_label_generation.sh → scripts/train_han.sh → scripts/test_han.sh | 中 | 部分可复现 | 数据获取复杂、三阶段训练流程、预训练依赖 |
| E2 | AVE数据集 | 音视频事件识别 | Accuracy: 80.6% (Table 2) | 相同流程但配置不同 | 中 | 部分可复现 | 特征提取器配置差异、数据集规模小 |

## 1. 论文概述

### 1.1 标题
UWAV: Uncertainty-weighted Weakly-supervised Audio-Visual Video Parsing [Paper: 标题]

### 1.2 方法一句话总结
输入是带有视频级标签的音视频片段，输出是每个时间段的音频、视觉、音视频事件预测，核心机制是通过预训练的时序感知Transformer生成不确定性加权的伪标签，并结合特征混合和类别平衡损失进行弱监督训练。

### 1.3 核心贡献
1. 提出时序感知的伪标签生成模块，考虑段间依赖关系生成更准确的伪标签 [Paper: 摘要、引言]
2. 引入不确定性加权机制，在训练中考虑伪标签的置信度 [Paper: 摘要]
3. 设计特征混合正则化策略，通过混合段特征和插值伪标签减少过拟合 [Paper: 摘要]
4. 采用类别平衡损失重加权解决负样本偏置问题 [Paper: 方法章节]
5. 在LLP和AVE数据集上达到最先进性能 [Paper: Table 1, Table 2]

## 2. 主实验复现详解

### 【E1 主实验标题：LLP数据集弱监督音视频视频解析】

#### A. 这个主实验在回答什么问题
- 实验目的：验证UWAV方法在标准AVVP基准数据集上的有效性，证明其相比现有方法的优越性
- 核心结论对应点：UWAV在几乎所有指标上超越先前方法，特别是在视觉事件检测上有显著提升
- 论文证据位置：[Paper: Table 1主结果表、5.2节Results段落]

#### B. 实验任务与工作原理
- 任务定义：输入10秒视频片段和视频级事件标签，输出每1秒段的音频(A)、视觉(V)、音视频(AV)事件预测
- 方法关键流程：视频分段 → 特征提取 → UnAV预训练 → LLP伪标签生成 → HAN模型训练 → 评测
- 最终设置：完整UWAV模型，包含不确定性加权、类别平衡重加权、特征混合的三阶段训练策略
- 实例说明：对于包含"割草机"和"说话"事件的10秒视频，模型需要准确定位割草机在视觉和听觉模态中的时间段

#### C. 数据
- 数据集名称与来源：Look, Listen, and Parse (LLP) dataset，来自YouTube视频 [Paper: 5.1节Datasets] [Repo: README.md数据集设置部分]
- 数据许可/访问限制：【未知】论文和代码未明确说明许可限制
- 数据结构示例：
  ```
  视频: 10秒，分为10个1秒段，每段8帧
  音频特征: VGGish (B, T, 128)
  视觉特征: ResNet-152 (B, 8*T, 2048), R(2+1)D (B, T, 512)
  CLIP/CLAP特征: (B, T, 768)/(B, T, 512)
  标签: 视频级25类事件标签，段级评测标签
  ```
- 数据量：训练集10,000视频，验证集649视频，测试集1,200视频 [Paper: 5.1节]
- 训练集构建：仅使用视频级弱标签，无段级标签 [Paper: 任务定义]
- 测试集构建：包含段级ground truth用于评测 [Paper: 5.1节]
- 预处理与缓存：需要下载官方预提取特征和CLIP/CLAP特征 [Repo: README.md数据设置]

#### D. 模型与依赖
- 基础模型/Backbone：VGGish(音频)、ResNet-152(2D视觉)、R(2+1)D(3D视觉)、CLIP、CLAP [Repo: scripts/train_transformer.sh参数]
- 关键模块：
  - TemporalTransformer: 5层编码器，1024隐藏维度，16注意力头 [Repo: scripts/train_transformer.sh]
  - HAN: 5层HAN，512隐藏维度，1层MMIL [Repo: scripts/train_han.sh]
- 训练策略：AdamW优化器，学习率1e-4，余弦退火调度，梯度裁剪，批大小64 [Repo: 训练脚本]
- 随机性控制：seed=1000(预训练)，seed=20000(HAN训练) [Repo: scripts/train_transformer.sh, scripts/train_han.sh]

#### E. 评价指标与论文主表预期结果
- 指标定义：
  - 段级：每段预测与真值比较的宏F1分数
  - 事件级：连续正段合并为事件，mIoU>0.5的F1分数
  - Type@AV：A、V、AV三类F1均值
  - Event@AV：所有事件不分模态的F1分数
- 论文主结果数值：
  - Segment-level: A=64.2, V=70.0, AV=63.4, Type=65.9, Event=63.9
  - Event-level: A=58.6, V=66.7, AV=57.5, Type=60.9, Event=57.4
  [Paper: Table 1]
- 复现预期：以论文主表数值为准，允许±1%的合理波动

#### F. 环境与硬件需求
- 软件环境：Python 3.10.12, PyTorch 1.12.1, CUDA 11.3, transformers 4.30.2 [Repo: environment.yaml]
- 硬件要求：NVIDIA RTX 3090 GPU, 32GB RAM [Paper: 附录9节] [Repo: README.md]
- 训练时长：【未知】论文未明确说明总训练时间

#### G. 可直接照做的主实验复现步骤

1. **获取代码与安装依赖**
   ```bash
   git clone <UWAV_REPO_URL>
   cd UWAV
   conda env create -f environment.yaml
   conda activate uwav
   ```

2. **获取数据与放置路径**
   ```bash
   # 下载LLP数据集注释文件到data/LLP/
   # 下载官方预提取特征到data/LLP/feats/
   # 下载CLIP/CLAP特征到data/LLP/feats_CLIP/和data/LLP/feats_CLAP/
   # 下载UnAV数据集到data/UnAV/
   ```
   [Repo: README.md数据设置部分详细说明文件结构]

3. **预训练时序感知Transformer**
   ```bash
   bash ./scripts/train_transformer.sh
   ```
   - 关键参数：UnAV数据集，80轮训练，批大小64
   - 预期生成：temp_train_logs/TemporalTransformer_UnAV_*/checkpoints/

4. **生成伪标签和阈值**
   ```bash
   # 修改scripts/pseudo_label_generation.sh中的PRETRAINED_TRANSFORMER_DIR路径
   bash ./scripts/pseudo_label_generation.sh
   ```
   - 预期生成：段级伪标签、logits、类别阈值文件

5. **训练HAN模型**
   ```bash
   # 修改scripts/train_han.sh中的PRETRAINED_TRANSFORMER_DIR路径
   bash ./scripts/train_han.sh
   ```
   - 关键参数：不确定性加权、重加权、混合损失，80轮训练
   - 预期生成：temp_train_logs/HAN_*/checkpoints/

6. **主实验评测**
   ```bash
   # 修改scripts/test_han.sh中的CHECKPOINT_DIR路径
   bash ./scripts/test_han.sh
   ```
   - 预期输出：test_logs/中的评测结果，包含所有指标的F1分数

#### H. 可复现性判断
- 结论：**部分可复现**
- 依据清单：
  - ✓ 代码完整，包含所有训练和评测脚本
  - ✓ 环境配置明确，依赖版本锁定
  - ✓ 模型架构和超参数详细说明
  - ⚠ 数据获取复杂，需要多个外部下载链接
  - ⚠ 三阶段训练流程，中间结果依赖性强
  - ⚠ 预训练模型路径需要手动配置
- 补救路径：
  - 确保所有数据下载链接可访问
  - 按顺序执行三个训练阶段
  - 仔细配置脚本中的路径参数

#### I. 主实验专属排错要点
- 路径配置：脚本中的PRETRAINED_TRANSFORMER_DIR和CHECKPOINT_DIR必须正确设置
- 数据完整性：确保所有特征文件按README要求的目录结构放置
- 内存管理：批大小64可能需要根据GPU显存调整
- 训练依赖：HAN训练必须在伪标签生成完成后进行
- 特征维度：确保音视频特征维度与脚本参数匹配

### 【E2 主实验标题：AVE数据集音视频事件识别】

#### A. 这个主实验在回答什么问题
- 实验目的：验证UWAV方法的泛化能力，在不同数据集上的有效性
- 核心结论对应点：UWAV在小规模数据集上也能超越现有方法
- 论文证据位置：[Paper: Table 2、5.2节AVE结果段落]

#### B. 实验任务与工作原理
- 任务定义：输入10秒视频片段，输出每1秒段的29类音视频事件预测（包括背景类）
- 方法关键流程：与E1相同的三阶段流程，但使用不同的特征提取器配置
- 最终设置：完整UWAV模型，针对AVE数据集优化的特征混合策略
- 实例说明：对于包含单一音视频事件的视频，模型需要准确识别事件类别和时间定位

#### C. 数据
- 数据集名称与来源：Audio Visual Event (AVE) dataset，来自YouTube [Paper: 5.1节]
- 数据许可/访问限制：【未知】
- 数据结构示例：
  ```
  视频: 10秒，分为10个1秒段
  音频特征: CLAP (B, T, 512)
  视觉特征: CLIP (B, T, 768), R(2+1)D (B, T, 512)
  标签: 29类事件（含背景类），每视频单一事件
  ```
- 数据量：训练集3,339视频，验证集402视频，测试集402视频 [Paper: 5.1节]
- 训练集构建：视频级标签，每视频单一音视频事件
- 测试集构建：段级ground truth用于评测
- 预处理与缓存：需要CLIP和CLAP特征提取

#### D. 模型与依赖
- 基础模型/Backbone：CLIP(视觉)、R(2+1)D(视觉)、CLAP(音频) [Paper: 实现细节]
- 关键模块：与E1相同的TemporalTransformer和HAN架构
- 训练策略：批大小16，其他参数与E1相同 [Paper: 实现细节]
- 随机性控制：【推断】使用相同的seed设置

#### E. 评价指标与论文主表预期结果
- 指标定义：准确率，段级预测与真值匹配的比例 [Paper: 5.1节Metrics]
- 论文主结果数值：Accuracy = 80.6% [Paper: Table 2]
- 复现预期：以论文主表数值为准

#### F. 环境与硬件需求
- 软件环境：与E1相同 [Repo: environment.yaml]
- 硬件要求：与E1相同，但数据集较小可能需要更少资源
- 训练时长：【未知】

#### G. 可直接照做的主实验复现步骤
1-6. **与E1相同的步骤，但需要：**
   - 使用AVE数据集配置
   - 调整特征提取器为CLIP/CLAP
   - 修改批大小为16
   - 使用ceiling操作的特征混合策略 [Paper: 实现细节]

#### H. 可复现性判断
- 结论：**部分可复现**
- 依据清单：与E1相似，但数据集规模较小，复现相对容易
- 补救路径：确保使用正确的特征提取器配置

#### I. 主实验专属排错要点
- 特征配置：确保使用CLIP/CLAP而非ResNet/VGGish特征
- 批大小：调整为16以适应数据集规模
- 混合策略：使用ceiling操作的特征混合变体

## 3. 主实验一致性检查

- **论文主表指标可复现性**：代码提供了完整的评测脚本，能够产出与论文Table 1和Table 2一致的指标格式
- **共享组件**：两个主实验共享相同的模型架构和训练流程，仅在特征提取器和数据集配置上有差异
- **最小复现路径**：建议先复现E1（LLP），因为它是主要基准，然后复现E2验证泛化能力

## 4. 未知项与我需要你补充的最小信息

1. **数据访问权限**：LLP和AVE数据集的具体下载权限和访问限制
   - 缺失后果：无法获取数据将导致完全无法复现
   
2. **预训练模型权重**：是否提供预训练好的TemporalTransformer权重
   - 缺失后果：需要完整的三阶段训练，增加复现时间和计算成本
   
3. **具体训练时长**：每个阶段的预期训练时间
   - 缺失后果：无法合理规划计算资源和时间安排