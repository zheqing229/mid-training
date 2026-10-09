# Nanochat Midtraining 实验报告

## **摘要**

实验结论如下：

1. OLMo 3 中训练数据不适合直接替代 ClimbMix 完成全程预训练，但适合作为后半程的能力强化数据。仅使用该数据从头训练时，CORE 为 0\.2159，明显低于 ClimbMix baseline 的 0\.2560；采用“前期 ClimbMix、后期 ClimbMix 40% \+ OLMo 3 中训练数据 60%”后，CORE 提升至 0\.2665。

2. 数据切换和学习率退火需要同步提前。在第 3500 步同时切换数据并开始退火的实验，CORE/CORE\-12 达到 0\.2665/0\.3918；若第 3500 步切换数据、但仍到第 4500 步才退火，结果降至 0\.2529/0\.3807。

3. 成分消融表明 Wiki\-to\-RCQA 和 Nemotron synthetic QA 是主要有效成分，Reddit\-to\-Flashcards 的贡献较弱，Geography 和 Science 在当前配方及训练预算下没有带来增益。由此将 OLMo 3 部分从 `Wiki:Reddit:Nemotron:Geography:Science = 35:35:20:5:5` 调整为 `40:25:35:0:0`。

4. 最终方案将总训练步数从 5568 缩短到 5000，减少 10\.20%。两次独立运行的 CORE 分别为 0\.2603 和 0\.2638，均超过 5568 步 baseline 的 0\.2560；CORE\-12 也均超过 baseline。训练时间同步减少约 10\.08%。

因此，OLMo 3 中训练数据带来的提升不仅能在原训练长度下观察到，也足以转化为约 10% 的训练步数和计算量节省。

## **1\. 实验范围**

本报告只覆盖引入 OLMo 3 中训练数据后的实验链，包括：

- OLMo 3 数据从头训练的负向对照；

- ClimbMix 与 OLMo 3 中训练数据的混合比例探索；

- Midtraining 开始时点和学习率退火时点探索；

- Midtraining 数据成分消融；

- 优化数据配方后的完整训练；

- 缩短至 5000 步后的复现实验。

## **2\. 公共实验设置与评价口径**

除特别说明外，实验使用相同的模型和训练设置：

|项目|设置|
|---|---|
|模型|nanochat d24|
|硬件|4 × NVIDIA H20|
|精度|FP8 训练，评测时关闭 FP8|
|序列长度|2048|
|全局 batch|1,048,576 tokens|
|MLP ratio|3|
|注意力短窗口|128，窗口模式 SSSL|
|学习率计划|Cosine warmdown|
|baseline 总步数|5568|
|中训练候选起点|第 3500 或 4500 步|
|主要指标|CORE、CORE\-12，越高越好|

CORE 是 22 个任务经随机基线校正后的平均分；CORE\-12 是其中 12 个对训练变化更敏感任务的平均分。

各混合数据目录拥有自己的 validation shard，因此不同配方之间的 `val/bpb` 不是同分布指标，不能用其绝对值判断配方优劣。报告以跨实验可比的 CORE 和 CORE\-12 为主。





原始 baseline 使用 ClimbMix 训练 5568 步，最终结果为：

|CORE|CORE\-12|训练时间|训练 FLOPs|
|---|---|---|---|
|0\.2560|0\.3818|27,898\.5 秒|2\.3169 × 10^19|

## **3\. Dolma 3 Dolmino Mix数据的使用方式**

数据集链接：https://huggingface\.co/datasets/allenai/dolma3\_dolmino\_mix\-100B\-1125

项目尝试仅使用Dolma 3 Dolmino Mix中训练数据从头训练作为对照实验。结果表明，这组数据具有明显的任务强化属性，但不足以覆盖通用预训练阶段所需的数据广度：

|实验|数据策略|CORE|CORE\-12|相对 baseline|
|---|---|---|---|---|
|baseline|ClimbMix，0→5568|0\.2560|0\.3818|—|
|**Dolma 3 Dolmino Mix** from scratch|Dolma 3 Dolmino Mix，0→5568|0\.2159<br>|0\.3167|CORE \-0\.0401|
|混合中训练|前期 ClimbMix；第 4500 步后使用 ClimbMix 40% \+ Dolma 3 Dolmino Mix 60%|0\.2617|0\.3887|CORE \+0\.0057|

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzUwNDI1MjEwZDJjMTE0ZTg1MzM4NTYyMjQzYWQ2NWRfMjczY2I3MDU0ZjEzMGM3NGI3M2VkNmYzYzk1OTQ3OGRfSUQ6NzY4NDEyMjg2NDc2MDQ0MTgyNV8xNzg5MTIyMjA1OjE3ODkyMDg2MDVfVjM)

从头训练降低了 0\.0401 CORE，而后程混合中训练提高了 0\.0057。这说明正确的策略不是用Dolma 3 Dolmino Mix 数据替换通用预训练，而是保留 ClimbMix 学到的通用语言能力，在训练后程引入问答、知识和推理导向的数据。

基础中训练配方中，顶层比例为 ClimbMix 40%、Dolma 3 Dolmino Mix 数据 60%；Dolma 3 Dolmino Mix部分内部为：

```Plain Text
Wiki-to-RCQA 35% + Reddit-to-Flashcards 35% +
Nemotron synthetic QA 20% + Geography 5% + Science 5%
```

数据构建器按文档进行确定性哈希采样和全局打散，在固定字符预算内避免简单拼接造成的局部数据偏置，并为不同实验保留相同的训练 horizon。

## **4\. Midtraining与退火时点探索**

这一组实验回答两个问题：Midtraining 数据应从何时引入，以及学习率退火是否需要与数据切换同步。

|实验|数据切换，开始Midtraining|退火开始|CORE|Δ CORE vs baseline|CORE\-12|Δ CORE\-12 vs baseline|
|---|---|---|---|---|---|---|
|原始 baseline|不切换|4500|0\.2560|—|0\.3818|—|
|早退火 control|不切换|3500|0\.2471|\-0\.0089|0\.3730|\-0\.0088|
|先退火后切换|4500|3500|0\.2617|\+0\.0057|0\.3887|\+0\.0068|
|同时提前|3500|3500|**0\.2665**|**\+0\.0105**|0\.3918|\+0\.0099|
|同时晚起|4500|4500|0\.2594|\+0\.0033|**0\.3937**|**\+0\.0118**|
|切换早于退火|3500|4500|0\.2529|\-0\.0031|0\.3807|\-0\.0011|

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Nzk5ZDU3N2U3N2QzNmRhMDViZDVhM2M5MzIyZjM0ZjBfODYwNDU0NWI3MjQ2NzgxNDZjMzA4MWQ2Zjk1NjZiZGVfSUQ6NzY4NDEyMzQ1OTg2NTEwMzMyOV8xNzg5MTIyMjA1OjE3ODkyMDg2MDVfVjM)

结果包含三个重要信息：

- 仅提前退火会使 baseline 下降，说明收益并非来自延长退火本身。

- 数据分布变化与学习率计划存在明显耦合：同时在3500步或者同时在4500步开始退火和Midtraining切换数据都能带来效果提升

- 同时晚启动的总体 CORE 较低，但 CORE\-12 略高，说明较短、较集中的中训练也能有效强化训练敏感任务；综合总体 CORE，仍选择第 3500 步同步切换和退火。

## **5\. 数据成分消融**

消融实验固定为第 4500 步切换数据，并从已经在第 3500 步开始退火的 checkpoint 继续训练。每次删除 OLMo 3 部分中的一个成分，再按原有比例将该成分的额度分配给其余成分。参照实验为完整的 `35:35:20:5:5` 配方。

Ps: 当时做实验的时候上面关于时间点的结果还没跑出来，所以没有使用3500同步切换和退火

|实验|删除成分|CORE|Δ CORE vs 完整配方|CORE\-12|Δ CORE\-12 vs 完整配方|
|---|---|---|---|---|---|
|完整配方`35:35:20:5:5`|无|0\.2617|—|0\.3887|—|
|S1|Wiki\-to\-RCQA|0\.2593|\-0\.0024|0\.3783|\-0\.0103|
|S2|Reddit\-to\-Flashcards|0\.2620|\+0\.0003|0\.3865|\-0\.0022|
|S3|Nemotron synthetic QA|0\.2595|\-0\.0022|0\.3825|\-0\.0062|
|S4|Geography|0\.2632|**\+0\.0014**|0\.3898|**\+0\.0012**|
|S5|Science|0\.2653|**\+0\.0036**|0\.3949|**\+0\.0063**|

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NDliOWU5OTY1ZjE4ZmMxOTAzNTEyZjlmMTFkY2IzYzlfYzA5OGEzOTU0MmYyYTA2NGJmY2Y0MmFmMTc2MTUwYzFfSUQ6NzY4NDEyMzY2NzIzMjAxNzM3MF8xNzg5MTIyMjA1OjE3ODkyMDg2MDVfVjM)

消融结果显示：

- Wiki\-to\-RCQA 和 Nemotron 被删除后两个指标都下降，是应当保留并提高占比的核心成分。

- Reddit 对总体 CORE 基本中性，对 CORE\-12 略有正贡献，适合保留但降低权重。

- 删除 Geography 带来小幅提升，删除 Science 带来这一组中最明显的提升。在当前小模型和有限中训练预算下，这两类数据的边际收益为负。

据此得到新的 OLMo 3 内部配方：

```Plain Text
Wiki-to-RCQA 40% + Reddit-to-Flashcards 25% +
Nemotron synthetic QA 35% + Geography 0% + Science 0%
```

换算到整个后程训练流，实际 manifest 权重为 ClimbMix 40%、Wiki\-to\-RCQA 24%、Reddit\-to\-Flashcards 15%、Nemotron synthetic QA 21%。

## **6\. 优化配方的完整训练结果**

将优化后的 `40:25:35` 配方用于 T1 的第 3500 步同步切换方案，结果如下：

|配方|总步数|CORE|CORE\-12|
|---|---|---|---|
|原始 baseline|5568|0\.2560|0\.3818|
|初始配方35:35:20:5:5|5568|0\.2665|0\.3918|
|优化配方40:25:35|5568|0\.2662|0\.3965|

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NWYyNjZlZDI0OTIyODRiYWE1YTFmNTgwZDVkMzgzZjBfNzY3ZWFhZTZjZTc1NjIxZjBjNzhlZWQ1ZTIyZmJiM2ZfSUQ6NzY4NDEyNDA3NjY2NzM5MDk1OV8xNzg5MTIyMjA1OjE3ODkyMDg2MDVfVjM)

优化配方的总体 CORE 与初始中训练配方基本持平，仅相差 0\.0003；CORE\-12 则进一步提高 0\.0047。相对原始 baseline，优化配方的 CORE 提高 0\.0102，CORE\-12 提高 0\.0147，分别约为 3\.98% 和 3\.84% 的相对提升。

从任务分项看，优化配方相对 baseline 的主要增益来自 `bigbench_qa_wikidata`（\+0\.098）、`commonsense_qa`（\+0\.083）、`squad`（\+0\.080）和 `coqa`（\+0\.068）。这与中训练数据偏向知识问答和合成 QA 的构成一致。部分任务仍有回退，例如 `boolq`、`hellaswag_zeroshot`、`arc_challenge` 和 `piqa`，说明后续仍有继续平衡数据分布的空间。

Ps: 后续可以继续研究不同数据配比，或者其他数据集

## **7\. 缩短 10% 训练步数**

完整训练已经提供约 0\.010 CORE 的余量，因此进一步将训练终点从 5568 提前到 5000，并把 cosine warmdown 调整为覆盖第 3500～5000 步。5000 步相对 baseline 减少 568 步，即 10\.20%。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDQzZjhmNWFiMzIyMWQ1NDhmMzAxNWMzODgwYWFlMDNfZmFjMDRlNTk5MGMxYTliMGNmMjE2N2EzMmJmMjU1OWNfSUQ6NzY4NDEyNDIyNjM1NzM4MjExNF8xNzg5MTIyMjA1OjE3ODkyMDg2MDVfVjM)

|实验|步数|CORE|Δ CORE vs 5568\-step baseline|CORE\-12|训练时间|训练 FLOPs|
|---|---|---|---|---|---|---|
|原始 baseline|5568|0\.2560|—|0\.3818|27,898\.5 秒|2\.3169 × 10^19|
|最优时间点\+最优数据配比（SOTA）|5568|0\.2662|\+0\.0102|0\.3965|27944\.7 秒|2\.3169 × 10^19<br>|
|5000step SOTA|5000|0\.2638|\+0\.0078|0\.3894|25,085\.7 秒|2\.0806 × 10^19 |

少训练 10\.20% 后，CORE 仍提高 0\.0078，CORE\-12 仍提高 0\.0076。训练 FLOPs 减少 10\.20%，实际训练时间减少 10\.08%，约节省 2813 秒，即 46\.9 分钟。

## **8\. 结论**

本阶段证明了 OLMo 3 中训练数据在 nanochat d24 上能够带来稳定提升，但其有效性依赖正确的训练方式：

- 前期保留 ClimbMix 通用预训练，后期使用 ClimbMix 40% \+ OLMo 3 中训练数据 60%；

- 在第 3500 步同时切换数据并启动 cosine warmdown；

- Dolma 3 Dolmino Mix 部分采用 Wiki\-to\-RCQA 40%、Reddit\-to\-Flashcards 25%、Nemotron synthetic QA 35%，不再使用 Geography 和 Science；

- 最终训练到第 5000 步即可停止。

最终方案在训练步数缩短 10\.20%、训练时间缩短约 10\.08% 的情况下，运行的 CORE 和 CORE\-12 都超过了原 5568 步 baseline。由此可以得出明确结论：引入并优化 Dolma 3 Dolmino Mix 中训练数据能够提升模型效果，而且提升幅度足以换取约 10% 的训练成本节省。

## **9\. 下一步讨论**

更好的数据配比

更好的数据集

一些其他trick

和退火怎么配合

