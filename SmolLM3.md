# SmolLM3

参考链接：https://huggingface\.co/spaces/HuggingFaceTB/smol\-training\-playbook

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTRkMzY5Mjg5MTc4OWQyYmM4ZTIyM2JjYzMxNWViZmRfOTI0OTNiMTI1YjA1Mzg3NWY4YWEzYjM3ODkwNzZjZWVfSUQ6NzY4NDIyMTY5NTQyMjg0NDE4N18xNzkxNTE2NjE3OjE3OTE2MDMwMTdfVjM)

### 训练三个阶段：

- **Stage 1—Base training \(8T tokens, 4k context\):** 主要数据集 FineWeb\-Edu, DCLM, FineWeb2, FineWeb2\-HQ, 代码数据集 The Stack v2 and StarCoder2, 数学数据集 FineMath3\+ and InfiWebMath3\+\. 数据集配比 Web : Code : Math = 85 : 12 : 3

- **Stage 2—High\-quality injection \(2T tokens, 4k context\):** 高质量数据集注入，代码 Stack\-Edu, 数学 FineMath4\+ and InfiWebMath4\+, 高质量数学推理 MegaMath \(使用 Qwen Q\&A, 转写合成, 文本代码交错块\)\. 数据集配比 Web : Code : Math = 75 : 15 : 10

- **Stage 3—LR decay with reasoning and Q\&A data \(1\.1T tokens, 4k context\):** 在学习率退火阶段，除了对高质量数学和代码数据进行上采样，还引入指令和推理数据，包括 OpenMathReasoning, OpenCodeReasoning, and OpenMathInstruct\. 数据集配比 Web : Code : Math = 63 : 24 : 13

> 💡 **关键细节**：Text\-Code Interleaved Blocks 是指将代码与自然语言交替混合的样本（如技术教程、README），而非纯代码或纯文本，这对提升模型的"代码解释"和"按需生成"能力至关重要。
> 
> 

### 数据配比的动态调整策略

Mid\-training 并非使用固定配比，而是采用 **“退火消融实验”** 来确定新数据的最佳混合比例：

- **实验方法**：从 Stage 1 后期（约 7T tokens 处）提取检查点，进行 50B tokens 的短时退火实验。

- **测试配置**：**40% Stage 1 基线数据 \+ 60% 待测新数据集**。

- **决策依据**：只有当新数据集在退火实验中证明能显著提升特定领域性能且不损害通用能力时，才会被正式纳入 Stage 2 配比。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjBjMGYwY2E1YzQ1OThmYTU0NWJmNjhjYTE0MmY2YjdfZjNiYjg4NzQ5YWRhNjQzMTRjNzFkMDE1ZDcwZThiM2JfSUQ6NzY4NDIyMTc2OTE4NzY0MjY2MF8xNzkxNTE2NjE3OjE3OTE2MDMwMTdfVjM)

### Insights

- **高质量数据后置原则**：小而精的数据集（如 Stack\-Edu、MegaMath）在训练中后期引入效果最佳，因为模型在末期对数据分布更敏感。

- **计划性与反应性结合**：

    - *计划性*：预先规划好 Stage 2 要引入的高质量数据集。

    - *反应性*：训练过程中持续监控 benchmark，若发现某项能力（如数学）停滞，则临时策划并注入针对性数据（这一机制在 SmolLM2 中已验证有效）。

- **避免过度重复**：即使高质量数据更好，也不能只用它们。Stage 2 仍保留大量 Stage 1 基线数据，防止因数据量不足导致的过拟合和重复训练伤害。

## Phi 4

参考链接：https://arxiv\.org/abs/2412\.08905

Midtraining阶段的主要目的是将模型的上下文长度从预训练阶段的 **4K 扩展至 16K**。

### 数据策略与组合

为了提升长上下文性能，团队进行了多项消融实验，发现使用“天然长文本”比“人工拼接填充的长文本”效果更好。基于此，Midtraining的数据混合策略如下：

- **天然长文本筛选：** 对高质量非合成数据集（如学术文献、书籍和代码）进行过滤，分离出超过8K上下文的样本，并对长度≥16K的数据子集进行加权（up\-weight）。

- **新增长文本合成数据：** 专门创建了满足 \>4K 序列要求的新合成数据集。

- **最终数据配比：**

    - **30%**：新策划的长上下文数据（包含上述筛选的天然长文本和新合成的长文本）。

    - **70%**：来自预训练阶段的回忆token（recall tokens），用于保持模型原有能力。

### 训练超参数与设置

- **位置编码调整：** 为适应更长的上下文，将RoPE（旋转位置编码）的基础频率（base frequency）增加至 **250K**。

- **学习率：** 最大学习率相比预训练阶段下降了 **10倍**。

- **训练规模：** Midtraining阶段总共训练了 **250B tokens**。

### 评估方法

团队强调使用反映真实应用场景的综合评估框架，而非仅依赖“大海捞针”等合成基准。主要采用 **HELMET** 评估套件，涵盖以下任务：

- **Recall（召回）：** 从长JSON文件中检索特定键值。

- **RAG：** 基于多篇打乱的维基百科文档回答问题。

- **Re\-rank（重排序）：** 根据查询对检索文档进行重排。

- **ICL（上下文学习）：** 多样本（many\-shot）上下文学习任务。

- **QA（问答）：** 基于长文档回答问题（如NarrativeQAv2）。

- **Summ（摘要）：** 对长篇法律文档进行摘要。

通过这些措施，Phi\-4在16K上下文长度下保持了强劲的性能，且在多个长文本基准测试中优于同量级甚至更大参数的模型。

## OLmo\-3

参考链接：https://arxiv\.org/abs/2512\.13961

### 目标

提升代码数学能力，通过QA实现更高效知识获取、指令遵循、思考推理

三个训练阶段token预算 

pretraining:midtraining:长上下文拓展=5\.9T:100B:50B\(OLMO3\-base\-7B\)/100B\(OLMO3\-base\-32B\)

### Signal\-to\-Noise 信噪比分析

核心想法：信噪比衡量的是能反映真实能力变化的信号vs评测本身的随机波动。如果某任务的分数主要在噪声里抖，就没法用来判断数据消融好坏。做了三类处理：

1. 整任务剔除
噪声太大、又没有独立价值的，直接移出套件。典型是 BoolQ 这类二分类：模型在多数类/少数类之间来回振荡，分数几乎不反映能力。

2. 移出宏平均，但不删
有独立能力价值、但噪声会污染平均分的任务（如 CruxEval）——从 cluster 平均里拿掉，单独跟踪。信号还在，只是不混进总分。

3. 改评测配置抬 SNR（Signal Noise Ratio）
例如：用完整评测集代替 1K 子集；调 generation 超参（如提高 pass@k 的 k）；多任务聚合本身也能把单任务噪声压下去。

### Midtraining 两套方法框架

- 用「分布式轻量反馈」快速筛数据源，再用「集中式集成测试」评估候选 mix 的整体质量与 post\-trainability。

- Microanneal: 并行试很多候选数据，判断这个数据本身有没有用

    1. 选择目标数据集，从中采样5B

    2. 和预训练的网络数据采样5B混合

    3. 消融实验，对比选择10B预训练的网络数据

- Integration Test：在 microanneal 有足够结果后，对 100B\-token midtraining mix 做完整 anneal 集成测试。评估两层：

    - Base 质量：在 OlmoBaseEval Main 上看能力簇

    - Post\-trainability：midtrain checkpoint 再做快速 SFT，看后续 post\-train 评测是否也有提升

### Midtraining数据集

100B token混合构成的最高质量的预训练数据，辅以大量针对数学与代码问题的专项任务数据，以及通用知识 QA、指令遵循等相关能力。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGE0NzI2M2ZiMTI4NzY2YjIxMTI1NmU5YzM3OWQ5YmNfYmU2MGY3ODI4OTU5ZWVlMDcyY2MwYTMwOTZlMmNhNjlfSUQ6NzY4NDIyMTgyNDA1MzY5MzQxOV8xNzkxNTE2NjE3OjE3OTE2MDMwMTdfVjM)

结合Nanochat项目的CORE任务，主要看他们在Mitraining阶段用来提升知识获取的数据集：

1\. Reddit\-to\-Flashcards（新合成）

- 动机：提升多选 QA 里多样的内容类别和问题结构。

- 做法：先筛出偏学术的 subreddit，再用 GPT\-4o\-mini 把「帖子 \+ 评论」改写成多选 QA，并用 7 种任务格式增加多样性。

- 效果：10B microanneal 里加入 5B tokens，相对 web\-only baseline，MC\-Non\-STEM 约 \+2，MMLU 约 \+3。

- 副作用：会抬高部分 code 指标，但可能拖累 math 和 GenQA。

2\. Wiki\-to\-RCQA（新合成）

- 动机：提升基于篇章的阅读理解 QA。

- 做法：取 Wikipedia 段落，用 Qwen2\.5 32B Instruct 生成 QA 对，并约束成类似人工标注阅读理解数据的要求。

- 效果：4\.2B tokens 放进 10B microanneal，GenQA 约 \+2，收益主要来自 DROP、SQuAD、CoQA。

3\. Nemotron Synth QA（外部数据子集）

- 来源：Nemotron CC 的 “diverse QA pairs” 合成子集。

- 筛选结果：这一子集有效——GenQA \+1\.5、MC\-Non\-STEM \+1\.9；MC\-STEM 与「用最高 5% 质量 web 数据」持平。

- 没用上的：同数据集里的 `distill`、`extract knowledge`、`knowledge list`、`wrap medium` 等子集都不如自然数据，因此剔除。

### Decontamination（去污染）

训练数据里不要“偷看”评测集，否则 benchmark 分数会虚高、不可信。

训练后期记忆效应最强，所以重点在 midtraining 和 long\-context extension（它们共用部分数据池）阶段去污染，而不是把同样力度铺到整个 pretraining。

对评测 harness 里 所有 benchmark 的所有 split 去污染。原因是有的评测为了降噪也会用 train split，污染 train 也可能抬高最终分数。

工具：`decon`（两阶段）检测阶段稀疏扫描，扩展阶段精细算污染分。

- 1\. Detection（检测）

对每篇 midtraining 文档，按固定 stride 采样 n\-gram，检查是否命中评测集里已知 n\-gram。

- 2\. Cluster expansion（簇扩展）

命中后向左右扩展，统计邻接 n\-gram 里也有多少污染；超过阈值就判定整篇文档被污染并删掉。

发现与影响

- 污染最集中的来源主要是 Flan、Nemotron 等现成数据集；很多是模板化整段泄漏，甚至包含完整 val/test split。

- 污染不总是抬高分数：有的任务（DROP、Minerva、SQuAD）去掉污染后分数明显下降；有的几乎不变（DeepSeek LeetCode 本来就很低，SQuAD\-MC 已饱和）。

- GSM8K 甚至相反：即便检测到完整泄漏，污染版也更差——可能因为泄漏格式与评测格式不一致。

***Ps\. 这部分在Nanochat里没来得及复现，后续可以考虑***



