# 构图任务从“最优构图”到“更好构图”的 Benchmark 可行性调研

调研日期：2026-07-01  
项目语境：`RIGCrop` / `GAICD` / `CPC` / `DACC` / VLM-assisted composition dataset builder  
核心问题：

1. 现有构图/裁剪方法多在候选空间中寻找 `top-K` 或最高分 crop，但搜索空间巨大，且“最优构图”可能并不唯一。
2. “不同测试群体对 top-K 的偏差很大”是否有文献支撑？
3. 能否把构图任务重新定义为“寻求更好的构图”，而不是“寻求唯一最优构图”，并据此构建 benchmark？
4. 是否已有相似工作？

## 0. 结论先行

### 0.1 对“不同测试群体对 top-K 偏差很大”的判断

结论：**有较强间接支撑，但裁剪/构图领域中直接把不同测试群体的 top-K crop 做系统对比的公开证据还不充分。**

更稳妥的表述是：

> 图像构图/裁剪的优质解具有非唯一性、主观性和多模态性；现有 top-K/MOS benchmark 往往通过 MOS、投票阈值、歧义样本剔除或专家筛选来压缩这种差异。因此，“不同测试群体的 top-K 会出现明显偏移”是高度合理的研究假设，但若要作为论文核心 claim，建议在 benchmark 中显式报告 group-wise top-K stability，而不是只引用现有文献。

支撑链条如下：

- **直接来自裁剪文献的证据**：FCDB/WACV 2017 的 pilot study 明确指出 photo cropping 有时非常主观，人们会认可与自己不同的 crop；CPC/CVPR 2018 和 GAICD/CVPR 2019 都把裁剪监督从单个 GT box 转为 pairwise ranking、MOS 或 top-N acceptable set，说明“唯一最优框”不是稳固假设。
- **来自审美评价的间接强证据**：PARA/CVPR 2022 指出个性化图像审美高度主观，审美偏好与评价者属性有关；MOS 只是 average opinion，会遮蔽个体/群体差异。
- **反向约束**：GAICD 报告在经过摄影/艺术背景筛选的 19 名标注者中，每个 crop 由 7 人评分，94.25% crop 的评分标准差小于 1。这说明在受控专家-ish 群体内，一致性可以被提高；它不直接证明“群体偏差很大”，反而提示必须区分“随机人群”“摄影/艺术背景人群”“目标用户群体”。

### 0.2 对“寻求更好构图” benchmark 的判断

结论：**可行，而且更符合构图任务的真实形态；但它不是完全无人做过的空白，需要把 novelty 放在 benchmark formulation 与 group-robust evaluation 上。**

已有相似路线：

- WACV 2017 FCDB：把裁剪看成 pairwise view ranking。
- CVPR 2018 CPC：大规模 comparative view pairs，核心监督是“哪个 view 更好”。
- CVPR 2019 GAICD：官方指标 `AccK/N` 本质上评估返回 crop 是否落入 top-N 优质集合，而不是唯一 top-1。
- CVPR 2022 Rethinking Image Cropping：把裁剪看成 set prediction，目标是从 global view 生成多个 good compositions。
- CVPR 2023 Spatial-Aware + Rank Consistency：继续沿用 GAICD 的 top-N/MOS 排序，并引入 pairwise ranking classifier。
- CVPR 2025 Cropper：VLM 生成多个候选，使用 scorer feedback iterative refinement，并做人评 pairwise preference。

所以更可发表的定位应是：

> 不只是提出 pairwise crop ranking，而是提出一个面向“composition improvement”的 benchmark：给定原图/初始构图/候选构图，评价模型是否能稳定找到“相对更好且不退化”的构图，并显式报告跨群体、跨用途、跨偏好的鲁棒性。

## 1. 文献检索范围与证据等级

### 1.1 检索范围

本次调研重点检索和核对了以下方向：

- automatic image cropping / photo cropping
- composition ranking / view finding / view proposal
- GAICD / CPC / FCDB / FLMS
- MOS、`AccK/N`、SRCC、PCC、IoU、BDE/Disp
- pairwise preference / learning-to-rank
- personalized image aesthetics assessment
- VLM image cropping / in-context learning cropping

### 1.2 证据等级

| 等级 | 含义 | 本调研中的使用方式 |
|---|---|---|
| A | 裁剪/构图论文中直接支持 | GAICD、CPC、FCDB、Rethinking、Cropper |
| B | 审美/主观评价论文中间接支持 | PARA、AADB/PIAA 等 |
| C | 方法论推断 | 从 MOS/top-N/pairwise 任务定义推断 top-K 不稳定风险 |
| D | 需要本项目补实证 | 不同群体 top-K Jaccard/Kendall/NDCG 的具体偏移幅度 |

## 2. 现有构图/裁剪 benchmark 的核心范式

### 2.1 早期单 GT/少量 GT：IoU/BDE 难以表达非唯一性

早期 image cropping 数据集常给每张图一个或少数几个人工 crop box，并用 IoU、boundary displacement error（BDE/Disp）评价预测框是否接近 GT。问题是：裁剪和目标检测不同，**一个不接近 GT 的 crop 也可能是好构图，一个接近 GT 的 crop 也可能因为边界细节而更差**。

GAICD 论文在引言中明确批评：已有数据库只提供 one or several human-annotated boxes，无法反映裁剪的 non-uniqueness 和 flexibility；IoU/BDE 也不能可靠反映裁剪质量。

FCDB/WACV 2017 也指出，photo cropping 的本质是对 visually similar proposal windows 做排序，而不只是回归一个位置。

### 2.2 FCDB/WACV 2017：裁剪是 pairwise ranking 问题

论文：Chen et al., *Quantitative Analysis of Automatic Image Cropping Algorithms: A Dataset and Comparative Study*, WACV 2017.  
链接：[PDF](https://www.cmlab.csie.ntu.edu.tw/~robin/docs/wacv17.pdf) / [dataset](https://github.com/yiling-chen/flickr-cropping-dataset)

关键点：

- 提出 Flickr Cropping Dataset，包含 `3,413` cropped images 与 `34,130` crop pairs。
- 从 `3,413` 个人工裁剪结果中，经 AMT 人评筛选出 `1,743` 个 highly ranked crops；其中 `348` 张用于测试。
- 每个 source/crop pair 由 `7` 个 AMT worker 评价，只有至少 `4` 人认为 crop 更好才保留。
- 每张图还随机生成 `10` 对 crop windows，由 `5` 个 worker 做 pairwise preference annotation。
- 论文 pilot study 观察到：photo cropping 有时非常主观；人们有时会认可与自己不同的 crop。
- 论文结论强调：ranking pairwise views 对 image cropping 很关键。

对本项目的启发：

- “更好构图”可以直接定义为 pairwise preference：`crop_a > crop_b`。
- “原图 vs 裁剪图”“baseline crop vs model crop”“bad crop vs improved crop”都可以成为监督单元。
- 与其让标注者寻找唯一最优，不如让其判断相对改进，任务负担更小，噪声也更可控。

### 2.3 CPC/CVPR 2018：大规模 comparative view pairs

论文：Wei et al., *Good View Hunting: Learning Photo Composition from Dense View Pairs*, CVPR 2018.  
链接：[CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Good_View_Hunting_CVPR_2018_paper.pdf)

关键点：

- 提出 Comparative Photo Composition（CPC）dataset。
- `10,800` images，每张图生成 `24` views。
- 每张图由 `6` 个 AMT workers 标注，最终形成超过 `1M` comparative view pairs。
- 两阶段标注：
  - Stage 1：在每个 aspect-ratio group 中选择 `2-5` 个 good views。
  - Stage 2：从候选 good views 中选择 overall top `3`。
- 论文特别指出，随机 views 的 pairwise labeling 会有二次复杂度，因此用两阶段流程降低认知负担。
- XPView 专家测试集中，`992` images、`24` candidate crops/image，由 `3` 位视觉艺术方向专家标注为 good/fair/poor，并保留 unanimously labeled 样本。
- 论文末尾提出 future work：探索 professional versus AMT annotations 对预测质量的影响。

对本项目的启发：

- CPC 已经把裁剪从“唯一 top-1 box”推进到了 comparative preference。
- 但 CPC 的重点是训练 dense view ranker，不是专门构建“相对于原始/初始构图的 improvement benchmark”。
- CPC 对群体差异的处理仍然较粗：主要是多 worker 投票、质量控制、歧义对 pruning，而不是显式发布 group-conditioned labels。

### 2.4 GAICD/CVPR 2019：top-N acceptable set，而不是唯一最优

论文：Zeng et al., *Reliable and Efficient Image Cropping: A Grid Anchor Based Approach*, CVPR 2019.  
链接：[CVF PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf)

关键点：

- 指出 crop search space 极大：从像素级候选的 `H(H-1)W(W-1)/4` 降到 grid anchor 的少量候选。
- GAIC formulation 把每张图候选 crop 降到不超过 `90`。
- GAICD conference version 包含 `1,236` source images 和 `106,860` annotated candidate crops。
- `19` 位通过摄影构图测试的标注者参与，来自摄影社区或艺术院系；每个 crop 由 `7` 人评分。
- 每个 crop 的 1-5 分 MOS 作为 ground-truth quality score；论文报告 `94.25%` crop 的评分标准差小于 `1`。
- 官方指标：
  - SRCC：预测分数与 MOS 排序的 Spearman correlation。
  - `AccK/N`：模型返回的 top-K crop 有多少落入 MOS 排名前 N 的集合。
  - 论文设置 `N in {5, 10}`，`K in {1,2,3,4}`，并汇总为 `Acc5`、`Acc10`。

对本项目的启发：

- GAICD 已经默认“好构图集合”存在，而不是只看 top-1。
- 但 GAICD 仍然依赖全局 MOS 排序，并把 group preference 压缩成一个平均分。
- 如果你的核心担心是不同测试群体 top-K 不稳定，那么 GAICD 协议需要扩展为 group-wise MOS / group-wise top-N / group-robust metrics。

### 2.5 Rethinking Image Cropping/CVPR 2022：多样 good compositions

论文：Jia et al., *Rethinking Image Cropping: Exploring Diverse Compositions from Global Views*, CVPR 2022.  
链接：[CVF PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Rethinking_Image_Cropping_Exploring_Diverse_Compositions_From_Global_Views_CVPR_2022_paper.pdf)

关键点：

- 论文把既有方法分为：
  - anchor evaluation methods：可以输出多样 crops，但预定义 anchors 难以覆盖全局好构图。
  - coordinate regression methods：从 global image 回归 crop，但常只输出一个 crop，忽视 diversity。
- 论文明确指出 learning one best crop may cause ambiguity，因为它潜在假设其他 crop 都不好。
- 提出 set prediction formulation：多个 learnable anchors 回归多个 crops，再用 validity classifier 选择有效子集。
- 在 GAICv1/GAICv2 中，将 quality score 高于阈值的 crops 定义为 good crops。
- 使用 AP 评估 top-K predictions，K 包括 `5, 10, 40`。
- 论文还讨论 hard validity label 与连续 crop quality 的不一致：若阈值 `s >= 4` 定义 good，`1.2` 和 `3.8` 都会被二值化为 bad，但质量明显不同。

对本项目的启发：

- 这是与你想法最接近的一篇：它已经把任务从“单个最优 crop”转向“找多个 good compositions”。
- 但它仍然以 quality-score threshold / AP / top-K validity 为核心，不是“相对于某个初始构图是否更好”的 benchmark。
- 你的 benchmark 可以继承 set/multiple-good-crops 的思想，但把监督单元改成 improvement pair 或 improvement set。

### 2.6 Spatial-Aware + Rank Consistency/CVPR 2023：继续沿用 GAICD top-N 与 pairwise rank

论文：Wang et al., *Image Cropping With Spatial-Aware Feature and Rank Consistency*, CVPR 2023.  
链接：[CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.pdf)

关键点：

- 使用 GAICD journal version：`3,336` images，训练/验证/测试为 `2,636/200/500`，共 `288,069` labeled crops。
- 指标沿用 `AccK/N`、`Acc5`、`Acc10`、SRCC、PCC。
- 方法中引入 pairwise ranking classifier，让 crop-level score ranking 与 pairwise ranking 保持一致。
- 论文指出标注 dozens of candidate crops 很昂贵，因此探索 unlabeled data / rank consistency。

对本项目的启发：

- 2023 的 SOTA 仍在 GAICD top-N/MOS 体系内做提升，说明该协议仍是主流。
- 但其 pairwise classifier 说明 pairwise preference 是可被模型学习并用于提升排序稳定性的。

### 2.7 Cropper/CVPR 2025：VLM 生成候选 + scorer feedback + 人评

论文：Lee et al., *Cropper: Vision-Language Model for Image Cropping through In-Context Learning*, CVPR 2025.  
链接：[CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf)

关键点：

- 使用 VLM in-context learning 做 free-form、subject-aware、aspect-ratio-aware cropping。
- GAICD 中每张训练图有多个人工 crop 和 MOS，Cropper 检索 top-S 相似 images，再选择 MOS top-ranked crops 作为 examples。
- 默认超参中，free-form cropping 使用 `S=30` 个 ICL examples、每轮 `R=6` 个 candidates、`L=2` 轮 refinement。
- 评价仍用 GAICD 的 `AccK/N`、`Acc5`、`Acc10`、SRCC、PCC；FCDB/SACD 使用 IoU/Disp。
- 进行 user study：在 GAICD `200` 张测试图上，与 A2RL、GAIC、CGS 做 pairwise preference，Cropper 获得约 `60.8%`、`62.2%`、`63.4%` 的偏好率。

对本项目的启发：

- VLM 时代的方法也没有完全摆脱 top-N/MOS；但 user study 已经以 pairwise win rate 呈现“更受偏好”。
- 你的 benchmark 可以把人评 pairwise win rate 从补充实验提升为官方主指标。

## 3. “不同测试群体 top-K 偏差大”的证据分析

### 3.1 裁剪领域的直接证据：存在主观性、多解和歧义

| 来源 | 支撑内容 | 对“群体 top-K 偏差”的证据强度 |
|---|---|---|
| FCDB/WACV 2017 | pilot study 发现裁剪有时非常主观；不同人的 crop 可以都被认可；最终 crop 需经 7 worker 投票筛选 | 中等：直接说主观和多解，但未系统比较群体 top-K |
| CPC/CVPR 2018 | 每图 6 AMT worker，选择 2-5 good views 与 top-3 views；保留/生成 comparative pairs；专家集保留 unanimous labels | 中等：说明需要投票和歧义处理，但没有按群体发布 top-K |
| GAICD/CVPR 2019 | 每 crop 7 人 MOS，top-N acceptable set；受控标注者群体中 std < 1 的比例较高 | 双向：说明 top-N 比 top-1 更合理，也说明专家群体内一致性可提高 |
| Rethinking/CVPR 2022 | 明确指出 one best crop 会带来 ambiguity；多 good crops 需要 set prediction | 强：支持“唯一最优构图”不稳 |
| Cropper/CVPR 2025 | 用 user study pairwise preference 评价模型输出 | 中等：支持人评偏好作为主指标，但不报告群体分歧 |

### 3.2 审美个性化领域的间接强证据：群体/个体偏好会变

论文：Yang et al., *Personalized Image Aesthetics Assessment With Rich Attributes*, CVPR 2022.  
链接：[CVF PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Personalized_Image_Aesthetics_Assessment_With_Rich_Attributes_CVPR_2022_paper.pdf) / [arXiv](https://arxiv.org/abs/2203.16754)

关键点：

- Personalized image aesthetics assessment 被定义为 highly subjective。
- MOS 代表 average opinion，但会忽略 aesthetic tastes 的主观性。
- PARA 数据集包含 `31,220` images、`438` subjects，平均每图约 `25.87` 次标注，并提供年龄、性别、教育、艺术经验、摄影经验、Big-Five personality traits 等 subject information。
- 论文统计分析显示，personalized aesthetic preference 可以由 human-oriented subjective attributes 反映。
- 示例中，同一张图由 3 位 subjects 给出 `4`、`2.5`、`3.5` 的不同分数。
- 使用 subject information 的 conditional PIAA model 优于 unconditional model。

对裁剪 benchmark 的推论：

- 构图裁剪是 aesthetic judgement 的子任务，因此很可能继承个性化审美偏好。
- 如果“top-K crop”基于 MOS 平均分，则它更像多数群体的平均偏好，而不是所有群体的稳定 top-K。
- 对某些用途，目标群体不是平均人群。例如社交媒体、商品图、新闻图、艺术摄影、手机相册、学术图像裁剪，它们的偏好准则可能不同。

### 3.3 目前证据的缺口

缺口不是“主观性是否存在”，而是：

> 在同一批候选 crops 上，不同评价群体得到的 top-K 集合到底偏移多大？

现有论文通常采用这些方式“吸收”差异：

- 多标注者 MOS 平均。
- 投票阈值，如至少 `4/7` worker 认为更好。
- 剔除 ambiguous pairs。
- 只保留 unanimous labels。
- 筛选摄影/艺术背景标注者。

这些做法有利于训练稳定模型，但会让 group bias 在公开指标中不可见。

因此，如果要把“不同测试群体 top-K 偏差很大”写成强 claim，建议本项目补充以下实证：

1. **Bootstrap annotator top-K stability**：对每张图随机抽取 m 个标注者计算 MOS/top-K，重复采样，报告 `Jaccard@K`、`Kendall tau`、`top1 in consensus top10`。
2. **Expert vs amateur split**：分别计算专家组、普通用户组、摄影经验组、无摄影经验组的 top-K。
3. **Use-case split**：例如“社交分享偏好”“商业展示偏好”“艺术构图偏好”“内容完整性偏好”。
4. **Cross-group NDCG**：用 A 组 MOS 排序评价 B 组 top-K，反之亦然。
5. **Worst-group top-K hit**：模型输出是否只服务平均偏好，而在某个群体上显著退化。

建议在论文中把现有文献表述为：

> Prior work strongly suggests non-unique and subjective crop preferences, but group-conditioned top-K instability has not been sufficiently benchmarked.

这比直接说“已有研究证明不同测试群体 top-K 偏差很大”更严谨。

## 4. 为什么“寻求更好构图”是可行的任务重定义

### 4.1 从优化目标看：最优构图是不适定问题

传统 formulation：

```text
给定图像 I，输出 crop c* = argmax_c Quality(I, c)
```

问题：

- `Quality(I, c)` 通常来自 MOS、审美模型或人工偏好，本身有噪声和群体依赖。
- 搜索空间极大；即使 GAIC grid anchor 降到约 90 个，也只是离散近似。
- 多个 crop 的质量可能非常接近，top-1 排名会受少量标注噪声影响。
- 真实应用中，用户往往需要“比现在好”“不破坏主体和语义”“适合某用途”，不一定需要全局最优。

改写 formulation：

```text
给定图像 I 和参考构图 c0，输出 crop c，使得 c 在目标偏好分布 P_g 下优于 c0：

P_g(c ≻ c0 | I, context) > τ
```

其中 `g` 可以是总体用户、专家、目标群体或使用场景。

这个定义的好处：

- 把绝对 MOS 回归改成相对 preference，标注更容易。
- 允许多个正确答案，只要优于参考构图即可。
- 可以自然处理 tie/indifference：若差异不显著，不强迫模型排序。
- 可以用 group-wise preference 明确建模偏好差异。
- 更贴近日常裁剪：从原图、机器初稿或用户给的初始 crop 改到更好。

### 4.2 与现有 benchmark 的区别

| 维度 | GAICD/CPC/Rethinking | “更好构图” benchmark |
|---|---|---|
| 监督对象 | 候选 crop 的 MOS/top-N 或 pairwise ranking | crop 相对 reference/baseline 是否改进 |
| 正确答案 | top-N 集合或 good crop set | 所有显著优于 reference 的 crops |
| 评价目标 | 排出最靠前的高 MOS crops | 提高构图且避免退化 |
| 主指标 | `AccK/N`、SRCC、PCC、AP、pairwise acc | pairwise win rate、better-than-baseline rate、non-degradation rate、group robustness |
| 群体差异 | 多被 MOS 平均吸收 | 作为一等公民显式报告 |
| 任务语义 | 找好 crop / 排序 crop | 改善当前构图 |

### 4.3 可能的官方任务定义

建议定义三个层级，便于 benchmark 从简单到完整扩展。

#### Task A：Binary Composition Improvement

输入：

```text
image I
reference crop c0
candidate crop c
optional context q: aspect ratio/use case/subject
```

输出：

```text
y ∈ {better, tie, worse}
```

模型任务：

```text
判断 c 是否比 c0 更好
```

适合训练 ranker、reward model、VLM judge、data filtering。

#### Task B：Improved Crop Retrieval

输入：

```text
image I
reference crop c0
candidate set C = {c1, ..., cn}
```

输出：

```text
top-K crops that improve c0
```

官方评价：

- `Improve@K`：top-K 中有多少被多数人认为优于 c0。
- `BestImprove@K`：top-K 中是否至少有一个显著优于 c0。
- `NoHarm@K`：top-K 中被认为 worse than c0 的比例越低越好。
- `TieAwareImprove@K`：better=1、tie=0.5、worse=0。

#### Task C：Free-form Composition Improvement

输入：

```text
image I
reference crop c0 or original image
optional constraints: aspect ratio / subject / platform / style
```

输出：

```text
model-generated crop c
```

评价方式：

- 将模型输出匹配到候选集中最近 crop，或直接做人评 pairwise。
- 与 `original`、`reference crop`、`strong baseline crop` 分别比较。
- 报告整体 win rate 和 group-wise win rate。

## 5. Benchmark 设计建议

### 5.1 数据构建流程

建议沿用你当前项目已有 pipeline：

```text
原图
  -> semantic/VLM understanding
  -> subject/key object/relation detection
  -> GAIC-style grid candidates
  -> directional/action candidates
  -> negative/perturbed candidates
  -> reference crop selection
  -> pairwise/group-wise human or VLM-assisted annotation
```

候选 crop 来源建议包含：

- GAIC-style grid anchors：保证可与 GAICD 主流指标对齐。
- 原图 / center crop / rule-of-thirds crop：作为 reference 或 baseline。
- 现有算法输出：GAIC、CGS、RIGCrop、Cropper-style VLM、rule scorer。
- 扰动 negatives：主体切断、关系破坏、过度放大、留白失衡、干扰物保留。
- VLM suggested crops：增加语义驱动的非 grid 候选。
- human drawn crops：少量专家自由框作为高质量 anchors。

### 5.2 标注协议

推荐 pairwise 标注问题：

```text
在给定用途下，右侧构图是否比左侧构图更好？

选项：
1. A 明显更好
2. A 略好
3. 差不多 / 都可接受
4. B 略好
5. B 明显更好
6. 都不好 / 无法判断
```

然后映射为：

```text
better: majority prefers candidate over reference with sufficient margin
tie: votes close or many "差不多"
worse: majority prefers reference over candidate
ambiguous: low agreement / unable-to-judge high
```

建议保留 vote distribution，而不是只发布硬标签。这样可以：

- 训练 BT/Thurstone/Plackett-Luce latent quality。
- 计算群体分歧。
- 构造 hard set 与 ambiguous set。
- 让 benchmark 同时支持 pairwise、listwise、MOS-like 任务。

### 5.3 群体信息

如果要验证“不同测试群体 top-K 偏差”，标注者 metadata 很关键。建议至少采集：

- 摄影经验：无 / 初级 / 爱好者 / 专业。
- 艺术/设计背景：无 / 有课程训练 / 职业相关。
- 使用场景偏好：社交分享、商品展示、新闻纪实、艺术摄影、学术/技术文档。
- 年龄段、设备使用习惯可选。
- 是否色盲/视觉障碍只用于质量控制和伦理合规，不建议公开可识别信息。

### 5.4 官方指标

#### 主指标 1：Pairwise Improvement Win Rate

```text
PIWR = # {model crop preferred over reference} / # valid comparisons
```

可以细分：

- `PIWR_original`：相对原图。
- `PIWR_center`：相对 center crop。
- `PIWR_strong`：相对强 baseline。

#### 主指标 2：No-Harm Rate

```text
NHR = 1 - # {model crop judged worse than reference} / # valid comparisons
```

原因：如果 benchmark 只奖励“变好”，模型可能冒险裁掉重要内容；No-Harm Rate 可以惩罚退化。

#### 主指标 3：Tie-aware Utility

```text
Utility = 1.0 * P(better) + 0.5 * P(tie) + 0.0 * P(worse)
```

原因：构图中存在大量“差不多都好”的情况，硬二分类会制造伪差异。

#### 主指标 4：Group Robustness

```text
MeanGroupWin = mean_g PIWR_g
WorstGroupWin = min_g PIWR_g
GroupGap = max_g PIWR_g - min_g PIWR_g
```

如果你要主打“群体 top-K 偏差”，这是最关键的新增指标。

#### 诊断指标：top-K stability

对同一张图、同一候选集：

```text
TopKJaccard(g1, g2) = |TopK_g1 ∩ TopK_g2| / |TopK_g1 ∪ TopK_g2|
KendallTau(g1, g2)
NDCG_cross(g1 -> g2)
Top1InOtherTopN
```

#### 兼容指标

为便于和现有论文对齐，建议仍报告：

- GAICD-style `Acc1/5 ... Acc4/10`, `Acc5`, `Acc10`
- SRCC / PCC
- FCDB-style IoU / Disp
- CPC-style pairwise accuracy / swap error / nDCG

但论文主张应清楚区分：

```text
传统指标：模型能否接近 MOS top-N。
新指标：模型能否稳定改善 reference composition，并在不同偏好群体中不退化。
```

### 5.5 防止 benchmark 太容易

“更好构图”如果只和原图比较，可能很容易饱和。建议设计三档难度：

| 难度 | reference | 目的 |
|---|---|---|
| Easy | 原图或明显差 crop | 测试基本裁剪能力 |
| Medium | center/rule baseline | 测试常规改进能力 |
| Hard | GAIC/RIGCrop/Cropper-style strong baseline | 测试细粒度构图判断 |

同时保留 hard negatives：

- IoU 很高但切掉关键语义。
- 主体完整但关系被破坏。
- 构图规则更好但内容表达更差。
- MOS 接近但不同群体偏好相反。

## 6. 与本项目 RIGCrop/DACC 的结合方式

### 6.1 现有项目可直接复用的资产

本 workspace 已经具备：

- `composition_dataset_builder/`：候选生成、VLM 理解、主体/关键物、关系、方向性 candidates、负样本 candidates、多维 scoring。
- `RIGCrop/docs/GAICD_EVALUATION_PROTOCOL.md`：GAICD official metrics 的实现协议。
- `scripts/eval_cpc_pairwise.py`、`scripts/train_pairwise_ranker.py`：CPC-style pairwise 训练/评估。
- `RIGCrop/test_script/eval_cpc_pairwise_metrics.py`：RIGCrop 对 pairwise metrics 的测试路径。
- DACC/RIGCrop 的中间态 JSONL schema，可以承载 semantic issue、action、utility、relation importance。

### 6.2 建议新增的数据字段

在现有 JSONL 中为每个 candidate 加：

```json
{
  "reference_crop_id": "original|center|baseline_xxx",
  "candidate_crop_id": "cand_001",
  "pairwise_votes": {
    "overall": {"better": 4, "tie": 1, "worse": 1},
    "expert": {"better": 3, "tie": 0, "worse": 0},
    "amateur": {"better": 1, "tie": 1, "worse": 1}
  },
  "preference_label": "better|tie|worse|ambiguous",
  "improvement_margin": 0.42,
  "target_context": "social_share|commercial|documentary|artistic|general",
  "failure_modes": ["cuts_subject", "breaks_relation", "too_tight", "keeps_distractor"]
}
```

### 6.3 模型训练方式

可以训练三个 head：

- `score_head`：兼容 GAICD MOS/ranking。
- `pairwise_head`：预测 candidate 是否优于 reference。
- `group_conditioned_head`：输入 group/context embedding，输出 group-specific preference。

损失函数：

```text
L = L_mos_or_rank + λ_pair * L_pairwise + λ_group * L_group_consistency + λ_noharm * L_regression_penalty
```

其中 `L_noharm` 可以专门惩罚模型把明显 worse crop 排到 reference 之上。

## 7. 相似工作与可区分 novelty

| 工作 | 已做什么 | 与 proposed benchmark 的差异 |
|---|---|---|
| FCDB/WACV 2017 | pairwise crop ranking dataset；强调 ranking pairwise views | 主要是算法比较和裁剪 GT，未以“改善 reference crop”为核心 |
| CPC/CVPR 2018 | 大规模 comparative view pairs；dense view ranking | 没有显式 group-wise top-K stability，也不是 improvement benchmark |
| GAICD/CVPR 2019 | dense grid anchors + MOS + `AccK/N` | 仍是全局 MOS top-N，不暴露群体差异 |
| Rethinking/CVPR 2022 | set prediction，多个 good compositions | 仍基于 good crop threshold/AP，不是 pairwise improvement |
| Spatial-Aware/CVPR 2023 | pairwise rank classifier + rank consistency | 是训练正则，不是 benchmark 重定义 |
| Cropper/CVPR 2025 | VLM multi-candidate + iterative refinement + user study | user study 是补充，不是带群体鲁棒性的官方协议 |
| PIAA/PARA/CVPR 2022 | 个性化审美、subject attributes | 不是裁剪，但支持 group-conditioned aesthetics |

可区分 novelty 建议写成：

1. **Task novelty**：从 “find the best crop” 转为 “improve a given composition under subjective preference distribution”。
2. **Evaluation novelty**：引入 group-wise improvement robustness、top-K stability、no-harm rate。
3. **Data novelty**：保留 vote distribution 与 group metadata，而不是只发布 MOS。
4. **Modeling novelty**：训练 image-only student 或 VLM-distilled ranker，使其学习 semantic/relation-aware improvement，而不是只学 MOS top-N。

## 8. 论文/benchmark claim 的推荐写法

### 8.1 稳妥 claim

```text
Image composition is intrinsically non-unique and preference-dependent.
Existing benchmarks reduce this ambiguity through MOS aggregation,
top-N acceptable sets, or pairwise ranking, but rarely expose
group-conditioned top-K instability. We therefore formulate cropping
as composition improvement rather than single optimum search.
```

中文：

```text
图像构图具有非唯一和偏好依赖的性质。现有 benchmark 通常通过 MOS 平均、
top-N 可接受集合或 pairwise ranking 缓解这种歧义，但很少显式评估不同
测试群体下 top-K 构图集合的稳定性。因此，我们将裁剪/构图重新定义为
composition improvement，而非唯一最优构图搜索。
```

### 8.2 不建议直接写的强 claim

```text
Existing studies have proved that different test groups have large top-K bias.
```

风险：

- 裁剪论文通常没有直接报告 group-conditioned top-K overlap。
- GAICD 的专家标注一致性结果可能被审稿人用作反例。
- 个性化审美论文支持主观差异，但不是 crop top-K 的直接证据。

### 8.3 可以通过新增实验变成强 claim 的写法

```text
We empirically show that top-K crop sets are unstable under annotator
resampling and across preference groups. On our benchmark, the average
Jaccard@5 between expert and non-expert top-5 sets is X, and the
worst-group win rate of MOS-optimized croppers drops by Y%.
```

这需要本项目实际统计 X/Y。

## 9. 推荐实验：验证 top-K 偏差

### 9.1 如果能拿到 GAICD raw ratings

每张图、每个 crop 有 7 个评分时：

```python
for image in dataset:
    for repeat in range(B):
        group_a = sample(annotators, m)
        group_b = remaining_or_sample(annotators, m)
        mos_a = mean_rating(group_a)
        mos_b = mean_rating(group_b)
        topk_a = top_k(mos_a, k)
        topk_b = top_k(mos_b, k)
        record_jaccard(topk_a, topk_b)
        record_kendall(rank(mos_a), rank(mos_b))
```

报告：

- `Jaccard@5/10` 均值和置信区间。
- `Top1Agreement`。
- `Top1 in Other Top10`。
- `KendallTau`。
- 按图像类型分桶：人像、风景、商品、建筑、复杂多人、多主体关系。

### 9.2 如果只有 MOS，没有 raw ratings

可以做新标注：

- 选 `300-500` 张图。
- 每张图取 `20-40` 个候选 crop：GAIC anchors + model outputs + negatives。
- 每个 pair 至少 `5-7` 票。
- 标注者分为 expert / amateur / target user。
- 计算 group-wise BT score 和 top-K。

### 9.3 如果只想快速做 proof-of-concept

使用 CPC/FCDB 风格：

- 每张图选 `reference crop` 和 `6-10` 个 candidate crops。
- 问：candidate 是否比 reference 更好？
- 用 `Improve@1`、`Improve@3`、`NoHarm@3` 做小规模 benchmark。
- 附带 `Jaccard@K` bootstrap 说明 top-K 不稳定。

## 10. 风险与应对

| 风险 | 具体表现 | 应对 |
|---|---|---|
| “更好”定义过弱 | 模型只要裁掉边缘就能赢 | 增加 strong baseline reference；报告 hard split |
| 群体差异被噪声误判 | 小样本 group top-K 不稳定 | 使用 bootstrap CI、最小票数、agreement filtering |
| pairwise 标注成本高 | `O(n^2)` 对比爆炸 | 两阶段标注、active sampling、BT adaptive comparison |
| VLM judge 偏差 | VLM 偏好不等于人类群体偏好 | VLM 只做预筛/teacher，官方指标保留人评 |
| 与 CPC/GAICD novelty 不清 | 审稿人认为只是 pairwise ranking | 强调 reference-conditioned improvement + group robustness + no-harm |
| 不适合所有裁剪任务 | subject-aware/aspect-ratio-aware 约束不同 | benchmark 拆 task，context 作为输入 |

## 11. 最终建议

### 11.1 是否值得做？

值得。原因：

- 它顺着现有领域趋势：从单 GT 到 MOS/top-N，再到 pairwise/set/multi-crop。
- 它能回应真实痛点：构图搜索空间大、top-K 易受偏好和标注噪声影响。
- 它能与 RIGCrop 的 semantic/relation/action 中间态天然结合，尤其适合做“为什么这个 crop 更好”的可解释 benchmark。
- 它有机会形成区别于 GAICD/CPC 的新协议，而不是只在老指标上卷 `Acc5/Acc10`。

### 11.2 最推荐的 benchmark 名称/定位

可选名称：

- **CIBench**：Composition Improvement Benchmark
- **BetterCropBench**
- **Preference-aware Composition Improvement Benchmark**
- **Group-Robust Image Cropping Benchmark**

推荐定位：

```text
A benchmark for reference-conditioned, preference-aware image composition improvement.
```

中文：

```text
一个面向参考构图、偏好感知、群体鲁棒的图像构图改进 benchmark。
```

### 11.3 最小可行版本

MVP：

- `1,000` images。
- 每张图 `1` 个 reference crop + `12-24` candidates。
- 每张图 `20-40` pairwise comparisons。
- 每个 pair `5` votes。
- 至少两个群体：摄影/设计经验组 vs 普通用户组。
- 官方指标：`Improve@1/3`、`NoHarm@3`、`TieAwareUtility@3`、`GroupGap`、`WorstGroupWin`。
- 兼容指标：GAICD-style `AccK/N` 和 CPC-style pairwise accuracy。

增强版：

- `3,000-5,000` images。
- 三个 use-case context：general、social share、commercial/product。
- 每张图保留 vote distribution 和 ambiguity split。
- 发布 raw pair votes、BT scores、group-specific rankings。

## 12. 参考文献与链接

1. Chen, Y.-L., Huang, T.-W., Chang, K.-H., Tsai, Y.-C., Chen, H.-T., & Chen, B.-Y. (2017). *Quantitative Analysis of Automatic Image Cropping Algorithms: A Dataset and Comparative Study*. WACV. [PDF](https://www.cmlab.csie.ntu.edu.tw/~robin/docs/wacv17.pdf), [Dataset](https://github.com/yiling-chen/flickr-cropping-dataset)
2. Wei, Z., Zhang, J., Shen, X., Lin, Z., Mech, R., Hoai, M., & Samaras, D. (2018). *Good View Hunting: Learning Photo Composition from Dense View Pairs*. CVPR. [PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Good_View_Hunting_CVPR_2018_paper.pdf)
3. Zeng, H., Li, L., Cao, Z., & Zhang, L. (2019). *Reliable and Efficient Image Cropping: A Grid Anchor Based Approach*. CVPR. [PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf)
4. Jia, G., Huang, H., Fu, C., & He, R. (2022). *Rethinking Image Cropping: Exploring Diverse Compositions from Global Views*. CVPR. [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Jia_Rethinking_Image_Cropping_Exploring_Diverse_Compositions_From_Global_Views_CVPR_2022_paper.pdf)
5. Yang, M., et al. (2022). *Personalized Image Aesthetics Assessment With Rich Attributes*. CVPR. [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Personalized_Image_Aesthetics_Assessment_With_Rich_Attributes_CVPR_2022_paper.pdf), [arXiv](https://arxiv.org/abs/2203.16754)
6. Wang, C., Niu, L., Zhang, B., & Zhang, L. (2023). *Image Cropping With Spatial-Aware Feature and Rank Consistency*. CVPR. [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Image_Cropping_With_Spatial-Aware_Feature_and_Rank_Consistency_CVPR_2023_paper.pdf)
7. Lee, J., et al. (2025). *Cropper: Vision-Language Model for Image Cropping through In-Context Learning*. CVPR. [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Cropper_Vision-Language_Model_for_Image_Cropping_through_In-Context_Learning_CVPR_2025_paper.pdf)
8. Murray, N., Marchesotti, L., & Perronnin, F. (2012). *AVA: A Large-Scale Database for Aesthetic Visual Analysis*. CVPR. [CVF entry](https://openaccess.thecvf.com/content_cvpr_2012/html/Murray_AVA_A_Large-Scale_2012_CVPR_paper.html)
9. Kong, S., Shen, X., Lin, Z., Mech, R., & Fowlkes, C. (2016). *Photo Aesthetics Ranking Network with Attributes and Content Adaptation*. ECCV. [Project](https://github.com/aimerykong/deepImageAestheticsAnalysis)

