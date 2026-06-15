# InstantRetouch 论文技术分析汇报

论文：InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space  
出处：CVPR 2026, pp. 8216-8226  
作者：Jiarui Wu, Yujin Wang, Ruikang Li, Fan Zhang, Mingde Yao, Tianfan Xue  
官方链接：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_InstantRetouch_Efficient_and_High-Fidelity_Instruction-Guided_Image_Retouching_with_Bilateral_Space_CVPR_2026_paper.html)  
补充材料：[CVPR Supplementary Material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wu_InstantRetouch_Efficient_and_CVPR_2026_supplemental.pdf)

## 1. 研究概述

InstantRetouch 是一种面向文本指令的高保真图像修图方法，目标是在理解自然语言修图意图的同时，尽量避免改变原图的主体结构、纹理细节和身份信息。

与直接生成图像的 diffusion 编辑方法不同，InstantRetouch 根据输入图像和文本指令预测低分辨率的 **bilateral grid**，再将其应用到原始高清图像上完成颜色和影调变换。该设计兼顾了文本指令理解、内容保真和高分辨率处理效率。

## 2. 研究背景与核心矛盾

当前 AI 修图存在一个核心矛盾：

- 传统 LUT、曲线、HDRNet 等方法推理速度快、内容保真度高，但缺乏对复杂自然语言指令的理解能力。
- Diffusion 模型具备较强的文本理解和视觉生成能力，但在修图场景中容易引入内容漂移，并且推理延迟较高。

InstantRetouch 的核心思路是：  
**利用 diffusion 模型提供语义和审美先验，同时使用 bilateral grid 执行稳定、高效、保真的颜色与影调编辑。**

## 3. 论文解决的问题

### 3.1 任务定义：instruction-guided retouching

输入：

- 一张原图
- 一句文字指令

输出：

- 一张修图后的照片

例子：

- “让照片更暖一点，像黄金时刻。”
- “让画面更梦幻、更柔和。”
- “让人物脸部更亮，背景更有电影感。”
- “增强天空和海水，让画面更像旅行杂志。”

需要强调的是，本文讨论的 retouching 并不是“换背景、添加物体、重绘人物”等生成式内容编辑，而是更接近摄影后期中的以下操作：

- 曝光调整
- 白平衡调整
- 对比度调整
- 饱和度调整
- 色调风格调整
- 局部明暗和氛围调整

### 3.2 现有 diffusion 编辑方法的不足

Diffusion 图像编辑模型具备较强的语义理解和生成能力，但在照片修图任务中存在以下不足：

| 问题 | 具体表现 | 对修图任务的影响 |
|---|---|---|
| 内容漂移 | 人脸、衣服纹理或背景物体发生非预期变化 | 修图应保留原图内容，不应重新创作 |
| 推理延迟高 | 多步采样导致 720p 输入也可能需要数秒到数十秒 | 不适合实时预览和高分辨率批处理 |
| 高分辨率处理困难 | 部分模型无法直接处理 2K/4K 图像 | 摄影后期通常需要高清输出 |
| 可控性不稳定 | 同一句指令可能多次结果差异很大 | 专业修图需要可预测、可回退 |

论文认为，修图本质上应该只改 **photometric information**，也就是颜色、亮度、影调，而不是改 geometry、texture 和 content。

因此，本文将修图任务限定为颜色和影调层面的外观调整，而不是开放式图像重绘。

## 4. 核心方法思想

InstantRetouch 的整体流程如下：

```text
文字指令 + 输入图片
        |
        v
低分辨率 diffusion 分支理解修图意图
        |
        v
预测低分辨率 bilateral grid
        |
        v
在原始高清图上查表并应用局部仿射颜色变换
        |
        v
输出高清、保真的修图结果
```

该设计带来以下优势：

- 模型可以理解文字指令。
- 输出仍然是对原图做颜色/影调变换。
- 不直接生成新像素，因此能降低内容漂移风险。
- bilateral grid 表示紧凑，适合处理 4K 高分辨率图片。

## 5. Bilateral Grid 机制说明

Bilateral grid 可以理解为一种结合空间位置、颜色信息和引导信息的局部颜色变换网格。

普通 LUT 主要根据像素颜色进行映射：  
输入 RGB 颜色对应一个输出 RGB 颜色。

Bilateral grid 进一步引入位置和引导信息：  
像素所在位置、亮暗层级以及邻域结构都会影响最终变换。

因此，bilateral grid 可以支持更细粒度的局部修图：

- 天空可以更蓝，但人脸不一起变蓝。
- 背景可以压暗，但主体保持明亮。
- 高光可以压住，阴影可以提亮。
- 边缘区域可以保持较好的结构连续性，减少明显的边界模糊。

在这篇论文里，每个 grid 单元里存的不是一个颜色，而是一个 **3 x 4 仿射矩阵**。对每个像素来说，模型会查到一个局部矩阵，然后做：

```text
新 RGB = 局部仿射矩阵 x [R, G, B, 1]
```

也就是说，每个像素都会根据自身位置和颜色信息获得一个局部颜色变换公式。

## 6. 方法整体结构

论文的模型由两个分支组成。

### 6.1 低分辨率 one-step diffusion 分支

作用：理解图片和文字指令。

它接收：

- 输入图片的低分辨率表示
- 文本指令
- diffusion 噪声 latent

它输出：

- 低分辨率的编辑理解结果
- 给 bilateral adapter 使用的特征

这个分支的重点不是输出最终高清图，而是提供“应该怎么修”的语义判断。

### 6.2 高分辨率 bilateral processing 分支

作用：把修图真正应用到原始高清图上。

它包含三个关键模块：

| 模块 | 做什么 | 作用 |
|---|---|---|
| Light Bilateral Adapter | 根据 diffusion 特征预测 affine bilateral grid | 把文字理解转成调色参数 |
| GuideNet | 为原图生成 guide map | 决定每个像素去 grid 哪里查表 |
| Guided Slice & Apply | 查出每个像素的仿射矩阵并应用 | 在高清图上完成最终修图 |

这也是它快的原因：重计算都在低分辨率或小 grid 上完成，高清图只做查表和矩阵变换。

## 7. 训练流程

论文的训练可以分成四步理解。

### 7.1 构建训练数据

作者构建了大约 200K 个训练三元组：

```text
(输入图 x, 高质量修图目标 x*, 文字指令 cT)
```

数据构造方式：

1. 先收集高质量图片作为目标图 `x*`。
2. 用随机摄影后期操作把目标图“降质”成输入图 `x`。
3. 这些操作包括曝光、gamma、白平衡、对比度、曲线、饱和度、阴影/高光、HSL 等。
4. 用 Grounding-SAM 和软 mask 生成局部区域，让训练数据包含局部修图。
5. 用 Qwen2.5-VL-72B 根据前后图生成自然语言修图指令。
6. 用规则过滤掉“加物体、删物体”等内容编辑指令，只保留修图类指令。

这样做的好处是：  
不需要人工逐张写修图指令，也能得到大量“前图、后图、指令”训练样本。

### 7.2 训练 multi-step diffusion teacher

先训练一个比较重的 diffusion teacher。

它类似 InstructPix2Pix 的思路：  
输入原图和文字指令，学习生成目标修图图。

这个 teacher 质量较好，但有两个问题：

- 推理慢
- 仍可能有内容漂移

所以它不是最终模型，而是知识来源。

### 7.3 蒸馏成 one-step bilateral student

接下来，作者把 teacher 的能力蒸馏到一个更快的 student 里。

student 不再多步生成图像，而是一次前向推理预测 bilateral grid。

这里用了几个损失：

| 损失 | 作用 | 简单理解 |
|---|---|---|
| VSD loss | 把 diffusion teacher 的审美和生成先验传给 student | 继承教师模型的审美先验和编辑方向 |
| Data loss | 让低分辨率输出接近目标图 | 保证训练稳定 |
| Prompt alignment loss | 强化文字指令和结果的一致性 | 指令说“更暖”，结果就应该真的更暖 |
| Bilateral loss | 约束最终高清输出质量 | 让 bilateral grid 结果自然、平滑、不溢色 |

### 7.4 两阶段训练策略

直接训练整个模型不稳定，所以论文采用两阶段：

| 阶段 | 训练内容 | 目的 |
|---|---|---|
| Stage 1 | 只训练低分辨率 one-step diffusion 分支 | 先学会理解指令和整体风格 |
| Stage 2 | 联合训练 diffusion 分支和 bilateral 分支 | 再学会生成高清可用的局部调色参数 |

这个策略可以理解为：

> 先学“我要怎么修”，再学“怎么把修图安全地应用到高清原图上”。

## 8. Prompt Alignment Loss 的作用

修图指令通常具有语义模糊性，例如：

- “更温暖”
- “更梦幻”
- “更电影感”
- “更清透”

这些词很难用像素误差直接监督。  
所以论文把一句复杂指令拆成多个简单属性。

例子：

```text
指令：让照片更温暖、更复古、更柔和

可拆成：
- temperature: warm
- style: vintage
- contrast: soft
```

然后用 CLIP 比较输出图像和正负文本提示的相似度：

```text
warm image  vs  cool image
bright image  vs  dark image
vintage image  vs  modern image
```

这样模型就更容易知道指令的方向。

如果没有这个 loss，模型可能生成一张“看起来还行”的图，但不一定真的按用户要求修。

## 9. 推理流程

实际使用时，流程非常简洁：

```text
输入：原图 + 文字指令

1. 模型低分辨率读取图片和文字
2. 一次前向推理预测 bilateral grid
3. GuideNet 为原图生成 guide map
4. Guided Slice 从 grid 中查出每个像素的仿射矩阵
5. Apply 把矩阵应用到原始 RGB 像素
6. 得到最终高清结果
```

推理时不需要多步 diffusion 采样，也不需要逐图优化。

## 10. 实验结果分析

### 10.1 Benchmark：iRetouch

论文提出了新的 iRetouch benchmark：

- 500 组真实 before-after 修图样本
- 来自 Adobe Lightroom community
- 指令经过人工修订
- 覆盖人像、风景、动物、静物、街景等
- 指令包括全局调整、电影感、梦幻感、黄金时刻、局部主体增强、背景压暗、天空/水面增强等

### 10.2 评价维度

论文不仅评估主观视觉效果，还从三类维度进行综合比较：

| 维度 | 评估内容 | 重要性 |
|---|---|---|
| Fidelity | 是否保留原图结构、纹理、身份 | 修图应避免改变原始主体和场景内容 |
| Instruction Following | 是否按文字指令修改 | 模型需要准确执行用户意图 |
| Efficiency | 推理速度和高分辨率能力 | 产品需要实时预览和 4K 输出 |

### 10.3 关键数字

在 iRetouch benchmark 上，论文报告：

| 方法 | 720p 速度 | 4K 速度 | SSIM | Overall Score |
|---|---:|---:|---:|---:|
| InstructPix2Pix | 4.632s | 不支持/未报告 | 0.742 | 7.34 |
| Step1X-Edit | 57.932s | 不支持/未报告 | 0.706 | 8.06 |
| GPT-Image-1 | 15.427s | 不支持/未报告 | 0.505 | 8.32 |
| Qwen-Image | 7.720s | 不支持/未报告 | 0.689 | 8.39 |
| FLUX.1-Kontext-Pro | 10.235s | 不支持/未报告 | 0.802 | 8.12 |
| Gemini-2.5-Flash | 14.440s | 不支持/未报告 | 0.676 | 8.74 |
| InstantRetouch | 0.065s | 0.068s | 0.989 | 8.54 |

从结果可以看出：

- InstantRetouch 的 Overall Score 不是最高，Gemini-2.5-Flash 的 Overall Score 略高。
- 但 InstantRetouch 在推理速度和内容保真度上优势明显。
- 4K 下仍然约 0.068s，表明该方法适用于高分辨率修图和实时预览场景。
- 对修图任务来说，保真度非常关键，所以它的综合产品价值很高。

### 10.4 消融实验分析

论文的消融实验说明了三个关键点：

| 对比项 | 结果含义 |
|---|---|
| 只用 bilateral grid | 保真度高，但复杂指令理解能力不足 |
| 只用 diffusion | 指令执行效果较好，但内容容易漂移，推理速度较慢 |
| 完整模型 | 同时兼顾语义理解、保真度和速度 |

损失函数消融也说明：

- 加入 VSD 后，编辑质量明显提升。
- 再加入 prompt alignment loss 后，指令遵循能力进一步提升。
- bilateral loss 中的平滑正则和 RGB overflow penalty 能减少空间伪影和颜色溢出。

## 11. 细粒度控制能力

InstantRetouch 还支持对修图强度进行连续调节，这一点对产品交互较为重要。

因为最终编辑是仿射变换，所以可以用一个强度系数 `s` 控制效果：

```text
s = 0：原图
s = 0.5：半强度修图
s = 1：默认效果
s > 1：更强效果
```

该机制适合对应到产品中的强度控制滑杆：

- “效果强度”
- “电影感强度”
- “暖色强度”
- “梦幻感强度”

补充材料还提到可以通过 bilateral grid blending 支持区域控制，例如前景和背景分别应用不同指令。

## 12. 和传统方法、生成式方法的区别

| 方法类型 | 代表 | 优点 | 缺点 | InstantRetouch 的位置 |
|---|---|---|---|---|
| 传统 LUT/曲线 | Lightroom、3D LUT | 推理快、稳定、可控 | 缺乏自然语言理解能力，局部表达能力有限 | 借鉴其稳定执行方式 |
| diffusion 图像编辑 | InstructPix2Pix、Gemini、FLUX | 文本理解能力强，视觉生成能力强 | 推理慢，可能改变原始内容 | 借鉴其语义先验 |
| InstantRetouch | 本论文 | 推理快、保真度高、支持文本指令 | 主要适合调色/影调，不适合换物体 | 在生成式能力和修图保真之间取得平衡 |

## 13. 优势总结

### 13.1 对技术的优势

- 使用 bilateral space，天然限制模型只做颜色和影调类编辑。
- 使用 diffusion teacher 蒸馏，保留了大模型的审美和语言理解能力。
- 一次前向推理即可完成，不需要多步采样。
- 低分辨率预测，高分辨率应用，适合 4K 图片。
- 支持强度控制和区域控制扩展。

### 13.2 对产品的优势

- 适合做“文字调色”功能。
- 适合做实时预览和滑杆调节。
- 相比纯生成式编辑，更有利于保持人脸身份和图像细节。
- 可以和现有编辑器参数、LUT、mask 系统结合。
- 对移动端、高分辨率批处理、摄影工作流更友好。

## 14. 局限性

这篇论文也有清晰边界。

### 14.1 不适合大范围生成式编辑

它主要做颜色和影调调整，不适合：

- 加一个物体
- 删除路人
- 换背景
- 改衣服形状
- 改人物动作

因为 bilateral grid 本质上不能创造新结构。

### 14.2 依赖数据构造质量

训练数据是通过“高质量目标图 -> 降质输入图 -> 生成指令”构造的。  
如果降质流程和真实用户照片差异很大，模型可能泛化不足。

### 14.3 指令理解仍有边界

它适合“更暖、更亮、更电影感、更梦幻”这类摄影后期指令。  
如果用户给的是复杂语义指令，例如“让这个人看起来像 90 年代摇滚明星”，模型可能不如通用 diffusion 编辑。

### 14.4 评价仍有主观性

论文使用 GPT-4o 参与编辑质量评分，也做用户偏好研究。  
这比单纯 PSNR 更合理，但审美评价仍然会受到数据、模型和人群偏好的影响。

## 15. 产品落地分析

### 15.1 适合做的功能

| 功能 | 是否适合 | 原因 |
|---|---|---|
| 文本指令调色 | 高度适合 | 指令到影调/颜色变换正是论文目标 |
| AI 滤镜生成 | 高度适合 | 可以把文字风格转化为可调滤镜效果 |
| 高分辨率批量修图 | 高度适合 | 4K 速度优势明显 |
| 人像轻修 | 适合 | 保身份，但最好配合人脸/肤色保护 |
| 复杂局部换物体 | 不适合 | bilateral grid 不生成新内容 |
| 祛痘/磨皮/液化 | 不直接适合 | 需要人像专用模型 |

### 15.2 推荐产品架构

```text
用户上传图片
  |
  |-- 图像分析：场景、人像、曝光、噪声、肤色、天空
  |
  |-- 指令理解：把用户语言判断为调色类/生成类/人像类
  |
  |-- 如果是调色/影调类：
  |      调用 InstantRetouch 类模型
  |      输出高清结果 + 强度滑杆
  |
  |-- 如果是局部人像类：
  |      叠加人脸 mask / 肤色保护 / 美颜模型
  |
  |-- 如果是换背景/加物体类：
         转给 diffusion inpainting / generative editing
```

### 15.3 产品交互建议

建议给用户这些控件：

- 文字输入框：描述想要的风格。
- 强度滑杆：控制修图幅度。
- 局部选择：主体、背景、天空、人脸。
- 前后对比：快速检查内容保真和视觉变化。
- 锁定区域：例如锁定人脸、锁定肤色、锁定文字。
- 历史记录：每次指令都能撤销。

## 16. 汇报用 8 页 PPT 提纲

### 第 1 页：论文基本信息

- 论文标题、作者、CVPR 2026
- 关键词：instruction-guided retouching、bilateral grid、diffusion distillation
- 核心定位：支持文本指令的高保真颜色与影调修图

### 第 2 页：问题背景

- 传统 LUT/曲线：速度快，但缺乏文本指令理解能力
- Diffusion 编辑：文本理解能力强，但推理慢且存在内容漂移风险
- 修图的核心要求：保留原图，只改颜色和影调

### 第 3 页：核心思想

- 不直接生成像素
- 预测 bilateral grid
- 在高清原图上应用局部仿射颜色变换

### 第 4 页：模型结构

- 低分辨率 one-step diffusion 分支
- Light Bilateral Adapter
- GuideNet
- Guided Slice & Apply

### 第 5 页：训练方法

- 200K 训练三元组
- multi-step diffusion teacher
- VSD 蒸馏
- prompt alignment loss
- bilateral loss
- 两阶段训练

### 第 6 页：实验结果

- iRetouch benchmark：500 组真实 Lightroom before-after
- 0.065s 到 0.068s
- 4K 仍可快速处理
- SSIM 0.989，Overall Score 8.54

### 第 7 页：优势与局限

- 优势：推理快、保真度高、支持文本指令、可控性较强
- 局限：不适合换物体、删物体、强生成编辑

### 第 8 页：产品启发

- 适合做文字调色、AI 滤镜、高分辨率批量修图
- 最好和人像模型、生成式编辑模型分工使用
- 更适合作为高保真文本调色与影调编辑模块，而不是通用生成式图像编辑模型

## 17. 汇报结论

InstantRetouch 的价值在于，它没有把 AI 修图简单等同于生成式编辑，而是回到摄影后期的本质：在保留原图内容的前提下，调整颜色、亮度和氛围。

它利用 diffusion 模型理解“梦幻、电影感、黄金时刻”等审美指令，并通过 bilateral grid 这一紧凑、可控、保真的参数空间执行编辑。因此，它在速度和内容保持上明显优于多数通用生成式编辑模型，同时又比传统 LUT 具备更强的语言交互能力。

从产品角度看，InstantRetouch 更适合作为“文本调色、智能滤镜、高保真自动修图”的核心模块，而不是用于换背景、添加物体、强人像改造等生成式任务。

## 18. 参考资料

- Wu, J., Wang, Y., Li, R., Zhang, F., Yao, M., & Xue, T. (2026). [InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_InstantRetouch_Efficient_and_High-Fidelity_Instruction-Guided_Image_Retouching_with_Bilateral_Space_CVPR_2026_paper.html). CVPR 2026.
- [InstantRetouch Supplementary Material](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wu_InstantRetouch_Efficient_and_CVPR_2026_supplemental.pdf)
