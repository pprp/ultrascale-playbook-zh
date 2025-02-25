# **《超大规模操作手册：在 GPU 集群上训练 》Part1(基础概念,DP,TP)**

作者：nanotron

校正：pprp

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image.png)


*我们在最多 512 个 GPU 上进行了超过 4000 次扩展实验，测量了吞吐量（标记大小）和 GPU 利用率（标记颜色）。请注意，这两个指标均按模型大小进行了归一化处理，以便更直观地展示。*

# Overview

> 成排的GPU集群发出整齐划一的轰鸣，这正是训练当代顶尖AI模型所需的场景——一场算力交响曲的演绎，而这般景象在不久前还只是顶尖实验室的专利。开源运动虽然打破了技术垄断，却未能完全消弭核心壁垒。如今，任何人都能自由下载最新的Llama或DeepSeek模型，研读其技术文档和实验报告。但真正的精要所在——那套驾驭GPU集群训练庞然智能体的工程体系，那些在分布式系统中精妙调谐万千计算单元的核心技艺——仍如深藏云端的圣殿，其奥义散落在晦涩难懂的学术论文与彼此割裂的私有代码库之间，构筑着难以逾越的技术鸿沟。

这本开源书籍旨在引领变革。从基础出发，我们将引导您掌握将大型语言模型训练从单 GPU 扩展到数十、数百甚至数千 GPU 所需的知识，并通过实际代码示例和可复现的基准测试来阐述理论，使内容更加自然易懂。

随着训练模型所用的集群规模不断扩大，人们发明了多种技术，包括数据并行(Data Parallel, DP)、张量并行(Tensor Parallel, TP)、流水线并行(Pipeline Parallel, PP)和上下文并行(Context Parallel，以及 ZeRO 或内核融合(Kernel Fusion)，以确保 GPU 始终高效运行。这大大缩短了训练周期，并充分利用了昂贵的硬件资源。

更有甚者，随着 AI 训练扩展的挑战超越了仅仅构建初始模型，团队发现，在特定数据上对大型模型进行微调往往能取得最佳效果，通常采用相同的分布式训练技术。 

在本书中，我们将逐步介绍所有这些技巧——从最简单的到最精致的——同时保持一个主线故事，以便理解每种技巧的来源。

我们假设您对当前LLM架构有一些基本的了解，并对深度学习模型的训练方法有所熟悉，但分布式训练可能对您来说还是陌生的。如有需要，您可以在 DeepLearning.ai 或 PyTorch 教程部分找到有关模型训练基础知识的优质课程。

本书可以看作是我们关于数据预处理预训练的第一篇博客（即所谓的“FineWeb blog post”[1]）的续篇。阅读完这两篇博客后，您应该已经掌握了理解当今如何构建LLMs所需的大部分核心知识，只是缺少一些关于数据融合和架构选择的细节来完善整个方案（敬请期待第三部分……）。

这本书立足于以下三个基本原则：

**一、 快速了解理论和概念**：在深入代码和实验之前，我们希望先了解每种方法在宏观层面上的运作原理及其优缺点。您将学习到哪些部分的语言模型会消耗内存资源，以及这种消耗通常发生在训练的哪个阶段。您还将了解到，我们可以通过并行化模型来克服内存限制，并通过增加 GPU 的规模来提升处理能力。因此，您将明白如何使用以下组件来计算 Transformer 模型的内存分解情况：（编者注：访问[https://huggingface.co/spaces/nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 进行体验）

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%201.png)

我们还开发了一个工具，可以用来预测训练过程中的内存消耗情况：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%202.png)

**二、 清晰的代码实现：**理论固然重要，但在实际编码过程中，我们会遇到各种边缘情况和关键细节。因此，我们会在可能的情况下提供实现参考链接。根据具体情况，我们可能会引用两种代码示例：

- Picotron [2] 代码库是为教育而构建的，因此它通常在单个, self-contained 的短文件中实现概念。
- 另一方面，为了查看可用于生产的代码，我们将参考 nanotron [3] 的实现，这是 Hugging Face 在生产训练中使用的代码库。

**三、 真实训练效率基准**：最后，如何根据您的硬件设施（例如芯片类型、互连等）实际扩展您的LLM训练规模，我们无法给出一个统一的解决方案。不过，我们将提供一种评估多种配置的方法，这正是我们在我们的集群上所进行的实践！我们进行了超过 4100 次分布式实验（包括测试运行超过 16k 次），并使用了最多 512 个 GPU 来扫描多种可能的分布式训练架构和模型规模。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%203.png)

如您所见，有很多内容需要探讨。在深入分布式训练的细节之前，我们先简要了解一下本书将要讨论的挑战。

# High level overview  **高级概览**

本书将探讨的所有技术都旨在解决以下三个主要挑战之一或多个，这些挑战将在全书的学习过程中不断遇到：

- **内存使用**：这是一个严格的限制——如果训练步骤无法装入内存，则训练无法继续进行
- **计算效率**：我们希望硬件大部分时间用于计算，因此需要降低数据传输时间或等待其他 GPU 执行任务的时间，以提升效率。
- **通信开销**：我们希望尽量减少通信开销，因为它会导致 GPU 闲置。为此，我们将努力最大化利用节点内（快速）和节点间（较慢）的带宽，并尽可能地将通信与计算过程重叠。

在许多地方，我们会发现可以用其中之一（计算、通信、内存）来交换另一个（例如，重新计算或张量并行）。找到合适的平衡是扩展训练的关键。这样的表述更加自然，易于理解。

由于这本书内容非常丰富，我们特别制作了一张速查表，方便您浏览书籍并掌握核心要点。在您面对挑战时，请将它牢记于心！

速查表：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%204.png)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%205.png)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%206.png)

并行策略速查表：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%207.png)

术语表：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%208.png)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%209.png)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2010.png)

## 第一步：在单个GPU上训练

让我们从快速回顾模型训练的基础知识开始，然后再扩展到多个 GPU。当模型在单个 GPU 上训练时，通常包括以下三个步骤：

1. 通过模型传递输入以产生输出的前向传播
2. 反向传播计算梯度
3. 利用梯度更新参数

它看起来大致如此：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2011.png)

在这个图中，

- 顶部行的方框可以看作是模型内部连续的层（最后一行也是如此）。
- 红色方框代表这些层在反向传播过程中计算得到的关联梯度。

批处理大小（bs）是模型训练中的一个关键超参数，它不仅影响模型的收敛速度，还会对模型的处理能力产生影响。

在训练初期，小批量（batch size）可以快速地在training landscape中移动，从而快速达到一个较优的学习点。然而，在训练的后期，小批量会导致梯度保持较高的噪声，模型可能无法收敛到最优的最终性能。

在另一个极端，大批量虽然能够提供非常准确的梯度估计，但会降低每个训练样本的利用率，导致收敛速度变慢，并且可能会浪费计算资源。关于这个话题的早期讨论，你可以参考OpenAI关于大批量训练的论文或者MiniMax技术报告[4]

Batch size会影响在特定文本数据集上训练所需的时间：batch越小，训练相同样本所需的优化器步数就越多。优化器步数（在计算时间上）成本较高，因此与使用较大的batch size相比，总的训练时间会增加。**但要注意**，batch size通常可以在最佳batch size附近进行较大调整，而对模型性能的影响不大，也就是说，模型性能对确切batch size值的敏感性通常在最佳batch size附近较低。

> 编者注：在训练机器学习模型时，选择一个合适的batch size是很重要的。尽管如此，当batch size接近于某个最佳值时，模型的最终性能通常不会对batch size的精确值特别敏感。这意味着，即使batch size稍微有所调整，只要在最佳值附近，模型的性能也不会有显著变化。这种情况使得在实际应用中，我们可以在一定范围内灵活地选择batch size，而不必过于担心对模型性能的影响。

在LLM预训练社区中，batch size通常以 token 数量来报告，而不是样本数量（bst=batch size tokens），这使得训练数量通常与训练时使用的具体输入序列长度无关。

在最简单的情况下，如果是在单机上进行训练，可以通过以下方式从模型输入序列长度（seq）计算出 bs（样本数量）和 bst(batch size tokens)：

$$
bst=bs\times seq
$$

从这里开始，我们将以样本为单位展示batch size的公式，但您始终可以通过将其乘以序列长度来获得其以token为单位的对应值。

最近的大型语言模型训练的理想点通常是每batch大约 4-60 million token per batch。batch size和训练语料库多年来一直在稳步增加：Llama 1 使用约 4M token的batch size训练了 1.4T tokens，而 DeepSeek 使用约 60M token的batch size训练了 14 T tokens。

**在将模型训练扩展到大规模batch时，我们面临的首要挑战是内存不足问题。当我们的 GPU 内存不足以容纳目标batch大小时，我们该如何应对？**

让我们先快速了解导致我们最初出现内存不足问题的原因。这将帮助我们获得一些关于训练模型内存需求的宝贵直觉。

### Transformer的内存使用情况

在训练神经网络模型时，人们会在内存中存储多个项目：

- Model weights 模型权重
- Model gradients 模型梯度
- Optimizer states 优化器状态
- Activations needed to compute the gradients 计算梯度所需的激活数量

> 你可能会觉得对于一个模型，理论上应该可以精确计算出内存需求，但实际中存在一些额外的内存占用因素，这使得精确计算变得困难：
>
> - CUDA 内核通常需要 1-2GB 的 GPU 内存，您可以通过运行 `import torch; torch.ones((1, 1)).to("cuda")` 并检查 `nvidia-smi` 中的 GPU 内存来快速确认。
> - 来自缓冲区、中间结果以及因碎片化而无法使用的部分内存的休息内存使用情况
>
> 我们将忽略以上两者，因为它们通常很小且为常数项。

这些项目以张量（Tensor)形式存储，具有不同的形状和精度。

- 形状由超参数确定，如batch size bs、序列长度 seq、模型隐藏维度hid、注意力头head、词汇量大小以及我们稍后将要看到的潜在模型分片(model sharding)。
- 精度指的是 FP32、BF16 或 FP8 等格式，分别需要 4、2 或 1 个字节来存储张量中的每个单独值。

我们将在混合精度训练部分全面讨论不同的精度及其权衡，现在只需记住，这些不同格式的内存需求将不同，这将影响我们需要存储的项目内存使用。

所以，我如何快速确定这些变量的内存使用情况？一种简单的方法是进行经验测量。

### **分析内存使用 Profiling**

使用 PyTorch Profiler，我们可以了解在整个训练过程中内存是如何分配的。我们可以看到，内存利用率不是一个静态的东西，而是在训练过程中以及训练步骤中变化很大。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2012.png)

显然，第一步看起来与后续步骤非常不同，但让我们首先看看一个步骤的一般结构：

- 在**前向**传播时**激活值迅速增加**，
- 在**反向**传播时**梯度累积**，
- 随着反向传播的进行，用于**计算梯度的存储激活值逐渐被清除。**
- 执行优化步骤，在此期间我们需要**所有的梯度**，
- **更新优化器状态**，
- 开始下一次前向传播。

为什么第一步看起来不同：激活值迅速增加然后保持一段时间。在这个第一步中，torch 缓存分配器做了很多准备工作，**准备内存分配以加快后续步骤**，这样就不需要在之后**搜索空闲内存块**（参见 Zach 的博客[5]）。在第一步之后，我们还看到优化器状态的出现，这通常会抵消进一步训练步骤的内存使用。

现在我们已经对内存有了初步的了解，让我们看看如何扩大训练规模通常是一个在保持这些各种项目（激活、参数、梯度、优化器状态）内存需求在 GPU 内存限制内的问题，即**最大化计算效率。**

**权重/梯度/优化器状态内存**

让我们从我们列表中的前 3 个项目开始：模型的权重Weight、梯度Gradient和优化器Optimizer状态。实际上，我们可以相当容易地估算出它们所需的内存。

对于简单的 Transformer LLM，参数数量由以下公式[6]给出：

$$
N=h\times v+L\times(12\times h^2+13\times h)+2 \times h
$$

在该方程中， *h* 是隐藏维度， *v* 是词汇量大小， *L* 是模型中的层数。

注意，观察方程我们可以看到，在大的隐藏维度下将占主导地位的项是 $h^2$ 项，因为它是我们调整参数时唯一一个呈二次增长的项。

内存需求仅是参数数量乘以每个参数的字节数。在传统的全精度（FP32）训练中，参数和梯度都需要 4 个字节，而如果使用 Adam 优化器，则需要存储动量和方差，这为每个参数额外增加了两个 4 个字节的存储。总之：

$$
m_{params} = 4 \times N \\
m_{grad} = 4 \times N \\
m_{opt} = (4 + 4) \times N
$$

现在让我们看看如果我们使用更低的精度会发生什么变化。出于稳定性的原因，我们通常不使用完全的低精度训练，而是使用称为“混合精度”, 即同时采用更高和更低精度的组合。

现在混合精度训练的默认做法是，对于大多数计算通常使用 BF16，每个参数和梯度需要 2 个字节，以及额外的 FP32 模型权重和梯度的副本，因此每个参数总共需要 12 个字节。除了参数和梯度，我们还需要存储优化器的状态：对于 Adam 优化器，这需要动量和方差，通常以 FP32 存储以提高数值稳定性，每个使用 4 个字节。如下图：

| 组件类型 | 数据类型 | 每个参数/分量的字节大小 | 说明 |
| --- | --- | --- | --- |
| 参数（主副本） | BF16 | 2 字节 | BF16用于大多数计算 |
| 参数（FP32副本） | FP32 | 4 字节 | 用于存储FP32精度的模型权重副本 |
| 梯度（主副本） | BF16 | 2 字节 | BF16用于梯度计算 |
| 梯度（FP32副本） | FP32 | 4 字节 | 用于存储FP32精度的梯度副本 |
| Adam优化器状态（动量） | FP32 | 4 字节 | 动量值，以FP32存储以提高数值稳定性 |
| Adam优化器状态（方差） | FP32 | 4 字节 | 方差值，以FP32存储以提高数值稳定性 |

总结如下：

$$
m_{\text{params}} = 2 \times N \\ 
m_{\text{grad}} = 2 \times N \\ 
m_{\text{params\_fp32}} = 4 \times N \\
m_{\text{opt}} = (4 + 4) \times N
$$

> 注意：
>
> 一些库将梯度存储在 fp32 中，这将需要额外的
>
> mparams_fp32=4∗N内存。例如在 nanotron 中这样做，因为bf16对于较小的值是损失性的，我们始终优先考虑稳定性。有关更多信息，请参阅此 DeepSpeed 问题[7]。FP32 参数副本也被称为 “主权重 Master Weights”

有趣的是，混合精度本身并不节省整体内存，因为它只是将内存以不同的方式分配到三个组件中，而且如果我们以 FP32 累积梯度，实际上还会比全精度训练多出 4 个字节。

尽管如此，它仍然具有优势，因为**以半精度计算正向/反向传递**使我们能够

（1）在 GPU 上使用优化的低精度操作，这些**操作更快**

（2）**减少正向传递期间的激活内存需求.**

让我们了解一个模型（全精度和混合精度给出相同总体值）需要多少通用内存：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2013.png)

随着我们可以看到，一旦我们达到 7B，权重和优化器需求已经开始显著增加，并超过典型 GPU 内存的大小，例如 H100 GPU 的 80GB。

但现在，让我们从仍然可以适应单个 GPU 的模型开始，看看对我们内存预算贡献最大的因素：**激活内存 Activation Memory**。

**Activation Memory 激活内存**

激活内存的计算比权重、梯度和优化器状态要复杂一些，部分原因在于它依赖于模型的输入。如果你不确定为什么我们需要存储反向传播中的激活，这篇参考资料[8]是一个很好的快速回顾。在仔细检查了反向传播的计算方式之后，我们可以估算出在混合精度下激活所需的**总内存**，并得出以下方程：

$$
m_{\text{act}} = L \cdot \text{seq} \cdot \text{bs} \cdot h \cdot \left(34 + \frac{5 \cdot n_{\text{heads}} \cdot \text{seq}}{h}\right)
$$

这里  L 是层数， $\text{seq}$  是序列长度， bs  是样本batch size， h  是模型的隐藏维度， $n_{\text{heads}}$  是注意力头数。详细推到请参考[9]

这里一个有趣的观察是，对于给定的模型，其**内存不是静态**的，而是与**序列长度和batch size成线性关系**。这意味着激活内存是当我们增加batch size或使用更长的序列进行训练时将会膨胀的部分。我们可以使用这个方程来查看不同序列长度（例如 Llama 模型）的内存使用情况（ `bs=1` ）。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2014.png)

这张图讲述了一个引人注目的故事：对于短序列（或类似的小batch size），激活几乎可以忽略不计，但大约在 2-4k 个token处，它们开始占用相当大的内存，而参数、梯度和优化器状态的使用与序列长度和batch size基本独立。

**对于大型输入标记（即大型batch size/序列），激活函数成为最大的内存负担。**

有没有办法驯服这种“激活爆炸”？是时候来解释我们的第一种技术——称为**激活重计算（Activation Recomputation）**——它将帮助我们限制激活内存占用。今天大型模型训练工具箱中的一个基本工具。

**Activation recomputation 激活重新计算 / Gradient Checkpointing**

激活重计算（也称为梯度检查点 **`Gradient Checkpointing`** 或重材料化Rematerialization）的**总体思路**是在**正向传播过程中丢弃一些激活以节省内存，并在反向传播过程中额外计算这些激活**。

如果没有重新计算，我们将在两个可学习操作（例如前馈、层归一化等）之间存储每个隐藏状态，以便在反向传播中使用它们来计算梯度。

当我们使用重新计算时，通常只会在模型架构的几个关键点存储激活，丢弃其余的激活，并在反向传播过程中从最近的保存激活动态重新计算它们，基本上再次执行正向传播的一部分以交换内存和计算。它通常看起来是这样的：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2015.png)

有几种策略可以选择要存储的关键激活：

- `Full`: 我们在 Transformer 模型的每一层之间的过渡点检查激活。这通常被称为 `full` 策略，因为它需要通过每一层进行正向传递，实际上在反向传递期间增加了一个完整的正向传递。**这种策略节省了最多的内存，但在计算方面是最昂贵的。它通常将计算成本和时间增加 30-40%**，这是非常明显的。
- `Selective`：总的来说，我们可以做得比 `full` 的更好。Recomputation 论文的作者对哪些部分**激活占用最大且计算成本很低**进行了详细分析。结果发现, **注意力计算属于这一类别，因此我们通常可以丢弃它们，并专注于检查点化昂贵的前馈计算**。对于 GPT-3（175B）模型来说，这意味着在 2.7%的计算成本下，激活内存减少了 70%。

让我们看看重计算策略在实践中如何大幅减少内存占用，以及选择性重计算如何在节省内存和重计算成本之间找到一个不错的平衡点(下图左侧 使用了selective, 右侧是不使用recomputation)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2016.png)

另一个明显可见的趋势是，**对于较小的模型，长序列的激活作用更大，因此重新计算的效果变得更加明显**。

> 当你测量你的训练设置使用 GPU/TPU/加速器的效率时，通常需要考虑**重新计算**来计算总 FLOPS（每秒浮点运算次数），并将其与 **GPU/TPU/加速器的理论最大 FLOPS** 进行比较。在计算训练步骤的 FLOPS 时考虑重新计算，得到一个称为“硬件 FLOPS”的值，这是在加速器上实际执行的操作数。将这个数字除以**训练步骤的持续时间和最大加速器 FLOPS**，得到**硬件 FLOPS 利用率** （Hardware FLOPS Utilization（HFU））。
>
> 然而，最终真正重要的是在给定数据集上训练模型所需的总时间。因此，当比较各种 GPU/TPU/加速器时，如果其中之一提供了足够的内存以跳过重新计算，从而每秒执行的操作更少（较低的 HFU），但能更快地训练，那么它应该得到奖励而不是惩罚。因此，一种替代方法是计算所谓的**模型 FLOPS 利用率（Model FLOPS Utilization, MFU）**，与 HFU 不同，它只考虑模型前向和反向传递所需的操作，不包括在测量的 FLOPs 中的重新计算。因此，这个值比训练实现更具体于模型。

大多数训练框架现在使用 FlashAttention，它通过在反向传播中重新计算注意力分数和矩阵而不是存储它们，在优化策略中本地集成激活重计算。因此，大多数使用 Flash Attention 的人已经在使用选择性重计算 (Selective Recomputation)。

**如您现在所理解的那样，由于Recomputation，激活重计算会增加 FLOPs 的数量，同时显著减少内存访问开销。**

这种权衡在具有小型高速内存的硬件上特别有利，如 GPU，因为访问内存通常比执行计算慢。尽管涉及额外的操作，但整体效果通常是计算更快，同时内存占用也大大降低。

现在我们了解了重新计算，我们可以像上面图表中看到的那样驯服激活内存使用！

然而，激活仍然与BS呈线性相关，上面条形图中我们所有的配置都使用了 `bs=1` ，因此当我们转向更大的BS时，activation可能会再次成为一个问题。不要绝望，我们还有另一个工具——**梯度累积 Gradient Accumulation**来解决这个问题！

**Gradient Accumulation 梯度累积**

**梯度累积**是一种非常直接的方法来避免内存爆炸，它包括**将我们的batch拆分为 `Micro batch`**。我们将对每个Micro batch**依次执行前向和反向传播，计算梯度**，正如其名所示，在执行优化器步骤之前，将所有Micro batch的梯度相加。在实践中，优化步骤是在梯度平均而不是梯度总和上进行的，这样结果就与梯度累积步骤的数量无关。

将每个前向传递的batch size称为 `micro batch size` （mbs）。将每个优化器步骤之间的总batch size称为 `global batch size` （**gbs**）。如果为每个 8 次前向/反向传递执行一个优化器步骤，那么 `global batch size` 将是 `micro batch size` 的 8 倍。

我们现在所称的 `global batch size` 因此对应于我们之前为了简便而称之为 `batch size` 的内容（我们现在使我们的术语更加精确以避免歧义）。

使用梯度累积，全局batch size可以简单地按以下方式计算：

$$
bs=gbs=mbs\times grad_{acc}
$$

**梯度累积**使我们能够有效地将batch size增加到无限大（甚至更多！）同时保持内存占用不变。梯度累积还与激活重计算兼容，以进一步减少内存占用。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2017.png)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2018.png)

梯度累积允许我们通过仅计算部分、Micro Batch Size的激活来减少与批次大小线性增长的激活内存。

**然而，一个缺点是梯度累积需要在每个优化步骤中进行多次连续的前向/反向传递，从而增加了计算开销并减慢了训练速度。No Free Lunch!**

但是如果你仔细观察，你可能已经注意到每个Micro Batch的正向/反向传递实际上可以并行运行。正向/反向传递彼此独立，唯一的区别是独立的输入样本。看起来是时候开始将我们的训练扩展到多个 GPU 上了！

在之前，让我们快速浏览一下如何可视化计算和通信，并了解分布式训练工具箱中最有用的工具之一： `Profiler`  这个工具将非常有助于理解和验证 GPU 与计算之间的通信以及瓶颈所在。

**Profiling GPU Compute and Communication 计算通信分析**

PyTorch 的剖析器允许我们在训练过程中精确追踪和可视化 CPU 和 GPU 上发生的情况。它原生集成在 PyTorch 中。让我们看看如何使用它：

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    with_stack=True
) as prof:
    for step in range(steps):
        train_step() 
        prof.step()
```

这生成一个我们可以在 TensorBoard 或 Chrome 的Trace查看器中可视化的Trace。Trace中显示了：

- CPU 线程异步启动内核到 GPU
- 多个 CUDA 流并行处理计算和通信
- 内核执行时间及内存分配

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2019.png)

*Trace显示 CPU 线程异步启动内核到 GPU，计算内核和通信在不同 CUDA 流中并行发生*

追踪有助于识别瓶颈，如：

- 顺序计算和通信可以重叠
- 空闲 GPU 时间等待数据传输
- CPU 与 GPU 之间的内存移动
- CPU 启动开销

理解这些模式对于优化分布式训练性能至关重要。例如，跟踪将清楚地显示梯度同步是否与后续讨论的向后计算正确重叠。

---

以上是单个GPU训练相关知识，现在我们转移到多个GPU，并开始研究第一种scaling technique - **数据并行 (Data Parallel, DP)**：可以视为梯度累计的并行版本。 

## Data Parallelism 数据并行

**数据并行（DP）**背后的思想是在多个 GPU 上复制模型（我们称复制品为“模型实例”），并对每个 GPU 上的不同Micro Batch Size数据并行执行前向和反向传播，因此得名数据并行。你可能已经在简单的训练示例中见过数据并行，但正如你很快就会看到的，我们将在本节中深入探讨，所以即使你知道一般方法，也要保持关注。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2020.png)

对每个GPU来说，使用不同的Micro Batch意味着每个GPU上会有不同的梯度，因此为了保持不同GPU上的模型实例同步，会在反向传播过程中，在优化器步骤之前，使用一种称为 `“all-reduce”` 的操作对模型实例的梯度进行平均。

这涉及到我们的第一个“分布式通信” 原语：**all-reduce**，用来处理 GPU 实例和节点之间的同步和通信。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2021.png)

一个简单的 DP 实现只会等待反向传播完成，以便我们获得所有梯度，然后触发所有 DP 进程的全量 reduce 操作，以同步这些梯度。但这样的计算顺序，随后是通信，是大忌！因为**我们不希望我们的 GPU 在通信时闲置**，就像上面的图中所示。

我们应尽可能尝试**重叠通信和计算**，以便它们尽可能同时发生。

让我们看看三种优化，这些优化使我们能够比我们最初的简单实现做得更好！

**第一次优化：将梯度同步与反向传播重叠** 

朴素 DDP 方法的主要缺点是，在反向传播（计算）之后，必须等待**梯度同步（通信）才能更新参数**。能否将这种通信与我们的计算重叠？答案是肯定的！

如下图所示，层的梯度可以在计算早期层的梯度之前就进行汇总和求和。例如，一旦最后一层的反向传播完成，这些梯度就可以在**反向计算继续进行早期层的同时进行汇总和求和**，同时向左移动。

通过在每个参数上附加一个 all-reduce  Hook 函数，在 PyTorch 中可以实现这一点。一旦该参数的梯度准备好，就会触发 all-reduce 操作，而其他参数的梯度仍在计算中。这种方法将大多数 all-reduce 操作与梯度计算重叠，从而提高了效率。以下是一个简单的附加钩子函数的示例：

```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model that 
    require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

重叠计算和通信可以减少等待整个模型梯度同步所需的时间。梯度同步可以（至少部分地）与反向传播并行进行，从而显著加速数据并行。以下是带有同步重叠的朴素数据并行（DP）的完整实现：

```python
class DataParallelNaive(nn.Module):
    """
    Naive Data Parallelism. Not used in practice. But it is a good starting point to understand how data parallelism works.
    It implements a simple all-reduce operation to synchronize gradients across multiple processes.
    And `no_sync` context manager to disable gradient synchronization.
    """
    def __init__(self, module):
        """
        Initializes the DataParallel wrapper for a given module.

        Args:
            module (nn.Module): The model to be wrapped for data parallelism.
            process_group (torch.distributed.ProcessGroup): The process group used for gradient synchronization. 
                                                            It could be a data parallel or context parallel group.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        """
        Registers a backward hook for all parameters of the model that require gradients.    
        """
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)
                
    def _allreduce_grads(self, grad):
        """
        Performs an all-reduce operation to synchronize gradients across multiple processes.    
        """
        # No synchronization needed during gradient accumulation, except at the final accumulation step.
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        """
        A context manager to temporarily disable gradient synchronization. 
        This is useful for performing multiple backward passes during gradient accumulation without synchronizing 
        gradients in between.
        """
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

```

这是我们的第一个“重叠计算与通信”示例，我们将在本博客文章中多次讨论，这是实现最大扩展效率的关键技术。但我们可以进一步提高效率！

**第二次优化：Bucketing Gradients 梯度分桶**

GPU 操作通常在大型张量上执行比在许多小张量上执行操作更高效。这也适用于通信操作。因此，我们可以将**梯度分组到桶**中，并为同一桶内的所有梯度启动单个 all-reduce 操作，而不是为每个梯度执行独立的 all-reduce 操作。它通常看起来如下：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2022.png)

将其视为在发货前将物品装箱。发送几个大箱子比发送许多小箱子更有效率。通过为每个桶执行单个全量减少操作，我们可以显著**减少通信开销并加快通信操作**。

这里是一个带有分桶的代码实现：

```python
class DataParallelBucket(nn.Module):
    """
    Data Parallelism with gradient grouped into buckets to reduce the communication overhead.
    """
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        """
        Initialize the DataParallelBucket module.
        
        Args:
            module (nn.Module): The model to be parallelized.
            process_group: The process group for gradient synchronization, which can be either 
                           a data parallel group or a context parallel group.
            bucket_cap_mb (int, optional): The maximum size of each gradient synchronization bucket in megabytes. 
                                           Defaults to 25 MB.
            grad_type (torch.dtype, optional): The data type of gradients, defaulting to float32.
        """
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True # whether to synchronize gradients during backward pass. Set to False when using gradient accumulation
        grad_size = 2 if grad_type == torch.bfloat16 else 4 # float32 gradient: 4 bytes
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size # number of gradients in one bucket
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False # whether the callback for wait gradient synchronization is set
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        """
        Registers a backward hook to manually accumulate and synchronize gradients.
        
        This hook serves two main purposes:
        1. PyTorch does not natively support gradient accumulation with mixed precision.
        2. After gradient accumulation, it flags parameters as ready for synchronization.
        
        The gradient accumulation functions are stored to prevent them from going out of scope.
        
        References:
        - https://github.com/NVIDIA/Megatron-LM/issues/690
        - https://pytorch.org/docs/stable/generated/torch.autograd.graph.Node.register_hook.html
        - https://arxiv.org/abs/2006.15704 (page 5)
        """
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        """
        Creates the a hook for each parameter to handle gradient accumulation and synchronization.
        """
        def param_hook(*unused):
            """
            The hook called after the gradient is ready. It performs the following:
            1. Accumulates the gradient into the main gradient.
            2. Adds a post-backward callback to wait for gradient synchronization completion.
            3. Marks the parameter as ready for synchronization.
            """
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) # accumulate the gradients
                param.grad = None
                
                # skip the gradient synchronization (gradient accumulation/PP micro batches)
                if self.require_backward_grad_sync:
                    # Add a callback to wait for gradient synchronization. Ensures the callback is added only once.
                    # Callback is executed after the backward pass. It should be added per backward pass.
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    # mark the parameter as ready for gradient synchronization. 
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        """
        A post-backward callback that waits for gradient synchronization to finish, then copies 
        the synchronized gradients back to the parameters' grad attribute.
        
        This method is called after the backward pass and before the optimizer step.
        """
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        # copy to params.grad so we can use the optimizer to update the parameters
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) # In PyTorch, you cannot assign a gradient with one data type to a tensor of another data type.

    def reset(self):
        """
        Reset the bucket manager and zero out gradients in the model
        """
        self.bucket_manager.reset() 

```

**第三次优化：DP与梯度累计的相互作用**

最后，正如我们之前所看到的，梯度累积通过在更新参数 `optimizer.step()` 之前执行多次前向和反向传播来实现。当将梯度累积与数据并行结合使用时，当我们想要同步梯度时，我们应该小心谨慎。

在朴素版本中，在accumulation过程中，每次反向传播后都会自动触发 `all-reduce` 操作，这并非最优解，因为最终步骤后的一次减少操作就能达到相同效果，同时还能减少开销。

在 PyTorch 中，这通常通过在不需要归一化的反向传播上添加 `model.no_sync()` 装饰器，来禁用梯度同步来解决。

> 💡
>
> 在进行通信操作时，张量在内存中必须连续，以避免冗余的内存复制。为了最优地执行此操作，通常**预先分配连续的缓冲区**，其大小为激活或模型参数的大小，专门用于通信。**虽然这可以加快通信速度，但它也部分导致了训练期间的过高的峰值内存使用**。

**重新审视 Global Batch Size** 

我们可以更新我们的batch size公式，加入我们新增加的 数据并行DP 和梯度累积参数：

$$
bs = gbs = mbs \times grad\_acc \times dp
$$

- mbs: micro batch size
- grad_acc: 是gradient accumulation step数量
- dp: 是并行实例数量

给定一个目标 全局batch size gbs，可以用数据并行DP过程来交换梯度累积步骤，以加快训练速度。

在实践中，人们倾向于尽可能最大化数据并行节点（DP）的数量，超过梯度累积，因为它是**固有的并行，与梯度累积的顺序性质不同**。然后在数据并行扩展不足以在用完 GPU 之前达到目标全局批次大小时，将梯度累积添加到数据并行之上。

DP 能够将训练分布在不同的样本上，提供了一个并行化的第一维度，从而实现了这一维度的并行性（我们将逐步涵盖另外四个维度）。

> 💡
>
> 小节：关于1D parallel Training recipe:
>
> 1. 首先应**确定最佳的Global Batch Size Tokens**（ `GBST` ），要么通过查阅文献，要么通过运行测量模型收敛性的实验。
> 2. 然后选**择训练的序列长度**，这同样可以通过查阅文献或进行实验来实现。通常，对于我们今天进行的评估，2-8k 个标记可以可靠地工作（**我们不会深入探讨训练方法，但团队通常会在训练结束时增加序列长度，混合一些更长的上下文数据样本，以达到今天的更长上下文大小**）。
> 3. 可以通过增加局部batch size直到内存耗尽，在单个 GPU 上找到maximum local batch size（mbs）。
> 4. 最后，确定目标 DP 可用的 GPU 数量。GBS 与 DP 的比率给出了达到所需 GBS 所需的剩余梯度累积步骤数。

如果梯度累积率低于 1，即拥有过多的 GPU，也就是所谓的有钱任性🤑，可以选择不使用所有 GPU，探索更大的全局批量大小，或者测试降低 MBS 是否会加快训练速度。在后一种情况下，我们将**优先考虑吞吐量**而非单个 GPU 的计算效率，使用比可能更小的 MBS 来加快训练速度。

具体举例说明：假设我们想要训练一个具有 4M 个 token 和 4k 序列长度的最新模型。因此，我们的批量大小将是 1024 个样本（选择最接近的 2 的幂）。

- 假设我们观察到单个 GPU 只能容纳 MBS=2 的内存，并且我们有 128 个 GPU 可供训练。这意味着通过 4 个梯度累积步骤，我们将达到我们的目标，即每个训练步骤 1024 个样本或 4M 个 token。
  
    $gbs(1024)=128\times MBS(2)\times grad\_acc(4)$
    
- 现在，如果我们突然有 512 个 GPU 可用呢？我们可以通过保持 MBS=2，将梯度累积步骤设置为 1，从而实现相同的 GBS 和相同的训练，但训练速度更快！
  
    $gbs(1024)=512\times MBS(2) \times grad\_acc(1)$
    

> 💡
>
> 请注意，在 512+个 GPU 的规模下，根据所使用的网络，通信操作将开始受到环形延迟Ring Latency（信号在环形中传播所需的时间）的限制，这意味着不能再**完全重叠 DP 通信**。这将降低我们的计算效率并影响我们的吞吐量。在这种情况下，应该开始探索其他并行化的维度。

虽然数据并行DP很好地与 `all-reduce` 梯度同步结合，但这种好处在大规模时开始减弱。为什么？因为添加越来越多的 GPU（数百或数千个）时，它们之间的**协调开销**显著增加，**网络需求**变得过大，以至于超过了好处。因此，随着向系统中添加更多的 GPU，现有的设置将变得越来越不高效。

让我们通过一些基准来观察这一现象在实际中的应用：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2023.png)

可以观察到，当超过某个限制时，吞吐量开始显著下降，而每个 GPU 的内存使用量保持恒定，并且增加更多 DP 级别不会受到影响。

**数据并行是最初的简单的策略，用于在更多 GPU 上扩展训练。这种技术的工作原理类似于梯度累积，但并行化了对Micro Batch的正向和反向传递，从而提高了吞吐量！**

敏锐的你可能已经注意到了，在DP的假设中，至少可以将一个输入样本的前向传递（mbs=1）放入GPU 内存中。这并不总是成立！即使是激活重新计算被激活，较大的模型也无法放入单个 GPU 中：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2024.png)

注意到，数据并行在达到一定规模后会开始出现通信开销的限制。对于这些更大的模型或更大的批量大小，我们有其他选择吗？幸运的是，确实有一些解决方案。这些方案要么是将一些张量转移到CPU上，要么是将权重、梯度以及优化器状态张量分散到多个GPU设备上！现在开始深入探讨这些方法吧。

有两种主要的分割方法：

- **并行（Parallelism）**，包括张量（Tensor）、上下文（Context）或流水线（Pipeline）并行；**侧重于计算加速**，将计算任务分配到多个设备，缩短训练或者推理时间；
- **共享（Sharing）**，例如 DeepSpeed Zero 或 PyTorch FSDP；**侧重于内存优化**，通过分片存储模型状态，减少每个设备的内存负担。

这两种方法在一定程度上是正交的，实际上还可以结合使用！下面将首先通过研究 ZeRO 方法来探讨它！

**ZeRO (Zero Redundancy Optimizer)**

本节将介绍 DeepSpeed ZeRO（零冗余优化器），这是一种旨在减少LLM训练中**内存冗余**的内存优化技术。

虽然数据并行是一种有效的扩展训练的方法，但将优化器状态、梯度和参数在**各个 DP  Rank上的简单复制引入了显著的内存冗余**。ZeRO 通过在数据并行维度上划分 优化器状态、梯度和参数 来消除内存冗余，同时仍然允许使用完整的参数集进行计算。这有时需要 DP 等级之间进行更多的通信，这些通信可能或可能不会完全重叠。具体分为三个优化阶段：

- ZeRO-1: 优化器 state partitioning
- ZeRO-2: 优化器 state + gradient partitioning
- ZeRO-3 (FSDP “Fully-Sharded Data Parallelism”): 优化器 state + gradient + parameter partitioning

注意到，以上并没有对激活进行分片(shard), 这是由于模型的每个 DP 副本接收不同的Micro Batch，因此每个 DP 排名上的激活也各不相同，所以它们不会被重复，因此不能进行分片！

**Zero Memory使用分析**

上一节中提到的在标准训练过程中优化器状态、梯度和参数的内存使用情况。让我们称我们模型的参数数量为Ψ （之前是 N，但在这里使用原始 ZeRO 论文的符号）。在混合精度训练（更多细节将在下一节中介绍）中使用 Adam 优化器时，需要存储的每个项目的内存使用情况为：

- 模型参数 Parameter（半精度即 bf16/fp16）： 2Ψ
- 模型梯度 Gradients（半精度即 bf16/fp16）： 2Ψ
- 模型参数（fp32）和优化器状态： 4Ψ+(4Ψ+4Ψ)
- 模型梯度在 fp32 中： 4Ψ （可选，仅在需要累积 fp32 梯度时计算）

如果不在 fp32 中累积梯度，这将导致总内存消耗为 2Ψ+2Ψ+12Ψ，而如果累积，将是 2Ψ+6Ψ+12Ψ。为了简化，我们先关注不使用 fp32 梯度累积的情况，但你只需将受 ZeRO-2 和 3 影响的额外字节添加到梯度项中即可。

ZeRO 的理念是将这些对象在 DP Rank 的各个进程中分片Shard，每个节点只存储一部分，当需要时再重建，从而按数据并行度将内存使用量分成 $N_d$ （DP degree)

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2025.png)

**Zero-1: 分区 优化器状态**

在标准 DP 中，所有Rank在反向传播后收集相同的梯度并同时执行相同的优化步骤。这似乎是很多重复的工作。能否避免它同时减少内存使用？

在 ZeRO-1 中，优化器状态被划分为 $N_d$ 个相等的部分，其中 $N_d$  是 DP degree。这意味着每个模型副本在每个 DP 排名上只跟踪优化器状态的 $1/N_d$  。在优化步骤中，只有  $1/N_d$ 的 float32 权重被更新。

然而，在正向传播过程中，每个副本都需要所有参，因此我们需要在优化器步骤之后添加一个额外的 `all-gather` 操作（这是我们遇到的第二种集体通信原语！）以确保每个模型副本都有完整的最新权重。

这解释了我们在上面图表中看到的  $2\Psi + 2\Psi + \frac{k\Psi}{N_d}$  的内存公式！以下是单个训练步骤的操作序列摘要

- **前向传播**，使用每个副本相同的完整 bf16 参数集，但副本间的Micro Batch Size不同
- **反向传播**，每个副本使用相同的完整梯度集，但副本间的Micro Batch Size不同
- 对梯度执行 `reduce-scatter` 操作（我们将在下面的图中解释 reduce-scatter 原语）
- 每个副本在其本地优化器步骤上执行一个优化器步骤（仅限于 $\frac{1}{N_d}$ 优化器状态），以获取更新的 $\frac{1}{N_d}$ fp32 参数，这些参数随后可以转换为 $\frac{1}{N_d}$完整 bf16 参数集。
- 执行 bf16 参数的 `all-gather` 操作，将缺失的切片发送回每个副本。这是 ZeRO 中的新操作，在 vanilla DP 中未使用。

> 💡
>
> Reduce-Scatter: 结合了归约（Reduction）和分散（Scatter）两个步骤，用于在多个进程或节点之间对数据进行计算并分发结果(如下图所示）。
>
> - **归约（Reduction）**：将多个进程的数据按照某种操作（如求和、求最大值、求最小值等）合并成一个结果。
> - **分散（Scatter）**：将结果分割并分发到各个进程，使得每个进程接收一部分数据。
> - Reduce-Scatter 把这两个操作组合起来：先对所有进程的数据进行归约，然后将归约后的结果分散到各个进程。
>
> All-Gather: 目标是收集所有进程的数据，并将这些数据分发给每个进程，使得每个进程最终拥有所有进程的完整数据副本:
>
> - **Gather（收集）**：将多个进程的数据集中到某个地方。
> - **All**：表示不仅收集数据，还要将收集到的完整数据广播给所有进程。
> - All-Gather 的本质是：每个进程贡献自己的数据，所有进程最终获得所有数据的组合。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2026.png)

在通信方面，与标准 DP 相比，Zero-1 将 `all-reduce` 梯度通信改为 `reduce-scatter` 操作，并在优化器步骤之后对所有参数执行 `all-gather` 操作。如下所示：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2027.png)

在之前章节中，介绍了vanilla DP 中可以将 all-reduce 梯度通信与反向传播计算重叠。在 ZeRO-1中，我们还可以研究如何高效地重叠新添加的 bf16 参数的 `all-gather` 。这里有两大策略：

- **在优化器步骤中**：可以在优化器更新部分参数后立即启动 `all-gather` 。这允许通信可能与其他参数更新重叠。
- **在正向传播过程中**：可以将每一层参数的 `all-gather` 与正向传播过程重叠。

**Zero-2: Add Gradient Partitioning** 

由于只需要在每个副本上拥有与优化器状态块对应的梯度块，因此将梯度也像优化器状态一样分块是有意义的。在反向传播过程中，我们不是对梯度执行 `all-reduce` ，而是只执行 **reduce-scatter** 操作！在这里，**只将显存中需要的 $\frac{1}{N_d}$的梯度进行 scatter**，从而比 ZeRO-1 节省更多内存。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2028.png)

现在很容易看出，梯度分片导致  $2\Psi + \frac{2\Psi + k\Psi}{N_d}$ ，随着 $N_d$ 的增加，我们可以节省高达 8 倍的Gradient和Optimizer State显存占用。在通信方面，与 ZeRO-1 相同的过程适用，唯一的区别是Zero-2边通信边释放。因此，ZeRO-2 在通信方面也与 vanilla DP 训练相当。

在通信方面，ZeRO-2 与 ZeRO-1 相似，它们都需要对梯度进行 `reduce-scatter` 操作，并对所有参数进行 `all-gather` 操作。

![相比Zero-1, Zero-2完全没有任何开销，所以Zero-2是更有选择；](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2029.png)

相比Zero-1, Zero-2完全没有任何开销，所以Zero-2是更有选择；

**Zero-3: 添加 Parameter Partitioning** 

对于第三阶段，我们扩展了上述在 DP 副本上对优化器状态和梯度进行分片的方法，直至对**模型参数**进行分片。

> 💡
>
> 这个阶段也被称为 PyTorch 原生实现中的 FSDP（完全共享数据并行）。在这篇博客文章中，我们只提到 ZeRO-3，但你可以理解为 FSDP。

所以，如果模型的所有部分都是分布式的，我们如何在实践中进行正向或反向传播？很简单，当我们需要时，我们会按需收集它们。在正向传播中，这看起来如下：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2030.png)

因此，当执行正向传播并按顺序遍历层时，我们会按需检索必要的参数，并在不再需要它们时立即从内存中清除。反向传播的工作方式相同，只是流程相反，我们生成梯度碎片：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2031.png)

其他问题是，我们需要在正向和反向步骤中持续进行所有聚集，这相当于在训练步骤中比 Zero-2 多出 2⋅num_layers−1 次 `all-gathers`，每次都伴随着我们可以在下图中看到的小型基础延迟开销

在正向传播过程中，我们需要参数时进行 `all-gather` 操作，因此产生一个Ψ 通信成本。由于我们在正向传播中使用参数后立即丢弃它们，因此在反向传播过程中也需要进行一次额外的 `all-gather` 从而产生另一个Ψ 通信成本。最后需要与 ZeRO-2 中相同的 `reduce-scatter` 操作来处理梯度，这也需要Ψ 通信成本，从而使总通信成本达到3Ψ 。而 Zero-2 的通信成本为2Ψ 。

这可能听起来通信开销很大，但实际上可以接受，因为可以在所谓的**预取 `prefetching`** 中重叠下一层的参数通信与当前层的正向传播。在预取中，我们在正向传播中对层 n+1 进行 `all-gather` 权重，同时进行层 n 的正向传播，同样，我们在反向传播层 n 时对层 n-1 进行 `all-gather` 权重。当然，这种重叠只有在我们没有过度 scale DP 的情况下才成立。（一般来说，DP 不应超过 512）

> 💡
>
> **预取（Prefetching）**：在当前计算任务完成之前，提前发起通信或加载下一批数据/参数到内存或缓存中。**隐藏延迟**：通过将通信操作与计算操作重叠（Overlap），避免通信成为瓶颈。**典型场景**：
> • 在分布式深度学习中，Prefetching 常用于在 GPU 计算当前批次梯度时，提前从其他设备或主机加载下一批数据的梯度或参数。
> • 在数据并行中，Prefetching 可以提前发起 All-Reduce 或 Reduce-Scatter 操作，减少等待时间。
>
> - **预取（Prefetching）**：在当前计算任务完成之前，提前发起通信或加载下一批数据/参数到内存或缓存中。
> - **隐藏延迟**：通过将通信操作与计算操作重叠（Overlap），避免通信成为瓶颈。
> - **典型场景**：
> - 在分布式深度学习中，Prefetching 常用于在 GPU 计算当前批次梯度时，提前从其他设备或主机加载下一批数据的梯度或参数。
> - 在数据并行中，Prefetching 可以提前发起 All-Reduce 或 Reduce-Scatter 操作，减少等待时间。

在内存方面，我们可以看到我们的方程现在达到了最终形式 

$$
\frac{2\Psi + 2\Psi + k\Psi}{N_d}
$$

这意味着如果我们能提高 DP Rank $N_d$，就可以无限期地降低内存使用。注意，这对**中间层激活**没有帮助，对于这一点，我们可以使用我们在前几章中看到的**激活检查点 Activation Checkpointing 和梯度累积 Gradient Accumulation**。

**总结一下到目前为止对 DP 和 ZeRO 的探索：我们看到了通过 DP，可以通过简单地通过添加更多模型副本来扩展训练，从而显著提高训练吞吐量。使用 ZeRO，可以通过在 DP 上分片参数、梯度和优化器状态，训练那些通常无法适应单个 GPU 的模型，同时产生较小的通信成本。**

然而，这里有一个限制，DP 仅在模型的一层适合单个 GPU 时才起作用，ZeRO 只能划分参数、梯度和优化器状态，**但不能划分激活内存**！激活内存与序列长度和batch size成比例。本可以设置这些参数来限制这些，但在实践中，我们不想因为硬件限制而只能使用短序列长度进行训练。

下图展示的是8B模型的Memory Usage，当seq length很长的时候，Activation将主导内存占用；

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2032.png)

为了克服这些问题，是时候探索一个新的、正交的策略——张量并行（Tensor Parallel, TP）。与依赖于大量参数通信的 ZeRO3 不同，TP 提出将参数、梯度、优化器状态以及激活在设备间进行分片，而**不需要在 GPU 之间进行任何模型参数的通信。**

## Tensor Parallel 张量并行

目前使用 ZeRO 对模型的参数、梯度和优化器状态进行了分片，但一旦激活Activation 内存超过内存预算，就遇到了瓶颈。

而 Tensor Parallelism（TP）方法 不仅对权重、梯度和优化器状态进行分片，还对激活进行分片，而且无需在计算前收集它们。让我们首先看看 Tensor Parallel 是如何与简单的矩阵乘法一起工作的。

张量并行利用矩阵乘法的数学特性 $A\times B$, 让我们考察两个使这种并行化成为可能的基本方程：

$$
A \cdot B = A \cdot \begin{bmatrix} B_1 & B_2 & \cdots \end{bmatrix} = \begin{bmatrix} AB_1 & AB_2 & \cdots \end{bmatrix}  \\ 
A \cdot B = \begin{bmatrix} A_1 & A_2 & \cdots \end{bmatrix} \begin{bmatrix} B_1 \\ B_2 \\ \vdots \end{bmatrix} = \sum_{i=1}^{n} A_i B_i 
$$

这意味着我们可以通过以下两种方式之一计算矩阵乘积：1) 分别乘以B的每一列；2) 分别乘以每一行并将结果组合。在神经网络中，矩阵乘积通常以以下格式表示：X×W，其中 X 代表输入或者激活；W代表nn.Linear的权重矩阵；举例如下：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2033.png)

让我们看看如何并行化这个操作！在张量并行中，张量将沿着特定维度分成 N 个碎片，并分布在 N 个 GPU 上。矩阵可以在列部分或行部分进行分割，从而实现行和列并行。在接下来的内容中，我们会看到选择行或列碎片化将需要不同的通信原语。

第一个选项是使用**按照列进行分片**（也称为 `Column-linear` ）：完整输入矩阵复制到每个工作节点，需要执行 `broadcast` 操作，并将权重矩阵拆分为列。然后，输入与部分权重矩阵相乘，最后使用 `all-gather` 操作合并结果。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2034.png)

代码如下：

```python
class ColumnParallelLinear(torch.nn.Module):
    """Column Parallel Linear layer
    Y = XW + b, where weight matrix W is parallelized along its second dimension. W = [W_1, ..., W_p]
    This module returns the results of Y_i = XW_i + b_i in the forward method, Y_i is parallelized in the second dimension.
    Arguments:
        in_features: first dimension of weight matrix W.
        out_features: second dimension of weight matrix W.
        bias: If true, add bias
        init_method: method to initialize weights
        gather_output: If true, gather the output from all the partitions. This is used for the last linear layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        async_all_reduce: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert out_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output
        self.async_all_reduce = async_all_reduce
        # Allocate space for the weight and bias
        # Note: torch.nn.functional.linear performs XW^T + b so we exchange the order of dimensions
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, self.in_features)) # W_i
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with the default initialization
        # method used for nn.Linear in PyTorch
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        torch.nn.init.uniform_(master_weight, -bound, bound)
        # 这里随机初始化权重，模拟主权重
        
        # Split the model into size of self.output_size_per_partition
        # 这里执行对weight的分片
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        # 根据TP Rank访问对应切片
        self.weight.data = weight_list[self.tp_rank].contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        if self.async_all_reduce:
            output = linear_with_async_all_reduce(x, self.weight, self.bias) 
        else:
            output = linear_with_all_reduce(x, self.weight, self.bias) 
        if self.gather_output:
		        # 这里执行all gather 
            output = GatherFromModelParallelRegion.apply(output)
        return output
```

> 💡
>
> 解释：
>
> 1. **Input 的 broadcast**：
> - 隐含在 forward 中假设 x 已广播到所有进程。这是列并行设计的标准假设
> 2. **对 W 进行分片**：
> - 在 reset_parameters 中，通过 torch.split(master_weight, self.output_size_per_partition, dim=0) 将权重按列分割，并根据 tp_rank 分配。
> 3. **All-gather**：
> - 在 forward 中，当 self.gather_output == True 时，通过 GatherFromModelParallelRegion.apply(output) 收集所有进程的输出。

第二种方案被称为行分片（也称为 `row-linear` )：行线性分片意味着将权重矩阵分成行块。然而，这也需要将输入进行分割，这需要一个 `scatter` 操作，而不是像 `column-linear sharding` 那样使用广播操作。每个工作节点的结果已经处于正确的形状，但需要求和以得到最终结果，因此在这种情况下需要 `all-reduce` 操作。 

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2035.png)

以下是行向张量并行的实现

```python

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.
    Y = XW + b. W is parallelized along its first dimension and X along its second dimension as:
               -   -
              | W_1 |
              | .   |
          W = | .   |        X = [X_1, ..., X_p]
              | .   |
              | W_p |
               -   -
    We assume that X is already parallelized. This is the case after ColumnParallelLinear.
    This module returns the results of Y = sum(X_i * W_i + b_i) in the forward method.
    Arguments:
        in_features: first dimension of matrix W.
        out_features: second dimension of matrix W.
        bias: If true, add bias
        init_method: method to initialize weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super(RowParallelLinear, self).__init__()

        self.tp_world_size = pgm.process_group_manager.tp_world_size
        self.tp_rank = pgm.process_group_manager.tp_rank 

        self.in_features = in_features
        self.out_features = out_features
        assert in_features % self.tp_world_size == 0, "Hidden dimension must be divisible by the tensor parallel world size"
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight tensor with same dtype and device as self.weight
        master_weight = torch.empty(
            self.out_features, 
            self.in_features, 
            dtype=self.weight.dtype,
            device=self.weight.device,
            requires_grad=False
        )
        
        # Calculate bound based on master weight's input dimension
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)    
        torch.nn.init.uniform_(master_weight, -bound, bound)
        
        # Split the model into size of self.input_size_per_partition
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        # 在这里切分weight，分为TP rank份
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, x):
        # X_i * W_i^T + b
        output_parallel = F.linear(x, self.weight)
        # All-reduce across all the partitions.
        # 执行all-reduce 
        output = ReduceFromModelParallelRegion.apply(output_parallel)
        return output if self.bias is None else output + self.bias

```

> 💡
>
> 解释：
>
> - **对 W 进行分片**：
> - 在 reset_parameters 中，通过 torch.split(master_weight, self.input_size_per_partition, dim=1) 将权重按行分割，并根据 tp_rank 分配。
> - **All-reduce**：
> - 在 forward 中，通过 ReduceFromModelParallelRegion.apply(output_parallel) 将每个进程的局部输出 X_i * W_i^T 求和，生成完整的 Y。

### **Transformer 块中的张量并行**

为了制定一个策略，让我们从Toy Example 示例过渡到真实模型构建块。Transformer 模型由两个主要构建块组成：前馈层（MLP）和多头注意力（MHA）。我们可以将张量并行应用于两者。

前馈部分可以通过先进行 `“Column Linear”` 操作，然后进行 `"Row Linear"` 操作来实现并行化，这相当于广播以复制输入，并在正向传播中进行 `all-reduce`  。

请注意，**在实际训练中不需要广播，因为已经确保输入在 TP ranks 之间已经同步**。这种设置比先进行 `Row-Linear` 操作，然后进行 `Column-Linear` 操作更高效，因为我们可以在两个分割操作之间跳过中间的 `all-reduce` 。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2036.png)

> 💡
>
> 可以从以上示意图中观察到，column linear的输出刚好符合row linear的输入要求，因此可以不进行 al-reduce，可以节省了通信。

现在我们已经找到了 Transformer FeedForward 部分的效率方案，让我们来看看多头注意力块（MHA）。

我们可以一般遵循类似的方法，其中 Q、K 和 V 矩阵以 **列并行**方式分割，输出投影沿**行维度**分割。在多头注意力中，列并行方法有非常自然的解释：**每个worker计算单个或一组头部的注意力**。同样的方法也适用于多查询（MQA）或分组查询注意力（GQA），其中键和值在查询之间共享。

> 💡
>
> **注意**
>
> - 张量并行度不应超过 Q/K/V 头数，因为我们需要每个 TP  Rank的完整头（否则无法独立在每个 GPU 上计算注意力，需要额外的通信操作）。
> - 如果使用 GQA，TP 度实际上应该小于 K/V 头数。例如，LLaMA-3 8B 有 8 个K/V 头，因此张量并行度应不超过 8。
> - 如果为这个模型使用 TP=16，我们需要在每个 GPU 上复制 K/V 头，并确保它们保持同步。

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2037.png)

最后需要注意的是，Tensor Parallelsim 仍然不是训练的万能解决方案。我们在模型的计算路径中直接添加了几个分布式通信原语，因此这些原语难以完全隐藏/重叠于计算（就像我们在 ZeRO 中所做的那样），最终性能将是计算和内存增益与增加的通信开销之间的权衡结果。让举例说明：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2038.png)

观察 张量并行 MLP（同样适用于注意力机制）的操作时间线，我们可以更好地理解其中涉及的 trade-off。在每个解码器层的正向传播中，会遇到一个与 AllReduce 操作同步的点，这个点不能与计算重叠。这种通信开销是必要的，以便在最终应用 LayerNorm 之前，将张量并行等级的局部结果组合起来(PS: 否则结果便不正确了。

> 💡
>
> 备注：
>
> 通过执行块矩阵乘法与异步通信/计算相结合，可以部分隐藏这种通信。例如，Megatron-LM/Nanotron 实现了 all-reduce 与 FC1 计算的局部重叠，其中矩阵乘法结果的一部分将开始发送到其他 GPU，而另一部分仍在计算中。

张量并行 TP 确实有助于减少矩阵乘法中的激活内存，因为中间激活被分散到多个 GPU 上。然而，我们仍然需要收集全量激活以进行如 **LayerNorm** 等操作，这意味着我们没有获得本可以得到的全部内存优势。

此外，TP 引入了显著的通信需求，这严重依赖于网络基础设施。无法完全隐藏这种特定的 All-Reduce 操作在计算背后的能力意味着它直接增加了前向传播的关键路径。

下面讨论 TP degree 对并行的影响：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2039.png)

尽管随着 TP 的增加会导致每 GPU 吞吐量降低（左侧图所示），TP 使得处理更大的batch size成为可能（右侧图片所示），这说明了在分布式训练中计算效率与内存可用性之间的trade-off。

在实践中，正如我们在左图上所看到的，**当扩展到超过 8 个 GPU 时，张量并行的通信开销变得尤为明显。**虽然单个**节点内的张量并行**可以利用**快速的 NVLink** 互连，但**跨节点**则需要**较慢的网络连接**。我们从 TP=8 到 TP=16 时观察到显著的下降，从 TP=16 到 TP=32 时下降更为陡峭。在更高的并行度下，通信开销变得如此之高，以至于它迅速主导了计算时间。

总的来说，张量并行通过在多个 GPU 上分布模型参数、梯度、优化器状态和激活（在一定程度上）来为内存使用提供重要优势。让我们以一个 70 B的模型为例来考察这种效果：

![](https://raw.githubusercontent.com/pprp/blogimagebed/main/image%2040.png)

增加张量并行度可以降低每个 GPU 上模型参数、梯度和优化器状态所需的内存，直至可以开始在单个 8 GPU 节点上训练LLM。

是否存在一种方法能从这项技术中获得更多好处？我们已经看到，LayerNorm和 dropout 仍然需要在每个 GPU 上收集完整的激活，这在一定程度上抵消了内存节省的效果。我们可以通过找到并行化这些剩余操作的方法来做得更好。

> 💡
>
> 关于张量并行训练中 Layer Normalization 一个点——由于每个 TP rank 在 all-gather 之后看到相同的激活，LayerNorm的权重实际上在反向传播后不需要 all-reduce 来同步它们的梯度。它们自然地在 Rank 间保持同步。然而，对于 dropout 操作，我们必须确保在 TP Rank间同步随机种子，以保持确定性。
>
> 编者注：在张量并行训练中，每个设备（rank）通过all-gather操作获取相同的激活值后，层归一化（layer normalization）的输入是相同的。因此，每个设备计算的层归一化权重（gamma和beta）的梯度是相同的，不需要额外的 all-reduce 操作来同步梯度，因为它们已经自然一致。这是因为所有设备基于相同的输入和梯度计算，梯度本身就是同步的。

# Reference

[1] [https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

[2] [https://github.com/huggingface/picotron](https://github.com/huggingface/picotron)

[3] [https://github.com/huggingface/nanotron](https://github.com/huggingface/nanotron)

[4] [https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf](https://filecdn.minimax.chat/_Arxiv_MiniMax_01_Report.pdf)

[5] [https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) 

[6] [https://michaelwornow.net/2024/01/18/counting-params-in-transformer](https://michaelwornow.net/2024/01/18/counting-params-in-transformer)

[7] [https://github.com/microsoft/DeepSpeed/issues/1773](https://github.com/microsoft/DeepSpeed/issues/1773)

[8] [https://www.determined.ai/blog/act-mem-2](https://www.determined.ai/blog/act-mem-2) 

[9] Reducing Activation Recomputation in Large Transformer Models
