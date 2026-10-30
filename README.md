# guonl 的技术博客

一个紧跟技术热点的博客。挑当下最值得讲清楚的方向深入写，AI 也好，后端、性能、数据也好，标准只有一个：原理要能讲透，代码要能跑通。

不做新闻搬运。一个方向刚火的时候往往只有概念和口号，等它沉淀出可讲的原理和可用的实践，再动手写。所以这里更新不快，但每篇都尽量写到能当参考用的程度。

## 写作取向

- **跟热点，但更看重沉淀**：判断一个方向值不值得写，看它有没有稳定的原理和真实的工程实践。
- **讲原理**：不只给结论和用法，把"为什么会这样""换一种做法会付出什么代价"讲清楚。
- **能实战**：关键环节给可运行的代码，把踩过的坑、边界条件和取舍一并写出来。

## 目录

### AI 工程

**AI Agent 入门与工程化**

- [从"对话"到"行动"：AI Agent 到底是什么](posts/ai-agent/from-chat-to-action.md)
  Agent 和聊天机器人差在哪，它由哪些零件组成，ReAct 循环是怎么转起来的。
- [动手写一个最小 Agent：Function Calling 原理与代码](posts/ai-agent/build-minimal-agent.md)
  用一段可运行的 Python 代码讲透工具调用的数据流，从单次调用写到多轮循环。

**MCP 协议实战**

- [MCP 是什么：给大模型插上 USB-C](posts/mcp/mcp-intro.md)
  为什么每个工具都要重写一遍适配代码，MCP 想解决什么，它的三个核心概念。
- [手把手写一个 MCP Server 与安全避坑](posts/mcp/build-mcp-server.md)
  基于官方 Python SDK 的完整示例，附提示注入、最小权限、路径穿越的防护清单。

**上下文工程与记忆**

- [提示词工程不够用了：Context Engineering 入门](posts/context-engineering/context-engineering-intro.md)
  从"怎么问"到"让它看到什么"，以及 Agent 场景下的四大上下文痛点。
- [Agent 的记忆系统：短期记忆与长期记忆怎么设计](posts/context-engineering/agent-memory.md)
  记忆怎么存、怎么取、怎么忘，附一个零依赖的检索示例。

**Agentic Engineering 与 Eval**

- [从 Vibe Coding 到 Agentic Engineering](posts/agentic-engineering/vibe-coding-to-agentic-engineering.md)
  凭感觉写代码在生产环境会怎么翻车，以及怎么把 AI 编程变成可验收的工程。
- [Eval：怎么知道你的 Agent 变好了](posts/agentic-engineering/agent-eval.md)
  没有评估的迭代就是瞎调，讲清评估的要素、方法学，附一个最小可运行的 Eval。

### 嵌入式与边缘 AI

**边缘 AI 入门**

- [边缘 AI 是什么：AI 为什么正在从云端搬到设备端](posts/edge-ai/edge-ai-intro.md)
  为什么要下沉到端侧，云边端怎么分工，端侧的能力边界在哪。
- [一个任务该放云端还是端侧：云边端分工的取舍](posts/edge-ai/cloud-or-edge.md)
  给一套可操作的判断方法，以及端侧过滤、边聚合、云端深处理的分层架构。

**模型部署与量化**

- [模型量化入门：为什么 INT8 又快又省](posts/edge-ai/quantization-intro.md)
  浮点与定点、对称与非对称、per-tensor 与 per-channel，量化省在哪、代价是什么。
- [动手：把 PyTorch 模型部署到端侧](posts/edge-ai/deploy-model-on-edge.md)
  从导出 ONNX 到动态/静态量化，跑通链路并对比精度、延迟与体积。

**端侧大模型**

- [在设备上跑大模型为什么难：内存墙与带宽瓶颈](posts/edge-ai/on-device-llm.md)
  端侧 LLM 卡的不是算力而是内存带宽，讲清 KV cache 与 decode 的内存账。
- [动手：用 llama.cpp 在本地跑一个量化小模型](posts/edge-ai/llama-cpp-hands-on.md)
  本地跑通一个 GGUF 量化模型，观察内存占用与生成速度。

**MCU 与 TinyML**

- [TinyML 入门：KB 级内存里怎么跑神经网络](posts/edge-ai/tinyml-intro.md)
  几百 KB 内存下的取舍：定点运算、算子精简、小网络结构。
- [动手：在 ESP32 上跑一个关键词唤醒](posts/edge-ai/esp32-keyword-spotting.md)
  从训练到量化到部署，在几十块钱的开发板上跑通离线唤醒。

**嵌入式 AI 的硬件与加速**

- [NPU、DSP、GPU：端侧推理到底靠什么加速](posts/edge-ai/npu-dsp-gpu.md)
  异构计算怎么分工，TOPS 该怎么看，算子支持为什么关键。
- [选板指南：Jetson / RK3588 / ESP32-S3 怎么选](posts/edge-ai/choose-edge-board.md)
  按任务、约束和预算选板，附对比表与常见误区。

**实时性与算控一体**

- [RTOS 与实时性：为什么 AI 和实时控制会打架](posts/edge-ai/rtos-realtime.md)
  实时性是确定性延迟，不是快；大模型推理为什么进不了控制环。
- [算控一体：在同一颗芯片上同时跑 AI 和控制](posts/edge-ai/compute-control-unified.md)
  CPU 跑控制、NPU 跑感知，实时域怎么隔离，数据通路怎么走。

**边缘工程化与安全**

- [边缘设备的 OTA 与模型热更新](posts/edge-ai/edge-ota.md)
  分发、校验、A/B 分区与回滚，模型热更新和固件更新有什么不同。
- [边缘安全：安全启动、固件签名与最小权限](posts/edge-ai/edge-security.md)
  安全启动链、固件验签、设备身份与最小权限的一份可落地清单。

### 具身智能

**具身智能入门**

- [具身智能入门：机器人是怎么学会干活的](posts/embodied-ai/embodied-ai-intro.md)
  感知、决策、控制怎么闭环，它和数字世界的 Agent 是什么关系。

**VLA 与三大技术路线**

- [VLA 与三大技术路线：视觉-语言-动作模型、世界模型、本能驱动](posts/embodied-ai/vla-world-model.md)
  三条路线的原理、取舍与融合趋势。

**数据与仿真闭环**

- [数据与仿真闭环：具身智能为什么卡在数据上](posts/embodied-ai/data-simulation-loop.md)
  仿真与真机数据的取舍，数据闭环是怎么回事。

**具身学习**

- [模仿学习：机器人怎么从示教里学会动作](posts/embodied-ai/imitation-learning.md)
  行为克隆为什么十分钟讲得完却很难用好：分布偏移、多模态动作分布与对策。
- [强化学习在机器人上为什么这么难](posts/embodied-ai/rl-for-robots.md)
  样本效率、奖励设计、真机试错的代价，以及模仿打底、强化精修的组合。
- [动手：从示教数据训练一个抓取策略](posts/embodied-ai/hands-on-imitation-policy.md)
  合成示教数据训练一个小策略，用开环与闭环对照看复合误差。

**身体与操作**

- [机器人的动作空间与运动规划](posts/embodied-ai/action-space-motion-planning.md)
  关节空间与笛卡尔空间、逆运动学，以及路径与轨迹规划的区别。
- [抓取与灵巧操作：手怎么把东西拿稳](posts/embodied-ai/grasping-dexterous-manipulation.md)
  形封闭与力封闭、抓取检测，以及灵巧手和触觉为什么绕不过去。

**仿真到真机**

- [Sim2Real：仿真里练好的技能，为什么到了真机就废](posts/embodied-ai/sim2real.md)
  落差从哪来，域随机化、系统辨识与真机微调各自解决什么。
- [动手：用 MuJoCo 搭一个仿真训练闭环](posts/embodied-ai/mujoco-training-loop.md)
  零依赖跑通环境、采样、训练、评估的闭环，再看怎么换成 MuJoCo。

**端侧实时控制**

- [端侧实时控制：为什么大模型进不了控制环](posts/embodied-ai/on-device-realtime-control.md)
  控制环只有几毫秒，分层架构怎么把慢规划与快控制分开。

**具身系统工程**

- [机器人的软件栈：ROS 2、实时中间件与数据流](posts/embodied-ai/robot-software-stack.md)
  话题、服务、动作怎么选，QoS 与实时域怎么共存。
- [安全与人机协作：机器人怎么保证不伤人](posts/embodied-ai/safety-human-robot.md)
  五层安全设计、三种协作模式，以及为什么安全逻辑不能交给大模型。

**具身评测**

- [具身评测：怎么衡量一个机器人模型的好坏](posts/embodied-ai/embodied-eval.md)
  评测难在哪，几类基准，以及怎么设计自己的评测。

### 接下来打算写

方向会随着技术热点扩展，目前排在这几个：

- 后端与系统设计：分布式、高并发、一致性、消息队列。
- 性能与稳定性：性能剖析、压测、可观测性、SRE 实践。
- 数据与存储：向量数据库、湖仓一体、流处理。
- AI 基础设施：推理服务、GPU 调度、推理成本优化。

有想看的题目也欢迎开 issue 说。

## 仓库结构

```
posts/
  <专题>/
    <文章>.md
    images/<文章>/      该文的封面与插图（相对路径引用）
```

图片和文章放在同一专题目录下，用相对路径引用，在 GitHub 网页上可以直接正常显示。目录里的板块分类（如 AI 工程）只是阅读时的归类，仓库按专题平铺。

## 说明

文章内容基于写作时的技术现状整理，代码示例以 Python 为主；涉及外部服务的部分用 mock 实现，本地无需 API Key 即可跑通。技术变化快，读到过时的内容欢迎提 issue 指正。
