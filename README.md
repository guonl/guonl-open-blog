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

### 接下来打算写

方向会随着技术热点扩展，目前排在这几个：

- 

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
