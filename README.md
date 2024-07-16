# 教程[¶](https://langchain-ai.github.io/langgraph/tutorials/#tutorials)-v0.1.5

欢迎来到LangGraph教程！这些笔记本通过构建各种语言代理和应用程序来介绍LangGraph。

## 快速入门[¶](https://langchain-ai.github.io/langgraph/tutorials/#quick-start)

通过综合快速入门学习LangGraph的基础知识，您将从头开始构建一个代理。

- [快速入门](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

## 用例[¶](https://langchain-ai.github.io/langgraph/tutorials/#use-cases)

从针对特定场景设计的图形的示例实现中学习，这些示例实现了常见的设计模式。

#### 聊天机器人[¶](https://langchain-ai.github.io/langgraph/tutorials/#chatbots)

- [客户支持](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/): 构建一个客户支持聊天机器人来管理航班、酒店预订、租车和其他任务
- [从用户需求生成提示](https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/): 构建一个信息收集聊天机器人
- [代码助手](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/): 构建一个代码分析和生成助手

#### 多代理系统[¶](https://langchain-ai.github.io/langgraph/tutorials/#multi-agent-systems)

- [协作](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/): 使两个代理协作完成任务
- [监督](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/): 使用LLM协调并委派给个别代理
- [层次团队](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/): 协调嵌套的代理团队解决问题

#### RAG[¶](https://langchain-ai.github.io/langgraph/tutorials/#rag)

- 自适应RAG
  - [使用本地LLM的自适应RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/)
- [代理RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- 纠正RAG
  - [使用本地LLM的纠正RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local/)
- 自我RAG
  - [使用本地LLM的自我RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag_local/)
- [SQL代理](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/)

#### 计划代理[¶](https://langchain-ai.github.io/langgraph/tutorials/#planning-agents)

- [计划与执行](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/): 实现一个基本的计划和执行代理
- [无观察推理](https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/): 通过将观察结果保存为变量来减少重新计划
- [LLM编译器](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/): 流式传输并积极执行来自计划者的任务DAG

#### 反思与批判[¶](https://langchain-ai.github.io/langgraph/tutorials/#reflection-critique)

- [基础反思](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/): 提示代理反思并修正其输出
- [反射](https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/): 批判缺失和多余的细节以指导下一步
- [语言代理树搜索](https://langchain-ai.github.io/langgraph/tutorials/lats/lats/): 使用反思和奖励驱动代理的树搜索
- [自我发现代理](https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover/): 分析一个了解自身能力的代理

#### 评估[¶](https://langchain-ai.github.io/langgraph/tutorials/#evaluation)

- [基于代理](https://langchain-ai.github.io/langgraph/tutorials/chatbot-simulation-evaluation/agent-simulation-evaluation/): 通过模拟用户交互评估聊天机器人
- [在LangSmith中](https://langchain-ai.github.io/langgraph/tutorials/chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation/): 在LangSmith上通过对话数据集评估聊天机器人

#### 实验性[¶](https://langchain-ai.github.io/langgraph/tutorials/#experimental)

- [网络研究（STORM）](https://langchain-ai.github.io/langgraph/tutorials/storm/storm/): 通过研究和多角度问答生成类似维基百科的文章
- [TNT-LLM](https://langchain-ai.github.io/langgraph/tutorials/tnt-llm/tnt-llm/): 构建丰富、可解释的用户意图分类系统，该系统由微软为其Bing Copilot应用程序开发
- [网络导航](https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/): 构建一个可以导航和与网站交互的代理
- [竞技编程](https://langchain-ai.github.io/langgraph/tutorials/usaco/usaco/): 构建一个具有少样本“情节记忆”和人类参与协作的代理，以解决来自美国计算机奥林匹克竞赛的问题；改编自Shi、Tang、Narasimhan和Yao的[“Can Language Models Solve Olympiad Programming?”](https://arxiv.org/abs/2404.10952v1)论文
- [复杂数据提取](https://langchain-ai.github.io/langgraph/tutorials/extraction/retries/): 构建一个可以使用函数调用进行复杂提取任务的代理
