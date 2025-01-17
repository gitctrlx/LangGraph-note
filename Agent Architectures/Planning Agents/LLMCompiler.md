# LLMCompiler

This notebook shows how to implement [LLMCompiler, by Kim, et. al](https://arxiv.org/abs/2312.04511) in LangGraph.

LLMCompiler is an agent architecture designed to **speed up** the execution of agentic tasks by eagerly-executed tasks within a DAG. It also saves costs on redundant token usage by reducing the number of calls to the LLM. Below is an overview of its computational graph:

![LLMCompiler Graph](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/img/llm-compiler.png)

It has 3 main components:

1. Planner: stream a DAG of tasks.
2. Task Fetching Unit: schedules and executes the tasks as soon as they are executable
3. Joiner: Responds to the user or triggers a second plan

This notebook walks through each component and shows how to wire them together using LangGraph. The end result will leave a trace [like the following](https://smith.langchain.com/public/218c2677-c719-4147-b0e9-7bc3b5bb2623/r).

**First,** install the dependencies, and set up LangSmith for tracing to more easily debug and observe the agent.

```
pip install -U --quiet langchain_openai langsmith langgraph langchain numexpr
```

```
import getpass
import os


def _get_pass(var: str):
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")


# Optional: Debug + trace calls using LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = "LLMCompiler"
_get_pass("LANGCHAIN_API_KEY")
_get_pass("OPENAI_API_KEY")
```



## Part 1: Tools

We'll first define the tools for the agent to use in our demo. We'll give it the class search engine + calculator combo.

If you don't want to sign up for tavily, you can replace it with the free [DuckDuckGo](https://python.langchain.com/v0.2/docs/integrations/tools/ddg/).

```
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo
from math_tools import get_math_tool

_get_pass("TAVILY_API_KEY")

calculate = get_math_tool(ChatOpenAI(model="gpt-4-turbo-preview"))
search = TavilySearchResults(
    max_results=1,
    description='tavily_search_results_json(query="the search query") - a search engine.',
)

tools = [search, calculate]
```

```
calculate.invoke(
    {
        "problem": "What's the temp of sf + 5?",
        "context": ["Thet empreature of sf is 32 degrees"],
    }
)
```

`37`

# Part 2: Planner

Largely adapted from [the original source code](https://github.com/SqueezeAILab/LLMCompiler/blob/main/src/llm_compiler/output_parser.py), the planner accepts the input question and generates a task list to execute.

If it is provided with a previous plan, it is instructed to re-plan, which is useful if, upon completion of the first batch of tasks, the agent must take more actions.

The code below composes constructs the prompt template for the planner and composes it with LLM and output parser, defined in [output_parser.py](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/output_parser.py). The output parser processes a task list in the following form:

```plaintext
plaintext
1. tool_1(arg1="arg1", arg2=3.5, ...)
Thought: I then want to find out Y by using tool_2
2. tool_2(arg1="", arg2="${1}")'
3. join()<END_OF_PLAN>"
```

The "Thought" lines are optional. The `${#}` placeholders are variables. These are used to route tool (task) outputs to other tools.

```
from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from output_parser import LLMCompilerPlanParser, Task

prompt = hub.pull("wfh/llm-compiler")
print(prompt.pretty_print())
```

![image-20240716121009160](./assets/image-20240716121009160.png)

```
def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    tool_descriptions = "\n".join(
        f"{i+1}. {tool.description}\n"
        for i, tool in enumerate(
            tools
        )  # +1 to offset the 0 starting index, we want it count normally from 1.
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools)
        + 1,  # Add one because we're adding the join() tool at the end.
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )
```

```
llm = ChatOpenAI(model="gpt-4-turbo-preview")
# This is the primary "agent" in our application
planner = create_planner(llm, tools, prompt)
```

```
example_question = "What's the temperature in SF raised to the 3rd power?"

for task in planner.stream([HumanMessage(content=example_question)]):
    print(task["tool"], task["args"])
    print("---")
```

![image-20240716121051720](./assets/image-20240716121051720.png)

## 3. Task Fetching Unit

This component schedules the tasks. It receives a stream of tools of the following format:

```
{
    tool: BaseTool,
    dependencies: number[],
}
```

The basic idea is to begin executing tools as soon as their dependencies are met. This is done through multi-threading. We will combine the task fetching unit and executor below:

![diagram](https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/img/diagram.png)

```
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union

from langchain_core.runnables import (
    chain as as_runnable,
)
from typing_extensions import TypedDict


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            # This will likely fail
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            + f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    # 1or1or1 or {1} -> 1
    ID_PATTERN = r"$\{?(\d+)\}?"

    def replace_match(match):
        # If the string is 123,match.group(0)is123,match.group(0)is{123}, match.group(0) is {123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # For dependencies on other tasks
    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback

        observation = traceback.format_exception()  # repr(e) +
    observations[task["idx"]] = observation


def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # Dependencies not yet satisfied
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps
                and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)
    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages

```

```
import itertools


@as_runnable
def plan_and_schedule(messages: List[BaseMessage], config):
    tasks = planner.stream(messages, config)
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        },
        config,
    )
    return scheduled_tasks
```



#### Example Plan

We still haven't introduced any cycles in our computation graph, so this is all easily expressed in LCEL.

```
tool_messages = plan_and_schedule.invoke([HumanMessage(content=example_question)])
```

```
tool_messages
```

```
[FunctionMessage(content='[]', additional_kwargs={'idx': 0}, name='tavily_search_results_json'),
 FunctionMessage(content='ValueError(\'Failed to evaluate "N/A". Raised error: KeyError(\\\'A\\\'). Please try again with a valid numerical expression\')', additional_kwargs={'idx': 1}, name='math'),
 FunctionMessage(content='join', additional_kwargs={'idx': 2}, name='join')]
```

## 4. "Joiner"

So now we have the planning and initial execution done. We need a component to process these outputs and either:

1. Respond with the correct answer.
2. Loop with a new plan.

The paper refers to this as the "joiner". It's another LLM call. We are using function calling to improve parsing reliability.

```
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field


class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  # You can optionally add examples
llm = ChatOpenAI(model="gpt-4-turbo-preview")

runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)
```

We will select only the most recent messages in the state, and format the output to be more useful for the planner, should the agent need to loop.

```
def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
    else:
        return response + [AIMessage(content=decision.action.response)]


def select_recent_messages(messages: list) -> dict:
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output
```

```
input_messages = [HumanMessage(content=example_question)] + tool_messages
```

```
joiner.invoke(input_messages)
```



```
[AIMessage(content='Thought: The search did not return any results, and the attempt to calculate the temperature in San Francisco raised to the 3rd power failed due to missing temperature information.'),
 SystemMessage(content='Context from last attempt: I need to find the current temperature in San Francisco before calculating its value raised to the 3rd power.')]
```

## 5. Compose using LangGraph

We'll define the agent as a stateful graph, with the main nodes being:

1. Plan and execute (the DAG from the first step above)
2. Join: determine if we should finish or replan
3. Recontextualize: update the graph state based on the output from the joiner

```
from typing import Dict

from langgraph.graph import END, MessageGraph, START

graph_builder = MessageGraph()

# 1.  Define vertices
# We defined plan_and_schedule above already
# Assign each node to a state variable to update
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)


## Define edges
graph_builder.add_edge("plan_and_schedule", "join")

### This condition determines looping logic


def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"


graph_builder.add_conditional_edges(
    start_key="join",
    # Next, we pass in the function that will determine which node is called next.
    condition=should_continue,
)
graph_builder.add_edge(START, "plan_and_schedule")
chain = graph_builder.compile()
```

#### Simple question

Let's ask a simple question of the agent.

```
for step in chain.stream([HumanMessage(content="What's the GDP of New York?")]):
    print(step)
    print("---")
```



```

{'plan_and_schedule': [FunctionMessage(content='[{\'url\': \'https://www.governor.ny.gov/programs/fy-2024-new-york-state-budget\', \'content\': "The $229 billion FY 2024 New York State Budget reflects Governor Hochul\'s bold agenda to make New York more affordable,  FY 2024 Budget Assets FY 2024 New York State Budget Highlights Improving Public Safety  GOVERNOR HOME GOVERNOR KATHY HOCHUL FY 2024 New York State Budget  Transformative investments to support New York\'s business community and boost the state economy.The $229 billion FY 2024 NYS Budget reflects Governor Hochul\'s bold agenda to make New York more affordable, more livable, and safer."}]', additional_kwargs={'idx': 0}, name='tavily_search_results_json')]}
---
{'join': [AIMessage(content="Thought: The information provided does not specify the Gross Domestic Product (GDP) of New York, but instead provides details about the state's budget for fiscal year 2024, which is $229 billion. This budget figure cannot be accurately equated to the GDP."), SystemMessage(content="Context from last attempt: The search results provided information about New York's state budget rather than its GDP. To answer the user's question, we need to find specific data on New York's GDP, not its budget.")]}
---
{'plan_and_schedule': [FunctionMessage(content="[{'url': 'https://en.wikipedia.org/wiki/Economy_of_New_York_(state)', 'content': 'The economy of the State of New York is reflected in its gross state product in 2022 of $2.053 trillion, ranking third  Contents Economy of New York (state)  New York City-centered metropolitan statistical area produced a gross metropolitan product (GMP) of $US2.0 trillion,  of the items in which New York ranks high nationally:The economy of the State of New York is reflected in its gross state product in 2022 of $2.053 trillion, ranking third in size behind the larger states of\\xa0...'}]", additional_kwargs={'idx': 1}, name='tavily_search_results_json')]}
---
{'join': [AIMessage(content="Thought: The required information about New York's GDP is provided in the search results. In 2022, New York had a Gross State Product (GSP) of $2.053 trillion."), AIMessage(content='The Gross Domestic Product (GDP) of New York in 2022 was $2.053 trillion.')]}
---
{'__end__': [HumanMessage(content="What's the GDP of New York?"), FunctionMessage(content='[{\'url\': \'https://www.governor.ny.gov/programs/fy-2024-new-york-state-budget\', \'content\': "The $229 billion FY 2024 New York State Budget reflects Governor Hochul\'s bold agenda to make New York more affordable,  FY 2024 Budget Assets FY 2024 New York State Budget Highlights Improving Public Safety  GOVERNOR HOME GOVERNOR KATHY HOCHUL FY 2024 New York State Budget  Transformative investments to support New York\'s business community and boost the state economy.The $229 billion FY 2024 NYS Budget reflects Governor Hochul\'s bold agenda to make New York more affordable, more livable, and safer."}]', additional_kwargs={'idx': 0}, name='tavily_search_results_json'), AIMessage(content="Thought: The information provided does not specify the Gross Domestic Product (GDP) of New York, but instead provides details about the state's budget for fiscal year 2024, which is $229 billion. This budget figure cannot be accurately equated to the GDP."), SystemMessage(content="Context from last attempt: The search results provided information about New York's state budget rather than its GDP. To answer the user's question, we need to find specific data on New York's GDP, not its budget. - Begin counting at : 1"), FunctionMessage(content="[{'url': 'https://en.wikipedia.org/wiki/Economy_of_New_York_(state)', 'content': 'The economy of the State of New York is reflected in its gross state product in 2022 of $2.053 trillion, ranking third  Contents Economy of New York (state)  New York City-centered metropolitan statistical area produced a gross metropolitan product (GMP) of $US2.0 trillion,  of the items in which New York ranks high nationally:The economy of the State of New York is reflected in its gross state product in 2022 of $2.053 trillion, ranking third in size behind the larger states of\\xa0...'}]", additional_kwargs={'idx': 1}, name='tavily_search_results_json'), AIMessage(content="Thought: The required information about New York's GDP is provided in the search results. In 2022, New York had a Gross State Product (GSP) of $2.053 trillion."), AIMessage(content='The Gross Domestic Product (GDP) of New York in 2022 was $2.053 trillion.')]}
---

```

```
# Final answer
print(step[END][-1].content)
```

```
The Gross Domestic Product (GDP) of New York in 2022 was $2.053 trillion.
```



#### Multi-hop question

This question requires that the agent perform multiple searches.

```
steps = chain.stream(
    [
        HumanMessage(
            content="What's the oldest parrot alive, and how much longer is that than the average?"
        )
    ],
    {
        "recursion_limit": 100,
    },
)
for step in steps:
    print(step)
    print("---")
```



```

{'plan_and_schedule': [FunctionMessage(content="[{'url': 'https://a-z-animals.com/blog/discover-the-worlds-oldest-parrot/', 'content': 'How Old Is the World’s Oldest Parrot?  Discover the World’s Oldest Parrot Advertisement  of debate, so we’ll detail some other parrots whose lifespans may be longer but are hard to verify their exact age.  Comparing Parrots’ Lifespans to Other BirdsSep 8, 2023 — Sep 8, 2023The oldest parrot on record is Cookie, a pink cockatoo that survived to the age of 83 and survived his entire life at the Brookfield Zoo.'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json'), FunctionMessage(content="HTTPError('502 Server Error: Bad Gateway for url: https://api.tavily.com/search')", additional_kwargs={'idx': 1}, name='tavily_search_results_json'), FunctionMessage(content='join', additional_kwargs={'idx': 2}, name='join')]}
---
{'join': [AIMessage(content='Thought: The oldest parrot on record is Cookie, a pink cockatoo, who lived to be 83 years old. However, there was an error fetching additional search results to compare this age to the average lifespan of parrots.'), SystemMessage(content='Context from last attempt: I found the age of the oldest parrot, Cookie, who lived to be 83 years old. However, I need to search again to find the average lifespan of parrots to complete the comparison.')]}
---
{'plan_and_schedule': [FunctionMessage(content='[{\'url\': \'https://www.turlockvet.com/site/blog/2023/07/15/parrot-lifespan--how-long-pet-parrots-live\', \'content\': "Parrot Lifespan  the lifespan of a parrot?\'.  Parrot Lifespan: How Long Do Pet Parrots Live?  how long they actually live and what you should know about owning a parrot.Jul 15, 2023 — Jul 15, 2023Generally, the average lifespan of smaller species of parrots such as Budgies and Cockatiels is about 5 - 15 years, while larger parrots such as\\xa0..."}]', additional_kwargs={'idx': 3}, name='tavily_search_results_json')]}
---
{'join': [AIMessage(content="Thought: I have found that the oldest parrot on record, Cookie, lived to be 83 years old. Additionally, I've found that the average lifespan of parrots varies by species, with smaller species like Budgies and Cockatiels living between 5-15 years, and larger parrots potentially living longer. This allows me to compare Cookie's age to the average lifespan of smaller parrot species."), AIMessage(content="The oldest parrot on record is Cookie, a pink cockatoo, who lived to be 83 years old. Compared to the average lifespan of smaller parrot species such as Budgies and Cockatiels, which is about 5-15 years, Cookie lived significantly longer. The average lifespan of larger parrot species wasn't specified, but it's implied that larger parrots may live longer than smaller species, yet likely still much less than 83 years.")]}
---
{'__end__': [HumanMessage(content="What's the oldest parrot alive, and how much longer is that than the average?"), FunctionMessage(content="[{'url': 'https://a-z-animals.com/blog/discover-the-worlds-oldest-parrot/', 'content': 'How Old Is the World’s Oldest Parrot?  Discover the World’s Oldest Parrot Advertisement  of debate, so we’ll detail some other parrots whose lifespans may be longer but are hard to verify their exact age.  Comparing Parrots’ Lifespans to Other BirdsSep 8, 2023 — Sep 8, 2023The oldest parrot on record is Cookie, a pink cockatoo that survived to the age of 83 and survived his entire life at the Brookfield Zoo.'}]", additional_kwargs={'idx': 0}, name='tavily_search_results_json'), FunctionMessage(content="HTTPError('502 Server Error: Bad Gateway for url: https://api.tavily.com/search')", additional_kwargs={'idx': 1}, name='tavily_search_results_json'), FunctionMessage(content='join', additional_kwargs={'idx': 2}, name='join'), AIMessage(content='Thought: The oldest parrot on record is Cookie, a pink cockatoo, who lived to be 83 years old. However, there was an error fetching additional search results to compare this age to the average lifespan of parrots.'), SystemMessage(content='Context from last attempt: I found the age of the oldest parrot, Cookie, who lived to be 83 years old. However, I need to search again to find the average lifespan of parrots to complete the comparison. - Begin counting at : 3'), FunctionMessage(content='[{\'url\': \'https://www.turlockvet.com/site/blog/2023/07/15/parrot-lifespan--how-long-pet-parrots-live\', \'content\': "Parrot Lifespan  the lifespan of a parrot?\'.  Parrot Lifespan: How Long Do Pet Parrots Live?  how long they actually live and what you should know about owning a parrot.Jul 15, 2023 — Jul 15, 2023Generally, the average lifespan of smaller species of parrots such as Budgies and Cockatiels is about 5 - 15 years, while larger parrots such as\\xa0..."}]', additional_kwargs={'idx': 3}, name='tavily_search_results_json'), AIMessage(content="Thought: I have found that the oldest parrot on record, Cookie, lived to be 83 years old. Additionally, I've found that the average lifespan of parrots varies by species, with smaller species like Budgies and Cockatiels living between 5-15 years, and larger parrots potentially living longer. This allows me to compare Cookie's age to the average lifespan of smaller parrot species."), AIMessage(content="The oldest parrot on record is Cookie, a pink cockatoo, who lived to be 83 years old. Compared to the average lifespan of smaller parrot species such as Budgies and Cockatiels, which is about 5-15 years, Cookie lived significantly longer. The average lifespan of larger parrot species wasn't specified, but it's implied that larger parrots may live longer than smaller species, yet likely still much less than 83 years.")]}
---

```

```
# Final answer
print(step[END][-1].content)
```

```

The oldest parrot on record is Cookie, a pink cockatoo, who lived to be 83 years old. Compared to the average lifespan of smaller parrot species such as Budgies and Cockatiels, which is about 5-15 years, Cookie lived significantly longer. The average lifespan of larger parrot species wasn't specified, but it's implied that larger parrots may live longer than smaller species, yet likely still much less than 83 years.
```



#### Multi-step math

```
for step in chain.stream(
    [
        HumanMessage(
            content="What's ((3*(4+5)/0.5)+3245) + 8? What's 32/4.23? What's the sum of those two values?"
        )
    ]
):
    print(step)
```



```

{'plan_and_schedule': [FunctionMessage(content='3307.0', additional_kwargs={'idx': 1}, name='math'), FunctionMessage(content='7.565011820330969', additional_kwargs={'idx': 2}, name='math'), FunctionMessage(content='3314.565011820331', additional_kwargs={'idx': 3}, name='math'), FunctionMessage(content='join', additional_kwargs={'idx': 4}, name='join')]}
{'join': [AIMessage(content="Thought: The calculations for each part of the user's question have been successfully completed. The first calculation resulted in 3307.0, the second in 7.565011820330969, and the sum of those two values was correctly found to be 3314.565011820331."), AIMessage(content='The result of ((3*(4+5)/0.5)+3245) + 8 is 3307.0, the result of 32/4.23 is approximately 7.565, and the sum of those two values is approximately 3314.565.')]}
{'__end__': [HumanMessage(content="What's ((3*(4+5)/0.5)+3245) + 8? What's 32/4.23? What's the sum of those two values?"), FunctionMessage(content='3307.0', additional_kwargs={'idx': 1}, name='math'), FunctionMessage(content='7.565011820330969', additional_kwargs={'idx': 2}, name='math'), FunctionMessage(content='3314.565011820331', additional_kwargs={'idx': 3}, name='math'), FunctionMessage(content='join', additional_kwargs={'idx': 4}, name='join'), AIMessage(content="Thought: The calculations for each part of the user's question have been successfully completed. The first calculation resulted in 3307.0, the second in 7.565011820330969, and the sum of those two values was correctly found to be 3314.565011820331."), AIMessage(content='The result of ((3*(4+5)/0.5)+3245) + 8 is 3307.0, the result of 32/4.23 is approximately 7.565, and the sum of those two values is approximately 3314.565.')]}

```

```
# Final answer
print(step[END][-1].content)
```

```
The result of ((3*(4+5)/0.5)+3245) + 8 is 3307.0, the result of 32/4.23 is approximately 7.565, and the sum of those two values is approximately 3314.565.
```



## Conclusion

Congrats on building your first LLMCompiler agent! I'll leave you with some known limitations to the implementation above:

1. The planner output parsing format is fragile if your function requires more than 1 or 2 arguments. We could make it more robust by using streaming tool calling.
2. Variable substitution is fragile in the example above. It could be made more robust by using a fine-tuned model and a more robust syntax (using e.g., Lark or a tool calling schema)
3. The state can grow quite long if you require multiple re-planning runs. To handle, you could add a message compressor once you go above a certain token limit.