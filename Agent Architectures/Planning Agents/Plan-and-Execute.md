# Plan-and-Execute

This notebook shows how to create a "plan-and-execute" style agent. This is heavily inspired by the [Plan-and-Solve](https://arxiv.org/abs/2305.04091) paper as well as the [Baby-AGI](https://github.com/yoheinakajima/babyagi) project.

The core idea is to first come up with a multi-step plan, and then go through that plan one item at a time. After accomplishing a particular task, you can then revisit the plan and modify as appropriate.

The general computational graph looks like the following:

![plan-and-execute diagram](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/img/plan-and-execute.png)

This compares to a typical [ReAct](https://arxiv.org/abs/2210.03629) style agent where you think one step at a time. The advantages of this "plan-and-execute" style agent are:

1. Explicit long term planning (which even really strong LLMs can struggle with)
2. Ability to use smaller/weaker models for the execution step, only using larger/better models for the planning step

The following walkthrough demonstrates how to do so in LangGraph. The resulting agent will leave a trace like the following example: ([link](https://smith.langchain.com/public/d46e24d3-dda6-44d5-9550-b618fca4e0d4/r)).

## Setup

First, we need to install the packages required.

```
pip install --quiet -U langgraph langchain-community langchain-openai tavily-python
```

Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
```

Optionally, we can set API key for LangSmith tracing, which will give us best-in-class observability.

```
os.environ["LANGCHAIN_TRACING_V2"] = "true"
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Plan-and-execute"
```

## Define Tools

We will first define the tools we want to use. For this simple example, we will use a built-in search tool via Tavily. However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/v0.2/docs/how_to/custom_tools) on how to do that.

```
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]
```

## Define our Execution Agent

Now we will create the execution agent we want to use to execute tasks. Note that for this example, we will be using the same execution agent for each task, but this doesn't HAVE to be the case.

```
from langchain import hub
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("wfh/react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools, messages_modifier=prompt)
```

![image-20240716120348106](./assets/image-20240716120348106.png)

```
agent_executor.invoke({"messages": [("user", "who is the winnner of the us open")]})
```



```

================================ System Message ================================

You are a helpful assistant.

============================= Messages Placeholder =============================

{{messages}}

{'messages': [HumanMessage(content='who is the winnner of the us open', id='7c491c9f-cdbe-4761-b93b-3e4eeb526c97'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_MMmwmxwxRH2hrmMbuBeMGsXW', 'function': {'arguments': '{"query":"US Open 2023 winner"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 97, 'total_tokens': 120}, 'model_name': 'gpt-4-turbo-preview', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-855f7cff-62a2-4dd8-b71b-707b507b00a4-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'US Open 2023 winner'}, 'id': 'call_MMmwmxwxRH2hrmMbuBeMGsXW'}]),
  ToolMessage(content='[{"url": "https://www.bbc.com/sport/tennis/66766337", "content": ": Stephen Nolan goes in to find out\\nRelated Topics\\nTop Stories\\nTen Hag on Rashford plus transfer news, WSL deadline day\\nSpinner Leach doubtful for second Test in India\\nMcIlroy \'changes tune\' on LIV players\' punishment\\nElsewhere on the BBC\\nDiscover the tropical paradise of Thailand\\nFrom the secrets of the South to the mysterious North...\\n Djokovic offered to help up Medvedev when the Russian fell to the court in the third set\\nDjokovic\'s relentless returning continued to draw mistakes out of Medvedev, who was serving poorly and making loose errors, at the start of the second set.\\n It was clear to see Medvedev had needed to level by taking that second set to stand any real chance of victory and the feeling of the inevitable was heightened by the Russian needing treatment on a shoulder injury before the third set.\\n Djokovic shows again why he can never be written off\\nWhen Djokovic lost to 20-year-old Carlos Alcaraz in the Wimbledon final it felt like a changing-of-the-guard moment in the men\'s game.\\n The inside story of Putin\\u2019s invasion of Ukraine\\nTold by the Presidents and Prime Ministers tasked with making the critical decisions\\nSurvival of the wittiest!\\n"}, {"url": "https://www.usopen.org/en_US/news/articles/2023-09-10/novak_djokovic_wins_24th_grand_slam_singles_title_at_2023_us_open.html", "content": "WHAT HAPPENED: Novak Djokovic handled the weight of history to defeat Daniil Medvedev on Sunday in the 2023 US Open men\'s singles final. With a 6-3, 7-6(5), 6-3 victory, the 36-year-old won his 24th Grand Slam singles title, tying Margaret Court\'s record and bolstering his case to be considered the greatest tennis player of all time."}, {"url": "https://apnews.com/article/us-open-final-live-updates-djokovic-medvedev-8a4a26f8d77ef9ab2fb3efe1096dce7e", "content": "Novak Djokovic wins the US Open for his 24th Grand Slam title by beating Daniil Medvedev\\nNovak Djokovic, of Serbia, holds up the championship trophy after defeating Daniil Medvedev, of Russia, in the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. (AP Photo/Manu Fernandez)\\nDaniil Medvedev, of Russia, sits on the court after a rally against Novak Djokovic, of Serbia, during the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. (AP Photo/Manu Fernandez)\\nDaniil Medvedev, of Russia, sits on the court after a rally against Novak Djokovic, of Serbia, during the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. (AP Photo/Manu Fernandez)\\nDaniil Medvedev, of Russia, sits on the court after a rally against Novak Djokovic, of Serbia, during the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York. Novak Djokovic, of Serbia, reveals a t-shirt honoring the number 24 and Kobe Bryant after defeating Daniil Medvedev, of Russia, in the men\\u2019s singles final of the U.S. Open tennis championships, Sunday, Sept. 10, 2023, in New York."}]', name='tavily_search_results_json', id='ca0ff812-6c7f-43c1-9d0e-427cfe8da332', tool_call_id='call_MMmwmxwxRH2hrmMbuBeMGsXW'),
  AIMessage(content="The winner of the 2023 US Open men's singles was Novak Djokovic. He defeated Daniil Medvedev with a score of 6-3, 7-6(5), 6-3 in the final, winning his 24th Grand Slam singles title. This victory tied Margaret Court's record and bolstered Djokovic's claim to be considered one of the greatest tennis players of all time.", response_metadata={'token_usage': {'completion_tokens': 89, 'prompt_tokens': 972, 'total_tokens': 1061}, 'model_name': 'gpt-4-turbo-preview', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-ef37a655-1ea6-470e-a310-8f125ca48015-0')]}
```

## Define the State

Let's now start by defining the state the track for this agent.

First, we will need to track the current plan. Let's represent that as a list of strings.

Next, we should track previously executed steps. Let's represent that as a list of tuples (these tuples will contain the step and then the result)

Finally, we need to have some state to represent the final response as well as the original input.

```
import operator
from typing import Annotated, List, Tuple, TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
```

## Planning Step

Let's now think about creating the planning step. This will use function calling to create a plan.

```
from langchain_core.pydantic_v1 import BaseModel, Field


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
```

```
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)
```

```
planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)
```

`Plan(steps=['Identify the current winner of the Australian Open.', 'Determine the hometown of the identified winner.'])`

## Re-Plan Step

Now, let's create a step that re-does the plan based on the result of the previous step.

```
from typing import Union


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)
```

## Create the Graph

We can now create the graph!

```
from typing import Literal


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": (task, agent_response["messages"][-1].content),
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    if "response" in state and state["response"]:
        return "__end__"
    else:
        return "agent"


```

```
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
```

![image-20240716120445027](./assets/image-20240716120445027.png)

```
config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the 2024 Australia open winner?"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```



```

{'plan': ['Identify the winner of the 2024 Australian Open.', 'Determine the hometown of the identified winner.']}
{'past_steps': ('Identify the winner of the 2024 Australian Open.', 'The winner of the 2024 Australian Open is Jannik Sinner. He claimed his first Grand Slam title in an epic comeback win over Daniil Medvedev.')}
{'plan': ['Determine the hometown of Jannik Sinner.']}
{'past_steps': ('Determine the hometown of Jannik Sinner.', "Jannik Sinner's hometown is not directly mentioned in the provided excerpts. To ensure accurate information, it's advisable to check a reliable source like his official ATP profile or a detailed biography which often includes personal background details such as hometown.")}
{'plan': ["Check Jannik Sinner's official ATP profile or a detailed biography to find his hometown.", 'Return the hometown of Jannik Sinner.']}
{'past_steps': ("Check Jannik Sinner's official ATP profile or a detailed biography to find his hometown.", "Jannik Sinner's official ATP profile can be found at this URL: [ATP Tour - Jannik Sinner](https://www.atptour.com/en/players/jannik-sinner/s0ag/overview). This profile will contain detailed information including his biography, rankings, playing activity, and potentially his hometown.")}
{'plan': ["Visit Jannik Sinner's official ATP profile or a detailed biography to find his hometown.", 'Return the hometown of Jannik Sinner.']}
{'past_steps': ("Visit Jannik Sinner's official ATP profile or a detailed biography to find his hometown.", "Jannik Sinner's official ATP profile and other reliable sources do not explicitly mention his hometown in the search results provided. For detailed information, visiting his ATP profile directly or consulting a comprehensive biography would be recommended to find this specific information.")}
{'plan': ["Visit Jannik Sinner's official ATP profile or a detailed biography to find his hometown.", 'Return the hometown of Jannik Sinner.']}
{'past_steps': ("Visit Jannik Sinner's official ATP profile or a detailed biography to find his hometown.", "Jannik Sinner's official ATP profile can be accessed [here](https://www.atptour.com/en/players/jannik-sinner/s0ag/overview), although it does not directly provide his hometown in the snippet. For detailed information, such as his hometown, it might be necessary to visit the profile directly or consult other detailed biographies like the one available on [Wikipedia](https://en.wikipedia.org/wiki/Jannik_Sinner), which often include personal details such as hometowns.")}
{'plan': ["Visit Jannik Sinner's official ATP profile or his Wikipedia page to find his hometown.", 'Return the hometown of Jannik Sinner.']}
{'past_steps': ("Visit Jannik Sinner's official ATP profile or his Wikipedia page to find his hometown.", "Jannik Sinner's official ATP profile and Wikipedia page did not directly mention his hometown in the provided excerpts. However, further information can typically be found by visiting the full pages directly through the provided links:\n\n- [Jannik Sinner's ATP Tour Profile](https://www.atptour.com/en/players/jannik-sinner/s0ag/overview)\n- [Jannik Sinner's Wikipedia Page](https://en.wikipedia.org/wiki/Jannik_Sinner)\n\nFor detailed information, including his hometown, I recommend checking these sources.")}
{'response': 'The necessary steps to find the hometown of the 2024 Australian Open winner, Jannik Sinner, have already been completed. His hometown is Innichen, Italy.'}
```

## Conclusion

Congrats on making a plan-and-execute agent! One known limitations of the above design is that each task is still executed in sequence, meaning embarrassingly parallel operations all add to the total execution time. You could improve on this by having each task represented as a DAG (similar to LLMCompiler), rather than a regular list.