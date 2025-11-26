from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

load_dotenv()


@dataclass
class Context:
    user_id: str


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore[type-arg] # noqa: ARG001
    context: Context = runtime.context  # type: ignore[assignment]
    user_id = context.user_id
    print(f"before model: user_id={user_id}")
    return None


agent: CompiledStateGraph = create_agent(  # type: ignore[type-arg]
    model="openai:gpt-5-mini",
    tools=[],
    middleware=[log_before_model],
)


def main() -> None:
    thread_id = uuid4().hex
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    user_id = "123"
    context = Context(user_id=user_id)

    while True:
        human_input = input("> ")

        for chunk in agent.stream(
            input={"messages": [HumanMessage(content=human_input)]},
            config=config,
            context=context,
            stream_mode="updates",
        ):
            print(chunk)


if __name__ == "__main__":
    main()
