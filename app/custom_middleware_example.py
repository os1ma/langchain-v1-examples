from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain.messages import HumanMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime


@dataclass
class Context:
    user_id: str


@before_model
def _log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:  # type: ignore[type-arg]
    context: Context = runtime.context  # type: ignore[assignment]
    user_id = context.user_id
    message = state["messages"][-1].content
    print(f"before model: user_id={user_id}, message={message}")
    return None


class CustomMiddlewareExampleAgent:
    def __init__(self) -> None:
        self.agent: CompiledStateGraph = create_agent(
            model="openai:gpt-5-mini",
            tools=[],
            middleware=[_log_before_model],
            checkpointer=InMemorySaver(),
        )

    def get_messages(self, thread_id: str) -> list[BaseMessage]:
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        state_snapshot = self.agent.get_state(config=config)

        if "messages" in state_snapshot.values:
            return state_snapshot.values["messages"]  # type: ignore[no-any-return]
        else:
            return []

    def stream(
        self,
        message: str,
        user_id: str,
        thread_id: str,
    ) -> Iterator[BaseMessage]:
        context = Context(user_id=user_id)
        stream_input = {"messages": [HumanMessage(content=message)]}
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for chunk in self.agent.stream(
            input=stream_input,
            config=config,
            context=context,
            stream_mode="updates",
        ):
            if "model" in chunk:
                messages = chunk["model"]["messages"]
                for m in messages:
                    yield m
            if "tools" in chunk:
                messages = chunk["tools"]["messages"]
                for m in messages:
                    yield m
