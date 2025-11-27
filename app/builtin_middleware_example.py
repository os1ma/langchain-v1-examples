from collections.abc import Iterator

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph


class BuiltinMiddlewareExampleAgent:
    def __init__(self) -> None:
        self.agent: CompiledStateGraph = create_agent(  # type: ignore[type-arg]
            model="anthropic:claude-sonnet-4-5-20250929",
            tools=[],
            middleware=[AnthropicPromptCachingMiddleware(ttl="5m")],
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
        thread_id: str,
    ) -> Iterator[BaseMessage]:
        stream_input = {"messages": [HumanMessage(content=message)]}
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        for chunk in self.agent.stream(
            input=stream_input,
            config=config,
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
