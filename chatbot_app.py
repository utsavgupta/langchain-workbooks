import dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, Graph, START, END
from langgraph.graph.graph import CompiledGraph
from colorama import Style, Fore
import sys

class State(TypedDict):
    messages: Annotated[list, add_messages]

class ChatBotFactory:

    def __init__(self, llm: LLM):
        self._llm = llm

    def build(self):
        return lambda state: {"messages": [ self._llm.invoke(state["messages"]) ]}

def main():
    dotenv.load_dotenv()

    llm = build_llm()
    chatbot = ChatBotFactory(llm).build()

    graph_builder = build_state_graph_builder()
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile()
    ret = run_graph(graph=graph)

    sys.exit(ret)

def run_graph(graph: CompiledGraph) -> int:

    while True:
        print(f"{Fore.LIGHTGREEN_EX}User: {Style.RESET_ALL}", end="")
        user_input = input()
        if user_input.lower() in ["exit", "quit"]:
            print(f"{Fore.LIGHTRED_EX}Assistant: {Style.RESET_ALL} Bye!", end="")
            return 0
        else:
            for event in graph.stream({"messages": ("user", user_input)}):
                for value in event.values():
                    print(f"{Fore.LIGHTRED_EX}Assistant: {Style.RESET_ALL}{value['messages'][-1].content}")

def build_state_graph_builder() -> Graph:

    graph_builder = StateGraph(State)

    return graph_builder

def build_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    return llm

def build_llm_with_tool(tools: list):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    llm.bind_tools(tools = tools)
    return llm

if __name__ == '__main__':
    main()