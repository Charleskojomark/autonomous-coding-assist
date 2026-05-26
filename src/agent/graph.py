import os
import logging
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.prompts import CODEBASE_QA_SYSTEM_PROMPT
from src.agent.tools import search_codebase, get_file, get_issues

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_name: str

def create_agent_graph():
    # Load LLM
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    if not api_key:
        logger.warning("No GROQ_API_KEY found. Agent calls will fail.")
        
    llm = ChatGroq(
        temperature=0.1,
        model_name=model_name,
        groq_api_key=api_key or "gsk_dummy_temp_testing_key"
    )
    
    # Register tools
    tools = [search_codebase, get_file, get_issues]
    llm_with_tools = llm.bind_tools(tools)
    
    # 1. Define Node: Agent call
    def call_model(state: AgentState):
        messages = list(state["messages"])
        repo_name = state["repo_name"]
        
        # Check if system message already exists, if not prepend it
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content=f"{CODEBASE_QA_SYSTEM_PROMPT}\nActive Repository: {repo_name}"
            )
            messages.insert(0, system_msg)
            
        logger.info(f"Invoking LLM with {len(messages)} messages...")
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    # 2. Define Node: Tools executor
    tool_node = ToolNode(tools)
    
    # 3. Define routing logic
    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"Agent requested tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
            return "tools"
        logger.info("Agent did not request further tools. Ending session.")
        return END

    # Wire up the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # Tool output flows back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile
    app = workflow.compile()
    return app
