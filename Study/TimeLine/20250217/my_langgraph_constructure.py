import langgraph.graph as lg
from typing import Annotated, Dict, List
import graphviz
import networkx as nx
import streamlit as st
import json
import faiss
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings

# Load tools from commands.json
with open("commands.json", "r") as file:
    tools = json.load(file)

def get_tool_by_title(title: str):
    """Find a tool by its title in the commands.json file."""
    for tool in tools:
        if tool["title"].lower() == title.lower():
            return tool["code"]
    return None

# Initialize LLM and Embeddings
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
embeddings = OpenAIEmbeddings()

# Initialize FAISS Vector Database
vector_dim = 1536  # OpenAI embedding size
index = faiss.IndexFlatL2(vector_dim)
execution_data = []  # Store execution results

# Streamlit UI for input
st.title("LangGraph Agent System")
st.write("Enter a command and see how the agents process it.")

# User input with a submit button
user_input = st.text_input("Enter your command:", label_visibility="visible")
execute_button = st.button("Execute")

# Define state class
data = Annotated[Dict[str, str], "State"]

graph = lg.StateGraph(data)

# Agent 1: LLM-based refinement of input
def agent1(input_text):
    response = llm.invoke([HumanMessage(content=f"Refine this user input for better understanding: {input_text}")])
    return response.content

# Agent 2: LLM-based execution planning
def agent2(plan):
    response = llm.invoke([HumanMessage(content=f"Generate an execution plan for the following task: {plan}")])
    return response.content

# Agent 3: LLM-based tool selection
def agent3(plan):
    response = llm.invoke([HumanMessage(content=f"Select the best tool for executing the following plan. Here are available tools: {[tool['title'] for tool in tools]}. Plan: {plan}")])
    return response.content

# Agent 4: Vector Database-based Learning
def agent4(result):
    query_vector = embeddings.embed_query(result)  # Convert result to embedding
    query_vector = np.array(query_vector).reshape(1, -1)
    if index.ntotal > 0:
        distances, indices = index.search(query_vector, 1)  # Find closest match
        best_match = execution_data[indices[0][0]] if distances[0][0] < 0.5 else "No similar past execution"
    else:
        best_match = "No previous data"
    return best_match

# Node 1: Process natural language input with Agent 1
def refine_input_node(state: data):
    state["input"] = agent1(state.get("input", ""))
    return {"plan_execution": state}

# Node 2: Plan execution using Agent 2
def plan_execution_node(state: data):
    state["plan"] = agent2(state.get("input", ""))
    return {"tool_selection": state}

# Node 3: Select appropriate tool using Agent 3
def tool_selection_node(state: data):
    selected_tool = agent3(state.get("plan", ""))
    state["selected_tool"] = selected_tool
    state["tool_code"] = get_tool_by_title(selected_tool)
    return {"execute_action": state}

# Node 4: Execute the selected tool (Simulation for now)
def execute_action_node(state: data):
    state["result"] = f"Simulated Execution using tool: {state.get('selected_tool', '')}"
    return {"train_agents": state}

# Node 5: Train Agents using FAISS Vector DB
def train_agents_node(state: data):
    result_text = state.get("result", "")
    result_vector = embeddings.embed_query(result_text)  # Convert to embedding
    result_vector = np.array(result_vector).reshape(1, -1)
    index.add(result_vector)  # Store embedding in FAISS
    execution_data.append(result_text)  # Store result text
    state["trained"] = agent4(result_text)
    return {"refine_input": state}

# Add nodes to graph
graph.add_node("refine_input", refine_input_node)
graph.add_node("plan_execution", plan_execution_node)
graph.add_node("tool_selection", tool_selection_node)
graph.add_node("execute_action", execute_action_node)
graph.add_node("train_agents", train_agents_node)

# Set entry point
graph.set_entry_point("refine_input")

# Define edges
graph.add_edge("refine_input", "plan_execution")
graph.add_edge("plan_execution", "tool_selection")
graph.add_edge("tool_selection", "execute_action")
graph.add_edge("execute_action", "train_agents")
graph.add_edge("train_agents", "refine_input")  # Loop for continuous interaction

# Compile and visualize with execution tracking
def visualize_graph_with_execution(graph, executed_nodes):
    final_graph = graph.compile()
    nx_graph = final_graph.get_graph()
    
    dot = graphviz.Digraph()
    
    for node in nx_graph.nodes:
        if node in executed_nodes:
            dot.node(node, style="filled", fillcolor="red")  # Highlight executed nodes in red
        else:
            dot.node(node)
    
    for edge in nx_graph.edges:
        dot.edge(edge[0], edge[1])
    
    return dot

# Streamlit visualization
st.title("LangGraph Execution Visualization")
executed_nodes = []

if execute_button and user_input:
    state = {"input": user_input}
    output = refine_input_node(state)
    executed_nodes.append("refine_input")
    output = plan_execution_node(output["plan_execution"])
    executed_nodes.append("plan_execution")
    output = tool_selection_node(output["tool_selection"])
    executed_nodes.append("tool_selection")
    output = execute_action_node(output["execute_action"])
    executed_nodes.append("execute_action")
    output = train_agents_node(output["train_agents"])
    executed_nodes.append("train_agents")
    
    st.write("### Final Output:")
    st.json(output["refine_input"])

# Display graph
graph_viz = visualize_graph_with_execution(graph, executed_nodes)
st.graphviz_chart(graph_viz)

