import streamlit as st
from langgraph.graph import StateGraph
from typing import Dict, List, TypedDict
import json
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import jamo
import subprocess
import os
import rospy
import moveit_commander
import rospy
import geometry_msgs.msg
from typing import Annotated, Dict

class RobotState(TypedDict):
    text: str
    decomposed_text: List[str]
    available_tools: List[Dict]
    selected_tools: List[Dict]
    adjusted_coordinates: Dict[str, Dict[str, float]]
    generated_code: List[Dict[str, float]]
    result: str


moveit_commander.roscpp_initialize([])
robot = moveit_commander.RobotCommander()
group = moveit_commander.MoveGroupCommander("panda_arm")

# Define Agents
class SelectToolAgent:
    def __init__(self, tools):
        """Initialize the agent with available tools and LLM."""
        self.tools = tools
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=200)

    def run(self, decomposed_text: List[str]):
        """Use LLM to select the necessary tools based on decomposed text."""
        prompt = f"""
        You are selecting tools for a robotic arm that draws Hangul characters.

        Given the following Korean consonants and vowels:
        {', '.join(decomposed_text)}

        Match each component **EXACTLY** to one of the tool names from the list below:
        {', '.join([tool['title'] for tool in self.tools])}

        **STRICT OUTPUT RULES:**
        - Return ONLY the tool names in a comma-separated format.
        - Do NOT add explanations, descriptions, or extra text.
        - The response MUST follow this format: `ㄱ, ㅏ, ㄴ`
        - If a character is not in the tool list, it must be omitted from the output.

        **Example Input & Output:**
        - Input: ['ㄱ', 'ㅏ', 'ㄴ']
        - Available Tools: ㄱ, ㄴ, ㄷ, ㅏ, ㅓ, ㅗ
        - Output: ㄱ, ㅏ, ㄴ
        """


        response = self.llm.invoke([
            SystemMessage(content="You are a tool selection assistant."),
            HumanMessage(content=prompt)
        ])

        selected_tools = [t.strip() for t in response.content.split(",")]

        st.write("[DEBUG] Parsed Selected Tools:", selected_tools)  

        tool_dict = {tool["title"]: tool for tool in self.tools}
        final_tools = [tool_dict[t] for t in selected_tools if t in tool_dict]

        return final_tools

# Node: Modify Coordinates
class ModifyCoordinatesAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", max_tokens=1000)

    def run(self, state: Dict[str, any]):
        original_text = state["text"]
        selected_tools = state["selected_tools"]
        tool_names = ", ".join([tool["title"] for tool in selected_tools])
        prompt = f"""
        You are an AI specializing in modifying Hangul character component positions for natural handwriting.
        
        **Character to Write:** '{original_text}'
        **Given Components:** {tool_names}
        
        **Your Task:**
        - Adjust the relative positions of the selected tools to form '{original_text}' naturally.
        - Ensure the spacing and alignment between components are correct.
        - Maintain consistent stroke flow without unnatural gaps.
        - Return ONLY a valid JSON object with the adjusted coordinates and nothing else.
        - When modifying coordinates, please apply the following rules:
        
        - For vowels:
            - If the vowel is a single vowel or the first component of a compound vowel (such as ㅏ, ㅑ, ㅓ, ㅕ, ㅐ, ㅒ, ㅔ, ㅖ, ㅣ), position it strictly to the right of the initial consonant by applying only a horizontal (rightward) offset.
            - If the vowel is the second component in a compound vowel (such as the second part of ㅘ, ㅙ, ㅚ, ㅝ, ㅞ, ㅟ, ㅢ), position it strictly downward by applying only a vertical (downward) offset.
        
        - For consonants:
            - For the first consonant (initial consonant), apply a slight leftward adjustment (only a horizontal offset to the left).
            - For the third consonant (final consonant, if present), apply an adjustment that moves it only downward (vertical offset).
        
        **Example Output Format:**
        {{
            "ㄱ": [
                {{"x": -0.2, "z": 0.0}}
            ],
            "ㅏ": [
                {{"x": 0.2, "z": 0.0}}
            ],
            "ㄴ": [
                {{"x": 0.1, "z": -0.2}}
            ]
        }}
        These value is offset that applied at tools.
        """
        response = self.llm.invoke([
            SystemMessage(content="You are a Hangul character positioning assistant."),
            HumanMessage(content=prompt)
        ])
        try:
            adjusted_coordinates = json.loads(response.content)
            state["adjusted_coordinates"] = adjusted_coordinates
            state["result"] = "Coordinates adjusted"
            return state
        except json.JSONDecodeError:
            state["adjusted_coordinates"] = {}
            return state



# Node: Create Tools
def create_tools(state):
    """Load available tools from commands_words.json and add them to state."""
    try:
        with open("commands_words.json", "r", encoding="utf-8") as file:
            tools_data = json.load(file)
    except Exception as e:
        st.write("[ERROR] Failed to load commands_words.json:", str(e))
        state["available_tools"] = []
        return state

    # Convert tools data into a list of dictionaries
    if isinstance(tools_data, dict) and "characters" in tools_data:
        available_tools = [
            {"title": key, "strokes": value.get("strokes", []), "y": value.get("y", 0.0)}
            for key, value in tools_data["characters"].items()
        ]
    elif isinstance(tools_data, list):
        available_tools = tools_data
    else:
        st.write("[ERROR] Unexpected format for tools")
        available_tools = []

    state["available_tools"] = available_tools
    st.write("[DEBUG] Available Tools Loaded into State:", [tool["title"] for tool in available_tools])
    return state

# Node: Decompose Text
def decompose_text(state):
    """Decompose Hangul syllables into individual jamo characters."""
    from jamo import h2j, j2hcj  
    state["decomposed_text"] = list(j2hcj(h2j(state["text"])))
    st.write("[DEBUG] Decomposed Text:", state["decomposed_text"])
    return state


# Node: Select Tools
def select_tools(state: Dict[str, any]) -> Dict[str, any]:
    """Select the necessary tools by mapping decomposed text to available tools.
    
    This node uses the existing SelectToolAgent to preserve maintainability.
    However, after obtaining the LLM output, it filters the tools based on the decomposed text,
    ensuring that only tools corresponding to the decomposed characters are selected.
    """
    if "decomposed_text" not in state:
        raise KeyError("[ERROR] 'decomposed_text' is missing from state!")
    if "available_tools" not in state:
        st.write("[ERROR] 'available_tools' is missing from state!")
        state["selected_tools"] = []
        return state

    available_tools = state.get("available_tools", [])
    if not available_tools:
        st.write("[ERROR] 'available_tools' is empty!")
        state["selected_tools"] = []
        return state


    agent = SelectToolAgent(available_tools)
    try:
        raw_selected = agent.run(state["decomposed_text"])
    except Exception as e:
        st.write("[ERROR] LLM Call Failed:", str(e))
        state["selected_tools"] = []
        return state
    tool_dict = {tool["title"]: tool for tool in available_tools}
    selected_tools = [tool_dict[char] for char in state["decomposed_text"] if char in tool_dict]

    st.write("Selected Tools:", [tool["title"] for tool in selected_tools])
    state["selected_tools"] = selected_tools
    return state



def modify_coordinates(state: Dict[str, any]) -> Dict[str, any]:
    """
    Use ModifyCoordinatesAgent to adjust coordinates and output the result.
    
    This function instantiates the ModifyCoordinatesAgent and calls its run method,
    preserving the agent structure for maintainability. After the coordinates are adjusted,
    it outputs the adjusted coordinates using st.write.
    
    Args:
        state (Dict[str, any]): The current state containing text and selected_tools.
        
    Returns:
        Dict[str, any]: Updated state with adjusted_coordinates.
    """
    agent = ModifyCoordinatesAgent()
    updated_state = agent.run(state)
    if "adjusted_coordinates" in updated_state:
        st.write("Modified Coordinates:", updated_state["adjusted_coordinates"])
    else:
        st.write("Modified Coordinates not found.")
    return updated_state



# Node: Generate Code
def generate_code(state: Dict[str, any]) -> Dict[str, any]:
    """Generate waypoints grouped by tool by applying offset values to original stroke coordinates from selected tools."""
    generated_code_by_tool = {}
    
    # Iterate over each selected tool
    for tool in state.get("selected_tools", []):
        title = tool["title"]
        original_strokes = tool.get("strokes", [])
        # Retrieve offset value (assumed as a list with one offset object)
        offset_data = state.get("adjusted_coordinates", {}).get(title)
        if offset_data and isinstance(offset_data, list) and len(offset_data) > 0:
            offset = offset_data[0]
        else:
            offset = {"x": 0.0, "z": 0.0}
        
        # Apply the offset to each stroke coordinate
        modified_strokes = []
        for stroke in original_strokes:
            modified_stroke = {
                "x": stroke["x"] + offset["x"],
                "y": tool.get("y", -0.05),  # Use tool's y value
                "z": stroke["z"] + offset["z"]
            }
            # Round each coordinate to 2 decimal places
            modified_stroke = {k: round(v, 2) for k, v in modified_stroke.items()}
            modified_strokes.append(modified_stroke)
        # Group modified strokes by tool title
        generated_code_by_tool[title] = modified_strokes
    
    state["generated_code_by_tool"] = generated_code_by_tool
    state["result"] = "Code generated"
    st.write("### Generated Code (Grouped by Tool):")
    st.write(generated_code_by_tool)
    return state

def execute_code(state: RobotState) -> RobotState:
    waypoints = state.get("generated_code", [])
    if not waypoints:
        return {"result": "No code to execute."}
    
    try:
        if not rospy.core.is_initialized():
            rospy.init_node("franka_wall_writing", anonymous=True, disable_signals=True)
        
        (plan, fraction) = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        if plan:
            group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()
        return {"result": "MoveIt execution completed successfully!"}
    except Exception as e:
        return {"result": f"Execution failed: {str(e)}"}

class WorkflowState(TypedDict):
    text: str
    decomposed_text: List[str]
    selected_tools: List[str]
    generated_code: str

# Define the LangGraph Workflow
workflow = StateGraph(RobotState)
workflow.add_node("create_tools", create_tools)
workflow.add_node("decompose_text", decompose_text)
workflow.add_node("select_tools", select_tools)
workflow.add_node("modify_coordinates", modify_coordinates)
workflow.add_node("generate_code", generate_code)
# workflow.add_node("execute_code", execute_code)

workflow.add_edge("create_tools", "decompose_text")
workflow.add_edge("decompose_text", "select_tools")
workflow.add_edge("select_tools", "modify_coordinates")
workflow.add_edge("modify_coordinates", "generate_code")
# workflow.add_edge("generate_code", "execute_code")

# Set the entry point to the create_tools node
workflow.set_entry_point("create_tools")
graph = workflow.compile()

# Streamlit UI
st.title("Franka Robot Wall Writing")
user_input = st.text_input("Enter a character:")
if st.button("Generate and Execute"):
    result = graph.invoke({"text": user_input})
    st.write("Execution Completed.")
    if "result" in result:
        st.write("MoveIt Execution Result:", result["result"])

