import os
import json
import difflib
import rospy
import moveit_commander
import geometry_msgs.msg
import streamlit as st
import langgraph.graph as lg
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import TypedDict

# OpenAI API 설정
os.environ["OPENAI_API_KEY"] = ""

# ROS 및 MoveIt 초기화
rospy.init_node("franka_web_controller", anonymous=True, disable_signals=True)
moveit_commander.roscpp_initialize([])
robot = moveit_commander.RobotCommander()
arm = moveit_commander.MoveGroupCommander("panda_arm")

# LLM 인스턴스
llm_gpt4 = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# JSON 예제 명령어 로드
def load_examples():
    """ JSON 파일에서 기존 예제 로드 """
    examples_path = "commands.json"
    if not os.path.exists(examples_path):
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
    with open(examples_path, "r") as file:
        return json.load(file)

examples = load_examples()

# LangGraph 상태 정의
class GraphState(TypedDict):
    input: str
    selected_command: str
    code: str
    result: str

graph = lg.StateGraph(GraphState)

def input_analyzer(state: GraphState) -> GraphState:
    prompt = PromptTemplate(
        input_variables=["input"],
        template="Extract the core action from the following command:\nCommand: {input}\nCore Action (one short sentence only):"
    )
    response = llm_gpt4.invoke(prompt.format(input=state["input"])).content.strip()

    new_state = {**state, "selected_command": response}
    return new_state


def command_matcher(state: GraphState) -> GraphState:
    selected_command = state["selected_command"].strip().rstrip(".")
    best_match = None
    best_score = 0

    for cmd in examples:
        json_title = cmd["title"].strip()
        similarity = difflib.SequenceMatcher(None, selected_command.lower(), json_title.lower()).ratio()

        if similarity > best_score:
            best_match = cmd
            best_score = similarity

    matched_code = best_match["code"] if best_match and best_score > 0.7 else None  # 🚨 빈 문자열이 아니라 None으로 설정

    new_state = {**state, "code": matched_code}
    return new_state

def code_converter(state: GraphState) -> GraphState:
    if state["code"]:
        return state  # 기존 코드가 있으면 변환 생략

    # 🎯 **가장 유사한 기존 명령어 찾기 (예: "draw a circle" 참고)**
    best_example = None
    best_score = 0
    for ex in examples:
        similarity = difflib.SequenceMatcher(None, state["selected_command"].lower(), ex["title"].lower()).ratio()
        if similarity > best_score:
            best_example = ex
            best_score = similarity

    example_code = best_example["code"] if best_example else ""

    prompt = f"""
    You are an expert in robotics and MoveIt for controlling a Franka Emika Panda robot.
    Generate a Python script using MoveIt to execute the following task: "{state['selected_command']}"

    🎯 **Existing Example Reference** (for guidance):
    ```
    {example_code}
    ```

    Ensure the generated script:
    - Uses MoveIt for motion planning (`set_pose_target`, `plan`, `go`)
    - Is optimized for simulation in Gazebo.
    - Does NOT include markdown formatting or explanations, ONLY return Python code.
    """
    response = llm_gpt4.invoke(prompt).content.strip()

    new_state = {**state, "code": response}
    return new_state

def code_optimizer(state: GraphState) -> GraphState:
    prompt = f"""
    Optimize the following MoveIt script for efficiency and correctness:
    ```
    {state['code']}
    ```
    Optimized Code:
    """
    optimized_code = llm_gpt4.invoke(prompt).content.strip()

    new_state = {**state, "result": optimized_code}
    return new_state


def output(state: GraphState) -> GraphState:
    code_to_execute = state["code"].strip()

    if not code_to_execute:
        return {**state, "result": "No code to execute"}

    try:
        exec_globals = {
            "moveit_commander": moveit_commander,
            "geometry_msgs": geometry_msgs,
            "rospy": rospy,
            "arm": arm,
            "robot": robot
        }

        exec(code_to_execute, exec_globals)
        return {**state, "result": "Execution successful in Gazebo!"}

    except Exception as e:
        return {**state, "result": f"Execution Failed: {e}"}


def check_code_exists(state: GraphState):
    return bool(state["code"])  

graph.set_entry_point("input_analyzer")

graph.add_node("input_analyzer", input_analyzer)
graph.add_node("command_matcher", command_matcher)
graph.add_node("code_converter", code_converter)
graph.add_node("code_optimizer", code_optimizer)
graph.add_node("output", output)

graph.add_edge("input_analyzer", "command_matcher")

graph.add_conditional_edges(
    "command_matcher",
    {
        True: "output",         
        False: "code_converter" 
    },
    check_code_exists
)

graph.add_edge("code_converter", "code_optimizer")
graph.add_edge("code_optimizer", "output")

executor = graph.compile()


def run_pipeline(user_command: str):
    initial_state = {"input": user_command, "selected_command": "", "code": "", "result": ""}
    final_state = executor.invoke(initial_state)
    return final_state


st.title("LangGraph-based Robot Controller")
user_input = st.text_area("Enter a natural language command:", height=100)

if st.button("Execute"):
    if user_input.strip():
        result = run_pipeline(user_input)
        st.write("Pipeline Execution Result:", result["result"])
    else:
        st.warning("Please enter a command!")
