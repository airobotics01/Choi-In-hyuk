![left to right](https://github.com/user-attachments/assets/c0f1880e-2a63-4087-9549-ad2186fcb133)
[진행한 내용]
streamlit을 이용해서 웹에서 원하는 동작을 말하면 json파일에서 해당코드를 읽고 rivz에서 프랑카 로봇이 움직이도록 해보았습니다.

[메인 코드]
```python
import os
import json
import streamlit as st
import rospy
from std_msgs.msg import String
from langchain.agents import Tool
from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = ''

def load_examples():
    examples_path = "Franka_ex.json"
    if not os.path.exists(examples_path):
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
    with open(examples_path, "r") as file:
        return json.load(file)

examples = load_examples()

pub = None

def send_motion_command(command):
    global pub
    if pub is None:
        rospy.init_node('langchain_command_sender', anonymous=True, disable_signals=True)
        pub = rospy.Publisher('/robot_motion_command', String, queue_size=10)
        rospy.sleep(1)
    pub.publish(command)
    print(f"Sent command: {command}")

def parse_llm_response(response):
    if isinstance(response, dict):
        if "choices" in response and len(response["choices"]) > 0:
            response_text = response["choices"][0].get("text", "")
        else:
            raise ValueError("Invalid response format: 'choices' key is missing.")
    else:
        response_text = response
    return response_text

def search_examples_with_llm(query: str) -> dict:
    examples_text = "\n".join(
        [f"Title: {example['title']}\nDescription: {example['description']}" for example in examples]
    )
    prompt = f"""
            You are a helpful assistant for finding the most relevant Franka robot command.
            Below are multiple examples of Franka robot-related code, each with a title and description.
            Identify the most relevant example based on the user's query.

            User Query: "{query}"

            Examples: {examples_text}
            Respond with the title of the most relevant example only.
            If no example is relevant, respond with "No matching example found."
            """

    response = llm(prompt)
    response_text = response.strip() if isinstance(response, str) else response["choices"][0].get("text", "").strip()

    if "No matching example found" in response_text or len(response_text) == 0:
        return {"title": "No matching example found", "code": None}

    for example in examples:
        if example["title"].lower() == response_text.lower():
            return example

    return {"title": "No matching example found", "code": None}

franka_tool_llm = Tool(
    name="SearchFrankaExamplesWithLLM",
    func=search_examples_with_llm,
    description="Find the most relevant Franka example using a language model."
)

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

st.title("Franka Robot Command Execution")

user_request = st.text_input("Enter the task you want to perform:")
if user_request:
    with st.spinner("Searching for the best matching code..."):
        selected_motion = search_examples_with_llm(user_request)

        if selected_motion["title"] != "No matching example found":
            st.success(f"Executing motion: {selected_motion['title']}")
            send_motion_command(selected_motion["title"])
            motion_code = selected_motion["code"]
            if motion_code:
                try:
                    print("DEBUG: 실행할 MoveIt 코드 ↓↓↓\n", motion_code)
                    exec(motion_code)
                    print("DEBUG: MoveIt 실행 완료!")
                except Exception as e:
                    print(f"ERROR: 실행 중 오류 발생 - {e}")
                    st.error(f"Execution failed: {e}")
        else:
            st.error("No matching example found. Please refine your request.")

```
[JSON파일]
```python
[
    {
        "title": "Move to Ready Position",
        "description": "Move the Franka robot arm to its ready position.",
        "code": "from moveit_commander import MoveGroupCommander\n\ndef move_to_ready():\n    group = MoveGroupCommander('panda_arm')\n    group.set_named_target('ready')\n    success, plan = group.plan() if isinstance(group.plan(), tuple) else (True, group.plan())\n    if not success or not plan:\n        print('Planning failed!')\n        return\n    group.execute(plan, wait=True)\n    group.stop()\n    group.clear_pose_targets()\n    print('Moved robot arm to ready position.')\n\nmove_to_ready()"
    },
    {
        "title": "Pick Object",
        "description": "Move the Franka robot arm to a predefined picking position and close the gripper.",
        "code": "from moveit_commander import MoveGroupCommander\n\ndef pick_object():\n    group = MoveGroupCommander('panda_arm')\n    pose_target = group.get_current_pose().pose\n    pose_target.position.x = 0.4\n    pose_target.position.y = 0.0\n    pose_target.position.z = 0.2\n    group.set_pose_target(pose_target)\n    success, plan = group.plan() if isinstance(group.plan(), tuple) else (True, group.plan())\n    if not success or not plan:\n        print('Planning failed!')\n        return\n    group.execute(plan, wait=True)\n    print('Closing gripper...')\n    print('Object picked!')\n    group.stop()\n    group.clear_pose_targets()\n\npick_object()"
    },
    {
        "title": "Move Object Left to Right",
        "description": "Pick an object from the left and place it to the right.",
        "code": "from moveit_commander import MoveGroupCommander\n\ndef move_left_to_right():\n    group = MoveGroupCommander('panda_arm')\n    group.set_named_target('ready')\n    plan = group.plan()\n    if isinstance(plan, tuple):\n        plan = plan[1] if plan[0] else None\n    if not plan:\n        print('Planning to ready position failed!')\n        return\n    group.execute(plan, wait=True)\n    pose_target = group.get_current_pose().pose\n    pose_target.position.x = 0.3\n    pose_target.position.y = 0.2\n    pose_target.position.z = 0.2\n    group.set_pose_target(pose_target)\n    plan = group.plan()\n    if isinstance(plan, tuple):\n        plan = plan[1] if plan[0] else None\n    if not plan:\n        print('Planning to pick position failed!')\n        return\n    group.execute(plan, wait=True)\n    print('Closing gripper...')\n    pose_target.position.y = -0.2\n    group.set_pose_target(pose_target)\n    plan = group.plan()\n    if isinstance(plan, tuple):\n        plan = plan[1] if plan[0] else None\n    if not plan:\n        print('Planning to place position failed!')\n        return\n    group.execute(plan, wait=True)\n    print('Opening gripper...')\n    print('Object placed to the right!')\n    group.stop()\n    group.clear_pose_targets()\n\nmove_left_to_right()"
    },
    {
        "title": "Move Object Right to Left",
        "description": "Pick an object from the right and place it to the left.",
        "code": "from moveit_commander import MoveGroupCommander\n\ndef move_right_to_left():\n    group = MoveGroupCommander('panda_arm')\n    group.set_named_target('ready')\n    plan = group.plan()\n    if isinstance(plan, tuple):\n        plan = plan[1] if plan[0] else None\n    if not plan:\n        print('Planning to ready position failed!')\n        return\n    group.execute(plan, wait=True)\n    pose_target = group.get_current_pose().pose\n    pose_target.position.x = 0.3\n    pose_target.position.y = -0.2\n    pose_target.position.z = 0.2\n    group.set_pose_target(pose_target)\n    plan = group.plan()\n    if isinstance(plan, tuple):\n        plan = plan[1] if plan[0] else None\n    if not plan:\n        print('Planning to pick position failed!')\n        return\n    group.execute(plan, wait=True)\n    print('Closing gripper...')\n    pose_target.position.y = 0.2\n    group.set_pose_target(pose_target)\n    plan = group.plan()\n    if isinstance(plan, tuple):\n        plan = plan[1] if plan[0] else None\n    if not plan:\n        print('Planning to place position failed!')\n        return\n    group.execute(plan, wait=True)\n    print('Opening gripper...')\n    print('Object placed to the left!')\n    group.stop()\n    group.clear_pose_targets()\n\nmove_right_to_left()"
    }
]
```
