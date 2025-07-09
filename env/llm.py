'''
Baseline : Centralized LLM

'''
import json
import re
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import time

from openai import OpenAI
from env_b import AI2ThorEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import save_to_video

client = OpenAI(api_key=Path('api_key.txt').read_text())

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="config/config.json")
    args = parser.parse_args()
    return args

args = get_args()
AGENT_NAMES_ALL = ["Alice", "Bob", "Charlie", "David", "Emma"]
with open(args.config_file, "r") as f:
        config = json.load(f)
        NUM_AGENTS = config["num_agents"]
AGENT_NAMES = AGENT_NAMES_ALL[:NUM_AGENTS]

BASE_PROMPT = f"""
You are an excellent task planner whose task is to help {NUM_AGENTS} robots to complete the final task of "{config["task"]}" in {config["scene"]}. 

# Task
Let's work this out in a step by step way to be sure we have the right answer.
"""

ai2thor_actions = {
    "move": [
        "MoveAhead",
        "MoveBack",
        "MoveRight",
        "MoveLeft"
    ],
    "rotate": [
        "RotateRight",
        "RotateLeft"
    ],
    "look": [
        "LookUp<angle>",
        "LookDown<angle>"
    ],
    "idle": [
        "Idle",
        "Done"
    ],
    "interact_with_object": [
        "PickupObject<object_id>",
        "PutObject<receptacle_id>",
        "OpenObject<object_id>",
        "CloseObject<object_id>",
        "ToggleObjectOn<object_id>",
        "ToggleObjectOff<object_id>",
        "BreakObject<object_id>",
        "CookObject<object_id>",
        "SliceObject<object_id>",
        "DirtyObject<object_id>",
        "CleanObject<object_id>",
        "FillObjectWithLiquid<object_id>",
        "EmptyLiquidFromObject<object_id>",
        "UseUpObject<object_id>"
    ],
    "interact_without_navigation": [
        "DropHandObject",
        "ThrowObject"
    ]
}
ACTIONS = f'''
## list of action that can be performed by the robot
{ai2thor_actions}
'''

ACTION_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions: {ACTIONS}

Here `<angle>` must be one of `[30, 60, 90, 120, 150, 180]`.  
“Done” is used to indicate that the robot has completed its task and is ready to stop.
“Idle” is used to let a robot wait in place for one step, if necessary.


You will get a description of the task robots are supposed to do. 
You will get the current state of the enviroment, including the list of objects in the environment.
You will get the subtasks that the robots are supposed to complete in order to achieve the final goal(Task), based on the objects in the environment and the ability of robots.

Based on the above information, you need to break down each subtask into smaller actions and assign these subtasks to the appropriate robot.

* Make sure your output length is correct: the length of the Actions list must equal the number of robots. 
For example, if there are two robots, the Actions list should contain exactly two elements, one for each robot agent.

### INPUT FORMAT ###
{{Task: a high level description of the final goal the robots are supposed to do/complete,
Subtasks: a list of subtasks that the robots are supposed to complete in order to achieve the final goal(Task),
Objects: a list of objects in the enviroment}}

* Note that the number of subtasks may be bigger than the number of robot agent.


### OUTPUT FORMAT ###
You will output a list of actions for each robot in the following format, in json format:
{{
"Actions": [[actions for robot 1], [actions for robot 2], ..., [actions for robot N]],
}}

where each actions is a list of strings in the format of "ActionName<arg1, arg2, ...>".
For example, if robot 1 should pick up an object with id "Object_1" and put the object on the table with id "Table_1", the output should be:
[[NavigateTo<Object_1>, PickupObject<Object_1>, NavigateTo<Table_1>, PutObject<Table_1>], [Idle], ...]
<Object_1> and <Table_1> are the name plus the number of the object in the environment. If there are multiple objects with the same name but different id, you can use the number to distinguish them, e.g. "Object_1", "Object_2", etc. don't use the id only.

* Example output:
if given INPUT subtasks are:
{{
"Subtasks": [
    ["pick up the vase and put it on the table"],
    ["pick up the tissue box and put it on the table"],
    ["pick up the remote control and put it on the table"]
]
}}

* The OUTPUT could be:
{{
"Actions": [
    ["PickupObject(Tomato_1)", "PutObject(CounterTop_1)"],
    ["PickupObject(Lettuce_1)", "PutObject(CounterTop_1)", "PickupObject<Bread>", "PutObject<CounterTop>"]
]
}}
which means:
- Robot 1 should pick up the tomato with "Tomato_1" and put it on the counter top with "CounterTop_1"
- Robot 2 should pick up the lettuce with "Lettuce_1" and put it on the counter top with "CounterTop_1", then pick up the bread with "Bread_1" and put it on the counter top with "CounterTop_1".


### Important Notes ###
* The `Actions` list must contain exactly {len(AGENT_NAMES)} sublists—one for each robot. Each sublist should include the full sequence of actions for that robot.
* The number of subtasks may be greater than the number of robots. You may distribute the subtasks in any way you deem optimal.
* A robot can handle multiple subtasks in a single plan, but each robot must only appear once in the `Actions` list.
* Each robot’s action list should be a sequential plan—actions should be performed in order and not interleaved with other robots.
* The robots can hold only one object at a time. Make sure that the actions you suggest do not require the robots to hold more than one object at a time.
* If you need to put an object on a receptacle, you need to pick it up first, then navigate to the receptacle, and then put it on the receptacle.
* If you open an object, please ensure that you close it before you move to a different place.
* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)
* Since there are {len(AGENT_NAMES)} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.
* Be aware that the robots will be performing actions in parallel, so make sure that the actions you suggest do not conflict with each other.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""

PLANNER_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions: {ACTIONS}
You will get a description of the task robots are supposed to do.
You will get the current state of the enviroment, including the list of objects in the environment.

You need to suggest the decompose the task into several subtasks for that robots to complete in order to achieve the final goal(Task), based on the objects in the environment and the ability of robots. 

For example, if the task is "Put the vase, tissue box, and remote control on the table", the subtasks could be:
1. pick up the vase and put it on the table
2. pick up the tissue box and put it on the table
3. pick up the remote control and put it on the table 

### Important Notes ###
Note that the subtasks should be independent to each other, i.e. the robots can complete them in any order; 
If the subtasks require the robots to complete them in a specific order, these subtask should be combined into one subtask.

### INPUT FORMAT ###
{{Task: a high level description of the final goal the robots are supposed to complete,
objects: a list of objects in the enviroment,}}

### OUTPUT FORMAT ###
You will output a list of subtasks for each robot in the following format, in json format:
{{
"Subtasks": [[subtask 1], [subtask 2], ..., [subtask n]],
}}
example:
{{
"Subtasks": [
    ["pick up the vase and put it on the table"],
    ["pick up the tissue box and put it on the table"],
    ["pick up the remote control and put it on the table"]
]
}}

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""



def convert_dict_to_string(input_dict) -> str:
    """
    add new lines for each key value pair
    Example output:
    {Task: bring a tomato, lettuce and bread to the countertop to make a sandwich
    Alice's observation: I see: ['Cabinet_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1', 'Ceiling_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'CounterTop_1']
    Alice's state: I am at co-ordinates: (-1.00, 0.90, 1.00) and I am holding nothing
    Bob's observation: I see: ['Cabinet_1', 'Sink_1', 'Mug_1', 'UpperCabinets_1', 'StandardCounterHeightWidth_1', 'Wall_1', 'Window_1', 'Fridge_1', 'Toaster_1', 'CoffeeMachine_1', 'PaperTowelRoll_1', 'CounterTop_1']
    Bob's state: I am at co-ordinates: (-1.00, 0.90, 0.00) and I am holding nothing
    History: {'Observations': {'Alice': [], 'Bob': []}, 'Actions': {'Alice': [], 'Bob': []}, 'Action Success': {'Alice': [], 'Bob': []}} }
    """
    return "{" + "\n".join(f"{k}: {v}, " for k, v in input_dict.items()) + "}"



def set_env_with_config(config_file: str):
    ''''
    config example:{
            "num_agents": 2,
            "scene": "FloorPlan1",
            "task": "bring a tomato, lettuce, and bread to the countertop to make a sandwich",
            "timeout": 120,
            "model": "gpt-4o-mini",
            "use_obs_summariser": false,
            "use_act_summariser": false,
            "use_action_failure": true,
            "use_shared_subtask": true,
            "use_separate_subtask": false,
            "use_shared_memory": true,
            "use_separate_memory": false,
            "use_plan": true,
            "force_action": false,
            "temperature": 0.7,
            "overhead": true
        }
    '''
    env = AI2ThorEnv(config_file)
    with open(config_file, "r") as f:
        config = json.load(f)
    return env, config


def get_llm_response(payload, model = "gpt-4o-mini", temperature= 0.7, max_tokens=1024) -> str:
    response = client.chat.completions.create(model=model, 
                                                messages=payload, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature,)

    return response, response.choices[0].message.content.strip()

def prepare_prompt(env: AI2ThorEnv, mode: str = "init", addendum: str = "", subtasks=[]) -> str:
    """
    mode: str, choose from planner, action
    planner: for decomposing the task into subtasks
    action: for generating the actions for each robot to perform
    addendum: additional information to be added to the user prompt
    """

    if mode == "planner":
        system_prompt = PLANNER_PROMPT
        input = env.get_center_planner_llm_input()
        user_prompt = convert_dict_to_string(input)
    elif mode == "action":
        system_prompt = ACTION_PROMPT
        if not subtasks:
            print("No subtasks provided")
            return None, None
        input = env.get_center_planner_llm_input()
        input["Subtasks"] = subtasks
        user_prompt = convert_dict_to_string(input)
    
    user_prompt += addendum
    return system_prompt, user_prompt

def prepare_payload(system_prompt, user_prompt):
    payload = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
    
    return payload


def process_planner_llm_output(res_content):
    try:
        return json.loads(res_content)["Subtasks"]
    except json.JSONDecodeError as e:
        print(f"[Warning] Initial JSON decode failed: {e}")

        try:
            # 去除結尾多餘逗號
            res_content = re.sub(r",\s*([\]}])", r"\1", res_content)
            # 單引號換雙引號（簡單處理）
            res_content = res_content.replace("'", '"')
            # 移除前後空白
            res_content = res_content.strip()

            data = json.loads(res_content)
            return data["Subtasks"]
        except Exception as e2:
            print(f"[Error] Failed to parse fixed JSON: {e2}")
            return None

def process_actions_llm_output(res_content):
    try:
        return json.loads(res_content)["Actions"]
    except json.JSONDecodeError as e:
        print(f"[Warning] Initial JSON decode failed: {e}")

        try:
            # 去除結尾多餘逗號
            res_content = re.sub(r",\s*([\]}])", r"\1", res_content)
            # 單引號換雙引號（簡單處理）
            res_content = res_content.replace("'", '"')
            # 移除前後空白
            res_content = res_content.strip()

            data = json.loads(res_content)
            return data["Actions"]
        except Exception as e2:
            print(f"[Error] Failed to parse fixed JSON: {e2}")
            return None

def run_test(env, high_level_tasks, test_name, test_id, task_name=None):
    """
    Run a test case with multiple steps and print results.
      - high_level_tasks: List[List[str]]，每個 agent 的 high-level 
      - test_name: 
      - test_id:  測試識別字串，會傳給 reset()
    """
    print(f"\n=== {test_name} (Test ID: {test_id}) ===")
    obs = env.reset(test_case_id=test_id)
    # print("Initial Observations:\n", obs)

    print("high_level_tasks: ", high_level_tasks)
    history = env.action_loop(high_level_tasks)
    for step_idx, (obs, succ) in enumerate(history, start=1):
        print(f"\n--- Step {step_idx} ---")
        # print("Observations:", obs)
        # print("Success flags:", succ)

    for agent_id, name in enumerate(env.agent_names):
        print(f"{name} POV path:", env.get_frame(agent_id, "pov"))
    if env.overhead:
        print("Shared overhead path:", env.get_frame(view="overhead"))

    if task_name:
        print('logs/' + task_name.replace(" ", "_") + f"/test_{test_id}")
        save_to_video('logs/' + task_name.replace(" ", "_") + f"/test_{test_id}")

def run_main():
    
    env, config = set_env_with_config(args.config_file)
    env.reset(test_case_id="1")
    # print(f"Environment set with config: {config}")
    # print(f"Number of agents: {env.num_agents}")
    planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")
    # print(f"Planner Prompt: {planner_prompt}")
    # print(f"Planner User Prompt: {planner_user_prompt}")
    planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    # print(f"Payload for LLM: {planner_payload}")
    # res, res_content = get_llm_response(planner_payload)
    
    res_content = '''{
        "Subtasks": [
            ["'pick up the tomato and put it on the countertop'"],
            ["pick up the lettuce and put it on the countertop"],
            ["pick up the bread and put it on the countertop"]
        ]
    }
    '''
    # print(f"Before LLM Response: {res_content}, type of res_content: {type(res_content)}")
    subtasks = process_planner_llm_output(res_content)
    # print(f"LLM Response (orig): {res}")
    # print(f"After LLM Response: {subtasks}, type of res_content: {type(subtasks)}")
    action_prompt, action_user_prompt = prepare_prompt(env, mode="action", subtasks=subtasks)
    # print(f"Action Prompt: {action_prompt}")
    # print(f"Action User Prompt: {action_user_prompt}")
    action_payload = prepare_payload(action_prompt, action_user_prompt)
    print(f"Payload for LLM: {action_payload}")
    # res, res_content = get_llm_response(action_payload)
    # print(f"LLM orig Response: {res}")
    # print(f"Before LLM Response: {res_content}")
    # high_level_tasks = process_actions_llm_output(res_content)
    # print(f"After LLM Response: {high_level_tasks}, type of res_content: {type(high_level_tasks)}")
    # testing
    high_level_tasks = [['PickupObject(Tomato|-00.39|+01.14|-00.81)', 'PutObject(CounterTop|+00.69|+00.95|-02.48)'], ['PickupObject(Lettuce|-01.81|+00.97|-00.94)', 'PutObject(CounterTop|+00.69|+00.95|-02.48)', 'PickupObject(Bread|-00.52|+01.17|-00.03)', 'PutObject(CounterTop|+00.69|+00.95|-02.48)']]
    # run_test(
    #     env,
    #     high_level_tasks=high_level_tasks,
    #     test_name="Test 1",
    #     test_id=1,
    #     task_name = config["task"] if "task" in config else "Test Task",
    # )
    # env.close()
    



if __name__ == "__main__":
    run_main()