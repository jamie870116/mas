'''
Baseline : Centralized LLM

'''
import json
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
Based on the above code plan and environment states, generate an improved and executable Python script. 
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
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. All {len(AGENT_NAMES)} robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
They can perform the following actions: {ACTIONS}

Here `<angle>` must be one of `[30, 60, 90, 120, 150, 180]`.  
“Done” is used to indicate that the robot has completed its task and is ready to stop.
“Idle” is used to let a robot wait in place for one step, if necessary.


You need to suggest the action that each robot should take at the current time step.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input.
To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
Robots' combined memory: description of robot's combined memory}}

### Important Notes ###
* The robots can hold only one object at a time.
* Even if the robot can see objects, it might not be able to interact with them if they are too far away. Hence you will need to make the robot move closer to the objects they want to interact with.
For example: An action like "pick up <object_id>" is feasible only if robot can see the object and is close enough to it. So you will have to move closer to it before you can pick it up.
So if a particular action fails, you will have to choose a different action for the robot.
* If you open an object, please ensure that you close it before you move to a different place.
* Opening object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)
* Since there are {len(AGENT_NAMES)} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""

PLANNER_PROMPT = f"""
"""


VERIFIER_PROMPT = f"""
you are a task planning expert. Your task is to verify if the task is completed or not. You will be given '''task''' which is the final goal, '''environment state''' which is the current state of environment, and '''ground truth''' which is the ground truth you should check in environment.
    
        ## Input
        ###task:
        {{Task: description of the task the robots are supposed to do,
        Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Robots' combined memory: description of robots' combined memory}}
    
        ## Output Format        
        Reason over the robots' task, image inputs, observations, previous actions, open subtasks, completed subtasks and memory, and then output the following:
        * Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.
        * Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.
        The "Completed Subtasks" should be in a list format where the completed subtasks are listed. For example: ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
        Your output should be in the form of a python dictionary as shown below.
        Example output with two agents (do it for {len(AGENT_NAMES)} agents): {{"reason": "{AGENT_NAMES[0]} placed the apple in the fridge in the previous step and was successful and {AGENT_NAMES[1]} picked up the the book from the table. Hence {AGENT_NAMES[0]} has completed the subtask of transporting the apple to the fridge, {AGENT_NAMES[1]} has picked up the book, but {AGENT_NAMES[1]} has still not completed the subtask of transporting the book to the table", "completed subtasks": ["picked up book from the table", "transport the apple to the fridge"]}}

        You should reason over the above information,
        and tell me if the task is complete or not, if not, tell me what is completed and what was not. 
        
        * Note: gound truth only show the type of objects, while environment state using object id. This meaning not specific object need to be activated.
        There might be multiple same type of objects in the environment, be tolerant. If at least one object of the same type satisfy the ground truth condition, then the subtask is completed.
        Noticed that '''ground truth''' and '''environment state''' might have different naming. For example, state of object in ground truth is Toggled, means that the isToggled field of the object in environment state is true.
        Please be tolerent and some of Criteria is given to you in above.
        If you are not sure whether object A contains object B, first check the contains list. If absent, verify if B's position is inside A’s bounding box (size + position tolerance).

        * Since there are {len(AGENT_NAMES)} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.
        * Don't output anything else other than what has been specified.

        When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.
        Let's work this out in a step by step way to be sure we have the right answer.
        
        your output should be in the following format in dictionary:
        
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


def get_llm_response(payload: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens=128) -> str:
    response = client.chat.completions.create(model=model, 
                                                messages=payload, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature,)

    return response, response.choices[0].message.content.strip()

def prepare_prompt(env: AI2ThorEnv, mode: str = "init", addendum: str = "") -> str:
    """
    module_name: str
    choose from planner, verifier, action
    """
    # Choose the appropriate prompt based on what module is being called
    # user_prompt = convert_dict_to_string(env.input_dict)
    if mode == "action":
        system_prompt = ACTION_PROMPT
        user_prompt = convert_dict_to_string(env.get_action_llm_input())
    elif mode == "planner":
        system_prompt = "PLANNER_PROMPT"
        user_prompt = convert_dict_to_string(env.get_planner_llm_input())
    elif mode == "verifier":
        system_prompt = VERIFIER_PROMPT
        user_prompt = convert_dict_to_string(env.get_verifier_llm_input())
    user_prompt += addendum
    return system_prompt, user_prompt

def prepare_payload(system_prompt, user_prompt):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ],
        "max_tokens": 1000,
        "temperature": config.temperature,
    }
    return payload

def process_action_llm_output():
    pass



def run_main():
    
    env, config = set_env_with_config(args.config_file)
    env.reset(test_case_id="1")
    print(f"Environment set with config: {config}")
    print(f"Number of agents: {env.num_agents}")
    action_prompt, action_user_prompt = prepare_prompt(env, mode="action")
    print(f"Action Prompt: {action_prompt}")
    print(f"Action User Prompt: {action_user_prompt}")
    verifier_prompt, verifier_user_prompt = prepare_prompt(env, mode="verifier")
    print(f"Verifier Prompt: {verifier_prompt}")
    print(f"Verifier User Prompt: {verifier_user_prompt}")
    # payload = prepare_payload(action_prompt, action_user_prompt)
    # res, res_content = get_llm_response(payload)
    env.close()
    

if __name__ == "__main__":
    run_main()