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
ai2thor_objs = {
    "Kitchen": {
        "AluminumFoil": ["Pickupable"],
        "Apple": ["Pickupable", "Sliceable"],
        "AppleSliced": ["Pickupable"],
        "Bottle": ["Pickupable", "Fillable", "Breakable"],
        "Bowl": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "Bread": ["Pickupable", "Sliceable"],
        "BreadSliced": ["Pickupable", "Cookable"],
        "ButterKnife": ["Pickupable"],
        "Cabinet": ["Openable", "Receptacle"],
        "CoffeeMachine": ["Toggleable", "Receptacle", "Moveable"],
        "CounterTop": ["Receptacle"],
        "Cup": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "DiningTable": ["Receptacle", "Moveable"],
        "DishSponge": ["Pickupable"],
        "Drawer": ["Openable", "Receptacle"],
        "Egg": ["Pickupable", "Sliceable", "Breakable"],
        "EggCracked": ["Pickupable", "Cookable"],
        "Faucet": ["Toggleable"],
        "Fork": ["Pickupable"],
        "Fridge": ["Openable", "Receptacle"],
        "GarbageCan": ["Receptacle", "Moveable"],
        "HousePlant": ["Fillable", "Moveable"],
        "Kettle": ["Openable", "Pickupable", "Fillable"],
        "Knife": ["Pickupable"],
        "Ladle": ["Pickupable"],
        "Lettuce": ["Pickupable", "Sliceable"],
        "LettuceSliced": ["Pickupable"],
        "LightSwitch": ["Toggleable"],
        "Microwave": ["Openable", "Toggleable", "Receptacle", "Moveable"],
        "Mug": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "Pan": ["Pickupable", "Receptacle", "Dirtyable"],
        "PaperTowelRoll": ["Pickupable", "UsedUp"],
        "PepperShaker": ["Pickupable"],
        "Plate": ["Pickupable", "Receptacle", "Breakable", "Dirtyable"],
        "Pot": ["Pickupable", "Receptacle", "Fillable", "Dirtyable"],
        "Potato": ["Pickupable", "Sliceable", "Cookable"],
        "PotatoSliced": ["Pickupable", "Cookable"],
        "SaltShaker": ["Pickupable"],
        "Shelf": ["Receptacle"],
        "SideTable": ["Receptacle", "Moveable"],
        "Sink": ["Receptacle"],
        "SinkBasin": ["Receptacle"],
        "Spatula": ["Pickupable"],
        "Spoon": ["Pickupable"],
        "SprayBottle": ["Pickupable"],
        "Statue": ["Pickupable", "Breakable"],
        "Stool": ["Moveable"],
        "StoveBurner": ["Toggleable", "Receptacle"],
        "StoveKnob": ["Toggleable"],
        "Toaster": ["Toggleable", "Receptacle", "Moveable"],
        "Tomato": ["Pickupable", "Sliceable"],
        "TomatoSliced": ["Pickupable"],
        "WineBottle": ["Pickupable", "Fillable", "Breakable"]
    },

    "LivingRoom": {
        "ArmChair": ["Receptacle", "Moveable"],
        "Blinds": ["Openable"],
        "Book": ["Openable", "Pickupable"],
        "Boots": ["Pickupable"],
        "Box": ["Openable", "Pickupable", "Receptacle"],
        "CoffeeTable": ["Receptacle", "Moveable"],
        "CounterTop": ["Receptacle"],
        "CreditCard": ["Pickupable"],
        "Cup": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "Desk": ["Receptacle", "Moveable"],
        "DeskLamp": ["Toggleable", "Moveable"],
        "DiningTable": ["Receptacle", "Moveable"],
        "Drawer": ["Openable", "Receptacle"],
        "Dresser": ["Receptacle", "Moveable"],
        "Faucet": ["Toggleable"],       # only if present (rare)
        "FloorLamp": ["Toggleable", "Moveable"],
        "GarbageCan": ["Receptacle", "Moveable"],
        "HousePlant": ["Fillable", "Moveable"],
        "KeyChain": ["Pickupable"],
        "Laptop": ["Openable", "Pickupable", "Toggleable", "Breakable"],
        "LightSwitch": ["Toggleable"],
        "Microwave": ["Openable", "Toggleable", "Receptacle", "Moveable"],  # rare
        "Mug": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "Newspaper": ["Pickupable"],
        "Ottoman": ["Receptacle", "Moveable"],
        "Pen": ["Pickupable"],
        "Pencil": ["Pickupable"],
        "PepperShaker": ["Pickupable"],
        "Pillow": ["Pickupable"],
        "Plate": ["Pickupable", "Receptacle", "Breakable", "Dirtyable"],
        "RemoteControl": ["Pickupable"],
        "RoomDecor": ["Moveable"],
        "Safe": ["Openable", "Receptacle", "Moveable"],
        "SaltShaker": ["Pickupable"],
        "Shelf": ["Receptacle"],
        "ShelvingUnit": ["Moveable"],
        "SideTable": ["Receptacle", "Moveable"],
        "Sofa": ["Receptacle", "Moveable"],
        "SprayBottle": ["Pickupable"],
        "Statue": ["Pickupable", "Breakable"],
        "Stool": ["Moveable"],
        "Television": ["Toggleable", "Breakable", "Moveable"],
        "TissueBox": ["Pickupable", "UsedUp"],
        "TVStand": ["Receptacle", "Moveable"],
        "Watch": ["Pickupable"],
        "WateringCan": ["Pickupable", "Fillable"],
        "Window": ["Breakable"],
        "WineBottle": ["Pickupable", "Fillable", "Breakable"]
    },

    "Bedroom": {
        "AlarmClock": ["Pickupable"],
        "ArmChair": ["Receptacle", "Moveable"],
        "BaseballBat": ["Pickupable"],
        "BasketBall": ["Pickupable"],
        "Bed": ["Receptacle", "Dirtyable"],
        "Blinds": ["Openable"],
        "Book": ["Openable", "Pickupable"],
        "Boots": ["Pickupable"],
        "Box": ["Openable", "Pickupable", "Receptacle"],
        "CD": ["Pickupable"],
        "CellPhone": ["Pickupable", "Toggleable", "Breakable"],
        "Chair": ["Moveable"],
        "Cloth": ["Pickupable", "Dirtyable"],
        "CreditCard": ["Pickupable"],
        "Cup": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "Desk": ["Receptacle", "Moveable"],
        "DeskLamp": ["Toggleable", "Moveable"],
        "Desktop": ["Moveable"],
        "Dresser": ["Receptacle", "Moveable"],
        "Drawer": ["Openable", "Receptacle"],
        "Dumbbell": ["Pickupable"],
        "GarbageBag": ["Moveable"],
        "GarbageCan": ["Receptacle", "Moveable"],
        "HandTowel": ["Pickupable"],           # present only in some bathrooms, included here for completeness
        "HousePlant": ["Fillable", "Moveable"],
        "KeyChain": ["Pickupable"],
        "Laptop": ["Openable", "Pickupable", "Toggleable", "Breakable"],
        "LaundryHamper": ["Receptacle", "Moveable"],
        "LightSwitch": ["Toggleable"],
        "Mug": ["Pickupable", "Receptacle", "Fillable", "Breakable", "Dirtyable"],
        "Pen": ["Pickupable"],
        "Pencil": ["Pickupable"],
        "Pillow": ["Pickupable"],
        "Plate": ["Pickupable", "Receptacle", "Breakable", "Dirtyable"],
        "Poster": [],                          # no actionable properties; omitted by design
        "Safe": ["Openable", "Receptacle", "Moveable"],
        "Shelf": ["Receptacle"],
        "ShelvingUnit": ["Moveable"],
        "SideTable": ["Receptacle", "Moveable"],
        "SprayBottle": ["Pickupable"],
        "Statue": ["Pickupable", "Breakable"],
        "Stool": ["Moveable"],
        "TableTopDecor": ["Pickupable"],
        "TeddyBear": ["Pickupable"],
        "Television": ["Toggleable", "Breakable", "Moveable"],
        "TennisRacket": ["Pickupable"],
        "TissueBox": ["Pickupable", "UsedUp"],
        "TVStand": ["Receptacle", "Moveable"],
        "VacuumCleaner": ["Moveable"],
        "Watch": ["Pickupable"],
        "Window": ["Breakable"]
    },

    "Bathroom": {
        "Bathtub": ["Receptacle"],
        "BathtubBasin": ["Receptacle"],
        "Blinds": ["Openable"],
        "Cabinet": ["Openable", "Receptacle"],
        "Candle": ["Pickupable", "Toggleable"],
        "Cloth": ["Pickupable", "Dirtyable"],
        "CounterTop": ["Receptacle"],
        "Drawer": ["Openable", "Receptacle"],
        "Faucet": ["Toggleable"],
        "GarbageCan": ["Receptacle", "Moveable"],
        "HandTowel": ["Pickupable"],
        "HandTowelHolder": ["Receptacle"],
        "LightSwitch": ["Toggleable"],
        "Mirror": ["Breakable", "Dirtyable"],
        "Plunger": ["Pickupable"],
        "ScrubBrush": ["Pickupable"],
        "Shelf": ["Receptacle"],
        "ShowerCurtain": ["Openable"],
        "ShowerDoor": ["Openable", "Breakable"],
        "ShowerGlass": ["Breakable"],
        "ShowerHead": ["Toggleable"],
        "SideTable": ["Receptacle", "Moveable"],
        "Sink": ["Receptacle"],
        "SinkBasin": ["Receptacle"],
        "SoapBar": ["Pickupable"],
        "SoapBottle": ["Pickupable", "UsedUp"],
        "SprayBottle": ["Pickupable"],
        "Statue": ["Pickupable", "Breakable"],
        "TissueBox": ["Pickupable", "UsedUp"],
        "Toilet": ["Openable", "Receptacle"],
        "ToiletPaper": ["Pickupable", "UsedUp"],
        "ToiletPaperHanger": ["Receptacle"],
        "Towel": ["Pickupable"],
        "TowelHolder": ["Receptacle"],
        "Window": ["Breakable"]
    }
}

def _scene_type(fpid: str) -> str | None:
    """根據 FloorPlan 編號直接判斷場景類型。"""
    num = int(fpid[9:])
    if   1   <= num <= 30:  return "Kitchen"
    elif 201 <= num <= 230: return "LivingRoom"
    elif 301 <= num <= 330: return "Bedroom"
    elif 401 <= num <= 430: return "Bathroom"
    return None

def get_objects_by_floorplan(fpid: str) -> list[str]:
    """回傳指定 FloorPlan 所屬場景的所有物件型別清單。"""
    scene = _scene_type(fpid)
    return list(ai2thor_objs[scene].items()) if scene else []

CURRENT_OBJECT_REFERENCE = get_objects_by_floorplan(config["scene"])

ai2thor_actions = {
    # "move": [
    #     "MoveAhead",
    #     "MoveBack",
    #     "MoveRight",
    #     "MoveLeft"
    # ],
    # "rotate": [
    #     "RotateRight",
    #     "RotateLeft"
    # ],
    # "look": [
    #     "LookUp<angle>",
    #     "LookDown<angle>"
    # ],
    "idle": [
        "Idle",
        "Done"
    ],
    "interact_with_object": [
        "NavigateTo<object_name>",
        "PickupObject<object_name>",
        "PutObject<receptacle_name>",
        "OpenObject<object_name>",
        "CloseObject<object_name>",
        "ToggleObjectOn<object_name>",
        "ToggleObjectOff<object_name>",
        "BreakObject<object_name>",
        "CookObject<object_name>",
        "SliceObject<object_name>",
        "DirtyObject<object_name>",
        "CleanObject<object_name>",
        "FillObjectWithLiquname<object_name>",
        "EmptyLiquidFromObject<object_name>",
        "UseUpObject<object_name>"
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

# You are given a list of all object types that may appear in a {_scene_type(config["scene"])} environment, along with their supported interactions.

# This is **not** the list of objects currently present in the scene, but a reference for what types of objects and actions are possible in such environments. 
# Use this knowledge to ensure that any actions you generate are logically valid and feasible for the given object types.

# Below is the reference list for the {_scene_type(config["scene"])} scene: {CURRENT_OBJECT_REFERENCE}

ACTION_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions: {ACTIONS}

“Done” is used to indicate that the robot has completed its task and is ready to stop.
“Idle” is used to let a robot wait in place for one step, if necessary.

### Task ###
You will get a description of the task robots are supposed to do. 
You will get the current state of the enviroment, including the list of objects in the environment.
You will get the subtasks that the robots are supposed to complete in order to achieve the final goal(Task), based on the objects in the environment and the ability of robots.

Based on the above information, you need to break down each subtask into smaller actions and assign these subtasks to the appropriate robot.

* Make sure your output length is correct: the length of the Actions list must equal the number of robots. 
For example, if there are two robots, the Actions list should contain exactly two elements, one for each robot agent.

### INPUT FORMAT ###
{{
Task: a high level description of the final goal the robots are supposed to do/complete,
Number of agents: the number of robots in the environment,
Subtasks: a list of subtasks that the robots are supposed to complete in order to achieve the final goal(Task),
Objects:  a list of objects in the current enviroment.
}}


### OUTPUT FORMAT ###
You will output a list of actions for each robot in the following format, in json format:
{{
"Actions": [[actions for robot 1], [actions for robot 2], ..., [actions for robot N]],
}}

where each actions is a list of strings in the format of "ActionName<arg1, arg2, ...>".
For example, if robot 1 should pick up an object with id "Object_1" and put the object on the table with id "Table_1", the output should be:
[[NavigateTo<Object_1>, PickupObject<Object_1>, NavigateTo<Table_1>, PutObject<Table_1>], [Idle], ...]
<Object_1> and <Table_1> are the name plus the number of the object in the environment. If there are multiple objects with the same name but different id, you can use the number to distinguish them, e.g. "Object_1", "Object_2", etc. don't use the id only.

* Example1:
if given INPUT subtasks are:
{{"
"Number of agents": 2,
"Task": "bring a vase, tissue box, and remote control to the counter top to make a sandwich",
"Subtasks": [
    ["pick up the vase and put it on the table"],
    ["pick up the tissue box and put it on the table"],
    ["pick up the remote control and put it on the table"]
],
}}

* The OUTPUT could be:
{{
"Actions": [
    ["NavigateTo(Vase_1)", "PickupObject(Vase_1)", "NavigateTo(Table_1)", "PutObject(Table_1)"],
    ["NavigateTo(TissueBox_1)", "PickupObject(TissueBox_1)", "NavigateTo(Table_1)", "PutObject(Table_1)", "NavigateTo(RemoteControl)", "PickupObject<RemoteControl>", "PutObject(Table_1)","PutObject<Table_1>"]
]
}}
which means:
- Robot 1 should pick up the Vase with "Vase_1" and put it on the table with "Table_1"
- Robot 2 should pick up the tissue box with "TissueBox_1" and put it on the counter top with "Table_1", then pick up the remote control with "RemoteControl_1" and put it on the table with "Table_1".


* Example2:
if given INPUT subtasks are:
{{
"Number of agents": 2,
"Task": "put the Book inside the drawer",
"Subtasks": ["put the Book inside the drawer"]
}}

* the OUTPUT could be:
{{ "Actions": 
[["NavigateTo(Drawer_1)", "OpenObject(Drawer_1)", "NavigateTo(Book_1)", "PickupObject(Book_1)",,"PutObject(Drawer_1)", "CloseObject(Drawer_1)"],["Idle"]]
}}

Note: This subtask combines opening, placing, and closing into one sequence, since the drawer must be opened before placing the item and closed afterward. The robot must have empty hands to operate the drawer.

### Important Notes ###
* The `Actions` list must contain exactly {len(AGENT_NAMES)} sublists—one for each robot. Each sublist should include the full sequence of actions for that robot.
* Each robot's action list should be a sequential plan—actions should be performed in order and not interleaved with other robots.
* Make sure that the objects you suggest the robots to interact with are actually in the environment.
* Plan actions that are consistent with the object's real-world behavior and usage.
* Do not use OpenObject on surfaces like tables or countertops.
* For any Openable receptacle (e.g., drawer, fridge, cabinet), you MUST:
    - Open the container before placing any object inside;
    - Ensure the robot has empty hands before opening or closing;
    - Close the container after placing the object.
* Robots cannot open objects (e.g., drawers, cabinets, fridge) while holding something. Ensure the robot's hand is empty before attempting to open or close objects.
* If an object needs to be placed inside an openable container, open it first before placing the item.
* Openable object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* If you open an object, please ensure that you close it before you move to a different place.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)
* If you need to put an object on a receptacle, you need to pick it up first, then navigate to the receptacle, and then put it on the receptacle.
* The robots can hold only one object at a time. Make sure that the actions you suggest do not require the robots to hold more than one object at a time.
* Since there are {len(AGENT_NAMES)} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.
* Be aware that the robots will be performing actions in parallel, so make sure that the actions you suggest do not conflict with each other.
* You should consider the position of the objects and the robots in the enviroment when assigning actions and receptacle to the robots.
* The number of subtasks may be greater than the number of robots. You may distribute the subtasks in any way you deem optimal.
* A robot can handle multiple subtasks in a single plan, but each robot must only appear once in the `Actions` list.
* Do NOT assume or default to common receptacles like CounterTop or Table unless explicitly specified in the task.



* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

Let's work this out in a step by step way to be sure we have the right answer.
"""
# You are given a list of all object types that may appear in a {_scene_type(config["scene"])} environment, along with their supported interactions.
# This is **not** the list of objects currently present in the scene, but a reference for what types of objects and actions are possible in such environments. 
# Use this knowledge to ensure that any actions you generate are logically valid and feasible for the given object types.
PLANNER_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions: {ACTIONS}

### TASK ###
You will get a description of the task robots are supposed to do.
You will get the current state of the enviroment, including the list of objects in the environment.

You need to suggest the decompose the task into several subtasks for that robots to complete in order to achieve the given Task, based on the objects in the environment and the ability of robots. 

For example, if the task is "Put the vase, tissue box, and remote control on the table", the subtasks could be:
1. pick up the vase and put it on the table
2. pick up the tissue box and put it on the table
3. pick up the remote control and put it on the table 




### INPUT FORMAT ###
{{Task: a high level description of the final goal the robots are supposed to complete,
Number of agents": the number of robots in the environment,
Objects: a list of objects in the current enviroment}}

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


### Important Notes ###
* Note that the subtasks should be independent to each other, i.e. the robots can complete them in any order; 
* If the subtasks require the robots to complete them in a specific order, these subtask should be combined into one subtask.
* Make sure that the objects you suggest the robots to interact with are actually in the environment.
* Plan actions that are consistent with the object's real-world behavior and usage.
* Do not use OpenObject on surfaces like tables or countertops.
* For any Openable receptacle (e.g., drawer, fridge, cabinet), you MUST:
    - Open the container before placing any object inside;
    - Ensure the robot has empty hands before opening or closing;
    - Close the container after placing the object.
* Robots cannot open objects (e.g., drawers, cabinets, fridge) while holding something. Ensure the robot’s hand is empty before attempting to open or close objects.
* If an object needs to be placed inside an openable container, open it first before placing the item.
* Openable object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* If you open an object, please ensure that you close it before you move to a different place.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)
* If you need to put an object on a receptacle, you need to pick it up first, then navigate to the receptacle, and then put it on the receptacle.
* The robots can hold only one object at a time. Make sure that the actions you suggest do not require the robots to hold more than one object at a time.
* Do NOT assume or default to common receptacles like CounterTop or Table unless explicitly specified in the task.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.
"""

EDITOR_PROMPT = f"""
You are an expert task planner and editor.

You will receive:
- A high-level task
- A list of subtasks generated by another agent
- A list of objects currently in the scene

Your job is to correct any incorrect subtasks so that the result matches the intended goal.

### Task Interpretation Rules ###
1. If the task says "put X in the fridge", make sure X is actually placed **inside** the fridge.
2. If a container Y (like Fridge or Drawer) is used, make sure it is opened before placing an item inside, and closed afterward. For example:
   - "open the Y, put the X inside, and close the Y"
3. Never place items on receptacles not mentioned in the task.
4. Preserve the robot capabilities (e.g. hand constraints, action limits), but you may restructure or combine subtasks if necessary.

### Input Format ###
{{
Task: a high level description of the final goal the robots are supposed to do/complete,
Number of agents: the number of robots in the environment,
Subtasks: A list of subtasks generated by another agent.
Objects:  a list of objects in the current enviroment.
}}

### Output Format ###
Return only the corrected list of subtasks in the following format,in json format:
{{
"Subtasks": [[subtask 1], [subtask 2], ..., [subtask n]],
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
        input = env.get_center_llm_input()
        user_prompt = convert_dict_to_string(input)
    elif mode == "action":
        system_prompt = ACTION_PROMPT
        if not subtasks:
            print("No subtasks provided")
            return None, None
        input = env.get_center_llm_input()
        input["Subtasks"] = subtasks
        user_prompt = convert_dict_to_string(input)
    elif mode == "editor":
        system_prompt = EDITOR_PROMPT
        input = env.get_center_llm_input()
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
    ## ----planner
    # planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")

    # print(f"Planner Prompt: {planner_prompt}")
    # print(f"Planner User Prompt: {planner_user_prompt}")

    # planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    # print(f"Payload for LLM: {planner_payload}")

    # res, res_content = get_llm_response(planner_payload)
    
    # res_content = '''{
    #     "Subtasks": [
    #         ["'pick up the tomato and put it on the countertop'"],
    #         ["pick up the lettuce and put it on the countertop"],
    #         ["pick up the bread and put it on the countertop"]
    #     ]
    # }
    # '''
    # print(f"Before Planner LLM Response: {res_content}, type of res_content: {type(res_content)}")
    # subtasks = process_planner_llm_output(res_content)

    # # print(f"LLM Response (orig): {res}")
    # print(f"After Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    ## ----Editor
    # editor_prompt, editor_user_prompt = prepare_prompt(env, mode="editor", subtasks=subtasks)

    # print(f"editor Prompt: {editor_prompt}")
    # print(f"editor User Prompt: {editor_user_prompt}")

    # editor_payload = prepare_payload(editor_prompt, editor_user_prompt)
    # # print(f"Payload for LLM: {editor_payload}")

    # res, res_content = get_llm_response(editor_payload)
    # # print(f"LLM orig Response: {res}")
    # print(f"Before Editor LLM Response: {res_content}")
    # subtasks = process_planner_llm_output(res_content)
    # print(f"After Editor LLM Response: {subtasks}, type of res_content: {type(subtasks)}")


    ## ----Action
    # action_prompt, action_user_prompt = prepare_prompt(env, mode="action", subtasks=subtasks)

    # print(f"Action Prompt: {action_prompt}")
    # print(f"Action User Prompt: {action_user_prompt}")

    # action_payload = prepare_payload(action_prompt, action_user_prompt)
    # # print(f"Payload for LLM: {action_payload}")

    # res, res_content = get_llm_response(action_payload)
    # # print(f"LLM orig Response: {res}")
    # print(f"Before Action LLM Response: {res_content}")
    # high_level_tasks = process_actions_llm_output(res_content)
    # print(f"After Action LLM Response: {high_level_tasks}, type of res_content: {type(high_level_tasks)}")

    # testing
    high_level_tasks = [['NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'PutObject(Fridge_1)', 'NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'PutObject(Fridge_1)', 'NavigateTo(Bread_1)', 'PickupObject(Bread_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'], ['Idle']]
    run_test(
        env,
        high_level_tasks=high_level_tasks,
        test_name="Test 2",
        test_id=2,
        task_name = config["task"] if "task" in config else "Test Task",
    )
    env.close()
    



if __name__ == "__main__":
    run_main()

'''
Terminal: 
Planner Prompt: 
You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice, and Bob carry out a task. 
They can perform the following actions: 
## list of action that can be performed by the robot
{'idle': ['Idle', 'Done'], 'interact_with_object': ['NavigateTo<object_name>', 'PickupObject<object_name>', 'PutObject<receptacle_name>', 'OpenObject<object_name>', 'CloseObject<object_name>', 'ToggleObjectOn<object_name>', 'ToggleObjectOff<object_name>', 'BreakObject<object_name>', 'CookObject<object_name>', 'SliceObject<object_name>', 'DirtyObject<object_name>', 'CleanObject<object_name>', 'FillObjectWithLiquname<object_name>', 'EmptyLiquidFromObject<object_name>', 'UseUpObject<object_name>'], 'interact_without_navigation': ['DropHandObject', 'ThrowObject']}


### TASK ###
You will get a description of the task robots are supposed to do.
You will get the current state of the enviroment, including the list of objects in the environment.

You need to suggest the decompose the task into several subtasks for that robots to complete in order to achieve the given Task, based on the objects in the environment and the ability of robots. 

For example, if the task is "Put the vase, tissue box, and remote control on the table", the subtasks could be:
1. pick up the vase and put it on the table
2. pick up the tissue box and put it on the table
3. pick up the remote control and put it on the table 




### INPUT FORMAT ###
{Task: a high level description of the final goal the robots are supposed to complete,
Number of agents": the number of robots in the environment,
Objects: a list of objects in the current enviroment}

### OUTPUT FORMAT ###
You will output a list of subtasks for each robot in the following format, in json format:
{
"Subtasks": [[subtask 1], [subtask 2], ..., [subtask n]],
}
example:
{
"Subtasks": [
    ["pick up the vase and put it on the table"],
    ["pick up the tissue box and put it on the table"],
    ["pick up the remote control and put it on the table"]
]
}


### Important Notes ###
* Note that the subtasks should be independent to each other, i.e. the robots can complete them in any order; 
* If the subtasks require the robots to complete them in a specific order, these subtask should be combined into one subtask.
* Make sure that the objects you suggest the robots to interact with are actually in the environment.
* Plan actions that are consistent with the object's real-world behavior and usage.
* Do not use OpenObject on surfaces like tables or countertops.
* For any Openable receptacle (e.g., drawer, fridge, cabinet), you MUST:
    - Open the container before placing any object inside;
    - Ensure the robot has empty hands before opening or closing;
    - Close the container after placing the object.
* Robots cannot open objects (e.g., drawers, cabinets, fridge) while holding something. Ensure the robot’s hand is empty before attempting to open or close objects.
* If an object needs to be placed inside an openable container, open it first before placing the item.
* Openable object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* If you open an object, please ensure that you close it before you move to a different place.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)
* If you need to put an object on a receptacle, you need to pick it up first, then navigate to the receptacle, and then put it on the receptacle.
* The robots can hold only one object at a time. Make sure that the actions you suggest do not require the robots to hold more than one object at a time.
* Do NOT assume or default to common receptacles like CounterTop or Table unless explicitly specified in the task.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.

Planner User Prompt: {Task: Put the bread, lettuce, and tomato in the fridge, 
Number of agents: 2, 
Objects: ['Apple_1', 'Book_1', 'Bottle_1', 'Bowl_1', 'Bread_1', 'ButterKnife_1', 'Cabinet_1', 'Cabinet_2', 'Cabinet_3', 'Cabinet_4', 'Cabinet_5', 'Cabinet_6', 'Cabinet_7', 'Cabinet_8', 'Cabinet_9', 'CoffeeMachine_1', 'CounterTop_1', 'CounterTop_2', 'CounterTop_3', 'CreditCard_1', 'Cup_1', 'DishSponge_1', 'Drawer_1', 'Drawer_2', 'Drawer_3', 'Drawer_4', 'Drawer_5', 'Drawer_6', 'Drawer_7', 'Drawer_8', 'Drawer_9', 'Egg_1', 'Faucet_1', 'Floor_1', 'Fork_1', 'Fridge_1', 'GarbageCan_1', 'HousePlant_1', 'Kettle_1', 'Knife_1', 'Lettuce_1', 'LightSwitch_1', 'Microwave_1', 'Mug_1', 'Pan_1', 'PaperTowelRoll_1', 'PepperShaker_1', 'Plate_1', 'Pot_1', 'Potato_1', 'SaltShaker_1', 'Shelf_1', 'Shelf_2', 'Shelf_3', 'ShelvingUnit_1', 'Sink_1', 'Sink_2', 'SoapBottle_1', 'Spatula_1', 'Spoon_1', 'Statue_1', 'Stool_1', 'Stool_2', 'StoveBurner_1', 'StoveBurner_2', 'StoveBurner_3', 'StoveBurner_4', 'StoveKnob_1', 'StoveKnob_2', 'StoveKnob_3', 'StoveKnob_4', 'Toaster_1', 'Tomato_1', 'Vase_1', 'Vase_2', 'Window_2', 'WineBottle_1'], }
Payload for LLM: [{'role': 'system', 'content': '\nYou are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice, and Bob carry out a task. \nThey can perform the following actions: \n## list of action that can be performed by the robot\n{\'idle\': [\'Idle\', \'Done\'], \'interact_with_object\': [\'NavigateTo<object_name>\', \'PickupObject<object_name>\', \'PutObject<receptacle_name>\', \'OpenObject<object_name>\', \'CloseObject<object_name>\', \'ToggleObjectOn<object_name>\', \'ToggleObjectOff<object_name>\', \'BreakObject<object_name>\', \'CookObject<object_name>\', \'SliceObject<object_name>\', \'DirtyObject<object_name>\', \'CleanObject<object_name>\', \'FillObjectWithLiquname<object_name>\', \'EmptyLiquidFromObject<object_name>\', \'UseUpObject<object_name>\'], \'interact_without_navigation\': [\'DropHandObject\', \'ThrowObject\']}\n\n\n### TASK ###\nYou will get a description of the task robots are supposed to do.\nYou will get the current state of the enviroment, including the list of objects in the environment.\n\nYou need to suggest the decompose the task into several subtasks for that robots to complete in order to achieve the given Task, based on the objects in the environment and the ability of robots. \n\nFor example, if the task is "Put the vase, tissue box, and remote control on the table", the subtasks could be:\n1. pick up the vase and put it on the table\n2. pick up the tissue box and put it on the table\n3. pick up the remote control and put it on the table \n\n\n\n\n### INPUT FORMAT ###\n{Task: a high level description of the final goal the robots are supposed to complete,\nNumber of agents": the number of robots in the environment,\nObjects: a list of objects in the current enviroment}\n\n### OUTPUT FORMAT ###\nYou will output a list of subtasks for each robot in the following format, in json format:\n{\n"Subtasks": [[subtask 1], [subtask 2], ..., [subtask n]],\n}\nexample:\n{\n"Subtasks": [\n    ["pick up the vase and put it on the table"],\n    ["pick up the tissue box and put it on the table"],\n    ["pick up the remote control and put it on the table"]\n]\n}\n\n\n### Important Notes ###\n* Note that the subtasks should be independent to each other, i.e. the robots can complete them in any order; \n* If the subtasks require the robots to complete them in a specific order, these subtask should be combined into one subtask.\n* Make sure that the objects you suggest the robots to interact with are actually in the environment.\n* Plan actions that are consistent with the object\'s real-world behavior and usage.\n* Do not use OpenObject on surfaces like tables or countertops.\n* For any Openable receptacle (e.g., drawer, fridge, cabinet), you MUST:\n    - Open the container before placing any object inside;\n    - Ensure the robot has empty hands before opening or closing;\n    - Close the container after placing the object.\n* Robots cannot open objects (e.g., drawers, cabinets, fridge) while holding something. Ensure the robot’s hand is empty before attempting to open or close objects.\n* If an object needs to be placed inside an openable container, open it first before placing the item.\n* Openable object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.\n* If you open an object, please ensure that you close it before you move to a different place.\n* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)\n* If you need to put an object on a receptacle, you need to pick it up first, then navigate to the receptacle, and then put it on the receptacle.\n* The robots can hold only one object at a time. Make sure that the actions you suggest do not require the robots to hold more than one object at a time.\n* Do NOT assume or default to common receptacles like CounterTop or Table unless explicitly specified in the task.\n\n* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED\nLet\'s work this out in a step by step way to be sure we have the right answer.\n'}, {'role': 'user', 'content': "{Task: Put the bread, lettuce, and tomato in the fridge, \nNumber of agents: 2, \nObjects: ['Apple_1', 'Book_1', 'Bottle_1', 'Bowl_1', 'Bread_1', 'ButterKnife_1', 'Cabinet_1', 'Cabinet_2', 'Cabinet_3', 'Cabinet_4', 'Cabinet_5', 'Cabinet_6', 'Cabinet_7', 'Cabinet_8', 'Cabinet_9', 'CoffeeMachine_1', 'CounterTop_1', 'CounterTop_2', 'CounterTop_3', 'CreditCard_1', 'Cup_1', 'DishSponge_1', 'Drawer_1', 'Drawer_2', 'Drawer_3', 'Drawer_4', 'Drawer_5', 'Drawer_6', 'Drawer_7', 'Drawer_8', 'Drawer_9', 'Egg_1', 'Faucet_1', 'Floor_1', 'Fork_1', 'Fridge_1', 'GarbageCan_1', 'HousePlant_1', 'Kettle_1', 'Knife_1', 'Lettuce_1', 'LightSwitch_1', 'Microwave_1', 'Mug_1', 'Pan_1', 'PaperTowelRoll_1', 'PepperShaker_1', 'Plate_1', 'Pot_1', 'Potato_1', 'SaltShaker_1', 'Shelf_1', 'Shelf_2', 'Shelf_3', 'ShelvingUnit_1', 'Sink_1', 'Sink_2', 'SoapBottle_1', 'Spatula_1', 'Spoon_1', 'Statue_1', 'Stool_1', 'Stool_2', 'StoveBurner_1', 'StoveBurner_2', 'StoveBurner_3', 'StoveBurner_4', 'StoveKnob_1', 'StoveKnob_2', 'StoveKnob_3', 'StoveKnob_4', 'Toaster_1', 'Tomato_1', 'Vase_1', 'Vase_2', 'Window_2', 'WineBottle_1'], }"}]
Before Planner LLM Response: {
        "Subtasks": [
            ["'pick up the tomato and put it on the countertop'"],
            ["pick up the lettuce and put it on the countertop"],
            ["pick up the bread and put it on the countertop"]
        ]
    }
    , type of res_content: <class 'str'>
After Planner LLM Response: [["'pick up the tomato and put it on the countertop'"], ['pick up the lettuce and put it on the countertop'], ['pick up the bread and put it on the countertop']], type of res_content: <class 'list'>
editor Prompt: 
You are an expert task planner and editor.

You will receive:
- A high-level task
- A list of subtasks generated by another agent
- A list of objects currently in the scene

Your job is to correct any incorrect subtasks so that the result matches the intended goal.

### Task Interpretation Rules ###
1. If the task says "put X in the fridge", make sure X is actually placed **inside** the fridge.
2. If a container Y (like Fridge or Drawer) is used, make sure it is opened before placing an item inside, and closed afterward. For example:
   - "open the Y, put the X inside, and close the Y"
3. Never place items on receptacles not mentioned in the task.
4. Preserve the robot capabilities (e.g. hand constraints, action limits), but you may restructure or combine subtasks if necessary.

### Input Format ###
{
Task: a high level description of the final goal the robots are supposed to do/complete,
Number of agents: the number of robots in the environment,
Subtasks: A list of subtasks generated by another agent.
Objects:  a list of objects in the current enviroment.
}

### Output Format ###
Return only the corrected list of subtasks in the following format,in json format:
{
"Subtasks": [[subtask 1], [subtask 2], ..., [subtask n]],
}


* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer.

editor User Prompt: {Task: Put the bread, lettuce, and tomato in the fridge, 
Number of agents: 2, 
Objects: ['Apple_1', 'Book_1', 'Bottle_1', 'Bowl_1', 'Bread_1', 'ButterKnife_1', 'Cabinet_1', 'Cabinet_2', 'Cabinet_3', 'Cabinet_4', 'Cabinet_5', 'Cabinet_6', 'Cabinet_7', 'Cabinet_8', 'Cabinet_9', 'CoffeeMachine_1', 'CounterTop_1', 'CounterTop_2', 'CounterTop_3', 'CreditCard_1', 'Cup_1', 'DishSponge_1', 'Drawer_1', 'Drawer_2', 'Drawer_3', 'Drawer_4', 'Drawer_5', 'Drawer_6', 'Drawer_7', 'Drawer_8', 'Drawer_9', 'Egg_1', 'Faucet_1', 'Floor_1', 'Fork_1', 'Fridge_1', 'GarbageCan_1', 'HousePlant_1', 'Kettle_1', 'Knife_1', 'Lettuce_1', 'LightSwitch_1', 'Microwave_1', 'Mug_1', 'Pan_1', 'PaperTowelRoll_1', 'PepperShaker_1', 'Plate_1', 'Pot_1', 'Potato_1', 'SaltShaker_1', 'Shelf_1', 'Shelf_2', 'Shelf_3', 'ShelvingUnit_1', 'Sink_1', 'Sink_2', 'SoapBottle_1', 'Spatula_1', 'Spoon_1', 'Statue_1', 'Stool_1', 'Stool_2', 'StoveBurner_1', 'StoveBurner_2', 'StoveBurner_3', 'StoveBurner_4', 'StoveKnob_1', 'StoveKnob_2', 'StoveKnob_3', 'StoveKnob_4', 'Toaster_1', 'Tomato_1', 'Vase_1', 'Vase_2', 'Window_2', 'WineBottle_1'], 
Subtasks: [["'pick up the tomato and put it on the countertop'"], ['pick up the lettuce and put it on the countertop'], ['pick up the bread and put it on the countertop']], }
Before Editor LLM Response: {
"Subtasks": [["open the Fridge_1"], ["pick up the tomato_1 and put it inside the Fridge_1"], ["pick up the lettuce_1 and put it inside the Fridge_1"], ["pick up the bread_1 and put it inside the Fridge_1"], ["close the Fridge_1"]]
}
After Editor LLM Response: [['open the Fridge_1'], ['pick up the tomato_1 and put it inside the Fridge_1'], ['pick up the lettuce_1 and put it inside the Fridge_1'], ['pick up the bread_1 and put it inside the Fridge_1'], ['close the Fridge_1']], type of res_content: <class 'list'>
Action Prompt: 
You are an excellent planner and robot controller who is tasked with helping 2 embodied robots named Alice, and Bob carry out a task. 
They can perform the following actions: 
## list of action that can be performed by the robot
{'idle': ['Idle', 'Done'], 'interact_with_object': ['NavigateTo<object_name>', 'PickupObject<object_name>', 'PutObject<receptacle_name>', 'OpenObject<object_name>', 'CloseObject<object_name>', 'ToggleObjectOn<object_name>', 'ToggleObjectOff<object_name>', 'BreakObject<object_name>', 'CookObject<object_name>', 'SliceObject<object_name>', 'DirtyObject<object_name>', 'CleanObject<object_name>', 'FillObjectWithLiquname<object_name>', 'EmptyLiquidFromObject<object_name>', 'UseUpObject<object_name>'], 'interact_without_navigation': ['DropHandObject', 'ThrowObject']}


“Done” is used to indicate that the robot has completed its task and is ready to stop.
“Idle” is used to let a robot wait in place for one step, if necessary.

### Task ###
You will get a description of the task robots are supposed to do. 
You will get the current state of the enviroment, including the list of objects in the environment.
You will get the subtasks that the robots are supposed to complete in order to achieve the final goal(Task), based on the objects in the environment and the ability of robots.

Based on the above information, you need to break down each subtask into smaller actions and assign these subtasks to the appropriate robot.

* Make sure your output length is correct: the length of the Actions list must equal the number of robots. 
For example, if there are two robots, the Actions list should contain exactly two elements, one for each robot agent.

### INPUT FORMAT ###
{
Task: a high level description of the final goal the robots are supposed to do/complete,
Number of agents: the number of robots in the environment,
Subtasks: a list of subtasks that the robots are supposed to complete in order to achieve the final goal(Task),
Objects:  a list of objects in the current enviroment.
}


### OUTPUT FORMAT ###
You will output a list of actions for each robot in the following format, in json format:
{
"Actions": [[actions for robot 1], [actions for robot 2], ..., [actions for robot N]],
}

where each actions is a list of strings in the format of "ActionName<arg1, arg2, ...>".
For example, if robot 1 should pick up an object with id "Object_1" and put the object on the table with id "Table_1", the output should be:
[[NavigateTo<Object_1>, PickupObject<Object_1>, NavigateTo<Table_1>, PutObject<Table_1>], [Idle], ...]
<Object_1> and <Table_1> are the name plus the number of the object in the environment. If there are multiple objects with the same name but different id, you can use the number to distinguish them, e.g. "Object_1", "Object_2", etc. don't use the id only.

* Example1:
if given INPUT subtasks are:
{"
"Number of agents": 2,
"Task": "bring a vase, tissue box, and remote control to the counter top to make a sandwich",
"Subtasks": [
    ["pick up the vase and put it on the table"],
    ["pick up the tissue box and put it on the table"],
    ["pick up the remote control and put it on the table"]
],
}

* The OUTPUT could be:
{
"Actions": [
    ["NavigateTo(Vase_1)", "PickupObject(Vase_1)", "NavigateTo(Table_1)", "PutObject(Table_1)"],
    ["NavigateTo(TissueBox_1)", "PickupObject(TissueBox_1)", "NavigateTo(Table_1)", "PutObject(Table_1)", "NavigateTo(RemoteControl)", "PickupObject<RemoteControl>", "PutObject(Table_1)","PutObject<Table_1>"]
]
}
which means:
- Robot 1 should pick up the Vase with "Vase_1" and put it on the table with "Table_1"
- Robot 2 should pick up the tissue box with "TissueBox_1" and put it on the counter top with "Table_1", then pick up the remote control with "RemoteControl_1" and put it on the table with "Table_1".


* Example2:
if given INPUT subtasks are:
{
"Number of agents": 2,
"Task": "put the Book inside the drawer",
"Subtasks": ["put the Book inside the drawer"]
}

* the OUTPUT could be:
{ "Actions": 
[["NavigateTo(Drawer_1)", "OpenObject(Drawer_1)", "NavigateTo(Book_1)", "PickupObject(Book_1)",,"PutObject(Drawer_1)", "CloseObject(Drawer_1)"],["Idle"]]
}

Note: This subtask combines opening, placing, and closing into one sequence, since the drawer must be opened before placing the item and closed afterward. The robot must have empty hands to operate the drawer.

### Important Notes ###
* The `Actions` list must contain exactly 2 sublists—one for each robot. Each sublist should include the full sequence of actions for that robot.
* Each robot's action list should be a sequential plan—actions should be performed in order and not interleaved with other robots.
* Make sure that the objects you suggest the robots to interact with are actually in the environment.
* Plan actions that are consistent with the object's real-world behavior and usage.
* Do not use OpenObject on surfaces like tables or countertops.
* For any Openable receptacle (e.g., drawer, fridge, cabinet), you MUST:
    - Open the container before placing any object inside;
    - Ensure the robot has empty hands before opening or closing;
    - Close the container after placing the object.
* Robots cannot open objects (e.g., drawers, cabinets, fridge) while holding something. Ensure the robot's hand is empty before attempting to open or close objects.
* If an object needs to be placed inside an openable container, open it first before placing the item.
* Openable object like drawers, cabinets, fridge can block the path of the robot. So open objects only when you think it is necessary.
* If you open an object, please ensure that you close it before you move to a different place.
* When possible do not perform extraneous actions when one action is sufficient (e.g. only do CleanObject to clean an object, nothing else)
* If you need to put an object on a receptacle, you need to pick it up first, then navigate to the receptacle, and then put it on the receptacle.
* The robots can hold only one object at a time. Make sure that the actions you suggest do not require the robots to hold more than one object at a time.
* Since there are 2 agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.
* Be aware that the robots will be performing actions in parallel, so make sure that the actions you suggest do not conflict with each other.
* You should consider the position of the objects and the robots in the enviroment when assigning actions and receptacle to the robots.
* The number of subtasks may be greater than the number of robots. You may distribute the subtasks in any way you deem optimal.
* A robot can handle multiple subtasks in a single plan, but each robot must only appear once in the `Actions` list.
* Do NOT assume or default to common receptacles like CounterTop or Table unless explicitly specified in the task.



* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

Let's work this out in a step by step way to be sure we have the right answer.

Action User Prompt: {Task: Put the bread, lettuce, and tomato in the fridge, 
Number of agents: 2, 
Objects: ['Apple_1', 'Book_1', 'Bottle_1', 'Bowl_1', 'Bread_1', 'ButterKnife_1', 'Cabinet_1', 'Cabinet_2', 'Cabinet_3', 'Cabinet_4', 'Cabinet_5', 'Cabinet_6', 'Cabinet_7', 'Cabinet_8', 'Cabinet_9', 'CoffeeMachine_1', 'CounterTop_1', 'CounterTop_2', 'CounterTop_3', 'CreditCard_1', 'Cup_1', 'DishSponge_1', 'Drawer_1', 'Drawer_2', 'Drawer_3', 'Drawer_4', 'Drawer_5', 'Drawer_6', 'Drawer_7', 'Drawer_8', 'Drawer_9', 'Egg_1', 'Faucet_1', 'Floor_1', 'Fork_1', 'Fridge_1', 'GarbageCan_1', 'HousePlant_1', 'Kettle_1', 'Knife_1', 'Lettuce_1', 'LightSwitch_1', 'Microwave_1', 'Mug_1', 'Pan_1', 'PaperTowelRoll_1', 'PepperShaker_1', 'Plate_1', 'Pot_1', 'Potato_1', 'SaltShaker_1', 'Shelf_1', 'Shelf_2', 'Shelf_3', 'ShelvingUnit_1', 'Sink_1', 'Sink_2', 'SoapBottle_1', 'Spatula_1', 'Spoon_1', 'Statue_1', 'Stool_1', 'Stool_2', 'StoveBurner_1', 'StoveBurner_2', 'StoveBurner_3', 'StoveBurner_4', 'StoveKnob_1', 'StoveKnob_2', 'StoveKnob_3', 'StoveKnob_4', 'Toaster_1', 'Tomato_1', 'Vase_1', 'Vase_2', 'Window_2', 'WineBottle_1'], 
Subtasks: [['open the Fridge_1'], ['pick up the tomato_1 and put it inside the Fridge_1'], ['pick up the lettuce_1 and put it inside the Fridge_1'], ['pick up the bread_1 and put it inside the Fridge_1'], ['close the Fridge_1']], }
Before Action LLM Response: {
"Actions": [
    ["NavigateTo(Fridge_1)", "OpenObject(Fridge_1)", "NavigateTo(Tomato_1)", "PickupObject(Tomato_1)", "PutObject(Fridge_1)", "NavigateTo(Lettuce_1)", "PickupObject(Lettuce_1)", "PutObject(Fridge_1)", "NavigateTo(Bread_1)", "PickupObject(Bread_1)", "PutObject(Fridge_1)", "CloseObject(Fridge_1)"],
    ["Idle"]
]
}
After Action LLM Response: [['NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'PutObject(Fridge_1)', 'NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'PutObject(Fridge_1)', 'NavigateTo(Bread_1)', 'PickupObject(Bread_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'], ['Idle']], type of res_content: <class 'list'>
'''