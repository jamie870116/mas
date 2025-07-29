'''
Baseline : Centralized LLM + replanning

Temporary structure:
1. initial planning (remain the same as previous method: given task, let planner and editor to generate a list of subtasks (this will be the open subtasks)
2. start a loop, until timeout or all the open subtasks is empty:
2.1 update open subtasks and completed subtask
2.2 allocate subtask to robot agents in the environment with llm
2.3 break down each assigned subtasks with llm into a list of smaller available actions
2.4 execute one subtask per agents
2.5 verify if the subtask is completed by two methods: one LLM (by observation of the env), another is rule-based
2.6 replan: similar to initial planning : given task and closed subtask, let planner and editor to generate a list of subtasks (this will be the new open subtasks)
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
from env_cen import AI2ThorEnv_cen as AI2ThorEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import save_to_video

client = OpenAI(api_key=Path('api_key_ucsb.txt').read_text())

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
        "Idle"
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

PLANNER_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions: {ai2thor_actions}

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
"Subtasks": [subtask 1, subtask 2, ..., subtask n],
}}
example:
{{
"Subtasks": [
    "pick up the vase and put it on the table",
    "pick up the tissue box and put it on the table",
    "pick up the remote control and put it on the table"
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

ALLOCATOR_PROMPT = f"""

You are an expert task allocator who is tasked with helping {len(AGENT_NAMES)} embodied robots carry out a task. 

### TASK ###
You will receive:
- a description of the overall task goal,
- the number of available robot agents,
- a list of open (pending) subtasks and a list of completed subtasks,
- a list of previously failed subtasks (optional),
- the failure reasons of those subtasks (optional),
- a list of failed actions (optional),
- and the current inventory (i.e., what each robot is holding, if anything).

Your job is to assign at most one subtask to each available agent at a time, in a way that helps fulfill the overall task.
If a subtask depends on the completion of another subtask, assign "Idle" to the agent instead, until the prerequisite subtask is completed.
Also output the subtasks that are not yet been assigned to any robot agent.
- *Executable* means: all of its prerequisite subtasks (if any) appear in the **Robots' completed subtasks** list.
- If a subtask is **not yet executable** (blocked by missing prerequisites), do **not** assign it. Instead, assign `"Idle"` to that agent (if no other executable subtask is available).
- Do not assign the **same subtask** to multiple agents unless the subtask text explicitly indicates it is parallelizable (e.g., contains a marker like `[MULTI]` or specifies distinct targets).
- If there are fewer open subtasks than agents, assign remaining agents `"Idle"`.

Also return the list of **unassigned subtasks** (these are the open subtasks that remain after this allocation—include both blocked and unallocated-but-executable ones that you did not assign this round).
In addition, include a short explanation of **why** the subtask was (or was not) assigned to each agent. These reasons will help the coordinator module understand your allocation logic.



### INPUT FORMAT ###
{{
  "Task": "High-level description of the task the robots are supposed to do",
  "Number of agents": <int>,
  "Robots' open subtasks": [ ... ] or None ,
  "Robots' completed subtasks": [ ... ] or None,
  "failure_subtasks": [ ... ] or None,
  "subtask_failure_reasons": [ ... ] or None,
  "failed_acts": [ ... ] or None,
  "inventory": {{
      "agent1": "object or None",
      "agent2": "object or None",
      ...
  }} or None
}}


### OUTPUT FORMAT ###
Return a JSON object:
{{
  "Allocation": {{
    "agent1": "subtask or Idle",
    "agent2": "subtask or Idle",
    ...
    "agentN": "subtask or Idle"
  }},
  "Remain": ["unassigned_subtask1", "unassigned_subtask2", ...],
  "Reason": ["reason"]
}}


### Important Notes ###
* Each agent can be assigned at most one subtask.
* A subtask can only be assigned if all its prerequisites have been completed.
* Do **not** assign the **same subtask** to multiple agents.
* If no subtasks are available for an agent, assign `"Idle"`.
* Do not assign a subtask to an agent if the agent's inventory would make it infeasible.
* Consider failure history and avoid repeating problematic assignments.
* **NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED**

Let's work this out in a step by step way to be sure we have the right answer.
"""

ACTION_PROMPT = f"""

You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions:
{ai2thor_actions}

"Idle" is used to let a robot wait in place for one step, if necessary.


### Task ###
You will receive:
- a high-level description of the overall task the robots are supposed to complete,
- a list of objects present in the environment,
- a list of subtasks that need to be completed.

Your job is to break down each subtask independently into a list of executable actions that a robot can perform to complete it.


### INPUT FORMAT ###
{{
  "Task": <string> — description of the overall goal,
  "Number of agents": <int> — number of available robots,
  "Objects": <list<string>> — list of object in the environment,
  "Subtasks": <list<string>> — list of subtasks; each subtask is to be broken down into actions
}}

### OUTPUT FORMAT ###
Return a JSON object in the following format:
{{
  "Actions": [
    [<action_1_for_subtask_1>, <action_2_for_subtask_1>, ...],
    [<action_1_for_subtask_2>, <action_2_for_subtask_2>, ...],
    ...
  ]
}}

# Note:
Each action must be a string in the format: ActionName<Object_ID or arguments>
For example, to pick up an object with ID "Object_1" and place it on a table with ID "Table_1":
[
  ["NavigateTo(Object_1)", "PickupObject(Object_1)", "NavigateTo(Table_1)", "PutObject(Table_1)"],
  ...
]
<Object_ID> should match the names in the "Objects" list. If there are multiple objects of the same type, use numbered identifiers like "Book_1", "Book_2", etc. — do not use object names without identifiers.


# Example1:
if given INPUT subtasks are:
{{"
"Subtasks": [
    ["pick up the vase and put it on the table"],
    ["pick up the tissue box and put it on the table"],
],
}}

* The OUTPUT could be:
{{
"Actions": [
    ["NavigateTo(Vase_1)", "PickupObject(Vase_1)", "NavigateTo(Table_1)", "PutObject(Table_1)"],
    ["NavigateTo(TissueBox_1)", "PickupObject(TissueBox_1)", "NavigateTo(Table_1)", "PutObject(Table_1)"]
]
}}


* Example2:
if given INPUT subtasks are:
{{
"Task": "put the Book inside the drawer",
"Subtasks": ["put the Book inside the drawer", "Idle"]
}}

* the OUTPUT could be:
{{ "Actions": 
[["NavigateTo(Drawer_1)", "OpenObject(Drawer_1)", "NavigateTo(Book_1)", "PickupObject(Book_1)",,"PutObject(Drawer_1)", "CloseObject(Drawer_1)"],["Idle"]]
}}
Note: The drawer must be opened before placing the item, and closed afterward. The robot must have empty hands to operate the drawer.


### Important Notes ###
* Each subtask must be broken down into a **sequential list of feasible atomic actions**, based on the available actions defined above.
* The output must contain one list of actions for each input subtask, in the same order. The i-th subtask must correspond to the i-th action list.
* Use only objects listed in the "Objects" input. Do not invent object names or assume availability.
* All object references in the action strings must include the object no. (e.g., "Book_1", not just "Book").
* If a subtask is "Idle", return a single action ["Idle"].
* Actions must follow physical and semantic feasibility:
    - Robots must navigate to an object before interacting with it.
    - To place an object inside or on another object, the robot must first pick up the object.
    - Robots can only hold **one object at a time**.
    - Do not open containers while holding an object—robots must have empty hands to open or close openable objects.
    - When placing an object inside an openable container (e.g., Drawer, Cabinet, Fridge), the container must be opened first and closed afterward.
    - Do not use OpenObject on non-openable objects like tables or countertops.
* Minimize redundant or unnecessary actions. For example:
    - Use only CleanObject when cleaning is needed—do not combine with unrelated actions.
    - Do not open containers unless it is required to fulfill the subtask.
    - Close openable containers only if the robot had opened them earlier in the same sequence.
* Do not assume default targets (e.g., Table or CounterTop) unless the subtask or task explicitly mentions them.
* Each subtask’s action list should be complete and self-contained. You should not assume external context, shared memory, or agent collaboration.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN JSON AS DESCRIBED
Let's work this out in a step by step way to be sure we have the right answer.
"""

REPLAN_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
They can perform the following actions: {ai2thor_actions}

### TASK ###
Your job is to reason over the current state of the environment and the progress history of previously executed plans, in order to **replan** the sequence of subtasks necessary to complete the overall goal.

This environment is partially observable, and robots may encounter execution failures due to occlusion, distance, blocked paths, or misidentified object locations. Therefore, the replanning should take into account:
- Which subtasks have already been completed successfully
- Which subtasks have failed (and the reasons why)
- The list of failed actions
- The current objects in the environment
- The current inventory (objects held by robots)
- The number of available robots

Your goal is to **revise or regenerate a valid and efficient sequence of subtasks** for the robots, using all the information above. When possible, retain subtasks that have not yet failed. Otherwise, adjust or replace the failed ones with alternative approaches.


### INPUT FORMAT ###
{{
Task: a high level description of the final goal the robots are supposed to complete,
Number of agents": the number of robots in the environment,
Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.,
failure_subtasks: a list of subtasks that have failed,
subtask_failure_reasons: a list of reasons why the subtasks failed,
inventory: what the robots are currently holding,
failed_acts: a list of actions that failed,
Objects: a list of objects in the current enviroment
}}

### OUTPUT FORMAT ###
You will output a list of subtasks for each robot in the following format, in json format:
{{
"Subtasks": [subtask 1, subtask 2, ..., subtask n],
}}
example:
{{
"Subtasks": [
    "pick up the vase and put it on the table",
    "pick up the tissue box and put it on the table",
    "pick up the remote control and put it on the table"
]
}}


### Important Notes ###
* Each subtask should be atomic and goal-directed (e.g., "pick up the apple and put it in the fridge").
* Avoid repeating failed subtasks unless their failure cause has been addressed.
* Ensure the plan is minimal-avoid unnecessary steps.
* The subtasks should not be robot-specific; centralized planning assigns tasks without binding them to specific agents.
* You can assume the robots are coordinated and will handle task allocation downstream.
* Consider object availability, spatial constraints, and object affordances when generating subtasks.
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


def get_llm_response(payload, model = "gpt-4o", temperature= 0.7, max_tokens=1024) -> str:
    response = client.chat.completions.create(model=model, 
                                                messages=payload, 
                                                max_tokens=max_tokens, 
                                                temperature=temperature,)

    return response, response.choices[0].message.content.strip()

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

def prepare_prompt(env: AI2ThorEnv, mode: str = "init", addendum: str = "", subtasks=[], info={}) -> str:
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
    elif mode == "editor":
        system_prompt = EDITOR_PROMPT
        input = env.get_center_llm_input()
        input["Subtasks"] = subtasks
        user_prompt = convert_dict_to_string(input)
    elif mode == "action":
        system_prompt = ACTION_PROMPT
        if not subtasks:
            print("No subtasks provided")
            return None, None
        input = env.get_center_llm_input()
        input["Subtasks"] = subtasks
        user_prompt = convert_dict_to_string(input)

    elif mode == "allocator":
        system_prompt = ALLOCATOR_PROMPT
        input = env.get_center_allocator_llm_input()
        if info:
            input['failure_subtasks'] = info['failure_subtasks']
            input['subtask_failure_reasons'] = info['subtask_failure_reasons']
            input['failed_acts'] = info['failed_acts']
        user_prompt = convert_dict_to_string(input)

    elif mode == "replan":
        system_prompt = REPLAN_PROMPT
        input = env.get_replan_llm_input()
        if info:
            input['failure_subtasks'] = info['failure_subtasks']
            input['subtask_failure_reasons'] = info['subtask_failure_reasons']
            input['failed_acts'] = info['failed_acts']
        user_prompt = convert_dict_to_string(input)

    user_prompt += addendum
    return system_prompt, user_prompt




def process_planner_llm_output(res_content):
    try:
        return json.loads(res_content)["Subtasks"]
    except json.JSONDecodeError as e:
        print(f"[Warning] Initial JSON decode failed: {e}")

        try:
            if res_content.startswith("```json"):
                res_content = res_content.removeprefix("```json").strip()
            elif res_content.startswith("```"):
                res_content = res_content.removeprefix("```").strip()
            if res_content.endswith("```"):
                res_content = res_content[:-3].strip()
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
            if res_content.startswith("```json"):
                res_content = res_content.removeprefix("```json").strip()
            elif res_content.startswith("```"):
                res_content = res_content.removeprefix("```").strip()
            if res_content.endswith("```"):
                res_content = res_content[:-3].strip()
            res_content = re.sub(r",\s*([\]}])", r"\1", res_content)
            res_content = res_content.replace("'", '"')
            res_content = res_content.strip()

            data = json.loads(res_content)
            return data["Actions"]
        except Exception as e2:
            print(f"[Error] Failed to parse fixed JSON: {e2}")
            return None

def process_allocator_llm_output(res_content):
    """
    Input:
    {
        "Allocation": {
            "agent1": "pick up the tomato and put it on the countertop",
            "agent2": "pick up the lettuce and put it on the countertop"
        },
        "Remain": [
            "pick up the bread and put it on the countertop"
        ]
        "Reason": [""]
    }
    Output:
    allocation: [subtask, subtask]
    remain: []
    """
    try:
        remain = json.loads(res_content)["Remain"]
        allocations = json.loads(res_content)["Allocation"]
        reasons = json.loads(res_content)["Reason"]
        res = []
        for i in range(NUM_AGENTS):
            key = 'agent' + str(i+1)
            if key in allocations:
                res.append(allocations[key].strip('"\''))
            else:
                res.append("Idle")
        return res, remain
    except json.JSONDecodeError as e:
        print(f"[Warning] Initial JSON decode failed: {e}")
        try:
            if res_content.startswith("```json"):
                res_content = res_content.removeprefix("```json").strip()
            elif res_content.startswith("```"):
                res_content = res_content.removeprefix("```").strip()
            if res_content.endswith("```"):
                res_content = res_content[:-3].strip()
            res_content = re.sub(r",\s*([\]}])", r"\1", res_content)
            res_content = res_content.replace("'", '"')
            res_content = res_content.strip()

            data = json.loads(res_content)
            remain = data["Remain"]
            allocations = data["Allocation"]
            reasons = json.loads(res_content)["Reason"]
            res = []
            for i in range(NUM_AGENTS):
                key = 'agent' + str(i+1)
                if key in allocations:
                    res.append(allocations[key].strip('"\''))
                else:
                    res.append("Idle")
            return res, remain
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

def initial_subtask_planning(env, config):
    """初始階段：由 LLM 進行任務分解與編輯"""
    # ---1. Planner LLM 產生初步 subtasks
    
    planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")
    planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    res, res_content = get_llm_response(planner_payload, model=config['model'])
    # print('init plan llm output', res_content)
    subtasks = process_planner_llm_output(res_content)
    # print(f"After Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    # ---2. Editor LLM 修正 subtasks
    editor_prompt, editor_user_prompt = prepare_prompt(env, mode="editor", subtasks=subtasks)
    editor_payload = prepare_payload(editor_prompt, editor_user_prompt)
    res, res_content = get_llm_response(editor_payload, model=config['model'])

    subtasks = process_planner_llm_output(res_content)
    print(f"After Editor LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    # for testing
    # subtasks = ['pick up the tomato and put it on the countertop', 'pick up the lettuce and put it on the countertop', 'pick up the bread and put it on the countertop']
    return subtasks, []

def allocate_subtasks_to_agents(env, info={}):
    """分配 open_subtasks 給各 agent"""
    
    allocator_prompt, allocator_user_prompt = prepare_prompt(env, mode="allocator", info=info)
    allocator_payload = prepare_payload(allocator_prompt, allocator_user_prompt)
    # print("allocator system prompt: ", allocator_prompt)
    # print("allocator user prompt: ", allocator_user_prompt)
    res, res_content = get_llm_response(allocator_payload, model=config['model'])
    print('llm allocator output', res_content)
    allocation, remain = process_allocator_llm_output(res_content)
    print('allocation: ', allocation)
    # for testing
    # allocation =  ['pick up the tomato and put it on the countertop', 'pick up the lettuce and put it on the countertop']
    # remain =  ['pick up the bread and put it on the countertop']
    
    return allocation, remain

def decompose_subtask_to_actions(env, subtasks):
    """將 subtask 拆解成 atomic actions（LLM）"""
    action_prompt, action_user_prompt = prepare_prompt(env, mode="action", subtasks=subtasks)
    action_payload = prepare_payload(action_prompt, action_user_prompt)
    # print("action prompt:", action_prompt)
    # print("action user prompt:", action_user_prompt)
    res, res_content = get_llm_response(action_payload, model=config['model'])
    # print('llm output', res_content)
    actions = process_actions_llm_output(res_content)

    # For testing 
    # actions = [['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)'], ['NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)']]
    return actions

def get_steps_by_actions(env, actions):
    steps = env.actions_decomp(actions)
    return steps


def verify_subtask_completion(env, info):
    '''
    info:
        {
            "step": self.action_step_num,
            "actions_success": self.subtask_success_history,
            "success_subtasks": succ,
            "failure_subtasks": fail,
            "subtask_failure_reasons": self.subtask_failure_reasons,
            "inventory": self.inventory.copy(),
            "failed_acts": self.agent_failure_acts,
        }
    '''
    open_subtasks = env.open_subtasks
    closed_subtasks = env.closed_subtasks
    completed_subtasks = info['success_subtasks']
    # print(open_subtasks)
    for c in completed_subtasks:
        if c != 'Idle':
            print(c)
            open_subtasks.remove(c)
            closed_subtasks.append(c)
    return open_subtasks, closed_subtasks

def replan_open_subtasks(env, info, completed_subtasks):
    replan_prompt, replan_user_prompt = prepare_prompt(env, mode="replan", info=info)
    # print("replan system prompt: ", replan_prompt)
    # print("replan user prompt: ", replan_user_prompt)
    replan_payload = prepare_payload(replan_prompt, replan_user_prompt)
    res, res_content = get_llm_response(replan_payload, model=config['model'])
    # print('llm output', res_content)
    subtasks = process_planner_llm_output(res_content)
    # print(f"After Re-Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    return subtasks, completed_subtasks


     
def bundle_task_plan(subtasks, actions, decomp_actions):
    """
    Pack corresponding subtask, actions, and decomposed actions into aligned dicts.
    [
        { # for each agent
            "subtask": str,       # 語意任務的自然語言描述
            "actions": List[str], # 高階動作清單（LLM 輸出）
            "steps": List[List[str]]  # 每個 high-level action 對應的原子步驟序列
        }
        ...
    ]

    Example:
    [
        {
            "subtask": "pick up the tomato and put it on the countertop",
            "actions": [
                "NavigateTo(Tomato_1)",
                "PickupObject(Tomato_1)",
                "NavigateTo(CounterTop_1)",
                "PutObject(CounterTop_1)"
            ],
            "steps": [
                ['MoveAhead', 'MoveAhead', ..., 'RotateRight', 'MoveAhead'],
                ['PickupObject(Tomato_1)'],
                ['MoveAhead', 'MoveAhead', ..., 'RotateLeft'],
                ['PutObject(CounterTop_1)']
            ]
        },
        {
            "subtask": "pick up the lettuce and put it on the countertop",
            "actions": [...],
            "steps": [...]
        }
    ]


    """
    assert len(subtasks) == len(actions) == len(decomp_actions), "Input lists must be of equal length"

    bundled = []
    for subtask, acts, decomp in zip(subtasks, actions, decomp_actions):
        bundled.append({
            "subtask": subtask,
            "actions": acts,
            "steps": decomp
        })
    return bundled

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

def run_main():
    # --- Init.
    env, config = set_env_with_config('config/config.json')
    agents = env.agent_names
    task = config["task"]
    timeout = config["timeout"]
    obs = env.reset(test_case_id=config['test_id'])
    # --- initial subtask planning
    open_subtasks, completed_subtasks = initial_subtask_planning(env, config)

    # --- loop start
    start_time = time.time()
    while open_subtasks and (time.time() - start_time < timeout):
        env.update_plan(open_subtasks, completed_subtasks)
        print("open_subtasks: ", env.open_subtasks)
        print("closed_subtasks: ", env.closed_subtasks)

        # 2. 任務分配
        agent_assignments, remain = allocate_subtasks_to_agents(env)
        print("agent_assignments: ", agent_assignments)
        print("remain unassigned subtasks: ", remain)

        # 3. 拆解每個 agent 當前 subtask 為 actions
        actions = decompose_subtask_to_actions(env, agent_assignments)
        # print("actions: ", actions)
        decomp_actions = get_steps_by_actions(env, actions)

        # 4. 執行動作，當所有agent完成當前子任務或是同時卡住時，更新子任務狀態並進入下一個循環
        # print("decomp_actions: ", decomp_actions)
        cur_plan = bundle_task_plan(agent_assignments, actions, decomp_actions)
        print("cur_plan: ", cur_plan)
        isSuccess, info = env.stepwise_action_loop(cur_plan)

        # if not isSuccess:
        #     print("Subtask failed. Need replan.")
        #     print("Failure info:", info)
        # elif isSuccess:
        #     print("All subtasks completed successfully.")
        #     print("Execution summary:", info)
        # for testing
        # isSuccess = True
        # info = {'step': 0, 'actions_success': {'Alice': ['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)'], 'Bob': ['NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)']}, 'success_subtasks': ['pick up the tomato and put it on the countertop', 'pick up the lettuce and put it on the countertop'], 'failure_subtasks': [], 'subtask_failure_reasons': {'Alice': [], 'Bob': []}, 'inventory': ['nothing', 'nothing'], 'failed_acts': {'Alice': [], 'Bob': []}}
        print('info', info)
        open_subtasks, completed_subtasks = verify_subtask_completion(env, info)
        print("after verify open_subtasks: ", open_subtasks)
        print("after verify closed_subtasks: ", completed_subtasks)
        env.update_plan(open_subtasks, completed_subtasks)
        if open_subtasks or not isSuccess:
            open_subtasks, completed_subtasks = replan_open_subtasks(env, info, completed_subtasks)
            print("replan open_subtasks: ", open_subtasks)
            print("replan closed_subtasks: ", completed_subtasks)
            start_time = time.time()
        else:
            break
        
    # env.save_log()
   
    env.close()

if __name__ == "__main__":
    run_main()

