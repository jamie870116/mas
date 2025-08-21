'''
Baseline : Centralized LLM + replanning + shared memory(llm based)

TBD:
- (v) wrap up: reachable positions(only x,z), agent states(current position(x,z), inventory, facing direction),  objects in view(objectId, position) 
- (v) wrap up:
    failure reasons:
    types of failures:
    1. navigation failure: cannot reach the target position (too far, blocked path, occluded)
    2. interaction failure: cannot interact with the target object (too far, occluded, not visible) 
    3. task failure: the action is not feasible (e.g. trying to open a non-openable object, or put an object inside a non-receptacle object, etc.)
    4. object not found: the target object is not in view (not visible, occluded, out of reachable range)
    5. object not exist: the target object is not in the environment
    6. other unknown reasons
- (v) ACTION_PROMPT: add more details about the environment, reachable positions, objects in view, etc.
   reachable position should filter out the positions of each agent.
- (v) REPLAN_PROMPT: add more details about the environment, reachable positions, objects in view, etc. And failures in order to make more procise intructions,

- () more test 


structure same as llm_c.py but with more information about the environment and positions, failures, etc.:
1. initial planning (remain the same as previous method: given task, let planner and editor to generate a list of subtasks (this will be the open subtasks)
2. start a loop, until timeout or all the open subtasks is empty:
2.1 update open subtasks and completed subtask
2.2 allocate subtask to robot agents in the environment with llm
2.3 break down each assigned subtasks with llm into a list of smaller available actions
2.4 execute one subtask per agents
2.5 verify if the subtask is completed and identify the failure reason and collect the history and suggest the next step
2.6 replan: similar to initial planning : given task and closed subtask
'''

import json
import re
import os
import sys
import argparse
from pathlib import Path
import time
import base64
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
        "ShelvingUnit": ["Moveable"],
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

EXAMPLE_PLAN = f"""
#**EXAMPLES:**
High-level task: "Put the vase, tissue box, and remote control on the table", with 2 agents:

example output:
[
"navigate to the vase, pick up the vase, navigate to the table, and put it on the table",
"navigate to the tissue box, pick up the tissue box, navigate to the table, and put it on the table",
"navigate to the remote control, pick up the remote control, navigate to the table, and put it on the table",
]

High-level task: "Put the lettuce and potato in the fridge" with 2 agents, example output:
[
"navigate to the fridge and open the fridge, without picking up any objects, before putting anything inside",
"navigate to the lettece, pick up the lettuce,  navigate to the fridge, put it in the fridge",
"navigate to the potato, pick up the potato, navigate to the fridge and put it in the fridge",
"navigate to the fridge and close the fridge, without picking up any objects"
]

High-level task: "Turn on the laptop and television and turn off the light switch" with 3 agents, example output:
[
'navigate to the laptop, open the laptop, turn on the laptop',
'navigate to the television, turn on the television',
'navigate to the light switch, turn off the light switch'
]

High-level task: "slice a tomato" with 2 agents, example output:
[
'navigate to the knife, pick up the knife, navigate to the tomato, slice the tomato with the knife',
]

"""

FALURE_EXAMPLES = f"""
Example input:
{{
  "failure reason": "None",
  "memory": "Alice is holding ButterKnife_1 and is near the kitchen counter. Bob is holding Apple_1 and is also close to the counter, with Bowl_1 visible nearby on Bob's right. Both robots are in a good position to cooperate. The apple and knife are both held by the robots, ready for slicing.",
  "reason": "To slice the apple, the robot holding the knife (Alice) must interact with the apple, which is currently held by Bob. Bob should place Apple_1 down on the counter so Alice can perform the slicing action. After slicing, one robot can pick up a slice and put it into the bowl.",
  "suggestion": "next, Bob should PutObject<CounterTop_2> to place Apple_1 on the counter, and Alice should wait (Idle) until the apple is available for slicing."
}}
"""

VERIFY_EXAMPLE = f"""
Possible failure reasons and how to handle them:
- Navigation: "no-path", "object-not-in-view", "distance-too-far": Use micro-movements (MoveAhead, RotateRight, etc.) to get in view/reach, using current/target positions and available observations. For example, you can have subtasks like "look down and navigate to potato" if previous failure reason of "pick up potato" was "object-not-in-view". Or you can have subtask like "use micro-movements to navigate to potato" if previous subtask "navigate to potato" was failured.
example: 
** when given input shows "no-path to Tomato_1" for subtask "navigate to tomato", and there do have Tomato_1 in the environment, but as shown in the image, the agent is block by the fridge door, then you can have subtask "Rotate twice in same direction and MoveAhead to bypass the firdge door, then try navigate to Tomato_1 again", this should based on not only the input reachable postion but also other input information.
** when given input shows "object-not-in-view" for subtask "navigate to tomato", and based on other input information, that the distance between the agent and the tomato is already close enough, then you can have subtask "Lookdown" or "Lookup" to find the tomato.
** when when given input shows "object-not-in-view" for subtask "navigate to tomato and pick up tomato", but you can still see a tomato based on the given point of view image, then try directly pick up the tomato without navigation.
** for any other unknown reason, you can try assigning subtasks to other agents.

- Object-related: "object-not-found", "object-not-reachable", "object-not-in-inventory", "object(<OBJ>)-not-picked-up":

example:
** when given input shows "object-not-found" and based on given object list of environment, that there's no target object, you can skipped the related subtask.
** when given input shows "object-not-in-inventory", means that the current agent didn't not pick up any object to perform the next step, you should have the subtask "pick up xxx, and do xxx". xxx depends on what is the subtask.

- Ensure necessary object prerequisites.

Example:
input:
{{
  "Task": "Place the mug and the knife into the cabinet",
  "Number of agents": 3,
  "Robots' open subtasks": ["put the knife in the cabinet", "close the cabinet"],
  "Robots' completed subtasks": ["pick up the knife", "pick up the mug", "open the cabinet", "put the mug in the cabinet"],
  "Objects in environment": ["Knife_1", "Mug_2", "Cabinet_1"],
  "Alice's observation": ["Knife_1", "Cabinet_1"],
  "Alice's state": "position: (1, 0.5), facing: north, inventory: [],
  "Alice's subtask_failure_reasons": [],
  "Alice's previous failures": [],
  "Bob's observation": ["Mug_2", "Cabinet_1"],
  "Bob's state": "position: (1, 0.25), facing: north, inventory: ["Mug_2"]",
  "Bob's subtask_failure_reasons": ["Attempted NavigateTo<Cabinet_1> but failed"],
  "Bob's previous failures": ["Alice and Charlie were blocking access to the cabinet"],
  "Charlie's observation": ["Cabinet_1"],
  "Charlie's state": "position: (1, 1), facing: north, inventory: []",
  "Charlie's subtask_failure_reasons": [],
  "Charlie's previous failures": []
}}
image input: (not available in this text-based interface), which shows point of view of Alice, Bob and Charlie.
output:
{{
  "failure reason": "Bob failed to navigate to the cabinet because Alice and Charlie were blocking access to it while Alice was putting in the knife.",
  "memory": "Alice has put the knife in the cabinet, and Bob has put the mug in the cabinet. The cabinet is now open.",
  "reason": "Alice should close the cabinet and move away. Charlie should move to another open space to reduce congestion. Bob should wait until the cabinet is accessible.",
  "suggestion": "next, Alice should move to other place, Bob should stay idle, Charlie should move to other place"
}}

"""

ACTION_EXAMPLES = f"""

Example 1: Given Assignement subtask:
[
"navigate to the vase, pick up the vase, navigate to the table, and put it on the table",
"wait in the current position.",
]
Output:
{{
  "Actions": [
    ["NavigateTo(Vase_1)", "PickupObject(Vase_1)", "NavigateTo(Table_1)", "PutObject(Table_1)"],
    ["Idle"],
  ]
}}

Example 2: Given Assignement:
[
"navigate to the fridge and open the fridge, without picking up any objects, before putting anything inside",
"navigate to the lettece, pick up the lettuce, and wait until the firde is openned",
]
Output:
{{
  "Actions": [
    ["NavigateTo(Fridge_1)", "OpenObject(Fridge_1)"],
    ["Idle"],
    ["NavigateTo(Lettece_1)", "PickupObject(Lettece_1)"]
  ]
}}

Example 3: Given Assignement:
[
'navigate to the laptop, open the laptop, turn on the laptop',
'rotates right to scan the main room for the Television'
]
Output:
{{
  "Actions": [
    ["NavigateTo(Laptop_1)", "ToggleObjectOn(Laptop_1)"],
    ["RotateRight"],
  ]
}}

"""

AI2THOR_ACTIONS = f"""
**Available Actions:**
A robot can perform the following actions:
- move: [MoveAhead, MoveBack, MoveRight, MoveLeft]
- rotate: [RotateRight, RotateLeft]
- look: [LookUp, LookDown]
- idle: [Idle]
- interact_with_object: [NavigateTo<object_name>, PickupObject<object_name>, PutObject<receptacle_name>, OpenObject<object_name>, CloseObject<object_name>, ToggleObjectOn<object_name>, ToggleObjectOff<object_name>, BreakObject<object_name>, CookObject<object_name>, SliceObject<object_name>, DirtyObject<object_name>, CleanObject<object_name>, FillObjectWithLiquid<object_name>, EmptyLiquidFromObject<object_name>, UseUpObject<object_name>]
- interact_without_navigation: [DropHandObject, ThrowObject]
"""

OUTPUT_FORMAT_PLAN = """
**Output Format (JSON):**
Return only a valid JSON object with this structure. 
Do not include explanations, comments, or markdown formatting.
{
    "Subtasks": [
    "subtask 1",
    "subtask 2",
    ...
    ]
}
"""

BASE_INPUT_FORMAT = """You will receive a JSON object with these fields:"""

TASK_INPUT_FORMAT = """
- "Task":  (string) A high-level description of the final goal.
- "Number of agents": (integer) Number of robots.
- "Objects in environment": List of objects currently available in the environment.
"""

SUBTASK_INPUT_FORMAT = """- "Subtasks": List of subtasks provided by another agent.
"""

ASSINMENG_INPUT_FORMAT = """- "Assignment": a list of subtasks, each corresponding to the action that an agent should perform in the upcoming step.
"""

REACABLE_POSITION_INPUT_FORMAT = """- "Reachable positions" (list of (x, z)) reachable positions in the environment.
"""

ROBOTS_SUBTASKS_INPUT_FORMAT = """- "Robots' open subtasks": (list of strings) list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
- "Robots' completed subtasks": (list of strings) list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
"""

AGENT_INPUT_FORMAT = """- "Agent's observation" (list):  list of objects and its postion the agent is observing.
- "Agent's state": Agent's position, facing, and inventory.
"""

MEMORY_HISTORY_INPUT_FORMAT = """- "Robots' memory": string of important information about the scene and action history that should be remembered for future steps,
- "suggestion": a string of reasoning for what each robot should do next and a description of the next actions each robot should take,
"""

FAILURES_INPUT_FORMAT = """- "Agent's subtask_failure_reasons": Previous step performance/failures, if any,
- "Agent's previous failures": Previous action failures, if any,
"""

PLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT
EDITOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + SUBTASK_INPUT_FORMAT
ALLOCATOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + FAILURES_INPUT_FORMAT
ACTION_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT
VERIFIER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT
REPLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT


PLANNER_PROMPT = f"""
You are a capable planner and robot controller overseeing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}, as they complete a specified task.
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

{AI2THOR_ACTIONS}


**Workflow:**
- You will be provided with a high-level description of the task, the number of agents, and the list of objects present in the environment.
- Decompose the main task into subtasks that the robots can complete, taking into account the robots' abilities and the objects available.
- Subtasks that must occur in sequence should be combined into a single subtask.

{EXAMPLE_PLAN}

**Input Format (JSON):**
{PLANNER_INPUT_FORMAT}

{OUTPUT_FORMAT_PLAN}

**Guidelines:**
- Subtasks must be independent unless a specific order is required; sequence-dependent actions must be merged into a single subtask.
- Only use objects that are present in the environment.
- Action plans must reflect real-world object affordances.
- Do not use OpenObject on surfaces such as tables or countertops.
- For openable receptacles (e.g. drawer, fridge, cabinet):
    - The container must be opened before placing any object inside.
    - The robot's hand must be empty before opening or closing any container.
    - The container should be closed after use.
- If placing an object inside an openable container: open it first, then pick up the object, then deposit the object.
- To put an object on a receptacle: pick up the object, navigate to the receptacle, then place the object.
- To cook an object, the object should be placed in a receptacle (e.g. pan, pot) and the receptacle should be on a stove burner and turn on stove knob.
- To slice an object, the robot must hold a knife.
- Robots can hold only one object at a time; ensure actions adhere to this constraint.
- Objects can be cleaned using CleanObject or by placing them under a running faucet. Only clean when explicitly required.
- Avoid unnecessary actions—only perform what is strictly required for the task.
- Do NOT assume or use default receptacles like CounterTop or Table unless they are mentioned in the task description.
- Output only the subtasks JSON as specified, with no extra explanations or formatting.

Proceed systematically to ensure the answer is correct.
"""

EDITOR_PROMPT = f"""
You are an expert task planner and editor responsible for coordinating {len(AGENT_NAMES)} embodied robots ({', '.join(AGENT_NAMES[:-1]) + f', and {AGENT_NAMES[-1]}'}) as they complete a specified task.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.


Your objective is to review and correct the sequence of subtasks to ensure all actions contribute precisely to achieving the given goal.

## Task Correction Rules
1. If instructed to put X in the fridge, verify that X is explicitly placed inside the fridge.
2. For any openable container (e.g., fridge, drawer), ensure it is opened prior to placing items inside and closed afterward instructions should appear as: "open Y, put X inside, and close Y."
3. Never place items on receptacles not identified in the original task.
4. Preserve robot capabilities (e.g., hand constraints, action limits), but restructure or merge subtasks for logical sequencing if required.
5. Do not specified which exact objects, use only the object names. (e.g. apple, not Apple_1)

## Input Format
{EDITOR_INPUT_FORMAT}

{OUTPUT_FORMAT_PLAN}

## Guidelines
- Subtasks should be independent unless a specific sequence is required; sequence-dependent operations should be grouped into one subtask.
- Reference only objects that are actually present in the environment.
- Plans must realistically account for object affordances.
- Do not attempt to open surfaces that aren't openable, such as tables or countertops.
- For openable containers: always open before placing objects, and ensure the robot's hand is empty before opening or closing. Close the container when finished.
- When placing an object inside an openable container: open the container, pick up the object, then deposit it inside.
- To place an object on a receptacle: pick up the object, navigate, then place it on the target.
- For cooking, place the item within a suitable container (e.g., pan, pot), set it on a stove burner, then turn on the stove knob.
- For slicing tasks, ensure the robot is holding a knife before slicing objects.
- Each robot can hold only one object at a time—enforce this constraint. At the very beginning, the robot's hand is always empty.
- Only clean objects when required, using CleanObject or a faucet.
- Actions should be minimal and strictly task-oriented avoid unnecessary steps.
- Never assume or use default receptacles (like countertops or tables) unless explicitly mentioned in the task.
- Output only the subtasks JSON—do not include explanations or extra formatting.


Proceed methodically to guarantee that your output matches the desired task outcome.
"""

ALLOCATOR_PROMPT = f"""
You are an expert task allocator, responsible for assigning tasks to a team of {len(AGENT_NAMES)} embodied robots, with name {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}. to efficiently achieve a specified goal.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

## Allocation Rules
- Assign at most one executable subtask to each available agent, maximizing overall task progress.
- A subtask is only executable if all its prerequisite subtasks are present in the list of completed subtasks. If a subtask is blocked by missing prerequisites, do not assign it.
- If no executable subtasks are available for an agent, assign "Idle" to that agent.
- Do not assign the same subtask to multiple agents, unless there are multiple same subtasks in the given list.
- If there are fewer open subtasks than available agents, assign "Idle" to any surplus agents.
- Do not assign a subtask if the agent's current inventory makes it infeasible.
- If there are no open subtasks (if open subtasks is None or empty), assign "Idle" to all agents and leave the unassigned subtasks list empty.
- After allocation, return a list of unassigned subtasks: any remaining open subtasks that are either blocked or unallocated but executable.

## Input Format
{ALLOCATOR_INPUT_FORMAT}

## Output Format
Produce a JSON object with this structure:

{{
  "Allocation": {{
    "agent1": "subtask or Idle",
    "agent2": "subtask or Idle",
    ...
    "agentN": "subtask or Idle"
  }},
  "Remain": ["unassigned_subtask1", "unassigned_subtask2", ...]
}}

## Guidelines
- Use sequential keys: "agent1", "agent2", ..., matching the agent order in input.
- If no open subtasks are present/valid, set all agents to "Idle" and "Remain": [].
- Keep all subtask values unaltered from input.
- Output only the "Allocation" and "Remain" JSON - do not include any explanatory text or extra fields.


** Think step-by-step internally, but do not show any intermediate thoughts. Output only the JSON.
** Respond with nothing but the exact JSON object. Any deviation from the format is considered incorrect.
"""

ACTION_PROMPT = f"""
# Role and Objective
- Manage {len(AGENT_NAMES)} embodied robots by generating executable action plans to fulfill assigned subtasks. Convert each Assignment subtask into an explicit, sequential action list for the agent, based on current state and environmental details.

# Checklist
- Begin each output with a concise checklist (3-7 bullets) summarizing the planned process before generating the output (e.g., extract subtasks, check object presence, plan concrete actions, verify conditions, format result).

# Planning and Execution
- For each "Assignment" subtask provided (one per agent), generate an ordered list of executable atomic actions using only the defined action set and schemas.
- Treat each subtask independently, without inferring new subtasks, changing intent, or considering inter-agent dependencies unless explicitly described in the input.
- Respect the affordances of objects, preconditions of actions, and environmental constraints in translating subtasks into actions.
- Assign ["Idle"] only if the subtask is "Idle".
- Generate a list of action plans (lists), one per agent, in the same order as the input subtasks.

{AI2THOR_ACTIONS}

# Context
- Robot names: {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}
- Inputs include: task description, agent states/observations, open and completed subtasks, reachable positions, all objects in the environment, failed diagnostics (if any), and subtasks to be executed.

# Input Format
{ACTION_INPUT_FORMAT}

# Planning and Validation
- Internally, reason step by step to extract and analyze each Assignment, confirm presence of referenced objects, consider the distance and position of agent and object, map actions using current environment and agent state, and validate required action conditions for each agent.
- After generating action plans, validate that each generated plan satisfies the requirements (object existence, feasible actions based on agent state/inventory, and completeness of required fields). If validation fails for a subtask, output as specified in Output Format.

# Guidelines
- If a referenced object is missing, inventory is full (when pickup is needed), agent state/observation is missing, or Subtask is None: output an empty list for that subtask (or {{ "Actions": [] }} if Subtasks is None).
- Assign only objects listed in the provided list. If multiple instances of the same type exist (e.g., Countertop_1, Countertop_2), select the most appropriate one based on the agent's current position and its proximity to the objects.
- when assigning actions which interact with objects and with navigation, always use NavigateTo<object_name> to approach the object first. Unless the targert obect is close enough and in the view of the agent.
- If the subtask requires micro-movements to approach an Object_name, use only Movement, Rotation, and Look actions based on position and failure reason—avoid using NavigateTo<Object_name> initially. Try one atomic movement at a time before attempting NavigateTo<Object_name> again.
    - When failure reason is "object-not-in-view", first try  Lookdown  or Lookup action based on the target object's most likely to be. (you can assume the agent always starts looking front. 
    - When failure reason is "no-plan", try moving around first—sometimes an obstacle (e.g., a door) may be blocking the path before re-attempting NavigateTo.


# Output Format
The output must be a single JSON object, with no extra explanation:
- Key: "Actions"
- Value: a list of lists. Each inner list contains the atomic actions for a subtask, matching the order of input subtasks.

**Example:**
{ACTION_EXAMPLES}

** The output must be a valid JSON object with no extra explanation.
** Do not include failure codes, error objects, or explanations. Encode error states strictly as empty lists at the corresponding subtask index.
** Output only the specified JSON object. Do not provide explanations or extra responses.


"""

REPLAN_PROMPT = f"""
You are a capable planner and multi-robot controller assigned to help {len(AGENT_NAMES)} embodied robots named {', '.join(AGENT_NAMES[:-1]) + f', and {AGENT_NAMES[-1]}'} accomplish a specific goal.
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

{AI2THOR_ACTIONS}


## TASK

Your goal is to replan a valid and efficient sequence of subtasks based on the current environment state and prior action history. You will be given:
- The original task description
- Robots' previous plans (open and completed subtasks)
- Their current observations and states
- Objects in environment
- Failure causes
- Reasoning and suggestions for next actions



You should reason step-by-step over the environment, agent memory, failure causes, and suggestions to derive a concise and updated subtask plan.
After reasoning, return only the updated `"Subtasks"` list in JSON format. Do **not** return any extra explanations or comments.


### INPUT FORMAT
{REPLANNER_INPUT_FORMAT}


{OUTPUT_FORMAT_PLAN}


{EXAMPLE_PLAN}



## Key Constraints and Guidelines
- Subtasks must be atomic and goal-directed; combine sequentially dependent actions into one subtask.
- Avoid repeating failed subtasks unless the situation has changed.
- Minimize and streamline plan; do only necessary actions.
- Robots are coordinated; explicit role allocation is not required.
- Only use objects present in the current environment list.
- Reflect real world affordances and limitations.
- Never OpenObject on non-openable surfaces (e.g. tables, countertops).
- For openable containers (e.g. drawers, fridge, cabinets):
    - Open first; hand must be empty; close after use.
    - To place inside: open → pick up object → deposit object.
- To put object on a receptacle: pick up → navigate → place.
- To cook: object must be in a receptacle (e.g. pan/pot), on stove, stove turned on.
- To slice: robot must hold a knife. Do not pick up the object to be sliced.
- Robots can carry one object at a time—enforce this strictly.
- Clean objects only when explicitly specified, using correct affordances.
- Do not assume or use default receptacles unless specified in the task.
- Output **only** the required subtasks JSON. No extra explanations, formatting, or commentary.

**Think step by step internally; return only the specified JSON.**

"""

# identify the failure reason for the last action and suggest what to do next.
VERIFIER_PROMPT = f"""
You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.


{AI2THOR_ACTIONS}

You need to verfify the previous failure and suggest the **next single action** that each robot should take in the current timestep.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input.
To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
- A task description
- A list of objects each robot can currently see (formatted as "<object_name>_<object_no>") with positions
- Observations, failure descriptions, and subtask progress

### INPUT FORMAT ###
{VERIFIER_INPUT_FORMAT}


### OUTPUT FORMAT ###
First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
* Failure reason: If any robot's previous action failed, use the previous history and your understanding of causality to think and rationalize about why it failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object and must explore to find, agent is doing the same redundant actions, etc.
* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
* Reason: The reasoning for what each robot is supposed to do next
* suggestion: The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that this suggested actions make these robots more efficient in completing the task as compared only one agent solving the task.
Your output should just be in the form of a json  as shown below.
Examples of output:
You must output a json dictionary with:
- "failure reason": explanation of why any previous action failed (or "None" if no failure)
- "memory": important information about the scene that should be remembered for future steps
- "reason": reasoning for what each robot should do next
- "suggestion": a description of the next actions each robot should take, formatted as "next, <robot_name and id> should ..., <robot_name and id> should ..."


{VERIFY_EXAMPLE}



### Important Notes ###
- Each robot can only hold **one object** at a time.
- If a robot opens an object (e.g., drawer), it should close it before leaving.
- Open objects (e.g., cabinets) can block paths.
- Avoid redundant or unnecessary actions.
- Plan actions to avoid robots blocking or colliding with each other.
- Reflect real world affordances and limitations.
- Never OpenObject on non-openable surfaces (e.g. tables, countertops).
- For openable containers (e.g. drawers, fridge, cabinets):
    - Open first; hand must be empty; close after use.
    - To place inside: open → pick up object → deposit object.
- To put object on a receptacle: pick up → navigate → place.
- To cook: object must be in a receptacle (e.g. pan/pot), on stove, stove turned on.
- To slice: robot must hold a knife. Do not pick up the object to be sliced.
- Don't use remote control to turn on or off the television.
- Robots can carry one object at a time—enforce this strictly.
- Clean objects only when explicitly specified, using correct affordances.
- DO NOT output anything beyond the specified dictionary format.

Let's work this out step by step to be sure we have the right answer.
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
    # print("using model:", model)
    # print("payload:", payload)
    if model.startswith("gpt-4"):
        # for models: gpt-4.1, gpt-4.1-2025-04-14, gpt-4o,
        response = client.chat.completions.create(model=model, 
                                                    messages=payload, 
                                                    max_tokens=max_tokens, 
                                                    temperature=temperature,)
    else:
        # for models: gpt-5-2025-08-07
        # max_tokens is replaced by max_completion_tokens; 
        # 'temperature' does not support 0.7 with this model. Only the default (1) value is supported."
        response = client.chat.completions.create(model=model, 
                                                    messages=payload, 
                                                    max_completion_tokens=max_tokens,)
    return response, response.choices[0].message.content.strip()

def prepare_payload(system_prompt, user_prompt, img_urls=None) -> list:
    # print("system_prompt:", system_prompt)
    # print("user_prompt:", user_prompt)
    if img_urls:
        # with image input
        payload = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}] + img_urls,
                }
            ]
    else:
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

def prepare_prompt(env: AI2ThorEnv, mode: str = "init", addendum: str = "", subtasks=[], info={}, verify_info={}) -> str:
    """
    mode: str, choose from planner, action
    planner: for decomposing the task into subtasks
    action: for generating the actions for each robot to perform
    addendum: additional information to be added to the user prompt
    """

    if mode == "planner":
        # for initial planning
        system_prompt = PLANNER_PROMPT
        input = env.get_center_llm_input()
        user_prompt = convert_dict_to_string(input)
    elif mode == "editor":
        # for editing the generated plan
        system_prompt = EDITOR_PROMPT
        input = env.get_center_llm_input()
        input["Subtasks"] = subtasks
        user_prompt = convert_dict_to_string(input)
    elif mode == "action":
        # for generating automic actions for each robot to perform
        system_prompt = ACTION_PROMPT
        if not subtasks:
            print("No subtasks provided")
            return None, None
        input = env.get_obs_llm_input(prev_info=info)
        input["Subtasks"] = subtasks
        del input["Robots' open subtasks"]
        del input["Robots' completed subtasks"]
        user_prompt = convert_dict_to_string(input)

    elif mode == "allocator":
        # for allocating subtasks to robots
        system_prompt = ALLOCATOR_PROMPT
        input = env.get_obs_llm_input(prev_info=info)
        user_prompt = convert_dict_to_string(input)

    elif mode == "replan":
        # for replanning the subtasks based on the current state of the environment
        system_prompt = REPLAN_PROMPT
        try:
            input = env.get_obs_llm_input(prev_info=info)
            del input['Reachable positions']
        except KeyError as e:
            print(f"[Error] Missing key in info for replan: {e}")
            return None, None
        
        user_prompt = convert_dict_to_string(input)
        # print("replan prompt:", system_prompt)
        # print("replan use prompt:", user_prompt)
    elif mode == "verifier":
        # for verifying the actions taken by the robots
        system_prompt = VERIFIER_PROMPT
        input = env.get_obs_llm_input(prev_info=info)
        del input['Reachable positions']

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
            elif re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL):
                match = re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL)
                res_content = match.group(1)
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
            elif re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL):
                match = re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL)
                res_content = match.group(1)
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
    }
    Output:
    allocation: [subtask, subtask]
    remain: []
    """
    try:
        remain = json.loads(res_content)["Remain"]
        allocations = json.loads(res_content)["Allocation"]
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
            elif re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL):
                match = re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL)
                res_content = match.group(1)
            if res_content.endswith("```"):
                res_content = res_content[:-3].strip()
            res_content = re.sub(r",\s*([\]}])", r"\1", res_content)
            res_content = res_content.replace("'", '"')
            res_content = res_content.strip()

            data = json.loads(res_content)
            remain = data["Remain"]
            allocations = data["Allocation"]
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
    
def process_verifier_llm_output(res_content):
    '''
    Input:
    {
    "failure reason": "None",
    "memory": "Alice is holding ButterKnife_1 and is near both Apple_1 and Bread_1, which are both on the counter in front of her. Bob is far from the action, not holding anything, and can assist with moving and toasting the bread after it is sliced. Toaster_1 is nearby on the left counter.",
    "reason": "Alice should slice the apple first since she is already holding the ButterKnife and is close to the apple. Bob should begin moving toward the bread to be ready to pick up a bread slice after Alice slices it, enabling efficient hand-off for toasting.",
    "suggestion": "next, Alice should SliceObject<Apple_1>, Bob should NavigateTo<Bread_1>"
    }
    output:
    failure_reason: str, memory: str, reason: str, suggestion: str
    '''
    try:
        failure_reason = json.loads(res_content)["failure reason"]
        memory = json.loads(res_content)["memory"]
        reason = json.loads(res_content)["reason"]
        suggestion = json.loads(res_content)["suggestion"]
        return failure_reason, memory, reason, suggestion
        
    except json.JSONDecodeError as e:
        print(f"[Warning] Initial JSON decode failed: {e}")
        try:
            if res_content.startswith("```json"):
                res_content = res_content.removeprefix("```json").strip()
            elif res_content.startswith("```"):
                res_content = res_content.removeprefix("```").strip()
            elif re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL):
                match = re.search(r"```json\s*(\{.*?\})\s*```", res_content, re.DOTALL)
                res_content = match.group(1)
            if res_content.endswith("```"):
                res_content = res_content[:-3].strip()
            res_content = re.sub(r",\s*([\]}])", r"\1", res_content)
            res_content = res_content.replace("'", '"')
            res_content = res_content.strip()

            data = json.loads(res_content)
            failure_reason = data["failure_reason"]
            memory = data["memory"]
            reason = data["reason"]
            suggestion = data["suggestion"]
            return failure_reason, memory, reason, suggestion
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
    # ---1. Planner LLM 產生初步 subtasks
    planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")
    planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    res, res_content = get_llm_response(planner_payload, model=config['model'])
    print('init plan llm output', res_content)
    subtasks = process_planner_llm_output(res_content)
    print(f"After Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    # # ---2. Editor LLM 修正 subtasks
    # editor_prompt, editor_user_prompt = prepare_prompt(env, mode="editor", subtasks=subtasks)
    # editor_payload = prepare_payload(editor_prompt, editor_user_prompt)
    # res, res_content = get_llm_response(editor_payload, model=config['model'])
    # print('editor llm output', res_content)
    # subtasks = process_planner_llm_output(res_content)
    # print(f"After Editor LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    # for testing
    # subtasks = ['open the fridge', 'pick up the apple and put it in the fridge', 'pick up the lettuce and put it in the fridge', 'pick up the tomato and put it in the fridge', 'close the fridge']
    # subtasks = ['pick up knife', 'slice apple', 'put down knife',  'slice bread', 'pick up one bread slice', 'insert bread slice into toaster', 'activate toaster']
    return subtasks, []

def allocate_subtasks_to_agents(env, info={}):
    """分配 open_subtasks 給各 agent"""
    allocator_prompt, allocator_user_prompt = prepare_prompt(env, mode="allocator", info=info)
    allocator_payload = prepare_payload(allocator_prompt, allocator_user_prompt)
    # print("allocator system prompt: ", allocator_prompt)
    # print("allocator user prompt: ", allocator_user_prompt)
    res, res_content = get_llm_response(allocator_payload, model=config['model'])
    # print('llm allocator output', res_content)
    allocation, remain = process_allocator_llm_output(res_content)
    # print('allocation: ', allocation)
    # for testing
    # allocation =  ['pick up the tomato and put it on the countertop', 'pick up the lettuce and put it on the countertop']
    # remain =  ['pick up the bread and put it on the countertop']
    
    return allocation, remain

def decompose_subtask_to_actions(env, subtasks, info={}):
    """將 subtask 拆解成 atomic actions（LLM）"""
    action_prompt, action_user_prompt = prepare_prompt(env, mode="action", subtasks=subtasks, info=info)
    action_payload = prepare_payload(action_prompt, action_user_prompt)
    # print("action prompt:", action_prompt)
    # print("action user prompt:", action_user_prompt)
    res, res_content = get_llm_response(action_payload, model=config['model'])
    print('action llm output', res_content)
    actions = process_actions_llm_output(res_content)

    # For testing 
    # actions = [['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)'], ['NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)']]
    return actions

def encode_image(image_path: str):
    # if not os.path.exists()
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def verify_actions(env, info):
    verify_prompt, verify_user_prompt = prepare_prompt(env, mode="verifier", info=info)
    base64_image = [encode_image(env.get_frame(i)) for i in range(len(AGENT_NAMES))]

    image_urls = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        for image in base64_image
    ]
    verify_payload = prepare_payload(verify_prompt, verify_user_prompt, img_urls=image_urls)
    print("verify prompt: ", verify_user_prompt)
    res, res_content = get_llm_response(verify_payload, model=config['model'])
    print('verify llm output', res_content)
    failure_reason, memory, reason, suggestion = process_verifier_llm_output(res_content)
    verify_res = {
        "failure reason": failure_reason,
        "memory": memory,
        "reason": reason,
        "suggestion": suggestion
    }
    return verify_res

def get_steps_by_actions(env, actions):
    print("get_steps_by_actions: ", actions)
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
    print('verifying open_subtasks')
    print(open_subtasks)
    for c in completed_subtasks:
        if c != 'Idle':
            print(c)
            open_subtasks.remove(c)
            closed_subtasks.append(c)
    return open_subtasks, closed_subtasks

def replan_open_subtasks(env, info, completed_subtasks, verify_info):
    replan_prompt, replan_user_prompt = prepare_prompt(env, mode="replan", info=info, verify_info=verify_info)
    # print("replan system prompt: ", replan_prompt)
    # print("replan user prompt: ", replan_user_prompt)
    replan_payload = prepare_payload(replan_prompt, replan_user_prompt)
    res, res_content = get_llm_response(replan_payload, model=config['model'])
    # print('replan llm output', res_content)
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
    info = {}
    # --- loop start
    cnt = 0
    start_time = time.time()
    logs = []
    filename = env.base_path / "logs_llm.txt"
    while open_subtasks and (time.time() - start_time < timeout):
        print(f"\n--- Loop {cnt + 1} ---")
        logs.append(f"\n--- Loop {cnt + 1} ---")

        env.update_plan(open_subtasks, completed_subtasks)
        logs.append(f"----")
        logs.append(f"open_subtasks: {env.open_subtasks}")
        logs.append(f'completed_subtasks: {env.closed_subtasks}')
        # print("open_subtasks: ", env.open_subtasks)
        # print("closed_subtasks: ", env.closed_subtasks)

        # 2. allocate subtasks to each agent
        logs.append(f"----allocating subtasks to agents----")
        agent_assignments, remain = allocate_subtasks_to_agents(env)
        print("agent_assignments: ", agent_assignments)
        logs.append(f"agent_assignments: {agent_assignments}")
        # print("remain unassigned subtasks: ", remain)
        
        # # 3. decompose subtask to smaller actions
        logs.append(f"----decomposing subtasks to agents----")
        if info:
            actions = decompose_subtask_to_actions(env, agent_assignments, info)
        else:
            actions = decompose_subtask_to_actions(env, agent_assignments)
        # print("actions: ", actions)
        logs.append(f"actions: {actions}")
        decomp_actions = get_steps_by_actions(env, actions)
        
        # # 4. execution
        # print("decomp_actions: ", decomp_actions)
        logs.append(f"decomp_actions: {decomp_actions}")
        cur_plan = bundle_task_plan(agent_assignments, actions, decomp_actions)
        # print("cur_plan: ", cur_plan)
        logs.append(f"cur_plan: {cur_plan}")
        logs.append(f"----executing subtasks to agents----")
        isSuccess, info = env.stepwise_action_loop(cur_plan)
        # print('info', info)
        logs.append(f"info: {info}")
  
        logs.append(f"----verifying subtasks to agents----")
        # # 5. verify which subtasks are done 
        open_subtasks, completed_subtasks = verify_subtask_completion(env, info)
        print("after verify open_subtasks: ", open_subtasks)
        print("after verify closed_subtasks: ", completed_subtasks)
        logs.append(f"after verify open_subtasks: {open_subtasks}")
        logs.append(f"after verify closed_subtasks: {completed_subtasks}")
        env.update_plan(open_subtasks, completed_subtasks)
        # 6. replan if needed
        if open_subtasks or not isSuccess:
            logs.append(f"----replanning subtasks to agents----")
            verify_res = verify_actions(env, info)
            # print("verify result: ", verify_res)
            logs.append(f"verify result: {verify_res}")
            env.update_memory(verify_res['memory'], suggestion=verify_res['reason'] + " Suggestion to do for next step: " + verify_res['suggestion'])
            open_subtasks, completed_subtasks = replan_open_subtasks(env, info, completed_subtasks, verify_res)
            # print("replan open_subtasks: ", open_subtasks)
            # print("replan closed_subtasks: ", completed_subtasks)
            logs.append(f"replan open_subtasks: {open_subtasks}")
            logs.append(f"replan closed_subtasks: {completed_subtasks}")
            
            start_time = time.time()
            get_object_dict = env.get_object_dict()
            logs.append(f"current Object dictionary: {get_object_dict}")
            # print("current Object dictionary:", get_object_dict)
            for i in range(len(env.agent_names)):
                state = env.get_agent_state(i)
                view = env.get_object_in_view(i)
                mapping = env.get_mapping_object_pos_in_view(i)

                # print(f"Agent {i} ({env.agent_names[i]}) observation: I see:", mapping) # list of object ids in view
                # print(f"Agent {i} ({env.agent_names[i]}) state: {state}") # I am at coordinates: (2.00, -1.50), facing west, holding nothing
                # print(f"Agent {i} ({env.agent_names[i]}) can see object:", view) # list of object ids in view
                logs.append(f"Agent {i} ({env.agent_names[i]}) observation: I see: {mapping}")
                logs.append(f"Agent {i} ({env.agent_names[i]}) state: {state}")
                logs.append(f"Agent {i} ({env.agent_names[i]}) can see object: {view}")

            # break
            with open(filename, "a", encoding="utf-8") as f:
                for log in logs:
                    f.write(str(log) + "\n")
            logs = []
       
        cnt += 1 
    env.save_log()
    
    
    env.close()

if __name__ == "__main__":
    run_main()

