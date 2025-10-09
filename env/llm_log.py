'''
Baseline : Centralized LLM + replanning + shared memory(log based)


structure same as llm_c.py but with more information about the environment and positions, failures, etc.:
1. initial planning (remain the same as previous method: given task, let planner and editor to generate a list of subtasks (this will be the open subtasks)
2. start a loop, until timeout or all the open subtasks is empty:
2.1 update open subtasks and completed subtask
2.2 allocate subtask to robot agents in the environment with llm
2.3 break down each assigned subtasks with llm into a list of smaller available actions
2.4 execute one subtask per agents
2.5 verify if the subtask is completed and identify the failure reason and collect the history and suggest the next step
2.6 replan: similar to initial planning : given task and closed subtask

log example in event.jsonl:
{"timestemp": 0, "agent_id": 0, "agent_name": "Alice", "curr_subtask": "NavigateTo(Lettuce_1)", "type": "Failed", "payload": {"last_action": "Idle", "failed_reason": "object(Lettuce_1)-not-exist (Object Lettuce_1 not found in object_dict)", "postion": "(-4.00, 1.50)", "rotation": "north", "inventory": "nothing", "observation": "I see: ['ArmChair_1', 'ArmChair_2', 'Bowl_1', 'Box_1', 'CoffeeTable_1', 'Curtains_1', 'Floor_1', 'Pillow_1', 'Shelf_2', 'Sofa_1', 'TVStand_1', 'Television_1', 'TissueBox_1', 'Window_2']"}}
{"timestemp": 17, "agent_id": 1, "agent_name": "Bob", "curr_subtask": "ToggleObjectOff(LightSwitch_1)", "type": "Success", "payload": {"last_action": "ToggleObjectOff(LightSwitch_1)", "postion": "(-3.50, 0.75)", "rotation": "south", "inventory": "nothing", "observation": "I see: ['HousePlant_1', 'LightSwitch_1', 'Painting_1', 'RemoteControl_1', 'SideTable_2', 'SideTable_3']"}}
{"timestemp": 17, "agent_id": 0, "agent_name": "Alice", "curr_subtask": "NavigateTo(FloorLamp_1)", "type": "Attempt", "payload": {"last_action": "MoveAhead", "postion": "(-1.25, 2.50)", "rotation": "east", "inventory": "nothing", "observation": "I see: ['Curtains_1', 'DeskLamp_1', 'KeyChain_1', 'SideTable_1', 'Vase_1', 'Window_1']"}}


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
from env_log import AI2ThorEnv_cen as AI2ThorEnv

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
# Error and Faulures handling
- Navigation: "no-path", "object-not-in-view", "distance-too-far": Use micro-movements (MoveAhead, RotateRight, etc.) to get in view/reach, using current/target positions and available observations. For example, you can have subtasks like "look down and navigate to potato" if previous failure reason of "pick up potato" was "object-not-in-view". Or you can have subtask like "use micro-movements to navigate to potato" if previous subtask "navigate to potato" was failured.
example: 
** If the input reports "no-path" to <Obj> while <Obj> exists and the images/observations indicate a physical occluder (e.g., an open fridge door between the agent and the target or blocked by other agent), issue a short detour macro before retrying. For example: RotateRight twice, then MoveAhead once, then retry NavigateTo(<Obj>).This decision must use multiple signals—images/2D detections, object states (e.g., isOpen for doors), recent actions, and visible objects—not just the Reachable positions list. Or navigate to somewhere far away to clean the path.
** When the input shows "no-path" and a likely cause is another agent blocking the aisle or target approach (same narrow corridor, same target, or the other agent is on the planned route), assign a yield/wait behavior to avoid deadlock: have the blocked agent Idle for 1 or 2 steps or take a small lateral/back step (MoveRight/MoveLeft/MoveBack) to clear space, or temporarily reassign the blocking agent to a different subtask/movement. After the yielding action, retry NavigateTo(<Obj>).
** when given input shows "object-not-in-view" for subtask "navigate to tomato", and based on other input information, that the distance between the agent and the tomato is already close enough, then you can have subtask "Lookdown" or "Lookup" to find the tomato or "RotateRight" or "RotateLetf".
** when when given input shows "object-not-in-view" for subtask "navigate to tomato and pick up tomato", but you can still see a tomato based on the given point of view image, then try directly pick up the tomato without navigation.
** for any other unknown reason or repeating failures, you can suggest assigning subtasks to other agents.

- Object-related: "object-not-found", "object-not-reachable", "object-not-in-inventory", "object(<OBJ>)-not-picked-up", "object cannot be interacted":
example:
** when given input shows "object-not-found" and based on given object list of environment, that there's no target object, you can skipped the related subtask.
** when given input shows "object-not-in-inventory", means that the current agent didn't not pick up any object to perform the next step, you should have the subtask "pick up xxx, and do xxx". xxx depends on what is the subtask.
** when given input shows "object cannot be interacted", means that the target object maybe  broken/disabled/locked, you should skip the related subtask, or try replan/choose an alternative..
** when given error shows "NullReferenceException: Target object not found within the specified visibility.", means that the target object may be inside the container and is not visiable to the agent, you should try open the container to find the target object.

- Ensure necessary object prerequisites.

# Example:
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
'rotates right and move forward to clear the path for other'
]
Output:
{{
  "Actions": [
    ["NavigateTo(Laptop_1)", "ToggleObjectOn(Laptop_1)"],
    ["RotateRight", "MoveAhead"],
  ]
}}

"""

AI2THOR_ACTIONS = f"""
# Available Actions:
A robot can perform the following actions:
- move: [MoveAhead, MoveBack, MoveRight, MoveLeft]
- rotate: [RotateRight, RotateLeft]
- look: [LookUp, LookDown]
- idle: [Idle]
- interact_with_object: [NavigateTo<object_name>, PickupObject<object_name>, PutObject<receptacle_name>, OpenObject<object_name>, CloseObject<object_name>, ToggleObjectOn<object_name>, ToggleObjectOff<object_name>, BreakObject<object_name>, CookObject<object_name>, SliceObject<object_name>, DirtyObject<object_name>, CleanObject<object_name>, FillObjectWithLiquid<object_name>, EmptyLiquidFromObject<object_name>, UseUpObject<object_name>]
- interact_without_navigation: [DropHandObject, ThrowObject]
"""

OUTPUT_FORMAT_PLAN = """
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

COMMON_GUIDELINES = """**Simulation note:** Agents operate in a simulator that mirrors real-world physics and affordances, but with explicit constraints enumerated below.
- Use only objects listed in the current environment and reflect real-world affordances.
- Do not search inside containers unless the task explicitly requires it.
- Never use OpenObject on non-openable surfaces (e.g., tables, countertops).
- For openable containers (e.g., fridge, drawer, cabinet):
  - Open before use; the robot's hand must be empty before opening or closing.
  - Close after use.
  - To place an object inside: open → pick up the object → deposit it inside → close.
- To place on a receptacle: pick up → navigate/approach → place.
- Cooking: item in a pan/pot → pan/pot on a stove burner → turn on the stove knob (turn off if the task requires).
- Slicing: the robot must be holding a knife, and do **not** pick up the item being sliced.
- After Slicing: When an object is sliced, it becomes multiple pieces that keep the same base name but receive new indices/unique IDs. E.g., slicing Apple_1 yields Apple_1, Apple_2, …
- Toasting (bread): first slice the bread using a butterknife or knife (do not pick up the bread before slicing); then pick up one slice(it should be Bread_1 ~ Bread_9 not BreadSliced), navigate to the toaster, put the slice into the toaster, and turn on the toaster.
- Each robot can hold only one object at a time. When holding an item, the robot **cannot** perform interactions that require a free hand (e.g., OpenObject, CloseObject, ToggleObjectOn/Off); empty the hand first (put on a surface or drop) before such actions.
- Clean only when explicitly required, using CleanObject or put it under a running faucet. Do not use a sponge or soap unless specified.
- Avoid unnecessary/redundant actions; minimize steps.
- Do not assume default receptacles (e.g., CounterTop, Table) unless explicitly mentioned/present.
- Close any opened object before leaving when appropriate.
- Avoid agents blocking each other where possible.
- Object cannot be given to other agent.
- Unless otherwise specified, assume robots start with empty hands.
- For electronic items (e.g., television, laptop, phone), toggle power directly on the device; **do not use remote controls** unless explicitly required.
- **Irreversible actions (non-repeatable per object):** `BreakObject(<object_name>)`, `CookObject(<object_name>)`, and `SliceObject(<object_name>)` are irreversible; each can be performed **at most once** on the same object. If the object is already broken/cooked/sliced, **skip** the related subtask and, if applicable, replan with an alternative instance.
"""

BASE_INPUT_FORMAT = """You will receive a JSON object with these fields:"""

TASK_INPUT_FORMAT = """
- "Task":  (string) A high-level description of the final goal.
- "Number of agents": (integer) Number of robots.
- "Objects in environment": List of objects currently available in the environment.
- "Objects in containers": A dictionary where each key is a container object (e.g., Fridge, Drawer), and its value is a list of objects currently inside that container.
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

AGENT_INPUT_FORMAT = """- "Agent's observation" (list):  list of objects and its postion the agent is currently observing.
- "Agent's state": Current agent's position, facing, and inventory.
"""

LOG_PROCESSED_INPUT_FORMAT = """- "Logs": The following is a list of execution log lines, each describing one agent's subtask execution result at a specific timestamp.
Each line follows this format: 
[t=timestamp] agent_name → curr_subtask → result_type (reason)
"""

LOF_ORIGINAL_INPUT_FORMAT = """
- "Logs": a list of JSON objects. Each object represents one action execution log for a specific agent at a certain timestamp.
JSON field description:
- "timestemp" (number): The timestamp of the action.
- "agent_id" (integer): The ID of the agent (e.g., 0 or 1).
- "agent_name" (string): The name of the agent.
- "curr_subtask" (string): The subtask the agent is currently executing.
- "type" (string): The result type of the action. Possible values: "Attempt", "Success", "Failed".
- "payload" (object): Detailed information about the agent's status during this action.
    - "last_action" (string): The last atomic action executed.
    - "failed_reason" (string, optional): If the type is "Failed", this field contains the error message.
    - "postion" (string): The agent's position in the environment.
    - "rotation" (string): The agent's facing direction.
    - "inventory" (string): The items the agent is currently holding.
    - "observation" (string): What the agent currently perceives in the environment.
"""

SUGGESTION_INPUT_FORMAT = """- "Reason": A detailed analysis of the failure, considering the task requirements, agent states, and environment conditions.
- "Suggestion": Clear, actionable recommendations for the next steps each agent should take to overcome the failure and progress towards completing the task.
"""


PREVIOUS_LOG_FORMAT = """- "Previous Actions: agent's subtask execution result for last action, in the following format [t=timestamp] agent_name → curr_subtask → result_type (reason)

"""

PLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT
EDITOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + SUBTASK_INPUT_FORMAT
ALLOCATOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + PREVIOUS_LOG_FORMAT
ACTION_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + AGENT_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT

VERIFIER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT
REPLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + SUGGESTION_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT

VERIFIER_ORG_LOG_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOF_ORIGINAL_INPUT_FORMAT
REPLANNER_ORG_LOG_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + SUGGESTION_INPUT_FORMAT + LOF_ORIGINAL_INPUT_FORMAT

PLANNER_PROMPT = f"""
# Role and Objective
You are a capable planner and robot controller overseeing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}, as they complete a specified task.
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

{AI2THOR_ACTIONS}

# Instructions
- You will be provided with a high-level description of the task, the number of agents, and the list of objects present in the environment.
- Decompose the main task into subtasks that the robots can complete, taking into account the robots' abilities and the objects available.
- Make sure to cover all the subtasks to complete the final goal.
- Merge steps that must occur in a strict sequence into a **single** subtask.
{COMMON_GUIDELINES}
- Output only the subtasks JSON as specified, with no extra explanations or formatting.

# Reasoning Steps
- Internally derive a concise checklist besed on the instruction and the given context.
- Verify objects exist in the provided environment list.
- Ensure each subtask is atomic yet sequence-consistent when required.

# Output Format
{OUTPUT_FORMAT_PLAN}

# Examples
{EXAMPLE_PLAN}

# Input Context 
{PLANNER_INPUT_FORMAT}

# Final instructions
First, think carefully step by step about the **subtask decomposition**, closely adhering to the **Instructions and Available Actions**. Then, **output only the required Subtasks JSON with no extra text**.
"""

EDITOR_PROMPT = f"""
# Role and Objective
You are an expert task planner and editor responsible for coordinating {len(AGENT_NAMES)} embodied robots ({', '.join(AGENT_NAMES[:-1]) + f', and {AGENT_NAMES[-1]}'}) as they complete a specified task.

# Instructions
- Review and correct the sequence of subtasks to ensure all actions contribute precisely to achieving the given goal.
{COMMON_GUIDELINES}

## Sub-categories
- Task Correction Rules:
  1) If asked to put X in fridge, ensure X is **inside** fridge.
  2) For openables: **open Y → put X → close Y**.
  3) Never place on receptacles not in the task.
  4) Preserve constraints; merge/reorder when logical.
  5) Use generic names (e.g., apple), **not** instance IDs (Apple_1).
- Minimality: only essential steps; hands start empty.

{AI2THOR_ACTIONS}

# Reasoning Steps
- Internally validate order/dependencies; guard affordances and constraints.

# Output Format
{OUTPUT_FORMAT_PLAN}


# Examples
{EXAMPLE_PLAN}

# Input Context 
{EDITOR_INPUT_FORMAT}

# Final instructions
First, think carefully step by step about the **plan correction** per the rules, closely adhering to the **Common Guidelines**. Then, **output only the corrected Subtasks JSON with no extra text**.
"""

ALLOCATOR_PROMPT = f"""
# Role and Objective
You are an expert task allocator, responsible for assigning tasks to a team of {len(AGENT_NAMES)} embodied robots, with name {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}. to efficiently achieve a specified goal.


# Instructions
- Assign at most one executable subtask to each available agent to maximize overall progress.
- A subtask is executable only if all its prerequisites are in the completed list; if blocked by missing prerequisites, do not assign it.
- Do not assign a subtask if the agent's current inventory/state makes it infeasible.
- Do not assign the same subtask to multiple agents, unless the list contains multiple identical subtasks.
- If an agent has no executable subtask (e.g., when open subtasks are fewer than agents), assign “Idle” to that agent.
- If no open/valid subtasks exist, assign “Idle” to all agents and return "Remain": [].
- After allocation, set "Remain" to any open subtasks that are blocked or unallocated but executable.
- Use sequential keys “agent1”, “agent2”, … in input order, and keep subtask strings unaltered.
- Do not change any word of each subtask.
- Output only the "Allocation" and "Remain" JSON—no extra text or fields.

# Guidlines
{COMMON_GUIDELINES}


# Reasoning Steps
- Internally test executability (prereqs, inventory, object existence).
- Object visibility is not a prerequisite—avoid search-related subtasks like “find” or “scan.” Use navigate to object.
- Consider primarily the open subtasks and the suggestions when deciding which agent should take which subtask.

# Context Input
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


# Final instructions
First, think carefully step by step about **which subtasks are executable now**, closely adhering to the **Allocation Rules**. Then, **return only the Allocation/Remain JSON with no extra text**.
"""

ACTION_PROMPT = f"""
# Role and Objective
You are an expert multi-robot controller, managing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}, by generating executable action plans to fulfill assigned subtasks. Convert each Assignment subtask into an explicit, sequential action list for the agent, based on current state and environmental details.

# Instructions
- Treat each assignment independently; do not invent new subtasks.
- For each "Assignment" subtask provided (one per agent), generate an ordered list of executable atomic actions using only the defined action set and schemas.
- Treat each subtask independently, without inferring new subtasks, changing intent, or considering inter-agent dependencies unless explicitly described in the input.
- Respect the affordances of objects, preconditions of actions, and environmental constraints in translating subtasks into actions.
- Assign ["Idle"] only if the subtask is "Idle".
- Generate a list of action plans (lists), one per agent, in the same order as the input subtasks.
- Navigation/Interaction:
  - Prefer NavigateTo<Object> before interaction unless object is near and in view.
  - For micro-approach, try Move/Rotate/Look first; then retry NavigateTo.
  - On "object-not-in-view": try LookDown/LookUp based on likely vertical location.
  - On "no-plan": move around to clear obstacles then retry.
- Feasibility:
  - If missing object/observation/inventory invalid/Subtask None → empty list at that index.
  - Prefer closest suitable instance if multiple exist.

# Guidelines
- If a referenced object is missing, inventory is full (when pickup is needed), agent state/observation is missing, or Subtask is None: output an empty list for that subtask (or {{ "Actions": [] }} if Subtasks is None).
- Assign only objects listed in the provided list. If multiple instances of the same type exist (e.g., Countertop_1, Countertop_2), select the most appropriate one based on the agent's current position and its proximity to the objects.
- when assigning actions which interact with objects and with navigation, always use NavigateTo<object_name> to approach the object first. Unless the targert obect is close enough and in the view of the agent.
- If the subtask requires micro-movements to approach an Object_name, use only Movement, Rotation, and Look actions based on position and failure reason—avoid using NavigateTo<Object_name> initially. Try one atomic movement at a time before attempting NavigateTo<Object_name> again.
    - When failure reason is "object-not-in-view", first try  Lookdown  or Lookup action based on the target object's most likely to be. (you can assume the agent always starts looking front. 
    - When failure reason is "no-plan", try moving around first—sometimes an obstacle (e.g., a door) may be blocking the path before re-attempting NavigateTo.
{COMMON_GUIDELINES}


# Context
- Robot names: {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}
{AI2THOR_ACTIONS}

# Input Context
- Inputs include: task description, agent states/observations, open and completed subtasks,  all objects in the environment, failed diagnostics (if any), and subtasks to be executed.
{ACTION_INPUT_FORMAT}

# Reasoning Steps
- Internally, reason step by step to extract and analyze each Assignment, confirm presence of referenced objects, consider the distance and position of agent and object, map actions using current environment and agent state, and validate required action conditions for each agent.
- After generating action plans, validate that each generated plan satisfies the requirements (object existence, feasible actions based on agent state/inventory, and completeness of required fields). If validation fails for a subtask, output as specified in Output Format.



# Output Format
The output must be a single JSON object, with no extra explanation:
- Key: "Actions"
- Value: a list of lists. Each inner list contains the atomic actions for a subtask, matching the order of input subtasks.

**Example:**
{ACTION_EXAMPLES}


# Final instructions
First, think carefully step by step about **mapping each assignment to atomic actions**, closely adhering to the **Instruction and Global Constraints and Navigation rules**. Then, **output only the Actions JSON with no explanations**.
"""



def get_verifier_prompt(need_process=False):
    if need_process:
        VERIFIER_PROMPT = f"""
        # Role and Objective
        You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. Both robots have a partially observable view of the environment. Hence, they may need to explore the environment to complete the task.
        You will get a description of the task, current observations (including images if available), and a recent slice of execution logs.

        You need to verify the previous outcome(s) and suggest the **next single action** that each robot should take in the current timestep.

        # Instructions
        - Use observations and execution logs to infer causes and decide next actions.
        - Do not generate subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred.
        - By default, object visibility is not required—always use NavigateTo(<Object>) directly when no navigation error is present.
        {COMMON_GUIDELINES}

        - Environment Hazards: open objects can block paths; avoid mutual blocking/collisions.
        - Electronics: operate directly on the device—do not use remotes unless required.
        - Use NavigateTo(<Object>) unless there is a navigation issue (e.g., no-path, distance-too-far, etc.).

        # Reasoning Steps
        - Internally reason over observations and states, using the execution logs to isolate the cause and fix.
        - You are supposed to reason over the image inputs (if any), the robots' observations, previous actions, execution logs (including any prior failures), current subtasks, and the available actions the robots can perform. Think step by step and then output the following:
        * Reason: The reasoning for what each robot is supposed to do next.
        * Suggestion: The actions the robots should take in the next step to make progress toward completing the task. Ensure these suggested actions improve efficiency compared to only one agent solving the task.

        # OUTPUT FORMAT
        You must output a JSON dictionary with:
        - "need_replan": boolean (true/false) indicating whether a replan is needed.
        - "reason": string
        - "suggestion": string (e.g., "next, Alice-0 should ..., Bob-1 should ...")
        


        # Errors Handling and Examples
        {VERIFY_EXAMPLE}

        # Context
        {VERIFIER_INPUT_FORMAT}
        {AI2THOR_ACTIONS}


        # Final instructions
        First, think carefully step by step about the **most likely failure cause and immediate fix**, closely adhering to the **Important Notes and Common Guidelines**. Then, **output only the specified dictionary**.
        """
    else:
        VERIFIER_PROMPT = f"""
        # Role and Objective
        You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. Both robots have a partially observable view of the environment. Hence, they may need to explore the environment to complete the task.
        You will get a description of the task, current observations (including images if available), and a recent slice of execution logs.

        You need to verify the previous outcome(s) and suggest the **next single action** that each robot should take in the current timestep.

        # Instructions
        - Use observations and execution logs to infer causes and decide next actions.
        - Do not generate subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred.
        - By default, object visibility is not required—always use NavigateTo(<Object>) directly when no navigation error is present.
        {COMMON_GUIDELINES}

        - Environment Hazards: open objects can block paths; avoid mutual blocking/collisions.
        - Electronics: operate directly on the device—do not use remotes unless required.
        - Use NavigateTo(<Object>) unless there is a navigation issue (e.g., no-path, distance-too-far, etc.).

        # Reasoning Steps
        - Internally reason over observations and states, using the execution logs to isolate the cause and fix.
        - You are supposed to reason over the image inputs (if any), the robots' observations, previous actions, execution logs (including any prior failures), current subtasks, and the available actions the robots can perform. Think step by step and then output the following:
        * need_replan: True or False. If you think the current plan is not efficient or not valid, output True. Otherwise, output False.
        * Reason: The reasoning for what each robot is supposed to do next.
        * Suggestion: The actions the robots should take in the next step to make progress toward completing the task. Ensure these suggested actions improve efficiency compared to only one agent solving the task.

        # OUTPUT FORMAT
        You must output a JSON dictionary with:
        - "need_replan": boolean (true/false) indicating whether a replan is needed.
        - "reason": string
        - "suggestion": string (e.g., "next, Alice-0 should ..., Bob-1 should ...")


        # Errors Handling and Examples
        {VERIFY_EXAMPLE}

        # Context
        {VERIFIER_ORG_LOG_INPUT_FORMAT}
        {AI2THOR_ACTIONS}


        # Final instructions
        First, think carefully step by step about the **most likely failure cause and immediate fix**, closely adhering to the **Important Notes and Common Guidelines**. Then, **output only the specified dictionary**.
        """
    return VERIFIER_PROMPT

def get_replanner_prompt(need_process=False):
    if need_process:
        REPLAN_PROMPT = f"""
        # Role and Objective
        You are a capable planner and multi-robot controller assigned to help {len(AGENT_NAMES)} embodied robots named {', '.join(AGENT_NAMES[:-1]) + f', and {AGENT_NAMES[-1]}'} accomplish a specific goal.
        Your goal is to replan a valid and efficient sequence of subtasks based on the current environment state and prior action history. You will be given:
        - The original task description
        - Robots' previous plans (open and completed subtasks)
        - Robots' current observations and states
        - Objects in environment
        - Failure causes
        - Reasoning and suggestions for next actions

        # Instructions
        - Avoid repeating success subtasks unless conditions have changed; minimize steps.
        - Do not generate subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred. 
        - By default, object visibility is not required—always use NavigateTo(<Object>) directly when no navigation error is present.
        {COMMON_GUIDELINES}

        - Plan Update Rules: atomic subtasks; combine strictly sequential steps; only necessary actions.
        - Please retain open subtasks that are not yet completed and are still required to accomplish the task.


        # Reasoning Steps
        - Internally analyze failures, memory, suggestions, and observations to adjust plan.


        # Output Format
        {OUTPUT_FORMAT_PLAN}

        # Examples
        {EXAMPLE_PLAN}

        # INPUT Context
        {REPLANNER_INPUT_FORMAT}
        {AI2THOR_ACTIONS}


        # Final instructions
        First, think carefully step by step about the **shortest valid subtask sequence** given the state, memory and failures, closely adhering to the **Common Guidelines**. Then, **return only the Subtasks JSON**.
        """

    else:
        REPLAN_PROMPT = f"""
        # Role and Objective
        You are a capable planner and multi-robot controller assigned to help {len(AGENT_NAMES)} embodied robots named {', '.join(AGENT_NAMES[:-1]) + f', and {AGENT_NAMES[-1]}'} accomplish a specific goal.
        Your goal is to replan a valid and efficient sequence of subtasks based on the current environment state and prior action history. You will be given:
        - The original task description
        - Robots' previous plans (open and completed subtasks)
        - Robots' current observations and states
        - Objects in environment
        - Failure causes
        - Reasoning and suggestions for next actions

        # Instructions
        - Avoid repeating success subtasks unless conditions have changed; minimize steps.
        - Do not generate subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred. 
        - By default, object visibility is not required—always use NavigateTo(<Object>) directly when no navigation error is present.
        {COMMON_GUIDELINES}

        - Plan Update Rules: atomic subtasks; combine strictly sequential steps; only necessary actions.
        - Please retain open subtasks that are not yet completed and are still required to accomplish the task.


        # Reasoning Steps
        - Internally analyze failures, memory, suggestions, and observations to adjust plan.


        # Output Format
        {OUTPUT_FORMAT_PLAN}

        # Examples
        {EXAMPLE_PLAN}

        # INPUT Context
        {REPLANNER_ORG_LOG_INPUT_FORMAT}
        {AI2THOR_ACTIONS}


        # Final instructions
        First, think carefully step by step about the **shortest valid subtask sequence** given the state, memory and failures, closely adhering to the **Common Guidelines**. Then, **return only the Subtasks JSON**.
"""

    return REPLAN_PROMPT

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

def prepare_prompt(env: AI2ThorEnv, mode: str = "init", addendum: str = "", subtasks=[], need_process=False, info=[]) -> str:
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
        input = env.get_obs_llm_input()
        input["Subtasks"] = subtasks
        del input["Robots' open subtasks"]
        del input["Robots' completed subtasks"]
        user_prompt = convert_dict_to_string(input)

    elif mode == "allocator":
        # for allocating subtasks to robots
        system_prompt = ALLOCATOR_PROMPT
        input = env.get_obs_llm_input(recent_logs=True)
        user_prompt = convert_dict_to_string(input)

    elif mode == "replan":
        # for replanning the subtasks based on the current state of the environment

        system_prompt = get_replanner_prompt(need_process)
        input = env.get_llm_log_input(need_process)
        input['Suggestion'] = info.get('suggestion', '')
        input['Reason'] = info.get('reason', '')
        
        user_prompt = convert_dict_to_string(input)
        # print("replan prompt:", system_prompt)
        # print("replan use prompt:", user_prompt)
    elif mode == "verifier":
        # for verifying the actions taken by the robots

        system_prompt = get_verifier_prompt(need_process)
        input = env.get_llm_log_input(need_process)

        user_prompt = convert_dict_to_string(input)
        user_prompt = json.dumps(input, ensure_ascii=False)
    user_prompt += addendum
    return system_prompt, user_prompt
  

def process_llm_output(res_content, mode):
    """
    mode in {"planner","action","allocator","verifier"}
    return 
      - planner  -> list[str]
      - action  -> list[str]
      - allocator-> (list[str], list[str])    # (allocations_by_agent, remain)
      - verifier -> (failure_reason, memory, reason, suggestion)
    """
    data = _to_json(res_content)
    if mode == "planner":
        return _get(data, "Subtasks") or []

    if mode == "action":
        return _get(data, "Actions") or []

    if mode == "allocator":
        alloc = _get(data, "Allocation") or {}
        remain = _get(data, "Remain") or []
        res = [(alloc.get(f"agent{i+1}", "Idle") or "Idle").strip('"\'' ) for i in range(NUM_AGENTS)]
        return res, remain

    if mode == "verifier":
        return (
            _get(data, "need_replan", aliases=["need_replan"]) or False,
            _get(data, "reason"),
            _get(data, "suggestion"),
        )

    return data

def _get(d: dict, key: str, aliases=None):
        if key in d: return d[key]
        if aliases:
            for a in aliases:
                if a in d: return d[a]
        return None

def _to_json(res_content):
    try:
        return json.loads(res_content)
    except json.JSONDecodeError:
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
        return json.loads(res_content)

def initial_subtask_planning(env, config):
    # ---1. Planner LLM 產生初步 subtasks
    planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")
    planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    res, res_content = get_llm_response(planner_payload, model=config['model'])
    print('init plan llm output', res_content)
    subtasks = process_llm_output(res_content, "planner")
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
    allocation, remain = process_llm_output(res_content, "allocator")
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
    actions = process_llm_output(res_content, mode="action")

    # For testing 
    # actions = [['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)'], ['NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)']]
    return actions

def encode_image(image_path: str):
    # if not os.path.exists()
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# def verify_actions(env, info={}, need_process=False):
#     verify_prompt, verify_user_prompt = prepare_prompt(env, mode="verifier", need_process=need_process)
#     base64_image = [encode_image(env.get_frame(i)) for i in range(len(AGENT_NAMES))]

#     image_urls = [
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
#         for image in base64_image
#     ]
#     verify_payload = prepare_payload(verify_prompt, verify_user_prompt, img_urls=image_urls)
#     # print("verify prompt: ", verify_user_prompt)
#     res, res_content = get_llm_response(verify_payload, model=config['model'])
#     # print('verify llm output', res_content)
#     need_replan, reason, suggestion = process_llm_output(res_content, mode="verifier")
#     verify_res = {
#         "need_replan": need_replan,
#         "reason": reason,
#         "suggestion": suggestion
#     }
#     return verify_res

def verify_actions(env, info={}, need_process=False):
    verify_prompt, verify_user_prompt = prepare_prompt(env, mode="verifier", need_process=need_process)
    verify_payload = prepare_payload(verify_prompt, verify_user_prompt)
    # print("verify prompt: ", verify_user_prompt)
    res, res_content = get_llm_response(verify_payload, model=config['model'])
    # print('verify llm output', res_content)
    need_replan, reason, suggestion = process_llm_output(res_content, mode="verifier")
    verify_res = {
        "need_replan": need_replan,
        "reason": reason,
        "suggestion": suggestion
    }
    return verify_res

def get_steps_by_actions(env, actions):
    print("get_steps_by_actions: ", actions)
    steps = env.actions_decomp(actions)
    return steps


def verify_subtask_completion(env, info):
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

import difflib

def verify_subtask_completion(env, info, similarity_cutoff: float = 0.62):
    """
    Args:
        env: 你的環境物件
        info: dict，至少包含 "success_subtasks"
        similarity_cutoff: 相似度比對的門檻（0~1），預設 0.62

    Returns:
        (open_subtasks, closed_subtasks)
    """

    def _normalize(s: str) -> str:
        if not isinstance(s, str):
            return s
        s = s.strip()
        if ":" in s:
            head, tail = s.split(":", 1)
            if head.strip() in env.agent_names:
                return tail.strip()
        return s

    def _best_fuzzy_match(query_norm: str, candidates_norm: list[str]) -> int | None:
        """
        回傳最佳匹配的索引（在 candidates_norm 的索引），找不到回傳 None。
        優先序：
          1) 完全等於
          2) startswith / in（子字串）
          3) difflib 相似度（>= similarity_cutoff）
        """
        # 1) 完全等於
        for i, cand in enumerate(candidates_norm):
            if query_norm == cand:
                return i

        # 2) 子字串 / 前綴（雙向嘗試，以覆蓋少字省略或附加狀況）
        for i, cand in enumerate(candidates_norm):
            if cand.startswith(query_norm) or query_norm.startswith(cand) or (query_norm in cand) or (cand in query_norm):
                return i

        # 3) difflib 相似度
        best_i, best_score = None, -1.0
        for i, cand in enumerate(candidates_norm):
            score = difflib.SequenceMatcher(None, query_norm, cand).ratio()
            if score > best_score:
                best_i, best_score = i, score
        if best_score >= similarity_cutoff:
            return best_i

        return None

    # 取列表副本避免原地修改帶來的索引問題
    open_subtasks = list(env.open_subtasks or [])
    closed_subtasks = list(env.closed_subtasks or [])
    completed_subtasks = list(info.get("success_subtasks", []))

    # 維護一份正規化後的 open_subtasks（同步更新）
    open_norm = [_normalize(s) for s in open_subtasks]

    for c in completed_subtasks:
        if c == "Idle":
            continue
        c_norm = _normalize(c)

        # 在 open_norm 裡找最佳匹配
        match_idx = _best_fuzzy_match(c_norm, open_norm)

        if match_idx is not None:
            removed_original = open_subtasks.pop(match_idx)
            open_norm.pop(match_idx)
            closed_subtasks.append(removed_original)
            if env.verbose:
                print(f"[verify_subtask_completion] Matched & closed: '{c}'  →  '{removed_original}'")
        else:
            if env.verbose:
                print(f"[verify_subtask_completion] No relaxed match for: '{c}' (norm='{c_norm}') "
                      f"in open={open_subtasks}")

    return open_subtasks, closed_subtasks

def replan_open_subtasks(env, completed_subtasks, verify_info, need_process=False):
    replan_prompt, replan_user_prompt = prepare_prompt(env, mode="replan", info=verify_info, need_process=need_process)
    # print("replan system prompt: ", replan_prompt)
    # print("replan user prompt: ", replan_user_prompt)
    replan_payload = prepare_payload(replan_prompt, replan_user_prompt)
    res, res_content = get_llm_response(replan_payload, model=config['model'])
    # print('replan llm output', res_content)
    subtasks = process_llm_output(res_content, "planner")
    # print(f"After Re-Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    return subtasks, completed_subtasks

    
def bundle_task_plan(subtasks, actions, decomp_actions):
    """
    Pack corresponding subtask, actions, and decomposed actions into aligned dicts.
    [
        { # for each agent
            "subtask": str,       
            "actions": List[str],
            "steps": List[List[str]]
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


def run_main(test_id = 0, config_path="config/config.json"):
    # --- Init.
    env, config = set_env_with_config(config_path)
    timeout = 250
    if test_id > 0:
        obs = env.reset(test_case_id=test_id)
    else:
        obs = env.reset(test_case_id=config['test_id'])
    # --- initial subtask planning
    open_subtasks, completed_subtasks = initial_subtask_planning(env, config)
    info = {}
    need_process = True

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
        print("open_subtasks: ", env.open_subtasks)
        print("closed_subtasks: ", env.closed_subtasks)

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
        # 6. verify the execution and update memory
        logs.append(f"----replanning subtasks to agents----")
        verify_res = verify_actions(env, info, need_process)
        print("verify result: ", verify_res)
        logs.append(f"verify result: {verify_res}")

        # 7. replan if needed
        if open_subtasks or not isSuccess or verify_res['need_replan']:
            
            open_subtasks, completed_subtasks = replan_open_subtasks(env, completed_subtasks, verify_res, need_process)

            # print("replan open_subtasks: ", open_subtasks)
            # print("replan closed_subtasks: ", completed_subtasks)
            logs.append(f"replan open_subtasks: {open_subtasks}")
            logs.append(f"replan closed_subtasks: {completed_subtasks}")
            
            # start_time = time.time()
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
        env.save_log()
        cnt += 1 
        
    iscomplete, report = env.run_task_check()
    obj_status = env.get_all_object_status()
    print("Final Report: ", report)
    with open(filename, "a", encoding="utf-8") as f:
        for log in logs:
            f.write(str(log) + "\n")
        f.write("Final Report: " + str(report) + "\n")
        f.write("Success: " + str(iscomplete) + "\n")
        f.write(f"Total steps: {env.step_num}\n")
        # f.write(f"Total action steps: {env.action_step_num}\n")
        f.write("\n")
        for obj_id, status in obj_status.items():
            f.write(f"{obj_id}: {status}\n")

    env.close()

def batch_run(tasks, base_dir="config", start = 1, end=5, sleep_after=2.0):
    """
    tasks: e.g. TASKS_1, TASKS_2 ...
    base_dir: the root config folder
    repeat: how many times each config runs
    sleep_after: seconds to sleep between runs
    """
    script_dir = Path(__file__).parent  # /mas/utils
    base_path = (script_dir / ".." / base_dir).resolve()

    for task in tasks:
        task_folder = task["task_folder"]
        for scene in task["scenes"]:
            cfg_path = base_path / task_folder / scene / "config.json"

            if not cfg_path.exists():
                print(f"[WARN] Config not found: {cfg_path}")
                continue

            print(f"==== Using config: {cfg_path} ====")

            for r in range(start, end + 1):
                print(f"---- Run {r}/{end} for {cfg_path} ----")
                run_main(test_id = r, config_path=str(cfg_path))
                time.sleep(sleep_after)

            print(f"==== Finished {cfg_path} ====")


if __name__ == "__main__":
    TASKS_1 = [
    # {
    #     "task_folder": "1_put_bread_lettuce_tomato_fridge",
    #     "task": "put bread, lettuce, and tomato in the fridge",
    #     "scenes": ["FloorPlan5"] #"FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", 
    # },
    # {
    #     "task_folder": "1_put_computer_book_remotecontrol_sofa",
    #     "task": "put laptop, book and remote control on the sofa",
    #     "scenes": ["FloorPlan201", "FloorPlan202","FloorPlan203", "FloorPlan209", "FloorPlan224"] 
    # },
    # {
    #     "task_folder": "1_put_knife_bowl_mug_countertop",
    #     "task": "put knife, bowl, and mug on the counter top",
    #     "scenes": ["FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"] # ,"FloorPlan1", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_put_plate_mug_bowl_fridge",
    #     "task": "put plate, mug, and bowl in the fridge",
    #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"] 
    # },
    # {
    #     "task_folder": "1_put_remotecontrol_keys_watch_box",
    #     "task": "put remote control, keys, and watch in the box",
    #     "scenes": ["FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan207","FloorPlan209", "FloorPlan215", "FloorPlan226", "FloorPlan228", ] # 
    # },
    # {
    #     "task_folder": "1_put_vase_tissuebox_remotecontrol_table",
    #     "task": "put vase, tissue box, and remote control on the side table1",
    #     "scenes": ["FloorPlan201", "FloorPlan203", "FloorPlan216"] # , "FloorPlan219"
    # },
    # {
    #     "task_folder": "1_put_vase_tissuebox_remotecontrol_table",
    #     "task": "put vase, tissue box, and remote control on the desk",
    #     "scenes": ["FloorPlan229"] #"FloorPlan201", "FloorPlan203", "FloorPlan216",
    # },
    # {
    #     "task_folder": "1_slice_bread_lettuce_tomato_egg",
    #     "task": "slice bread, lettuce, tomato, and egg with knife",
    #     "scenes": ["FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"] #"FloorPlan1", 
    # },
    # {
    #     "task_folder": "1_turn_off_faucet_light",
    #     "task": "turn off the sink faucet and turn off the light switch",
    #     "scenes": [ "FloorPlan4", "FloorPlan5"] #"FloorPlan1", "FloorPlan2", "FloorPlan3",
    # },
    {
        "task_folder": "1_wash_bowl_mug_pot_pan",
        "task": "clean the bowl, mug, pot, and pan",
        "scenes": ["FloorPlan3", "FloorPlan4", "FloorPlan5"] #"FloorPlan1", "FloorPlan2", 
    },
]

    TASKS_2 = [
        # {
        #     "task_folder": "2_open_all_cabinets",
        #     "task": "open all the cabinets",
        #     "scenes": [ "FloorPlan10"]#"FloorPlan1", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9",
        # },
        # {
        #     "task_folder": "2_open_all_drawers",
        #     "task": "open all the drawers",
        #     "scenes": [ "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9"] # "FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4",
        # },
        {
            "task_folder": "2_put_all_creditcards_remotecontrols_box",
            "task": "put all credit cards and remote controls in the box",
            "scenes": ["FloorPlan201", "FloorPlan203","FloorPlan204", "FloorPlan205"]
        },
        {
            "task_folder": "2_put_all_vases_countertop",
            "task": "put all the vases on the counter top",
            "scenes": ["FloorPlan1", "FloorPlan5"]
        },
        {
            "task_folder": "2_put_all_tomatoes_potatoes_fridge",
            "task": "put all tomatoes and potatoes in the fridge",
            "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        },
        
        {
            "task_folder": "2_turn_on_all_stove_knobs",
            "task": "turn on all the stove knobs",
            "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9"]
        },  
    ]

    TASKS_3 = [
        # {-
        #     "task_folder": "3_clear_table_to_sofa",
        #     "task": "Put all readable objects on the sofa",
        #     "scenes": ["FloorPlan201", "FloorPlan203", "FloorPlan204", "FloorPlan208", "FloorPlan223"]
        # },
        # {
        #     "task_folder": "3_put_all_food_countertop",
        #     "task": "Put all food on the countertop",
        #     "scenes": [ "FloorPlan4", "FloorPlan5"] # "FloorPlan1", "FloorPlan2", "FloorPlan3",
        # },
        # {
        #     "task_folder": "3_put_all_groceries_fridge",
        #     "task": "Put all groceries in the fridge",
        #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        # },
        # {
        #     "task_folder": "3_put_all_kitchenware_box",
        #     "task": "Put all kitchenware in the cardboard box",
        #     "scenes": ["FloorPlan201"]
        # },
        # {
        #     "task_folder": "3_put_all_school_supplies_sofa",
        #     "task": "Put all school supplies on the sofa",
        #     "scenes": ["FloorPlan201", "FloorPlan202", "FloorPlan203","FloorPlan209", "FloorPlan212"]
        # },
        # {
        #     "task_folder": "3_put_all_shakers_fridge",
        #     "task": "Put all shakers in the fridge",
        #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        # },  
        # {
        #     "task_folder": "3_put_all_shakers_tomato", # on countertop
        #     "task": "put all shakers and tomato on the counter top",
        #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        # },  
        # {
        #     "task_folder": "3_put_all_silverware_drawer",
        #     "task": "Put all silverware in the drawer",
        #     "scenes": [ "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6"]
        # },  
        # {
        #     "task_folder": "3_put_all_tableware_countertop",
        #     "task": "Put all tableware on the countertop",
        #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        # },  
        # {-
        #     "task_folder": "3_transport_groceries",
        #     "task": "put_all_food_countertops",
        #     "scenes": ["FloorPlan1"]
        # },  
        
    ]

    TASKS_4 = [
    {
        "task_folder": "4_clear_couch_livingroom",
        "task": "Clear the couch by placing the items in other appropriate positions ",
        "scenes": ["FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan209", ] #"FloorPlan212" hen
    },
    {
        "task_folder": "4_clear_countertop_kitchen",
        "task": "Clear the countertop by placing items in their appropriate positions",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan30", "FloorPlan10", "FloorPlan6"]
    },
    {
        "task_folder": "4_clear_floor_kitchen",
        "task": "Clear the floor by placing items at their appropriate positions",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "4_clear_table_kitchen",
        "task": "Clear the table by placing the items in their appropriate positions",
        "scenes": ["FloorPlan4", "FloorPlan11", "FloorPlan15", "FloorPlan16", "FloorPlan17"]
    },
    # {
    #     "task_folder": "4_make_livingroom_dark",
    #     "task": "Make the living room dark",
    #     "scenes": ["FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan204", "FloorPlan205"]
    # },
    {
        "task_folder": "4_put_appropriate_storage",
        "task": "Place all utensils into their appropriate positions",
        "scenes": ["FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6"]
    },  
]
    
    batch_run(TASKS_2, base_dir="config", start=1, end=1, sleep_after=50)
    # run_main(test_id = 1, config_path="config/config.json")

