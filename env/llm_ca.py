'''
Baseline : Centralized LLM + replanning

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

- () still need to adjust the prompt, 
- () reset the agent failure subtasks and reasons after replan
- () testing

model: gpt-4.1-2025-04-14, gpt-4o

structure same as llm_c.py but with more information about the environment and positions, failures, etc.:
1. initial planning (remain the same as previous method: given task, let planner and editor to generate a list of subtasks (this will be the open subtasks)
2. start a loop, until timeout or all the open subtasks is empty:
2.1 update open subtasks and completed subtask
2.2 allocate subtask to robot agents in the environment with llm
2.3 break down each assigned subtasks with llm into a list of smaller available actions
2.4 execute one subtask per agents
2.5 verify if the subtask is completed
2.6 replan: similar to initial planning : given task and closed subtask, let planner and editor to generate a list of subtasks (this will be the new open subtasks)
'''

import json
import re
import os
import sys
import argparse
from pathlib import Path
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
You are a capable planner and robot controller overseeing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}, as they complete a specified task.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

**Available Actions:**
- move: [MoveAhead, MoveBack, MoveRight, MoveLeft]
- rotate: [RotateRight, RotateLeft]
- look: [LookUp<angle>, LookDown<angle>]
- idle: [Idle]
- interact_with_object: [NavigateTo<object_name>, PickupObject<object_name>, PutObject<receptacle_name>, OpenObject<object_name>, CloseObject<object_name>, ToggleObjectOn<object_name>, ToggleObjectOff<object_name>, BreakObject<object_name>, CookObject<object_name>, SliceObject<object_name>, DirtyObject<object_name>, CleanObject<object_name>, FillObjectWithLiquid<object_name>, EmptyLiquidFromObject<object_name>, UseUpObject<object_name>]
- interact_without_navigation: [DropHandObject, ThrowObject]

**Workflow:**
- You will be provided with a high-level description of the task, the number of agents, and the list of objects present in the environment.
- Decompose the main task into subtasks that the robots can complete, taking into account the robots' abilities and the objects available.
- Subtasks that must occur in sequence should be combined into a single subtask.

**EXAMPLES:**
If the task is "Put the vase, tissue box, and remote control on the table", example subtasks:
[
"pick up the vase and put it on the table",
"pick up the tissue box and put it on the table",
"pick up the remote control and put it on the table"
]

If the task is "Put the lettuce and potato in the fridge", example subtasks:
[
"open the fridge",
"pick up the lettuce and put it in the fridge",
"pick up the potato and put it in the fridge",
"close the fridge"
]

**Input Format (JSON):**
{{
"Task": "A high-level description of the final goal",
"Number of agents": <number of robots>,
"Objects": [list of objects in the environment]
}}

**Output Format (JSON):**
{{
"Subtasks": [
"subtask 1",
"subtask 2",
...
]
}}

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

You will receive:
- A high-level task description
- A list of subtasks generated by another agent
- A list of objects currently present in the environment

Your objective is to review and correct the sequence of subtasks to ensure all actions contribute precisely to achieving the given goal.

## Task Correction Rules
1. If instructed to put X in the fridge, verify that X is explicitly placed inside the fridge.
2. For any openable container (e.g., fridge, drawer), ensure it is opened prior to placing items inside and closed afterward instructions should appear as: "open Y, put X inside, and close Y."
3. Never place items on receptacles not identified in the original task.
4. Preserve robot capabilities (e.g., hand constraints, action limits), but restructure or merge subtasks for logical sequencing if required.
5. Do not specified which exact objects, use only the object names. (e.g. apple, not Apple_1)

## Input Format
{{
  Task: Description of the overall goal for the robots,
  Number of agents: Number of robots,
  Subtasks: List of subtasks provided by another agent,
  Objects: List of objects currently available in the environment
}}

## Output Format
Return only the corrected subtasks in this exact JSON structure:
{{
  "Subtasks": [subtask 1, subtask 2, ..., subtask n]
}}

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
You are an expert task allocator, responsible for assigning tasks to a team of {len(AGENT_NAMES)} embodied robots to efficiently achieve a specified goal.

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
You will receive a JSON object with these required fields:
- "Task" (string): Main task description.
- "Number of agents" (integer).
- "Robots' open subtasks" (list of strings, can be None or empty if not planned).
- "Robots' completed subtasks" (list of strings, can be None or empty if not started).
- "Reachable positions" (list).
- "Agent's observation" (list): Observed objects and their positions.
- "Agent's state" (dict): Agent's position, facing, and inventory.
Optional fields:
- "Agent's subtask_failure_reasons" (string/list): Previous step performance/failures.
- "Agent's previous failures" (string/list): Previous action failures, if any.


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
- Manage {len(AGENT_NAMES)} embodied robots by generating executable action plans to fulfill assigned subtasks. Convert each subtask into an explicit, sequential action list for the agent, based on current state and environmental details.

# Checklist
- Begin each output with a concise checklist (3-7 bullets) summarizing the planned process before generating the output (e.g., extract subtasks, check object presence, plan concrete actions, verify conditions, format result).

# Planning and Execution
- For each subtask provided (one per agent), generate an ordered list of executable atomic actions using only the defined action set and schemas.
- Treat each subtask independently, without inferring new subtasks, changing intent, or considering inter-agent dependencies unless explicitly described in the input.
- Respect the affordances of objects, preconditions of actions, and environmental constraints in translating subtasks into actions.
- Assign ["Idle"] only if the subtask is "Idle".
- Generate a list of action plans (lists), one per agent, in the same order as the input subtasks.

## Action Set
- **Movement**: [MoveAhead, MoveBack, MoveRight, MoveLeft]
- **Rotation**: [RotateRight, RotateLeft]
- **Look**: [LookUp<angle>, LookDown<angle>] where angle is 30 or 60
- **Idle**: [Idle]
- **Object Interaction**:
  - With navigation: [NavigateTo<object_name>, PickupObject<object_name>, PutObject<receptacle_name>, OpenObject<object_name>, CloseObject<object_name>, ToggleObjectOn<object_name>, ToggleObjectOff<object_name>, BreakObject<object_name>, CookObject<object_name>, SliceObject<object_name>, DirtyObject<object_name>, CleanObject<object_name>, FillObjectWithLiquid<object_name>, EmptyLiquidFromObject<object_name>, UseUpObject<object_name>]
  - Without navigation: [DropHandObject, ThrowObject]

# Context
- Robot names: {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}
- Inputs include: task description, agent states/observations, open and completed subtasks, reachable positions, all objects in the environment, failed diagnostics (if any), and subtasks to be executed.

# Input Format
Provide inputs as a JSON object containing:
- Task (string)
- Number of agents (int)
- Robots' open subtasks (list[string]|None)
- Robots' completed subtasks (list[string]|None)
- Reachable positions (list[string])
- Objects in environment (list[string])
- Agent's observations (list[object/position])
- Agent's state (location, facing, inventory)
- Subtasks (list of subtasks to execute)
- (optional): agent subtask_failure_reasons, previous failures

# Planning and Validation
- Internally, reason step by step to extract and analyze each subtask, confirm presence of referenced objects, map actions using current environment and agent state, and validate required action conditions for each agent.
- After generating action plans, validate that each generated plan satisfies the requirements (object existence, feasible actions based on agent state/inventory, and completeness of required fields). If validation fails for a subtask, output as specified in Output Format.

# Guidelines
- If a referenced object is missing, inventory is full (when pickup is needed), agent state/observation is missing, or Subtask is None: output an empty list for that subtask (or {{ "Actions": [] }} if Subtasks is None).
- Do not output or format anything beyond the defined JSON.


# Output Format
The output must be a single JSON object, with no extra explanation:
- Key: "Actions"
- Value: a list of lists. Each inner list contains the atomic actions for a subtask, matching the order of input subtasks.
- Do not include failure codes, error objects, or explanations. Encode error states strictly as empty lists at the corresponding subtask index.
- Output only the specified JSON object. Do not provide explanations or extra responses. Reason in detail internally to maximize output correctness.

**Example:**
Given subtasks: ["pick up apple", "Idle", "slice bread"]

{{
  "Actions": [
    ["NavigateTo(Apple_1)", "PickupObject(Apple_1)"],
    ["Idle"],
    ["NavigateTo(Bread_1)", "SliceObject(Bread_1)"]
  ]
}}

** The output must be a valid JSON object with no extra explanation.

"""

REPLAN_PROMPT = f"""
You are a capable planner and multi-robot controller assigned to help {len(AGENT_NAMES)} embodied robots named {', '.join(AGENT_NAMES[:-1]) + f', and {AGENT_NAMES[-1]}'} accomplish a specific goal.
Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

## Available Actions
- Movement: MoveAhead, MoveBack, MoveRight, MoveLeft
- Rotation: RotateRight, RotateLeft
- Camera: LookUp<angle>, LookDown<angle> (angle: 30, 60)
- Idle: Idle
- Object Interaction (with navigation):
- NavigateTo<object_name>, PickupObject<object_name>, PutObject<receptacle_name>
- OpenObject<object_name>, CloseObject<object_name>
- ToggleObjectOn<object_name>, ToggleObjectOff<object_name>
- BreakObject<object_name>, CookObject<object_name>, SliceObject<object_name>
- DirtyObject<object_name>, CleanObject<object_name>
- FillObjectWithLiquid<object_name>, EmptyLiquidFromObject<object_name>
- UseUpObject<object_name>
- Object Interaction (without navigation): DropHandObject, ThrowObject

## TASK
You must reason about the current environment state and prior plan history to **replan** a valid, efficient subtask sequence for the robots to achieve the overall goal. Environment is partially observable, and robots may experience execution failures (occlusion, distance, blocked paths, misidentified objects). Plans should:
- Retain subtasks not yet failed.
- Revise or replace failed subtasks according to the failure cause.
- Decompose the main task into executable, atomic subtasks per robot.
For each subtask in 'open' or 'completed' lists:
- If open, check feasibility with current state and capabilities; reuse if viable.
- If completed, acknowledge and adapt the plan accordingly.
- If failed, determine if circumstances have changed; if not, substitute an alternate strategy.
After completing the checklist and plan, validate result in 1-2 lines and proceed or self-correct if validation fails, but don't output anything for this self-validation.

### INPUT FORMAT
{{
Task: description of task,
Number of agents: int,
Robots' open subtasks: [list of pending subtasks or None],
Robots' completed subtasks: [list of completed subtasks or None],
Reachable positions: [list],
Objects in environment: [list],
agent name's observation: [list of perceived objects/positions],
agent name's state: position, facing, inventory,
agent name's subtask_failure_reasons: [recent actions, success/failure with explanations],
agent name's previous failures: [recent failure details],
}}

### OUTPUT FORMAT
Return only the following JSON:
{{
"Subtasks": ["subtask 1", "subtask 2", ..., "subtask n"]
}}

Example:
{{
"Subtasks": [
"pick up the vase and put it on the table",
"pick up the tissue box and put it on the table",
"pick up the remote control and put it on the table"
]
}}

## Failure Handling
Possible causes and adjustments for subtask failures:
- Navigation: "no-path", "object-not-in-view", "distance-too-far": Use micro-movements (MoveAhead, RotateRight, etc.) to get in view/reach, using current/target positions and available observations. For example, you can have subtasks like "look down and navigate to potato" if previous failure reason of "pick up potato" was "object-not-in-view". Or you can have subtask like "use micro-movements to navigate to potato" if previous subtask "navigate to potato" was failured.
- Object-related: "object-not-found", "object-not-reachable", "object-not-in-inventory", "object(<OBJ>)-not-picked-up":
- Ensure necessary object prerequisites (e.g., pick up before use, confirm object presence).

## Key Constraints
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
- To slice: robot must hold a knife.
- Robots can carry one object at a time—enforce this strictly.
- Clean objects only when explicitly specified, using correct affordances.
- Do not assume or use default receptacles unless specified in the task.
- Output **only** the required subtasks JSON. No extra explanations, formatting, or commentary.

**Think step by step internally; return only the specified JSON.**

"""


# REPLAN_PROMPT = f"""
# You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
# They can perform the following actions: 
# **Available Actions:**
# - move: [MoveAhead, MoveBack, MoveRight, MoveLeft]
# - rotate: [RotateRight, RotateLeft]
# - look: [LookUp<angle>, LookDown<angle>], angle can be chosen from [30, 60]
# - idle: [Idle]
# - interact_with_object: [NavigateTo<object_name>, PickupObject<object_name>, PutObject<receptacle_name>, OpenObject<object_name>, CloseObject<object_name>, ToggleObjectOn<object_name>, ToggleObjectOff<object_name>, BreakObject<object_name>, CookObject<object_name>, SliceObject<object_name>, DirtyObject<object_name>, CleanObject<object_name>, FillObjectWithLiquid<object_name>, EmptyLiquidFromObject<object_name>, UseUpObject<object_name>]
# - interact_without_navigation: [DropHandObject, ThrowObject]

# ### TASK ###
# Your job is to reason over the current state of the environment and the progress history of previously executed plans, in order to **replan** the sequence of subtasks necessary to complete the overall goal.

# This environment is partially observable, and robots may encounter execution failures due to occlusion, distance, blocked paths, or misidentified object locations. Therefore, the replanning should take into account:


# Your goal is to **revise or regenerate a valid and efficient sequence of subtasks** for the robots, using all the information above. When possible, retain subtasks that have not yet failed. Otherwise, adjust or replace the failed ones with alternative approaches.
# Decompose the main task into subtasks that the robots can complete, taking into account the robots' abilities and the objects available.
# - For each subtask(from Robots' open subtasks and completed subtasks), evaluate its status:
#   - If it is in open subtask, meaning it has not been executed yet, check if it is feasible to execute it based on the current state of the environment and the robots' capabilities, consider reusing it directly.
#   - If it has been completed, meaning that this part of task is already been fulfilled and you should take into account to replan.
#   - If it has failed, evaluate whether the cause has been resolved. If not, replace it with an alternative subtask that circumvents the failure.

# ### INPUT FORMAT ###
# {{
#     Task: description of the task the robots are supposed to complete,
#     Number of agents: number of agents in the environment,
#     Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
#     Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
#     Reachable positions: list of reachable positions in the environment,
#     Objects in environment: list of all objects in the environment,
#     agent name's observation: list of objects and its postion the agent name is observing,
#     agent name's state: description of agent name's state(position, facing direction and inventory),
#     agent name's subtask_failure_reasons: description of what agent name did in the previous time step and whether it was successful and if it failed, why it failed,
#     agent name's previous failures: if agent name's few previous actions failed, description of what failed,
# }}  

# ### OUTPUT FORMAT ###
# You will output a list of subtasks for each robot in the following format, in json format:
# {{
# "Subtasks": [subtask 1, subtask 2, ..., subtask n],
# }}
# example:
# {{
# "Subtasks": [
#     "pick up the vase and put it on the table",
#     "pick up the tissue box and put it on the table",
#     "pick up the remote control and put it on the table"
# ]
# }}

# ## Failure reasons:
# Types of that can cause a subtask to fail:
# - navigation related: "no-path", "object-not-in-view", or "distance-too-far" etc, when this type of failure occurs, the robot should try to navigate to the object by using micro actions(such as MoveAhead, MoveBack, RotateRight, RotateLeft) to reach the object instead of just NavigateToObject (which this is sometime not feasible by unknow reason). take account of the given reachable positions and the agent's current position and the obect view in the environment.
# - object related: "object-not-found", "object-not-reachable", "object-not-in-inventory", or "object(<OBJ>)-not-picked-up" etc, when this type of failure occurs, often means that the robot intent to interact with an object but the prior action or reqirement was not met(such as the robot did not pick up the object before trying to put it in a receptacle, or the object is not in the environment).

# ### Important Notes ###
# * Each subtask should be atomic and goal-directed (e.g., "pick up the apple and put it in the fridge").
# * Avoid repeating failed subtasks unless their failure cause has been addressed.
# * Ensure the plan is minimal-avoid unnecessary steps.
# * You can assume the robots are coordinated and will handle task allocation downstream.
# * Consider object availability, spatial constraints, and object affordances when generating subtasks.
# - Subtasks must be independent unless a specific order is required; sequence-dependent actions must be merged into a single subtask.
# - Only use objects that are present in the environment.
# - Action plans must reflect real-world object affordances.
# - Do not use OpenObject on surfaces such as tables or countertops.
# - For openable receptacles (e.g. drawer, fridge, cabinet):
#     - The container must be opened before placing any object inside.
#     - The robot's hand must be empty before opening or closing any container.
#     - The container should be closed after use.
# - If placing an object inside an openable container: open it first, then pick up the object, then deposit the object.
# - To put an object on a receptacle: pick up the object, navigate to the receptacle, then place the object.
# - To cook an object, the object should be placed in a receptacle (e.g. pan, pot) and the receptacle should be on a stove burner and turn on stove knob.
# - To slice an object, the robot must hold a knife.
# - Robots can hold only one object at a time; ensure actions adhere to this constraint.
# - Objects can be cleaned using CleanObject or by placing them under a running faucet. Only clean when explicitly required.
# - Avoid unnecessary actions—only perform what is strictly required for the task.
# - Do NOT assume or use default receptacles like CounterTop or Table unless they are mentioned in the task description.
# - Output only the subtasks JSON as specified, with no extra explanations or formatting.
# * **NOTE**: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
# Let's work this out in a step by step way to be sure we have the right answer.
# """


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
    print("using model:", model)
    # print("payload:", payload)
    if model.startswith("gpt-4"):
        # for models: gpt-4.1-2025-04-14, gpt-4o,
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

def prepare_payload(system_prompt, user_prompt):
    print("system_prompt:", system_prompt)
    print("user_prompt:", user_prompt)
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
        user_prompt = convert_dict_to_string(input)

    elif mode == "allocator":
        # for allocating subtasks to robots
        system_prompt = ALLOCATOR_PROMPT
        input = env.get_obs_llm_input(prev_info=info)
        # if info:
        #     input['failure_subtasks'] = info['failure_subtasks']
        #     input['subtask_failure_reasons'] = info['subtask_failure_reasons']
        #     input['failed_acts'] = info['failed_acts']
        user_prompt = convert_dict_to_string(input)

    elif mode == "replan":
        # for replanning the subtasks based on the current state of the environment
        system_prompt = REPLAN_PROMPT
        try:
            input = env.get_obs_llm_input(prev_info=info)
        except KeyError as e:
            print(f"[Error] Missing key in info for replan: {e}")
            return None, None
        
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
    # planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")
    # planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    # res, res_content = get_llm_response(planner_payload, model=config['model'])
    # print('init plan llm output', res_content)
    # subtasks = process_planner_llm_output(res_content)
    # print(f"After Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    # # ---2. Editor LLM 修正 subtasks
    # editor_prompt, editor_user_prompt = prepare_prompt(env, mode="editor", subtasks=subtasks)
    # editor_payload = prepare_payload(editor_prompt, editor_user_prompt)
    # res, res_content = get_llm_response(editor_payload, model=config['model'])
    # print('editor llm output', res_content)
    # subtasks = process_planner_llm_output(res_content)
    # print(f"After Editor LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

    # for testing
    # subtasks = ['open the fridge', 'pick up the apple and put it in the fridge', 'pick up the lettuce and put it in the fridge', 'pick up the tomato and put it in the fridge', 'close the fridge']
    subtasks = ['pick up knife', 'slice apple', 'put down knife',  'slice bread', 'pick up one bread slice', 'insert bread slice into toaster', 'activate toaster']
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
    print('verifying open_subtasks')
    print(open_subtasks)
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
    print('replan llm output', res_content)
    subtasks = process_planner_llm_output(res_content)
    print(f"After Re-Planner LLM Response: {subtasks}, type of res_content: {type(subtasks)}")

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

        # 2. allocate subtasks to each agent
        agent_assignments, remain = allocate_subtasks_to_agents(env)
        print("agent_assignments: ", agent_assignments)
        # print("remain unassigned subtasks: ", remain)
        
        # # 3. decompose subtask to smaller actions
        actions = decompose_subtask_to_actions(env, agent_assignments)
        print("actions: ", actions)
        decomp_actions = get_steps_by_actions(env, actions)
        
        # # 4. execution
        print("decomp_actions: ", decomp_actions)
        cur_plan = bundle_task_plan(agent_assignments, actions, decomp_actions)
        print("cur_plan: ", cur_plan)
        isSuccess, info = env.stepwise_action_loop(cur_plan)
        print('info', info)
        '''
        {'step': 0, 'actions_success': {'Alice': [], 'Bob': ['Idle']}, 'success_subtasks': ['Idle'], 'failure_subtasks': ['pick up knife'], 'subtask_failure_reasons': {'Alice': [{'subtask': 'pick up knife', 'reason': 'no-path', 'at_action': 'NavigateTo(Knife_1)'}, {'subtask': 'pick up knife', 'reason': 'object-not-in-view', 'at_action': 'NavigateTo(Knife_1)'}], 'Bob': []}, 'inventory': ['nothing', 'nothing'], 'failed_acts': {'Alice': ['NavigateTo(Knife_1)'], 'Bob': []}}
        '''
        # break

        

        # # 5. verify which subtasks are done 
        open_subtasks, completed_subtasks = verify_subtask_completion(env, info)
        print("after verify open_subtasks: ", open_subtasks)
        print("after verify closed_subtasks: ", completed_subtasks)
        env.update_plan(open_subtasks, completed_subtasks)
        # 6. replan if needed
        if open_subtasks or not isSuccess:
            open_subtasks, completed_subtasks = replan_open_subtasks(env, info, completed_subtasks)
            print("replan open_subtasks: ", open_subtasks)
            print("replan closed_subtasks: ", completed_subtasks)
            start_time = time.time()
            # break
        else:
            break
        
    env.save_log()
   
    env.close()

if __name__ == "__main__":
    run_main()

