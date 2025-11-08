AGENT_NAMES_ALL = ["Alice", "Bob", "Charlie", "David", "Emma"]

NUM_AGENTS = 2
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
"navigate to the lettece, pick up the lettuce,  navigate to the fridge,  open the fridge, put it in the fridge, and close the fridge",
"navigate to the potato, pick up the potato,  navigate to the fridge,  open the fridge, put it in the fridge, and close the fridge",
]

High-level task: "Put the spoon and knife in any drawer" with 2 agents, example output (Note: "any drawer" means any suitable drawer in the environment):
[
"navigate to the spoon, pick up the spoon,  navigate to the drawer,  open the drawer, put it in the drawer, and close the drawer",
"navigate to the knife, pick up the knife,  navigate to the drawer,  open the drawer, put it in the drawer, and close the drawer",
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
** when given input shows "a held item: <object_id> with something if agent <navigation related action: e.g., rotates Left 30 degrees>", it means that the agent is currently in a narrow space and cannot do certain navigation(often happen when rotation), you can suggest the agent to do "MoveBack" or "MoveAhead" first to get out of the narrow space.
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
- For openable containers (do not close a box):
  - Open before use and always Close after use.
  - To place an object inside: navigate to the target object → pick up the object → navigate to the container → open the contatiner →  deposit it inside → close.
- To place on a receptacle: pick up → navigate/approach → place.
- Cooking: item in a pan/pot → pan/pot on a stove burner → turn on the stove knob (turn off if the task requires).
- Slicing: the robot can slice either with or without a knife, and do **not** pick up the item being sliced.
- After Slicing: When an object is sliced, it becomes multiple pieces that keep the same base name but receive new indices/unique IDs. E.g., slicing Apple_1 yields Apple_1, Apple_2, …
- Each robot can hold only one object at a time. When holding an item, the robot **cannot** perform interactions that require a free hand (e.g., OpenObject, CloseObject, ToggleObjectOn/Off); empty the hand first (put on a surface or drop) before such actions.
- When cleaning is required, you must only use the CleanObject action. Do not use any other methods (e.g., placing under faucet, using soap, or sponge). No need to pick up the dirty object first; clean it in place.
- Do not assume default receptacles (e.g., CounterTop, Table) unless explicitly mentioned/present.
- Close any opened object before leaving when appropriate.
- Avoid agents blocking each other where possible.
- Object cannot be given to other agent.
- Unless otherwise specified, assume robots start with empty hands.
- For electronic items (e.g., television, laptop, phone), toggle power directly on the device; **do not use remote controls** unless explicitly required.
- For storage selection: use appropriate receptacles based on the item size and context.
  - Small items (e.g., utensils, tools, food ingredients) can be stored in drawers or cabinets.
  - Larger items (e.g., books, vases, boxes) can be placed on shelves, tables, or other stable surfaces.
  - Use common sense when determining suitable storage.
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

LOG_ORIGINAL_INPUT_FORMAT = """
- "Logs": a list of JSON objects. Each object represents one action execution log for a specific agent at a certain timestamp.
JSON field description:
- "timestemp" (number): The timestamp of the action.
- "agent_id" (integer): The ID of the agent (e.g., 0 or 1).
- "agent_name" (string): The name of the agent.
- "curr_subtask" (string): The subtask the agent is currently executing.
- "type" (string): The result type of the action. Possible values: "Attempt", "Success", "Failed". "Attempt" indicates the subtask is still ongoing (not yet completed), which means that the action was interrupted because of other agent's failure.
- "payload" (object): Detailed information about the agent's status during this action.
    - "last_action" (string): The last atomic action executed.
    - "last_action_status" (string, optional): The status of the last action, e.g., "Success" or "Failed". This status do not represent the overall subtask status.
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

# for SUMMARY only
MEMORY_AGENT_INPUT_FORMAT = """
For each agent:
- "name": (string) The agent's name.
- "Agent's state": Agent's position, facing, and inventory.
- "Agent's last_action": The most recent action the agent attempted.
- "Agent's last_action_success": (boolean) Whether the last action succeeded.
- "Agent's recent_actions": (list) The last N actions the agent attempted.
- "Agent's recent_success_flags": (list[bool]) Whether those recent actions succeeded.
- "Agent's subtask_failure_reasons": Previous step performance/failures, if any.
- "Agent's previous failures": Action-level failures, if any.
- "Agent's last_check_reason": Latest environment-based failure diagnosis, if any.
- "Agent's observation": (list) List of visible objects with positions.
"""
# for SUMMARY only
MEMORY_HISTORY_INPUT_FORMAT = """- "Robots' memory": string of important information about the scene and action history that should be remembered for future steps,
- "suggestion": a string of reasoning for what each robot should do next and a description of the next actions each robot should take,
"""
# for SUMMARY only
HISTORY_INPUT_FORMAT = """
- "subtask_success_history": (dict) Mapping from agent name to list of past successful subtasks.
- "subtask_failure_reasons": (dict) Mapping from agent name to failure reason payloads.
"""
# for SUMMARY only
FAILURES_INPUT_FORMAT = """- "Agent's subtask_failure_reasons": Previous step performance/failures, if any,
- "Agent's previous failures": Previous action failures, if any,
"""



# LOG Input Formats
# PLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT
# EDITOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + SUBTASK_INPUT_FORMAT
# ALLOCATOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + PREVIOUS_LOG_FORMAT
# ACTION_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + AGENT_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT

# VERIFIER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT
# REPLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + SUGGESTION_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT

# VERIFIER_ORG_LOG_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_ORIGINAL_INPUT_FORMAT
# REPLANNER_ORG_LOG_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + SUGGESTION_INPUT_FORMAT + LOG_ORIGINAL_INPUT_FORMAT

# SUMMARY Input Formats
PLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT
EDITOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + SUBTASK_INPUT_FORMAT
# ALLOCATOR_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + FAILURES_INPUT_FORMAT
# ACTION_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT
# VERIFIER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT
# REPLANNER_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT
MEMORY_GATE_INPUT_FORMAT = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + MEMORY_AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + HISTORY_INPUT_FORMAT


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
- *Navigate directly to the target object*, not to the container or receptacle it is in or on.
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

# ACTION_PROMPT = f"""
# # Role and Objective
# You are an expert multi-robot controller, managing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}, by generating executable action plans to fulfill assigned subtasks. Convert each Assignment subtask into an explicit, sequential action list for the agent, based on current state and environmental details.

# # Instructions
# - Treat each assignment independently; do not invent new subtasks.
# - For each "Assignment" subtask provided (one per agent), generate an ordered list of executable atomic actions using only the defined action set and schemas.
# - Treat each subtask independently, without inferring new subtasks, changing intent, or considering inter-agent dependencies unless explicitly described in the input.
# - Respect the affordances of objects, preconditions of actions, and environmental constraints in translating subtasks into actions.
# - Assign ["Idle"] only if the subtask is "Idle". Do not assign empty lists.
# - Generate a list of action plans (lists), one per agent, in the same order as the input subtasks.
# - Navigation/Interaction:
#   - Prefer NavigateTo<Object> before interaction unless object is near and in view or currently held by the agent.
#   - For micro-approach, try Move/Rotate/Look first; then retry NavigateTo.
#   - On "object-not-in-view": try LookDown/LookUp based on likely vertical location.
#   - On "no-plan" or "no-path": move around to clear obstacles then retry.
# - Feasibility:
#   - Prefer closest suitable instance if multiple exist.

# # Guidelines
# - If a referenced object is missing, inventory is full (when pickup is needed), agent state/observation is missing, or Subtask is None: output an empty list for that subtask (or {{ "Actions": [] }} if Subtasks is None).
# - Assign only objects listed in the provided list. If multiple instances of the same type exist (e.g., Countertop_1, Countertop_2), select the most appropriate one based on the agent's current position and its proximity to the objects.
# - when assigning actions which interact with objects and with navigation, always use NavigateTo<object_name> to approach the object first. Unless the targert obect is close enough and in the view of the agent.
# - Avoid assigning the same object to multiple agents in the same step. 
# - If there are multiple same object type, assign the most reachable one according the given observation of the agent.
# - If the subtask requires micro-movements to approach an Object_name, use only Movement, Rotation, and Look actions based on position and failure reason—avoid using NavigateTo<Object_name> initially. Try one atomic movement at a time before attempting NavigateTo<Object_name> again.
#     - When failure reason is "object-not-in-view", first try  Lookdown  or Lookup action based on the target object's most likely to be. (you can assume the agent always starts looking front. 
#     - When failure reason is "no-plan" or "no-path", try moving around first—sometimes an obstacle (e.g., a door) may be blocking the path before re-attempting NavigateTo.
# {COMMON_GUIDELINES}


# # Context
# - Robot names: {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}
# {AI2THOR_ACTIONS}

# # Input Context
# - Inputs include: task description, agent states/observations, open and completed subtasks,  all objects in the environment, failed diagnostics (if any), and subtasks to be executed.
# {ACTION_INPUT_FORMAT}

# # Reasoning Steps
# - Internally, reason step by step to extract and analyze each Assignment, confirm presence of referenced objects, consider the distance and position of agent and object, map actions using current environment and agent state, and validate required action conditions for each agent.
# - After generating action plans, validate that each generated plan satisfies the requirements (object existence, feasible actions based on agent state/inventory, and completeness of required fields). If validation fails for a subtask, output as specified in Output Format.



# # Output Format
# The output must be a single JSON object, with no extra explanation:
# - Key: "Actions"
# - Value: a list of lists. Each inner list contains the atomic actions for a subtask, matching the order of input subtasks.

# **Example:**
# {ACTION_EXAMPLES}


# # Final instructions
# First, think carefully step by step about **mapping each assignment to atomic actions**, closely adhering to the **Instruction and Global Constraints and Navigation rules**. Then, **output only the Actions JSON with no explanations**.
# """

# errorMessage: StandardCounterHeightWidth is blocking Agent 0 from moving by (-0.2500, 0.0000, 0.0000).
#   - If missing object/observation/inventory invalid/Subtask None → empty list at that index.

# for SUMMARY
MEMORY_PROMPT = f"""
# Role and Objective
You are a lightweight memory gate for a multi-robot AI2-THOR system with {len(AGENT_NAMES)} robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}. 
Your sole job is to decide whether the **current memory and recent history** should be **carried into the next planning round**, and if so, to output a concise shared memory string plus compact success/failure highlights. 
You do **not** plan actions.

You will receive:
- The task description
- Open/closed subtasks
- Per-agent observations/states
- Recent action history and failure reasons
- a verifier summary already containing a short narrative memory

# Instructions
- Judge whether the provided memory/history materially improves **next-step planning**.
- Prefer including memory when it captures:
  - Dynamic world changes (objects moved/broken/opened; blocked paths/doorways)
  - Stable object locations or containment relations critical to the goal
  - Repeated/navigation-structural failures (e.g., no-path, distance-too-far) that change strategy
  - Tool/device states (e.g., appliance on/off, door open/closed) that alter reachable space
  - Cross-agent dependencies (handoffs, role assignments)
- Exclude noisy/ephemeral facts (one-off view lists, transient distances, obvious successes that don't change strategy).
- Object visibility is **not** a prerequisite—avoid “find/scan/explore” phrasing; prefer “NavigateTo(<object>)”.
- Be concise. If memory is carried, keep it to a **short, task-relevant** string.

# Reasoning Steps
- Internally analyze task, observations, inventories, success/failure patterns, verifier summary, and subtask progress. 
- Decide if including memory benefits replanning.
- If yes, produce a compact `common_memory` capturing only strategic, reusable facts (locations, device states, constraints, dependencies).
- Summarize **recent key failures** and **notable successes** per agent to help the planner avoid repeats.

# OUTPUT FORMAT
You must output a JSON dictionary with:
{{
  "use_in_next_plan": <true|false>,        
  "why": "<≤50 words reason>",        
  "common_memory": "<string>",     // the memory string to carry forward.
  "failure_history": {{                      // compact per-agent failures using natural language
    "<AgentName>": [""],
    ...
  }},
  "success_history": {{                      // compact per-agent successes using natural language
    "<AgentName>": [""],
    ...
  }}
}}
* why: A brief reasoning for what each robot is supposed to do next.
* common_memory: whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.


# Input Context
{MEMORY_GATE_INPUT_FORMAT}

# Notes:
- Available atomic actions: {AI2THOR_ACTIONS}
- Environment hazards: open objects may block paths; avoid mutual blocking/collisions.
- Electronics: operate devices directly; avoid remotes unless required.
- Prefer NavigateTo(<object>) unless navigation is structurally failing (no-path, distance-too-far, obstructed).
{COMMON_GUIDELINES}

# Final instructions
First, think carefully step by step about whether carrying memory helps the next plan and what minimal facts are worth persisting. 
Then, **output only the specified JSON dictionary**.
"""
def get_memory_prompt():
    return MEMORY_PROMPT

# for LOG
OLD_LOG_PROMPT = f"""
# Role and Objective
You are the Memory Aggregator within a multi-robot AI2-THOR system with {len(AGENT_NAMES)} robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}. 
Your task is to condense the full execution histories of all agents into a concise, structured summary that captures only facts relevant to future planning.

You will receive:
- The task description
- Open/closed subtasks
- Objects in the environment
- Per-agent POV images and an overhead image (use them to infer current states succinctly) 
- Per-agent FULL logs up to now

# Instructions
- Keep only what materially improves next-step planning.
- When interpreting logs, do not assume the entire subtask is complete just because one action in it succeeded. The subtask is only complete when the field "type" is "Success".
- Prefer including:
  - Dynamic world changes (on/off, open/close, place/pick, moved, blocked)
  - Stable object locations or containment relations essential to the goal
  - Structural/repeated navigation failures (no-path, blocked, distance-too-far)
  - Device/tool states that alter reachable space or task preconditions
  - Cross-agent dependencies (handoffs, role assignments)
  - Per-agent subtask progress and execution state (e.g., success, attempting or failed), especially when relevant for replanning or task reassignment.
	- Logs with type "Attempt" indicate ongoing subtasks, which may be interrupted by other agents' failures, always prioritize the latest available status.
	- The "last_action_status" field reflects only the outcome of the most recent atomic action and does not necessarily imply that the subtask is complete.

    
- Exclude noise:
  - One-off view lists, verbose inventories, generic successes with no strategic effect
  - Redundant pose details already evident from the image unless tied to a failure/success
- Be concise. Each output field must contain only essential changes, expressed in 0-3 sentences (under 50 words total). Avoid repetition or unnecessary details.

# Reasoning Steps
- Internally analyze task, observations, inventories, failures, subtask progress, and images.
- Decide the minimal facts to carry forward.
- Use images to confirm current states but avoid restating raw visuals; summarize only what helps replanning.

# OUTPUT FORMAT
Output a single JSON object with EXACTLY these keys:
{{
  "timestamp": <int>,                                   # the latest timestamp
  "environment_changes": "<0-3 sentences, <50 words>",  # persistent changes only
  "agent_action": {{                                     # one entry per agent
    "<AgentName>": "<0-3 sentences, <50 words: action history + current state + result/fail reason>",
    "...": "..."
    }}
}}

# Writing Rules
- Natural, compact English; no bullet points; no lists.
- Do not exceed 3 sentences or 50 words per field/agent entry.
- Do not invent facts. If uncertain, omit.

# Input Context
{LOG_ORIGINAL_INPUT_FORMAT}


# Final instructions
Think carefully about what minimal facts help the next plan. 
Then output ONLY the specified JSON object with the three fields and their length limits, nothing else.
"""


LOG_PROMPT = f""" 
# Role
You are the Memory Aggregator for a multi-robot AI2-THOR system with {len(AGENT_NAMES)} robots ({", ".join(AGENT_NAMES)}).  
Condense all execution logs into a factual summary that supports future planning.

# Input
You will receive:
- Task description
- Open/closed subtasks
- Objects in the environment
- Per-agent images and full logs

# Guidelines
- Keep only facts relevant for next-step planning.
- Do NOT infer completion unless a subtask’s "type" == "Success".
- Prefer dynamic world changes, stable object relations, failures, and per-agent subtask progress.
- Omit redundant inventories, view lists, or generic successes.
- Never assume all misplaced items are cleared unless explicitly verified by the object list.

# Output
Return ONLY one JSON object:
{{
  "timestamp": <int>,                                   
  "environment_changes": "<≤3 sentences, ≤50 words>", 
  "agent_action": {{                                     # one entry per agent
    "<AgentName>": "<≤3 sentences, ≤50 words describing actions + current state>",
    "...": "..."
    }}
}}

# Writing Rules
- Natural, compact English. No lists or speculation.
- ≤3 sentences, ≤50 words per field.
- Exclude uncertainty; omit unknowns."""

def get_log_prompt():
    return LOG_PROMPT

def get_planner_prompt():
    return PLANNER_PROMPT

def get_editor_prompt():
    return EDITOR_PROMPT

def get_allocator_prompt(mode='summary'):
    if mode == 'summary':
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + FAILURES_INPUT_FORMAT

    else:
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + PREVIOUS_LOG_FORMAT


    prompt = f"""
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
        {input_format}

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


    return prompt

def get_action_prompt(mode="summary"):
    base_prompt = f"""
    # Role and Objective
    You are an expert multi-robot controller, managing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}" }, by generating executable action plans to fulfill assigned subtasks. Convert each Assignment subtask into an explicit, sequential action list for the agent, based on current state and environmental details.

    # Instructions
    - Treat each assignment independently; do not invent new subtasks.
    - For each "Assignment" subtask provided (one per agent), generate an ordered list of executable atomic actions using only the defined action set and schemas.
    - Treat each subtask independently, without inferring new subtasks, changing intent, or considering inter-agent dependencies unless explicitly described in the input.
    - Respect the affordances of objects, preconditions of actions, and environmental constraints in translating subtasks into actions.
    - Assign ["Idle"] only if the subtask is "Idle". Do not assign empty lists.
    - Generate a list of action plans (lists), one per agent, in the same order as the input subtasks.
    - Navigation/Interaction:
      - Prefer NavigateTo<Object> before interaction unless object is near and in view or currently held by the agent.
      - For micro-approach, try Move/Rotate/Look first; then retry NavigateTo.
      - On "object-not-in-view": try LookDown/LookUp based on likely vertical location.
      - On "no-plan" or "no-path": move around to clear obstacles then retry.
    - Feasibility:
      - Prefer closest suitable instance if multiple exist.

    # Guidelines
    - If a referenced object is missing, inventory is full (when pickup is needed), agent state/observation is missing, or Subtask is None: output an empty list for that subtask (or {{ "Actions": [] }} if Subtasks is None).
    - Assign only objects listed in the provided list. If multiple instances of the same type exist (e.g., Countertop_1, Countertop_2), select the most appropriate one based on the agent's current position and its proximity to the objects.
    - when assigning actions which interact with objects and with navigation, always use NavigateTo<object_name> to approach the object first. Unless the target object is close enough and in the view of the agent.
    - Avoid assigning the same object to multiple agents in the same step. 
    - If there are multiple same object type, assign the most reachable one according the given observation of the agent.
    - If the subtask requires micro-movements to approach an Object_name, use only Movement, Rotation, and Look actions based on position and failure reason—avoid using NavigateTo<Object_name> initially. Try one atomic movement at a time before attempting NavigateTo<Object_name> again.
        - When failure reason is "object-not-in-view", first try LookDown or LookUp action based on the target object's most likely vertical location. (you can assume the agent always starts looking front.)
        - When failure reason is "no-plan" or "no-path", try moving around first—sometimes an obstacle (e.g., a door) may be blocking the path before re-attempting NavigateTo.
    {COMMON_GUIDELINES}

    # Context
    - Robot names: {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"}
    {AI2THOR_ACTIONS}
    """

    if mode == "summary":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT

        input_section = f"""
        # Input Context
        - Inputs include: task description, agent states/observations, open and completed subtasks, reachable positions, all objects in the environment, failed diagnostics (if any), and subtasks to be executed.
        {input_format}
        """
    else:  # mode == "log"
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + AGENT_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT

        input_section = f"""
        # Input Context
        - Inputs include: task description, agent states/observations, open and completed subtasks, all objects in the environment, failed diagnostics (if any), and subtasks to be executed.
        {input_format}
        """

    output_section = f"""
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
    return base_prompt + input_section + output_section




def get_verifier_prompt(mode: str = "summary", need_process: bool = False) -> str:
    """
    mode:
        - "summary": full diagnostic version (default)
        - "log": simplified log-based version
    need_process:
        - for mode="log" only:
            True  -> use VERIFIER_INPUT_FORMAT
            False -> use VERIFIER_ORG_LOG_INPUT_FORMAT
    """

    base_prompt = f"""
        # Role and Objective
        You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
        You will get a description of the task robots are supposed to do. You will get an image of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input.
        To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
        So, along with the image inputs you will get the following information:
        - A task description
        - A list of objects each robot can currently see (formatted as "<object_name>_<object_no>") with positions
        - Observations, failure descriptions, and subtask progress

        You need to verfify the previous failure and suggest the **next single action** that each robot should take in the current timestep.
        You must **not assume or infer missing facts**. Only evaluate based on explicitly provided data.

        # Instructions
        - Use observations, failure descriptions, memory, and progress to infer causes.
        - Do not suggest subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred. 
        - By default, object visibility is not required—always use NavigateTo(<Object>) directly when no navigation error is present.
        {COMMON_GUIDELINES}

        - Environment Hazards: open objects can block paths; avoid mutual blocking/collisions.
        - Electronics: operate directly on the device—do not use remotes unless required.
        - Use navigate to the object, unless something wrong while navigation. (no-path, distance-too-far..etc)

        # Reasoning Steps
        - Internally reason over images/observations/states/failures to isolate the cause and fix.
        - you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things.
        """

    if mode == "summary":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT

        output_section = f"""
        * Failure reason: If any robot's previous action failed, use the previous history and your understanding of causality to think and rationalize about why it failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
        * Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
        * Reason: The reasoning for what each robot is supposed to do next
        * suggestion: The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that this suggested actions make these robots more efficient in completing the task as compared only one agent solving the task.
        * need_plan: True or False. If you think the current plan is not efficient or not valid, output True. Otherwise, output False.

        #OUTPUT FORMAT
        You must output a JSON dictionary with:
        - "failure reason": string or "None"
        - "memory": string
        - "reason": string
        - "suggestion": string (e.g., "next, Alice-0 should ..., Bob-1 should ...")
        - "need_plan": <true|false>

        # Errors Handling and Examples
        {VERIFY_EXAMPLE}

        # Context
        {input_format}
        {AI2THOR_ACTIONS}

        # Final instructions
        First, think carefully step by step about the **most likely failure cause and immediate fix**, closely adhering to the **Important Notes and Common Guidelines**. Then, **output only the specified dictionary**.
        """
        return base_prompt + output_section

    elif mode == "log":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT
        input_format_org = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_ORIGINAL_INPUT_FORMAT

        output_section = f"""
        * Failure reason: Describe why the last action failed, if any. If all were successful, output "None".
        * Reason: The reasoning for what each robot is supposed to do next.
        * Suggestion: The actions the robots should take in the next step to make progress toward completing the task.
        * need_replan: True or False. If you think the current plan is not valid or efficient, output True.

        # OUTPUT FORMAT
        You must output a JSON dictionary with:
        - "need_replan": boolean (true/false)
        - "failure reason": string or "None"
        - "reason": string
        - "suggestion": string (e.g., "next, Alice-0 should ..., Bob-1 should ...")

        # Errors Handling and Examples
        {VERIFY_EXAMPLE}

        # Context
        {input_format if need_process else input_format_org}
        {AI2THOR_ACTIONS}

        # Final instructions
        First, think carefully step by step about the **most likely failure cause and immediate fix**, closely adhering to the **Important Notes and Common Guidelines**. Then, **output only the specified dictionary**.
        """
        return base_prompt + output_section

    else:
        raise ValueError(f"Unknown verifier mode: {mode}")

def get_replanner_prompt(mode: str = "summary", need_process: bool = False) -> str:
    base = f"""
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
        - Do not generate subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred. If the target object is not in visibility try RotateLeft/Right or LookUp/Down first.
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
        """
    final_block = """
    # Final instructions
    First, think carefully step by step about the **shortest valid subtask sequence** given the state, memory and failures, closely adhering to the **Common Guidelines**. Then, **return only the Subtasks JSON**.
    """
    if mode == "summary":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT
        
        context_block = f"""
        # INPUT Context
        {input_format}
        {AI2THOR_ACTIONS}
        """
    elif mode == "log":
        if need_process:
            input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + SUGGESTION_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT

            context_block = f"""
            # INPUT Context
            {input_format}
            {AI2THOR_ACTIONS}
            """
        else:
            input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + SUGGESTION_INPUT_FORMAT + LOG_ORIGINAL_INPUT_FORMAT
            context_block = f"""
            # INPUT Context
            {input_format}
            {AI2THOR_ACTIONS}
            """
    else:
        return base + final_block

    

    return base + context_block + final_block


if __name__ == "__main__":
    print("Testing prompt generation... Summary version:")
    # print(get_planner_prompt())
    # print("--------------------------------")
    # print(get_allocator_prompt())
    # print("--------------------------------")
    # print(get_action_prompt())
    # print("--------------------------------")
    # print(get_verifier_prompt())
    # print("--------------------------------")
    # print(get_replanner_prompt())
    # print("--------------------------------")
    # print(get_memory_prompt())
    # print("--------------------------------")
    
    print("Testing prompt generation... Log version:")
    # print(get_planner_prompt())
    # print("--------------------------------")
    # print(get_allocator_prompt(mode='log'))
    # print("--------------------------------")
    # print(get_action_prompt(mode='log'))
    # print("--------------------------------")
    # print(get_verifier_prompt(mode='log', need_process=False))
    # print("--------------------------------")
    # print(get_replanner_prompt(mode='log', need_process=False))
    # print("--------------------------------")
    # print(get_log_prompt())
    # print("--------------------------------")