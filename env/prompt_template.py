AGENT_NAMES_ALL = ["Alice", "Bob", "Charlie", "David", "Emma"]

NUM_AGENTS = 2
AGENT_NAMES = AGENT_NAMES_ALL[:NUM_AGENTS]

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
** If the input reports "no-path" to <Obj> while <Obj> exists and the images/observations indicate a physical occluder (e.g., an open fridge door between the agent and the target or blocked by other agent), issue a short detour macro before retrying. For example: RotateRight twice, then MoveAhead once, then retry NavigateTo(<Obj>).This decision must use multiple signals—images/2D detections, object states (e.g., isOpen for doors), recent actions, and visible objects—not just the Reachable positions list. Or navigate to somewhere far away to clean the path.
** When the input shows "no-path" and a likely cause is another agent blocking the aisle or target approach (same narrow corridor, same target, or the other agent is on the planned route), assign a yield/wait behavior to avoid deadlock: have the blocked agent Idle for 1 or 2 steps or take a small lateral/back step (MoveRight/MoveLeft/MoveBack) to clear space, or temporarily reassign the blocking agent to a different subtask/movement. After the yielding action, retry NavigateTo(<Obj>).
** when given input shows "object-not-in-view" for subtask "navigate to tomato", and based on other input information, that the distance between the agent and the tomato is already close enough, then you can have subtask "Lookdown" or "Lookup" to find the tomato or "RotateRight" or "RotateLetf".
** when when given input shows "object-not-in-view" for subtask "navigate to tomato and pick up tomato", but you can still see a tomato based on the given point of view image, then try directly pick up the tomato without navigation.
** when given input shows "a held item: <object_id> with something if agent <navigation related action: e.g., rotates Left 30 degrees>", it means that the agent is currently in a narrow space and cannot do certain navigation(often happen when rotation), you can suggest the agent to do "MoveBack" or "MoveAhead" first to get out of the narrow space.
** for any other unknown reason or repeating failures, you can suggest assigning subtasks to other agents.

- Object-related: "object-not-found", "object-not-reachable", "object-not-in-inventory", "object(<OBJ>)-not-picked-up", "object cannot be interacted":
** when given input shows "object-not-found" and based on given object list of environment, that there's no target object, you can skipped the related subtask.
** when given input shows "object-not-in-inventory", means that the current agent didn't not pick up any object to perform the next step, you should have the subtask "pick up xxx, and do xxx". xxx depends on what is the subtask.
** when given input shows "object cannot be interacted", means that the target object maybe  broken/disabled/locked, you should skip the related subtask, or try replan/choose an alternative..
** when given error shows "NullReferenceException: Target object not found within the specified visibility.", means that the target object may be inside the container and is not visiable to the agent, you should try open the container to find the target object.

- Ensure necessary object prerequisites.
"""

# Example:
# input:
# {{
#   "Task": "Place the mug and the knife into the cabinet",
#   "Number of agents": 3,
#   "Robots' open subtasks": ["put the knife in the cabinet", "close the cabinet"],
#   "Robots' completed subtasks": ["pick up the knife", "pick up the mug", "open the cabinet", "put the mug in the cabinet"],
#   "Objects in environment": ["Knife_1", "Mug_2", "Cabinet_1"],
#   "Alice's observation": ["Knife_1", "Cabinet_1"],
#   "Alice's state": "position: (1, 0.5), facing: north, inventory: [],
#   "Alice's subtask_failure_reasons": [],
#   "Alice's previous failures": [],
#   "Bob's observation": ["Mug_2", "Cabinet_1"],
#   "Bob's state": "position: (1, 0.25), facing: north, inventory: ["Mug_2"]",
#   "Bob's subtask_failure_reasons": ["Attempted NavigateTo<Cabinet_1> but failed"],
#   "Bob's previous failures": ["Alice and Charlie were blocking access to the cabinet"],
#   "Charlie's observation": ["Cabinet_1"],
#   "Charlie's state": "position: (1, 1), facing: north, inventory: []",
#   "Charlie's subtask_failure_reasons": [],
#   "Charlie's previous failures": []
# }}
# image input: (not available in this text-based interface), which shows point of view of Alice, Bob and Charlie.
# output:
# {{
#   "failure reason": "Bob failed to navigate to the cabinet because Alice and Charlie were blocking access to it while Alice was putting in the knife.",
#   "memory": "Alice has put the knife in the cabinet, and Bob has put the mug in the cabinet. The cabinet is now open.",
#   "reason": "Alice should close the cabinet and move away. Charlie should move to another open space to reduce congestion. Bob should wait until the cabinet is accessible.",
#   "suggestion": "next, Alice should move to other place, Bob should stay idle, Charlie should move to other place"
# }}


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
- "timestamp" (number): The timestamp of the action.
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
- Per-agent point of view and full logs

# Guidelines
- Keep only facts relevant for next-step planning.
- Do NOT infer completion unless a subtask's "type" == "Success".
- Prefer dynamic world changes, stable object relations, failures, and per-agent subtask progress.
- Never assume all misplaced items are cleared unless explicitly verified by the object list.
- Avoid restating raw inventories, long view lists, or generic “success” messages unless they change planning.

# Output
Return ONLY one JSON object:
{{
"timestamp": <int>,
"history": "<compact factual narrative mixing key environment changes and agent progress, suitable for future planning>"
}}

# Writing Rules
- "history" should focus on world changes, object relations, failures, and meaningful agent progress.
- No speculation or uncertainty; only describe facts present in the given information.
- Keep the content short and consice no more than 100 words.
"""
# {{
#   "timestamp": <int>,                                   # the latest timestamp
#   "environment_changes": "<0-3 sentences, <50 words>",  # persistent changes only
#   "agent_action": {{                                     # one entry per agent
#     "<AgentName>": "<0-3 sentences, <50 words: action history + current state + result/fail reason>",
#     "...": "..."
#     }}
# }}
LOG_DECEN_PROMPT = f""" 
# Role
You are the Memory Aggregator and Messenger for a multi-robot AI2-THOR system.
You operate **for one agent at a time** (“current agent”) and condense its execution logs into a factual summary that supports future replanning, and optionally produce a short message for other agents.

# Input
You will receive:
- Task description
- Objects in the environment
- This agent's past memory entries: a list of JSON objects, each with:
  {{ "timestamp": <int>, "history": "<previous factual summary>" }} or {{"timestamp": <int: the original timestamp when the message sent from other>, "received_at": <int: the timestamp received the message, it may have delay> , "history": "<message from other agent>" }}
- This agent's new execution log: a single JSON object in the action-log format:
  {{
    "timestamp": <int>,
    "agent_id": <int>,
    "agent_name": <string>,
    "curr_subtask": <string>,
    "curr_high_level_action": <string or null>,
    "type": "Attempt" | "Success" | "Failed",
    "payload": 
      "last_action": <string>,
      "last_action_status": <string, optional>,
      "failed_reason": <string, optional>,
      "postion": <string>,
      "rotation": <string>,
      "inventory": <string>,
      "observation": <string>
    
  }}

# Guidelines
- Use the **new execution log** as the primary source of new information.
- Use past memory entries only as context; do **not** rewrite or restate them.
- Keep only facts that are useful for **future planning / replanning** for this agent:
  - dynamic world changes (object moved, stored, opened/closed, cooked, sliced, broken, cleaned, etc.),
  - stable object relations (on/in/next to),
  - failures and their causes (e.g., navigation blocked, object missing),
  - meaningful progress on the current subtask.
- Do NOT infer task completion unless the log's `"type"` is `"Success"` for the relevant subtask.
- Never assume all misplaced or remaining items are handled unless explicitly supported by the environment/object list or logs.
- Avoid restating:
  - full inventories,
  - large raw observation lists,
  - generic “success” messages,
  unless they directly affect planning decisions.

# Message to Other Agents
Decide whether the new log contains information that other agents need for their own subtask planning.
Share only short, factual updates that may affect other agents' decisions, such as:
- this agent is blocked by another agent,
- an important object changed state or location,
- this agent's subtask succeeded, failed, or is being abandoned,
- this agent is switching to a different object or path.

Keep the message brief and directly useful for planning.
Do not mention delays or timestamps.
If nothing is relevant, return "".

# Output
Return ONLY one JSON object with this structure:
{{
  "log":
    {{"timestamp": "<int>  copy from the new log's timestamp" ,
    "history": "<compact factual narrative for this agent>"}}
  ,
  "message_to_others": "<short message for other agents or empty string>"
}}

# Writing Rules
- "history" should:
  - be ≤ 100 words,
  - be a concise, factual narrative emphasizing world changes, object relations, failures, and subtask progress that matter for future replanning.
- "message_to_others" should:
  - be one short sentence or a very short paragraph,
  - explicitly mention key facts that other agents should know (e.g., who is blocking whom, which object is already handled),
  - be empty string "" if no broadcast is needed.
- No speculation or uncertainty; only describe facts present in the given information.
- Do not include any explanations, comments, or extra keys outside the specified JSON format.
"""


def get_decen_planner_prompt(agent_id=0):
    cur_agent = AGENT_NAMES[agent_id]
    others = [name for i, name in enumerate(AGENT_NAMES) if i != agent_id]

    prompt = f"""# Role and Objective
    You are the planner and controller for the current agent {cur_agent} in a multi-agent environment, alongside other agents {others} who are working toward the same task.

    Your job in each call is to generate the shortest valid next subtask for this single agent only, given:
    - The high-level task specification
    - The current environment (objects and containers)
    - The current agent's state and history logs
    - Optional suggestions or failure analyses

    You do not plan for other agents; you may consider them only through the information reflected in the current agent's logs.

    Internally plan your steps but do not show any reasoning in the output.


    # Instructions (Decentralized, Single-Agent)
    You will receive a JSON user input describing:
    - "Task":  (string) A high-level description of the final goal.
    - "Objects in environment": List of objects currently available in the environment.
    - "Objects in containers": A dictionary where each key is a container object (e.g., Fridge, Drawer), and its value is a list of objects currently inside that container.
    - "Agent's state": current agent's position, facing, and inventory, and observation.
    - "Agent's log": (OPTIONAL)  a time-ordered log of this agent's action and observation history, and message from other agents in the same environment.
      - format will be either  {{ "timestamp": <int>, "history": "<previous factual summary>" }} or {{"sent_at": <int: the original timestamp when the message sent from other>, "received_at": <int: the timestamp received the message, it may have delay> , "msg": "<message from other agent>" }}
    - "Suggestion": (OPTIONAL) natural language hint about what to do next (e.g., after a failure or delayed message).


    Your job:
    - For the **current agent only**, produce a **short, atomic yet meaningful subtask description** that moves the global task closer to completion.
    - A **subtask** is a high-level instruction that can be expanded into one or several low-level actions from the Available Actions list.
    - For initial planning, history logs and suggestions may be empty or absent; you should still return a valid subtask for this agent based on the Task and environment.
    - For replanning, use the logs and suggestion to:
      - Avoid repeating actions that this agent or other agent has already successfully completed unless the environment has changed and repetition is necessary.
      - Adjust the subtask to account for failures, blocked paths, missing objects, or new information about other agents.
    - Internally interpret the Task, environment, state, logs, and Suggestion to choose the shortest valid next subtask.
    - Do **not** include your reasoning steps in the output.
    -Also, produce short messages to other agents in the form of a mapping {{"AgentName": "message"}}, which may:
      - Telling other agent that you are taking over the subtask.
      - Sharing remaining task status or completed subtasks, when such information is relevant for coordination.
      - Ask another agent to start or take over a subtask.
      - Request help or coordination.
      - Be an empty string or the entire mapping may be {{}} if no message is needed.

    {AI2THOR_ACTIONS}
    # Common Guidelines and Simulation Note
    - Always navigate directly to the target object, not to its surface or container.
    {COMMON_GUIDELINES}

    # Replanning and Failure Handling
    When logs and/or Suggestion indicate failures, delayed messages, or environment changes:
    - Prefer the **shortest corrective subtask** that gets the agent unstuck and back toward the global goal.
    - Avoid subtasks like “find”, “scan”, “explore”, or “look for” unless there is a navigation failure suggesting the target cannot be directly navigated to.
    - Reuse previous progress:
      - Do not repeat previously successful subtasks for this agent unless the environment state has changed in a way that invalidates earlier progress.
    - Use “Suggestion” as a hint, not a strict command; ensure the subtask still respects all constraints and Available Actions.


    # Output Format
    Return only a valid JSON object with this structure. 
    Do not include explanations, comments, or markdown formatting.

    "The JSON must contain the next subtask for this single agent only, plus an set of messages to other agents." 
    {{
      "Subtask": "description",
      "Messages":{ {
        "<AgentName>": "<message string or empty>"
      }}
    }}

    The subtask should be a concise natural-language description that can be expanded into low-level actions, for example:
    - "navigate to the vase, pick up the vase, navigate to the table, and put it on the table"
    - "navigate to the fridge, open the fridge, put the lettuce inside, and close the fridge"

    # Examples

    Example 1 - initial planning, no logs
    Input (informal description):
    - Task: "Put the vase, tissue box, and remote control on the table."
    - Objects in environment: [Vase_1, TissueBox_1, RemoteControl_1, Table_1]
    - Objects in containers: 
    - Agent's state: agent sees all objects, hands empty.
    - Agent's log: [] (no history; initial step)
    - Suggestion: (none)

    Possible output:
    {{
      "Subtask": "navigate to the vase, pick up the vase, navigate to the table, and put it on the table",
      "Messages": {{}}
    }}

    Example 2 - replanning with logs and suggestion
    Input (informal description):
    - Task: "Put the vase, tissue box, and remote control on the table."
    - Objects in environment: [Vase_1, TissueBox_1, RemoteControl_1, Table_1]
    - Objects in containers: {{Table_1: [Vase_1]}}
    - Agent's state: facing TissueBox_1, hands empty.
    - Agent's log: shows that the agent already placed Vase_1 on the table successfully. And agent Bob is currently holding RemoteControl_1.
    - Suggestion: "The vase is already correctly placed; focus on the tissue box next. "

    Possible output:
    {{
      "Subtask": "navigate to the tissue box, pick up the tissue box, navigate to the table, and put it on the table",
      "Messages": {{
        "Bob": "Alice starting subtask: place the tissue box. No remaining subtasks assigned to Bob; Bob may stay idle."
      }}
    }}

    Example 3 - replanning after navigation failure
    Input (informal description):
    - Task: "Put the lettuce in the fridge."
    - Objects in environment: [Lettuce_1, Fridge_1]
    - Objects in containers: {{ "Fridge_1": ["Apple_1"] }}
    - Agent's state: near Lettuce_1, some previous NavigateToFridge failed due to distance-too-far.
    - Agent's log: includes a failed navigation to Fridge_1 attempt and error reason.
    - Suggestion: "Since the Agent is holding Lettuce in hand, try rotating to adjust orientation before navigating to the fridge again."

    Possible output:
    {{
      "Subtask": 
        "rotate to face the fridge, navigate to the fridge, open the fridge, put the lettuce in the fridge, and close the fridge",
      "Messages": {{}}
    }}

    Example 4 -  coordination request due to blocking
    Input (informal description):
    - Task: "Put all the food in the fridge."
    - Objects in environment: [Lettuce_1, Bread_1, Tomato_1, Egg_1, Apple_1, Fridge_1]
    - Objects in containers: {{}}
    - Agent's state: Alice is facing the fridge while holding Lettuce_1, but Bob is standing directly in the path.
    - Agent's log: indicates Alice's previous NavigateToFridge attempts failed due to Bob blocking the approach route.
    - Suggestion: "Alice is blocked by Bob; Bob should clear the path."

    Possible output:
    {{
      "Subtask": "wait briefly and retry navigating to the fridge to place the lettuce inside",
      "Messages":{{
        "Bob": "Move aside to clear Alice's path and take over any pending food-placement subtasks when free."
      }}
    }}

    # Final Instructions
    Carefully consider the Task, environment, agent's state, logs, and optional Suggestion.  
    Then output **only** the required JSON object under the specified "Subtasks" format, containing the **next subtask sequence for this single agent**. No additional text, explanations, or formatting.

    """

    return prompt

def get_log_prompt():
    return LOG_PROMPT

def get_decen_log_prompt():
    return LOG_DECEN_PROMPT


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
    if mode == 'decen':
      first_line = "You are an expert robot controller, managing a single embodied robots by generating executable action plans to fulfill assigned subtasks. Convert the assigned subtask into an explicit, sequential action list for the agent, based on current state and environmental details."
    else:
      first_line = f"""You are an expert multi-robot controller, managing {len(AGENT_NAMES)} embodied robots, {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}" }, by generating executable action plans to fulfill assigned subtasks. Convert each Assignment subtask into an explicit, sequential action list for the agent, based on current state and environmental details."""
        
    base_prompt = f"""
    # Role and Objective
    {first_line}

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
    {AI2THOR_ACTIONS}
    """

    if mode == "summary":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + REACABLE_POSITION_INPUT_FORMAT + AGENT_INPUT_FORMAT + MEMORY_HISTORY_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT

        input_section = f"""
        # Input Context
        - Inputs include: task description, agent states/observations, open and completed subtasks, reachable positions, all objects in the environment, failed diagnostics (if any), and subtasks to be executed.
        {input_format}
        """
    elif mode == "log":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + AGENT_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT

        input_section = f"""
        # Input Context
        - Inputs include: task description, agent states/observations, open and completed subtasks, all objects in the environment, failed diagnostics (if any), and subtasks to be executed.
        {input_format}
        """
    else:
        # mode == 'decen'
         input_section = f"""
        # Input Context
          - Inputs include: task description, agent states/observations, all objects in the environment, and subtasks to be executed.
          {BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + AGENT_INPUT_FORMAT + ASSINMENG_INPUT_FORMAT}
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
    base_prompt = f"""# Role and Objective
        You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. 
        You will get a description of the task robots are supposed to do. You will get an object list of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input.
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

    elif mode == "log":
        input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_PROCESSED_INPUT_FORMAT
        if not need_process:
          input_format = BASE_INPUT_FORMAT + TASK_INPUT_FORMAT + ROBOTS_SUBTASKS_INPUT_FORMAT + AGENT_INPUT_FORMAT + LOG_ORIGINAL_INPUT_FORMAT

    output_section = f"""
        # OUTPUT FORMAT
        You must output a JSON dictionary with:
        - "need_replan": boolean (true/false), If you think the current plan is not valid or efficient, output True.
        - "failure reason": string or "None", Describe why the last action failed, if any. If all were successful, output "None".
        - "reason": string, The reasoning for what each robot is supposed to do next.
        - "suggestion": string (e.g., "next, Alice-0 should ..., Bob-1 should ...") The actions the robots should take in the next step to make progress toward completing the task. keep it short and concise.   

        {VERIFY_EXAMPLE}

        # Context
        {input_format}

        # Final instructions
        First, think carefully step by step about the **most likely failure cause and immediate fix**, closely adhering to the **Important Notes and Common Guidelines**. Then, **output only the specified dictionary**.
        """
    return base_prompt + output_section

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


def get_decen_verifier_prompt():
    prompt = f"""
    # Role and Objective
    You are an excellent planner and robot controller tasked with helping a **single embodied robot (the current agent)** carry out a task in a multi-agent AI2-THOR environment.  
    The current agent has a partially observable view of the environment and may be affected by other agents’ behavior (e.g., blocking paths or competing for the same object).

    You will get:
    - A description of the overall task.
    - The current agent's state.
    - A **log history** for this agent, each entry containing a timestamp and a compact factual history string.

    Your goal is to:
    - Verify and interpret the most recent failures or issues for this agent based on its log history and state.
    - Suggest a **concise, actionable recommendation ("suggestion")** that will help guide the next replanning step for this agent.

    You must **not assume or infer missing facts**. Only evaluate based on explicitly provided data.

    # Instructions
    - Use observations, failure descriptions, memory/log history, and progress to infer causes.
    - Do not suggest subtasks like “find”, “scan”, “explore”, or “look for” unless a navigation failure has occurred. 
    - By default, object visibility is not required — always use NavigateTo(<Object>) directly when no navigation error is present.
    {COMMON_GUIDELINES}
    - Environment Hazards: open objects can block paths; avoid mutual blocking/collisions.
    - Electronics: operate directly on the device—do not use remotes unless required.
    - Use navigate to the object, unless something wrong while navigation. (no-path, distance-too-far..etc)
    - If the best decision is for the current agent to remain idle / keep yielding / keep its current behavior and **no new subtask is needed**, set `"need_replan": false` and give a short suggestion like “remain idle near the fridge to yield space”.
    - If the log suggests that this agent could **usefully change its behavior** (e.g., try a different motion pattern, take over a remaining object, or help complete a task another agent is struggling with), set `"need_replan": true` and describe that adjustment in "suggestion".

    # Reasoning Steps
    - Internally reason over:
      - the task description,
      - the current agent's state,
      - the agent's log history (timestamp + history strings),
      to isolate the most likely failure cause(s) and the most helpful corrective direction.
    - You are supposed to reason over the agent's previous actions, previous failures, previous memory/logs, subtasks, and the available actions the agent can perform, and think step by step.
    - When the agent is free or stuck in its current subtask, also consider whether it should yield, stay idle, or take over a remaining goal that another agent may be struggling to complete.
    - Then output a **single textual "suggestion"** that will guide the next replanning step for this agent.

    # OUTPUT FORMAT
    You must output a JSON dictionary with:
    - "need_replan": boolean.  
      - Use `false` when the agent should just keep its current plan or remain idle/yield (no new subtask or plan change is needed).  
      - Use `true` when the agent's plan should be updated (e.g., change navigation strategy, retry with a different motion, skip this subtask, or take over a remaining task from another agent).
    - "suggestion": string.  
      A concise, actionable recommendation for what this **single agent** should do or how its plan should be adjusted next (for example: change navigation strategy, wait/yield because of blocking, choose another object instance, skip the subtask, or hand the responsibility to another agent).
    The suggestion should be written so that a planner/replanner module can use it as a hint to update subtasks and actions for this agent.

    {VERIFY_EXAMPLE}

    # Input Context
    You will receive a JSON object with these fields:
    - "Task":  (string) A high-level description of the final goal.
    - "Agent's state": Current agent's position, facing, and inventory. (string or structured description)
    - "Log": a list of JSON objects for **this agent only**, each with:
      - "timestamp": (number) the time of the summarized history entry.
      - "history": (string) a compact factual description of what happened and how the world changed, including any important information received about other agents, suitable for planning.

    # Final instructions
    First, think carefully step by step about the **most likely failure cause and the most helpful next adjustment** for this agent, closely adhering to the **Important Notes and Common Guidelines**.  
    Then, **output only the specified dictionary with the single "suggestion" field**.
    """
    return prompt

if __name__ == "__main__":
    # print("Testing prompt generation... Summary version:")
    # print(get_planner_prompt()) # 1500+ token
    # print("--------------------------------")
    # print(get_allocator_prompt()) # 1400+ token
    # print("--------------------------------")
    # print(get_action_prompt()) # 2200+ token
    # print("--------------------------------")
    # print(get_verifier_prompt()) # 2500+ token
    # print("--------------------------------")
    # print(get_replanner_prompt()) # 1800+ token
    # print("--------------------------------")
    # print(get_memory_prompt()) # 1800+ token
    # print("--------------------------------")
    
    # print("Testing prompt generation... Log version:")
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
    # print(get_log_prompt()) # 250+ token
    # print("--------------------------------")

    # print("Testing Decentralized planner prompt generation...")
    print(get_decen_planner_prompt()) # 2000+ token
    # print(get_decen_log_prompt())