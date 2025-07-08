'''
Baseline : LlaMAR

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

# planner
# v0 - add additional example of cleaning in plans - avoiding extraneous actions

PLANNER_OBS_STR = ",\n".join(
    [
        f"{name}'s observation: list of objects the {name} is observing"
        for name in AGENT_NAMES
    ]
)

PLANNER_PROMPT = f"""You are an excellent planner who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input. To help you with detecting objects in the image, you will also get a list of objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{PLANNER_OBS_STR}
Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' combined memory: description of robots' combined memory}}

Reason over the robots' task, image inputs, observations, open subtasks, completed subtasks and memory, and then output the following:
* Reason: The reason for why new subtasks need to be added.
* Subtasks: A list of open subtasks the robots are supposed to take to complete the task. Remember, as you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed.
The "Plan" should be in a list format where the actions are listed sequentially.
For example:
    ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
    ["locate the cup", "go to cup", "clean cup"]
When possible do not perform additional steps when one is sufficient (e.g. CleanObject is sufficient to clean an object, no other actions need to be done)
Your output should be in the form of a python dictionary as shown below.
Example output: {{"reason": "since the subtask list is empty, the robots need to transport the apple to the fridge and transport the book to the table", "plan": ["transport the apple to the fridge", "transport the book to the table"]}}

Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand.
Do not assign subtasks to any particular robot. Try not to modify the subtasks that already exist in the open subtasks list. Rather add new subtasks to the list.
Since there are {len(AGENT_NAMES)} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
Let's work this out in a step by step way to be sure we have the right answer."""

# verifier
# moves subtasks from open to completed
# v0 - add statement to encourage it append rather than overwrite completed subtasks

VERIFIER_OBS_STR = ",\n".join(
    [
        f"{name}'s observation: list of objects the {name} is observing,\n{name}'s state: description of {name}'s state,\n{name}'s previous action: the action {name} took in the previous step,"
        for name in AGENT_NAMES
    ]
)

VERIFIER_PROMPT = f"""You are an excellent planner who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. Both robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input. To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{VERIFIER_OBS_STR}
Robots' open subtasks: list of open subtasks the robots in the previous step. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' combined memory: description of robots' combined memory}}

Reason over the robots' task, image inputs, observations, previous actions, open subtasks, completed subtasks and memory, and then output the following:
* Reason: The reason for why you think a particular subtask should be moved from the open subtasks list to the completed subtasks list.
* Completed Subtasks: The list of subtasks that have been completed by the robots. Note that you can add subtasks to this list only if they have been successfully completed and were in the open subtask list. If no subtasks have been completed at the current step, return an empty list.
The "Completed Subtasks" should be in a list format where the completed subtasks are listed. For example: ["locate the apple", "transport the apple to the fridge", "transport the book to the table"]
Your output should be in the form of a python dictionary as shown below.
Example output with two agents (do it for {len(AGENT_NAMES)} agents): {{"reason": "{AGENT_NAMES[0]} placed the apple in the fridge in the previous step and was successful and {AGENT_NAMES[1]} picked up the the book from the table. Hence {AGENT_NAMES[0]} has completed the subtask of transporting the apple to the fridge, {AGENT_NAMES[1]} has picked up the book, but {AGENT_NAMES[1]} has still not completed the subtask of transporting the book to the table", "completed subtasks": ["picked up book from the table", "transport the apple to the fridge"]}}

### Important Notes ###
* Since there are {len(AGENT_NAMES)} agents moving around in one room, make sure to plan subtasks and actions so that they don't block or bump into each other as much as possible. This is especially important because there are so many agents in one room.
* DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED

When you output the completed subtasks, make sure to not forget to include the previous ones in addition to the new ones.
Let's work this out in a step by step way to be sure we have the right answer.
"""

FAILURE_REASON = """
If any robot's previous action failed, use the previous history and your understanding of causality to think and rationalize about why it failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object and must explore to find, agent is doing the same redundant actions, etc.
"""

FAILURE_REASON = """
If any robot's previous action failed, use the previous history, your current knowledge of the room (i.e. what things are where), and your understanding of causality to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
Common failure reasons to lookout for include: one agent blocking another so must move out of the way, agent can't see an object or its destination and must explore (such as move, rotate, or look in a different direction) to find it, agent doing extraneous actions (such as drying objects when cleaning), etc. If the previous action was successful, output "None".
"""

action_wrapper = lambda name, action: f'"{name}\'s action" : "{action}"'

action_agents_1 = [
    "close the Cabinet_1",
    "stay idle",
    "move to the right",
    "pick up the plate",
    "slice the orange",
]
ACTION_1 = ",\n".join(
    action_wrapper(AGENT_NAMES[i], action_agents_1[i]) for i in range(len(AGENT_NAMES))
)

action_agents_2 = [
    "clean the dish object",
    "navigate to the cup",
    "slice the pineapple",
    "open the drawer",
    "pick up the pencil",
]
ACTION_2 = ",\n".join(
    action_wrapper(AGENT_NAMES[i], action_agents_2[i]) for i in range(len(AGENT_NAMES))
)


# ----- example 1 (failure) ------
# "failure reason" - 1
and_charlie = f"and {AGENT_NAMES_ALL[2]}" if len(AGENT_NAMES) > 2 else ""
charlie_fix = (
    f", {AGENT_NAMES_ALL[2]} should move away to a different open area than {AGENT_NAMES_ALL[0]} to avoid congestion,"
    if len(AGENT_NAMES_ALL) > 2
    else ""
)
other_agents = (
    f" The other agents didn't affect the failure." if len(AGENT_NAMES) > 3 else ""
)
FAILURE_REASON_EX_1 = f"{AGENT_NAMES_ALL[1]} failed to put the mug in the cabinet earlier because {AGENT_NAMES_ALL[0]} {and_charlie} were blocking it when she was putting the knife. To fix this, {AGENT_NAMES_ALL[0]} should close the cabinet and move away {charlie_fix} and {AGENT_NAMES_ALL[1]} should wait until the next timestep until {AGENT_NAMES_ALL[0]} {and_charlie} can move aside.{other_agents}"

# "memory" - 1
memory_list_1 = [
    f"{AGENT_NAMES_ALL[0]} finished putting the knife in the cabinet when {AGENT_NAMES_ALL[0]} was at co-ordinates (1, .5) and was facing north.",
    f"{AGENT_NAMES_ALL[1]} wanted to put the mug in the cabinet when {AGENT_NAMES_ALL[1]} was at co-ordinates (1, 0.25) and was facing north.",
    f"{AGENT_NAMES_ALL[2]} finished navigating to the cabinet when {AGENT_NAMES_ALL[2]} was at coordinates (1, 1) and was facing north.",
    f"{AGENT_NAMES_ALL[3]} finished navigating to the plate when {AGENT_NAMES_ALL[3]} was at co-ordinates (-1, 0) and was facing east.",
    f"{AGENT_NAMES_ALL[4]} finished navigating to the orange when {AGENT_NAMES_ALL[4]} was at co-ordinates (-1, -1) and was facing west.",
]
MEMORY_EX_1 = " ".join(memory_list_1[: len(AGENT_NAMES)])

# "reason" - 1
REASON_EX_1 = (
    f"{AGENT_NAMES_ALL[0]} can close the cabinet door and then later back out in order help {AGENT_NAMES_ALL[1]} with completing the task. {AGENT_NAMES_ALL[1]} can be idle until the next timestep when {AGENT_NAMES_ALL[0]} moves aside, by then {AGENT_NAMES_ALL[1]} can navigate to the cabinet."
    + " ".join(
        [
            f" {AGENT_NAMES_ALL[2]} can move away to an open space that isn't blocking any agent to not interfere with any of the other agents.",
            f"{AGENT_NAMES_ALL[3]} can pick up the plate to progress towards completing the task since he has navigated to it.",
            f"{AGENT_NAMES_ALL[4]} can slice the orange to progress towards completing the task since she has navigated to it.",
        ][: len(AGENT_NAMES) - 2]
    )
)

# "subtask" - 1
subtask_list_1 = [
    f"{AGENT_NAMES_ALL[0]} is currently closing the cabinet door,",
    f"{AGENT_NAMES_ALL[1]} is currently waiting to get to navigate to the cabinet,",
    f"{AGENT_NAMES_ALL[2]} is currenly moving to the right,",
    f"{AGENT_NAMES_ALL[3]} is currently picking up the plate,",
    f"{AGENT_NAMES_ALL[4]} is currently slicing the orange",
]
SUBTASK_EX_1 = " ".join(subtask_list_1[: len(AGENT_NAMES)])


# ----- example 2 (failure) -----
# "failure reason" - 1
other_agents = (
    f" The other agents didn't affect the failure." if len(AGENT_NAMES) > 2 else ""
)
FAILURE_REASON_EX_2 = f"{AGENT_NAMES_ALL[1]} failed to clean the cup earlier because {AGENT_NAMES_ALL[1]} had not navigated to it, {AGENT_NAMES_ALL[1]} assumed the cup to be in the sink which was erroneous. To fix this, {AGENT_NAMES_ALL[1]} should navigate to the cup and in the next step clean cup.{other_agents}"

# "memory" - 2
memory_list_2 = [
    f"{AGENT_NAMES_ALL[0]} finished navigating to the dish when {AGENT_NAMES_ALL[0]} was at co-ordinates (-.5, .5) and was facing east.",
    f"{AGENT_NAMES_ALL[1]} was not able to clean the cup  in the cabinet when {AGENT_NAMES_ALL[1]} was at co-ordinates (1, .25) and was facing north.",
    f"{AGENT_NAMES_ALL[2]} finished navigating to the pineapple when {AGENT_NAMES_ALL[2]} was at co-ordinates (.5, -.5) and was facing west.",
    f"{AGENT_NAMES_ALL[3]} finished navigating to the drawer when {AGENT_NAMES_ALL[3]} was at co-ordinates (1, 2) and was facing west.",
    f"{AGENT_NAMES_ALL[4]} finished navigating to the couch when {AGENT_NAMES_ALL[4]} was at co-ordinates (1, -1) and was facing south.",
]
MEMORY_EX_2 = " ".join(memory_list_2[: len(AGENT_NAMES)])

# "reason" - 2
REASON_EX_2 = (
    f"{AGENT_NAMES_ALL[0]} can now clean the dish since {AGENT_NAMES_ALL[0]} has navigated to it. {AGENT_NAMES_ALL[1]} should navigate to the cup in order to be close enough to clean the cup."
    + " ".join(
        [
            f" {AGENT_NAMES_ALL[2]} can now slice the pineapple since {AGENT_NAMES_ALL[2]} has navigated to it.",
            f"{AGENT_NAMES_ALL[3]} can now open the drawer since {AGENT_NAMES_ALL[3]} has navigated to it.",
            f"{AGENT_NAMES_ALL[4]} can now pick up the pencil since {AGENT_NAMES_ALL[4]} has navigated to it.",
        ][: len(AGENT_NAMES) - 2]
    )
)

# "subtask" - 2
subtask_list_2 = [
    f"{AGENT_NAMES_ALL[0]} is currently trying to clean the dish,",
    f"{AGENT_NAMES_ALL[1]} is currently trying to navigate to the cup,",
    f"{AGENT_NAMES_ALL[2]} is currenly trying to slice the pineapple,",
    f"{AGENT_NAMES_ALL[3]} is currently opening the drawer,",
    f"{AGENT_NAMES_ALL[4]} is currently picking up the pencil",
]
SUBTASK_EX_2 = " ".join(subtask_list_2[: len(AGENT_NAMES)])

# -- construct failure example from this ---
FAILURE_EXAMPLE = f"""
Example 1:
{{
"failure reason": "{FAILURE_REASON_EX_1}",
"memory": "{MEMORY_EX_1}",
"reason": "{REASON_EX_1}",
"subtask": "{SUBTASK_EX_1}",
{ACTION_1}
}}

Example 2:
{{
"failure reason": "{FAILURE_REASON_EX_2}",
"memory": "{MEMORY_EX_2}",
"reason": "{REASON_EX_2}",
"subtask": "{SUBTASK_EX_2}",
{ACTION_2}
}}
"""

ACTION_OBS_STR = ", ".join(
    [
        f"{name}'s observation: list of objects the {name} is observing,\n{name}'s state: description of {name}'s state,\n{name}'s previous action: description of what {name} did in the previous time step and whether it was successful,\n{name}'s previous failures: if {name}'s few previous actions failed, description of what failed,"
        for name in AGENT_NAMES
    ]
)

# action planner
# v0 - change done part of the prompt (not finishing)
# v1 - add to "excellent planner" phrase "and robot controller"
ACTION_PROMPT = f"""You are an excellent planner and robot controller who is tasked with helping {len(AGENT_NAMES)} embodied robots named {", ".join(AGENT_NAMES[:-1]) + f", and {AGENT_NAMES[-1]}"} carry out a task. All {len(AGENT_NAMES)} robots have a partially observable view of the environment. Hence they have to explore around in the environment to do the task.
They can perform the following actions: ["navigate to object <object_id>", "rotate in <rotation> direction", "pick up object <object_id>", "put object on <receptacle_id>", "open object <object_id>", "close object <object_id>", "slice object <object_id>", “toggle object <object_id> on”, “toggle object <object_id> off”, "clean object <object_id>", "look up by angle <angle>", "look down by angle <angle>", “move in <translation> direction", "stay idle", "Done"]
Here "Done" is used when all the robots have completed the main task. Make sure to always say "Done" when you finish with the task.
"stay idle" is used when you want the robot to stay idle for one time step. This could be used to wait for the other robot to complete its subtask. Use it only when you think it is necessary.
Here <rotation> can be one of ["Right", "Left"].
Here <angle> is the angle in degrees and can only be one of [30, 60, 90, 120, 150, 180].
Here <translation> can be one of ["Ahead", "Back", "Left", "Right”].

You need to suggest the action that each robot should take at the current time step.

You will get a description of the task robots are supposed to do. You will get an image of the environment from {", ".join([f"{name}'s perspective" for name in AGENT_NAMES[:-1]]) + f", and {AGENT_NAMES[-1]}'s perspective"} as the observation input.
To help you with detecting objects in the image, you will also get a list objects each agent is able to see in the environment. Here the objects are named as "<object_name>_<object_id>".
So, along with the image inputs you will get the following information:
### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
{ACTION_OBS_STR}
Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
Robots' combined memory: description of robot's combined memory}}

First of all you are supposed to reason over the image inputs, the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
* Failure reason: {FAILURE_REASON}
* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. So, you should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
* Reason: The reasoning for what each robot is supposed to do next
* Subtask: The subtask each robot should currently try to solve, choose this from the list of open subtasks.
* Action: The actions the robots are supposed to take just in the next step such that they make progress towards completing the task. Make sure that this suggested actions make these robots more efficient in completing the task as compared only one agent solving the task.
Your output should just be in the form of a python dictionary as shown below.
Examples of output:
{FAILURE_EXAMPLE}
Note that the output should just be a dictionary similar to the example outputs.

### Important Notes ###
* The robots can hold only one object at a time.
For example: If {AGENT_NAMES[0]} is holding an apple, it cannot pick up another object until it puts the apple down.
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


def get_llm_response(env: AI2ThorEnv, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    pass

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
        system_prompt = "VERIFIER_PROMPT"
        user_prompt = convert_dict_to_string(env.get_verifier_llm_input())
    user_prompt += addendum
    return system_prompt, user_prompt

def prepare_payload():
    pass    

def process_action_llm_output():
    pass

def print_relevant_info():
    pass


def run_main():
    
    env, config = set_env_with_config(args.config_file)
    env.reset(test_case_id="1")
    print(f"Environment set with config: {config}")
    print(f"Number of agents: {env.num_agents}")
    action_prompt, action_user_prompt = prepare_prompt(env, mode="action")
    print(f"Action Prompt: {action_prompt}")
    print(f"Action User Prompt: {action_user_prompt}")
    planner_prompt, planner_user_prompt = prepare_prompt(env, mode="planner")
    print(f"Planner Prompt: {planner_prompt}")
    print(f"Planner User Prompt: {planner_user_prompt}")
    verifier_prompt, verifier_user_prompt = prepare_prompt(env, mode="verifier")
    print(f"Verifier Prompt: {verifier_prompt}")
    print(f"Verifier User Prompt: {verifier_user_prompt}")
    env.close()
    

if __name__ == "__main__":
    run_main()