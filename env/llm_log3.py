'''
Baseline : Centralized LLM + replanning + shared memory(log based) llm decide whether to append the log or not


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
import difflib

import ai2thor
from openai import OpenAI
from env_log import AI2ThorEnv_cen as AI2ThorEnv
from prompt_template import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

client = OpenAI(api_key=Path('api_key_ucsb.txt').read_text())

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="config/config.json")
    args = parser.parse_args()
    return args

# args = get_args()
test_config_path = "config/config.json"
AGENT_NAMES_ALL = ["Alice", "Bob", "Charlie", "David", "Emma"]
with open(test_config_path, "r") as f:
        config = json.load(f)
        NUM_AGENTS = config["num_agents"]
AGENT_NAMES = AGENT_NAMES_ALL[:NUM_AGENTS]




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



def get_llm_response(payload, model="gpt-4.1", temperature=0.7, max_tokens=2048, max_retries=5) -> str:
    attempt = 0
    while attempt < max_retries:
        try:
            if model.startswith("gpt-4"):
                response = client.chat.completions.create(
                    model=model,
                    messages=payload,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=payload,
                    max_completion_tokens=max_tokens,
                )
            return response, response.choices[0].message.content.strip()

        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower() or "429" in msg:
                wait_time = 10
                if "Please try again in" in msg:
                    try:
                        wait_time = float(msg.split("in ")[1].split("s")[0])
                    except:
                        pass
                print(f"[RateLimit] {msg} -> sleep {wait_time:.2f}s then retry...")
                time.sleep(wait_time)
                attempt += 1
                continue
            else:
                raise e

    raise RuntimeError(f"Failed after {max_retries} retries due to repeated rate limits.")



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
        system_prompt = get_planner_prompt()
        input = env.get_center_llm_input()
        user_prompt = convert_dict_to_string(input)
        # print("planner input:", user_prompt)
    elif mode == "editor":
        # for editing the generated plan
        system_prompt = get_editor_prompt()
        input = env.get_center_llm_input()
        input["Subtasks"] = subtasks
        user_prompt = convert_dict_to_string(input)
    elif mode == "action":
        # for generating automic actions for each robot to perform
        system_prompt = get_action_prompt(mode='log')
        if not subtasks:
            print("No subtasks provided")
            return None, None
        input = env.get_obs_llm_input(recent_logs=True)
        input["Subtasks"] = subtasks
        del input["Robots' open subtasks"]
        del input["Robots' completed subtasks"]
        user_prompt = convert_dict_to_string(input)

    elif mode == "allocator":
        # for allocating subtasks to robots
        system_prompt = get_allocator_prompt(mode='log')
        input = env.get_obs_llm_input(recent_logs=True)
        user_prompt = convert_dict_to_string(input)

    elif mode == "replan":
        # for replanning the subtasks based on the current state of the environment

        system_prompt = get_replanner_prompt(mode='log', need_process=need_process)
        input = env.get_llm_log_input(need_process)
        input['Suggestion'] = info.get('suggestion', '')
        input['Reason'] = info.get('reason', '')
        
        user_prompt = convert_dict_to_string(input)
        # print("replan prompt:", system_prompt)
        # print("replan use prompt:", user_prompt)
    elif mode == "verifier":
        # for verifying the actions taken by the robots

        system_prompt = get_verifier_prompt(mode='log', need_process=need_process)
        input = env.get_llm_log_input(need_process)

        user_prompt = convert_dict_to_string(input)
        user_prompt = json.dumps(input, ensure_ascii=False)
    elif mode == "log":
        system_prompt = get_log_prompt()
        input = env.get_llm_log_input(need_process, mode='agent') # use agent mode to get full logs of each agent
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
            _get(data, "failure_reason"),
            _get(data, "reason"),
            _get(data, "suggestion"),
        )
    if mode == 'log':
        return data

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
    need_replan, failure_reason, reason, suggestion = process_llm_output(res_content, mode="verifier")
    verify_res = {
        "need_replan": need_replan,
        "failure_reason": failure_reason,
        "reason": reason,
        "suggestion": suggestion
    }
    return verify_res

def log_summariser(env):
    log_prompt, log_user_prompt = prepare_prompt(env, mode="log")
    # print("log system prompt: ", log_prompt)
    # print("log user prompt: ", log_user_prompt)
    base64_image = [encode_image(env.get_frame(i)) for i in range(len(AGENT_NAMES))]
    image_urls = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        for image in base64_image
    ]
    log_payload = prepare_payload(log_prompt, log_user_prompt, img_urls=image_urls)
    res, res_content = get_llm_response(log_payload, model=config['model'])
    # print('log summariser llm output', res_content)
    summarized_log = process_llm_output(res_content, "log")
    return summarized_log

def get_steps_by_actions(env, actions):
    print("get_steps_by_actions: ", actions)
    steps = env.actions_decomp(actions)
    return steps



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

def set_env_with_config(controller,config_file: str):
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
    env = AI2ThorEnv(controller, config_file)
    with open(config_file, "r") as f:
        config = json.load(f)
    return env,config


def run_main(controller, test_id = 0, config_path="config/config.json", delete_frames=False, timeout=250):
    # --- Init.
    env, config = set_env_with_config(controller, config_path)
    timeout = timeout
    if test_id > 0:
        obs = env.reset(test_case_id=test_id)
    else:
        obs = env.reset(test_case_id=config['test_id'])
    # --- initial subtask planning
    print("\n--- Initial Subtask Planning ---")
    open_subtasks, completed_subtasks = initial_subtask_planning(env, config)
    info = {}
    need_process = False

    # return 
    # --- loop start
    cnt = 0
    start_time = time.time()
    logs = []
    filename = env.base_path / "logs_llm.txt"
    while open_subtasks:
        if time.time() - start_time > timeout:
            print("Timeout reached, ending loop.")
            logs.append(f"""Timeout ({timeout} second) reached, ending loop.""")
            break
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

        # 6. log summariser
        summarized_log = log_summariser(env)
        # print("summarized_log: ", summarized_log)
        env.save_log_result(summarized_log) # append to event.json
        logs.append(f"saved log: {summarized_log}")


        # 7. verify the execution 
        logs.append(f"----replanning subtasks to agents----")
        verify_res = verify_actions(env, info, need_process)
        print("verify result: ", verify_res)
        logs.append(f"verify result: {verify_res}")

        # 8. replan if needed
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

    env.save_to_video(delete_frames=delete_frames)
    # env.close()



def batch_run(tasks, base_dir="config", start = 1, end=5, sleep_after=2.0, delete_frames=False, timeout=250):
    """
    tasks: e.g. TASKS_1, TASKS_2 ...
    base_dir: the root config folder
    repeat: how many times each config runs
    sleep_after: seconds to sleep between runs
    """
    with ai2thor.controller.Controller(width=1000, height=1000, gridSize=0.25) as controller:

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
                    run_main(controller, test_id = r, config_path=str(cfg_path), delete_frames=delete_frames, timeout=timeout)
                    time.sleep(sleep_after)

                print(f"==== Finished {cfg_path} ====")


if __name__ == "__main__":
    TASKS_1 = [
    {
        "task_folder": "1_put_bread_lettuce_tomato_fridge",
        "task": "put bread, lettuce, and tomato in the fridge",
        "scenes": ["FloorPlan1"] # "FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    },
    # {
    #     "task_folder": "1_put_computer_book_remotecontrol_sofa",
    #     "task": "put laptop, book and remote control on the sofa",
    #     "scenes": ["FloorPlan201"] #,"FloorPlan201", "FloorPlan202""FloorPlan203", "FloorPlan209", "FloorPlan224"
    # },
    {
        "task_folder": "1_put_knife_bowl_mug_countertop",
        "task": "put knife, bowl, and mug on the counter top",
        "scenes": [ "FloorPlan5"] #"FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    },
    # {
    #     "task_folder": "1_put_plate_mug_bowl_fridge",
    #     "task": "put plate, mug, and bowl in the fridge",
    #     "scenes": [ "FloorPlan4"] #"FloorPlan1", "FloorPlan2",,"FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_put_remotecontrol_keys_watch_box",
    #     "task": "put remote control, keys, and watch in the box",
    #     "scenes": ["FloorPlan201"] # "FloorPlan201", "FloorPlan202", "FloorPlan203", ,"FloorPlan209", "FloorPlan215", "FloorPlan226", "FloorPlan228", "FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan207"
    # },
    {
        "task_folder": "1_put_vase_tissuebox_remotecontrol_table",
        "task": "put vase, tissue box, and remote control on the side table1",
        "scenes": [ "FloorPlan201"] # "FloorPlan201", "FloorPlan219", "FloorPlan203", "FloorPlan216", "FloorPlan219"
    },
   
    # {
    #     "task_folder": "1_slice_bread_lettuce_tomato_egg",
    #     "task": "slice bread, lettuce, tomato, and egg with knife",
    #     "scenes": [ "FloorPlan4"] #"FloorPlan1", , "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_turn_off_faucet_light",
    #     "task": "turn off the sink faucet and turn off the light switch",
    #     "scenes": ["FloorPlan5"] #"FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_wash_bowl_mug_pot_pan",
    #     "task": "clean the bowl, mug, pot, and pan",
    #     "scenes": [ "FloorPlan3"] #"FloorPlan3","FloorPlan1",  "FloorPlan2", "FloorPlan4", "FloorPlan5"
    # },
]


    TASKS_2 = [
        # {
        #     "task_folder": "2_open_all_cabinets",
        #     "task": "open all the cabinets",
        #     "scenes": [  "FloorPlan7", "FloorPlan8", "FloorPlan10"]# "FloorPlan1", "FloorPlan6", "FloorPlan7", "FloorPlan8",
        # },
        # {
        #     "task_folder": "2_open_all_drawers",
        #     "task": "open all the drawers",
        #     "scenes": [ "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9"] # "FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5",
        # },
        # {
        #     "task_folder": "2_put_all_creditcards_remotecontrols_box",
        #     "task": "put all credit cards and remote controls in the box",
        #     "scenes": ["FloorPlan202","FloorPlan203","FloorPlan204", "FloorPlan205"] #"FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan204", "FloorPlan205"
        # },
        # {
        #     "task_folder": "2_put_all_vases_countertop",
        #     "task": "put all the vases on the counter top",
        #     "scenes": ["FloorPlan1", "FloorPlan5"]
        # },
        # {
        #     "task_folder": "2_put_all_tomatoes_potatoes_fridge",
        #     "task": "put all tomatoes and potatoes in the fridge",
        #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"] #"
        # },
        
        # {
        #     "task_folder": "2_turn_on_all_stove_knobs",
        #     "task": "turn on all the stove knobs",
        #     "scenes": [ "FloorPlan7", "FloorPlan8"] #"FloorPlan1", "FloorPlan2","FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan7", "FloorPlan8"
        # },  
    ]

    TASKS_3 = [
        # {
        #     "task_folder": "3_clear_table_to_sofa",
        #     "task": "Put all readable objects on the sofa",
        #     "scenes": ["FloorPlan201", "FloorPlan203", "FloorPlan204", "FloorPlan208", "FloorPlan223"]
        # },
        # {
        #     "task_folder": "3_put_all_food_countertop",
        #     "task": "Put all food on the countertop",
        #     "scenes": [ "FloorPlan1"] #  "FloorPlan1", "FloorPlan2", "FloorPlan3","FloorPlan4","FloorPlan5"
        # },
        # {
        #     "task_folder": "3_put_all_groceries_fridge",
        #     "task": "Put all groceries in the fridge",
        #     "scenes": [ "FloorPlan3", "FloorPlan4", "FloorPlan5"] #"FloorPlan1", "FloorPlan2",
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
        #     "scenes": [ "FloorPlan4", "FloorPlan5"] #"FloorPlan1", "FloorPlan2", "FloorPlan3",
        # },  
        # {
        #     "task_folder": "3_put_all_shakers_tomato", # on countertop
        #     "task": "put all shakers and tomato on the counter top",
        #     "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        # },  
        # {
        #     "task_folder": "3_put_all_silverware_drawer",
        #     "task": "Put all silverware in the drawer",
        #     "scenes": [ "FloorPlan3", "FloorPlan4", "FloorPlan5",  "FloorPlan6"] #"FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", 
        # },  
        {
            "task_folder": "3_put_all_tableware_countertop",
            "task": "Put all tableware on the countertop",
            "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
        },  
        # {
        #     "task_folder": "3_transport_groceries",
        #     "task": "put_all_food_countertops",
        #     "scenes": ["FloorPlan1"]
        # },  
        
    ]

    TASKS_4 = [
    # {
    #     "task_folder": "4_clear_couch_livingroom",
    #     "task": "Clear the couch by placing the items in other appropriate positions ",
    #     "scenes": ["FloorPlan209"] #"FloorPlan212" hen "FloorPlan201",  "FloorPlan202","FloorPlan203","FloorPlan209", 
    # },
    # {
    #     "task_folder": "4_clear_countertop_kitchen",
    #     "task": "Clear the countertop by placing items in their appropriate positions",
    #     "scenes": ["FloorPlan1"] # "FloorPlan1", "FloorPlan2", "FloorPlan30", "FloorPlan10", "FloorPlan6"
    # },
    {
        "task_folder": "4_clear_floor_kitchen",
        "task": "Clear the floor by placing items at their appropriate positions",
        "scenes": ["FloorPlan1", "FloorPlan2","FloorPlan3", "FloorPlan4", "FloorPlan5"]# "FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    },
    # {
    #     "task_folder": "4_clear_table_kitchen",
    #     "task": "Clear the table by placing the items in their appropriate positions",
    #     "scenes": ["FloorPlan15", "FloorPlan16", "FloorPlan17"] #"FloorPlan4", "FloorPlan11", "FloorPlan15", "FloorPlan16", "FloorPlan17"
    # },
    
    # {
    #     "task_folder": "4_put_appropriate_storage",
    #     "task": "Place all utensils into their appropriate positions",
    #     "scenes": [  "FloorPlan2"] # "FloorPlan2",  "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6
    # }, 
    # {
    #     "task_folder": "4_make_livingroom_dark",
    #     "task": "Make the living room dark",
    #     "scenes": ["FloorPlan201"] #"FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan204","FloorPlan205"
    # }, 
]
    
    # batch_run(TASKS_1, base_dir="config", start=50, end=50, sleep_after=50, delete_frames=True)
    # batch_run(TASKS_2, base_dir="config", start=51, end=51, sleep_after=50, delete_frames=True)
    # batch_run(TASKS_3, base_dir="config", start=52, end=52, sleep_after=50, delete_frames=True)
    batch_run(TASKS_4, base_dir="config", start=52, end=52, sleep_after=50, delete_frames=True)
    # run_main(test_id = 3, config_path="config/config.json", delete_frames=True)
    # run_main(test_id = 2, config_path="config/config.json")
   