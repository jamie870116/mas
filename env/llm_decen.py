'''
Decentralized LLM for multi-agent task planning and execution

1. (LLM)planning subtasks for each agent(run initial_subtask_planning per agent) -> run once at the begining
    
LOOP(until all subtasks are done or timeout):
2. (LLM)decompose subtask to smaller actions (decompose_subtask_to_actions per agent)
3. combine each agent's subtasks into -> subtasks: {agent_id: (subtask, [actions(NavigateToObejct, PickupObject, ...)]), ...
4. execution (stepwise_action_loop per agent) while maintaing seperate logs with messege queue for delayed messaging
    4.5. (LLM)Log summariser (log_summariser per agent): decide what to log for each agent. (trigger when subtask is success/failure or delay messaging arrived)
5. (LLM)verify and replan if needed (verify_actions, replan_open_subtasks independently per agent)
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
import numpy as np

from openai import OpenAI
from env_decen import AI2ThorEnv_cen as AI2ThorEnv
from prompt_template import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

client = OpenAI(api_key=Path('api_key_ucsb.txt').read_text())

def get_delays(mode='poisson'):
        if mode == 'poisson':
            return np.random.poisson(lam=3)
        elif mode == 'geometric':
            return np.random.geometric(p=0.3)
        else:
            return 0
        
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

def prepare_prompt(env: AI2ThorEnv, mode: str = "init", agent_id: int = 0, addendum: str = "", subtasks=[], need_process=False, info="") -> str:
    """
    mode: str, choose from planner, action
    planner: for decomposing the task into subtasks
    action: for generating the actions for each robot to perform
    addendum: additional information to be added to the user prompt
    """

    if mode == "planner":
        # for initial planning
        system_prompt = get_decen_planner_prompt(agent_id)
        input = env.get_planner_llm_input(agent_id)
        user_prompt = convert_dict_to_string(input)
        # print("planner input:", user_prompt)
    elif mode == "replan":
        # for replanning the subtasks based on the current state of the environment
        system_prompt = get_decen_planner_prompt(agent_id)
        input = env.get_planner_llm_input(agent_id)
        if info:
            input['Suggestion'] = info
        
        user_prompt = convert_dict_to_string(input)
        # print("replan prompt:", system_prompt)
        # print("replan use prompt:", user_prompt)
    elif mode == "action":
        # for generating automic actions for each robot to perform
        system_prompt = get_action_prompt(mode='decen')
        if not subtasks:
            print("No subtasks provided")
            return None, None
        input = env.get_obs_llm_input(recent_logs=False, agent_id=agent_id)
        input["Subtasks"] = subtasks
        del input["Robots' open subtasks"]
        del input["Robots' completed subtasks"]
        user_prompt = convert_dict_to_string(input)

   
    elif mode == "verifier":
        # for verifying the actions taken by the robots

        system_prompt = get_decen_verifier_prompt()
        input = env.get_llm_decen_verify_input(agent_id)

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
    if mode == "planner" or mode == "replan":
        return [_get(data, "Subtask"), _get(data, "Messages")] or ""

    if mode == "action":
        return data["Actions"] if data else []

    
    if mode == "verifier":
        # return (
        #     _get(data, "need_replan", aliases=["need_replan"]) or False,
        #     _get(data, "failure_reason"),
        #     _get(data, "reason"),
        #     _get(data, "suggestion"),
        # )
        return { _get(data, "need_replan"), _get(data, "suggestion")}
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


def decompose_subtask_to_actions(env, subtasks, info={}):
    """將 subtask 拆解成 atomic actions LLM"""
    action_prompt, action_user_prompt = prepare_prompt(env, mode="action", subtasks=subtasks, info=info)
    action_payload = prepare_payload(action_prompt, action_user_prompt)
    # print("action prompt:", action_prompt)
    # print("action user prompt:", action_user_prompt)
    res, res_content = get_llm_response(action_payload, model=config['model'])
    # print('action llm output', res_content)
    actions = process_llm_output(res_content, mode="action")
    print("after llm processed output action: ", actions)
    # For testing 
    # actions = [['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)'], ['NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)']]
    return actions

def encode_image(image_path: str):
    # if not os.path.exists()
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def verify_actions(env, need_process=False, agent_id=0):
    # 
    verify_prompt, verify_user_prompt = prepare_prompt(env, mode="verifier", need_process=need_process, agent_id=agent_id)
    verify_payload = prepare_payload(verify_prompt, verify_user_prompt)
    # print("verify prompt: ", verify_user_prompt)
    res, res_content = get_llm_response(verify_payload, model=config['model'])
    print('verify llm output', res_content)
    need_replan, suggestion = process_llm_output(res_content, mode="verifier")
    verify_res = {
        "need_replan": need_replan,
        "suggestion": suggestion
    }
    return verify_res



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


def decen_subtask_planning(env, config, agent_id, is_initial=False, info=""):
    # planning subtasks based on environment and logs and agent's observation
    # Only Log when initial planning (Agent_{} start {subtask} )
    if is_initial:
        mode = "planner"
    else:
        mode = "replan"
    planner_prompt, planner_user_prompt = prepare_prompt(env, mode=mode, agent_id=agent_id, info=info)
    planner_payload = prepare_payload(planner_prompt, planner_user_prompt)
    res, res_content = get_llm_response(planner_payload, model=config['model'])
    # print('init plan llm output', res_content)
    subtasks, msg = process_llm_output(res_content, mode=mode)
    print(f"After Planner LLM Response subtasks: {subtasks}, msg: {msg}")
    return subtasks, msg

def get_agent_subtask(env: AI2ThorEnv, config, agent_id, is_initial=False, info=""):
    """
    Will be called one by one for each agent to plan their own subtasks
    return a list of subtasks for this agent
    """
    # planning subtasks based on environment and logs and agent's observation
    subtask, msg = decen_subtask_planning(env, config, agent_id, is_initial=is_initial, info=info)
    
    # decompose subtask to smaller actions
    actions = decompose_subtask_to_actions(env, subtask)
    actions = actions[0]  # get the first agent's actions
# 
    if is_initial:
        # log: agent start subtask
        log_dict = {}
        
        print(f"Agent_{agent_id} start {subtask}")
        log_dict = {
                'timestamp':0,
                'history': f"Agent_{agent_id} {AGENT_NAMES[agent_id]} starts {subtask}."
            }
        # boardcast log to all agents
        for aid in range(config['num_agents']):
            env.save2log_by_agent(aid, log_dict)

    else:
        # log with delay messaging
        for aid in range(config['num_agents']):
            if aid == agent_id:
                continue
            
            target_name = AGENT_NAMES[aid]
            
            if not msg.get(target_name):
                continue
            
            delay_sec = get_delays(mode='poisson')
            if delay_sec == 0:
                log_dict = {
                    'timestamp': env.get_cur_ts(),
                    'history': msg[target_name]
                }
                env.save2log_by_agent(aid, log_dict)
            else:
                env.schedule_message(aid, msg[target_name], delay_sec)
        # for aid in range(config['num_agents']):
        #     if aid != agent_id:
        #         delay_sec = get_delays(mode='poisson')
        #         if delay_sec == 0:
        #             log_dict = {
        #                 'timestamp':env.get_cur_ts(),
        #                 'history': f"Agent_{agent_id} {AGENT_NAMES[agent_id]} starts {subtask}."
        #             }
        #             env.save2log_by_agent(aid, log_dict)
        #         else:
        #             env.schedule_message(aid, f"Agent_{agent_id} {AGENT_NAMES[agent_id]} starts {subtask}.", delay_sec)
    # return a subtask and a list of actions of this subtask
    return subtask, actions, msg

def decen_main(test_id = 0, config_path="config/config.json", delete_frames=False, timeout=250):
    # Init. Env & config
    env, config = set_env_with_config(config_path)
    if test_id > 0:
        _ = env.reset(test_case_id=test_id)
    else:
        _ = env.reset(test_case_id=config['test_id'])

    num_agent = config['num_agents']
    logs_llm = []
    logs_llm_path = env.base_path / "logs_llm.txt"

    # run initial subtask planning for each agent
    print("\n--- Initial Subtask Planning for each agent---")
    logs_llm.append("\n--- Initial Subtask Planning for each agent---")
    subtasks = {}
    msg_list = []
    for aid in range(num_agent):
        subtask, actions, msg = get_agent_subtask(env, config, agent_id=aid, is_initial=True)
        subtasks[aid] = (subtask, actions)
        msg_list.append(msg)
    print("Initial subtasks for each agent: ", subtasks, "with msg:", msg_list)
    """
    self.current_subtask: {0: 'navigate to the bread, pick up the bread, navigate to the fridge, open the fridge, put the bread in the fridge, and close the fridge', 1: 'navigate to the bread, pick up the bread, navigate to the fridge, open the fridge, put the bread in the fridge, and close the fridge'}
    self.pending_high_level: defaultdict(<class 'collections.deque'>, {0: deque(['NavigateTo(Bread_1)', 'PickupObject(Bread_1)', 'NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)']), 1: deque(['NavigateTo(Bread_1)', 'PickupObject(Bread_1)', 'NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'])})
    self.current_hl: {0: None, 1: None}
    self.action_queue: defaultdict(<class 'collections.deque'>, {0: deque([]), 1: deque([])})
    
    """
    logs_llm.append(f"Initial subtasks for each agent: {subtasks} with msg: {msg_list}")

    # save the subtasks to env
    env.update_decen_plan(subtasks)

    # loop start
    cnt = 0
    start_time = time.time()
    # TBD: (more test) 
    while True:
        if time.time() - start_time > timeout:
            print("Timeout reached, ending loop.")
            logs_llm.append(f"""Timeout ({timeout} second) reached, ending loop.""")
            break
        print(f"\n--- Loop {cnt + 1} ---")
        logs_llm.append(f"\n--- Loop {cnt + 1} ---")

        if env.check_if_done():
            print("No more subtask --- End")
            logs_llm.append("No more subtask --- End")
            break
        
        if env.check_if_all_idle():
            print("All agent remain Idle --- End")
            logs_llm.append("All agent remain Idle--- End")
            break

        # execute
        isSuccess, info, succ, old_msg = env.stepwise_decen_loop()
        print('info', info)
        print('old_msg', old_msg)
        print('succ', succ)
        logs_llm.append(f"info: {info}")
        logs_llm.append(f"action exe status: {succ}")
        logs_llm.append(f"message status: {old_msg}")

        # which agent need replan (due to failure or/and new msg)
        for aid in range(num_agent):
            
            failed = not succ[aid]
            new_msg = not old_msg[aid]

            done_all = env.check_if_done_byagent(aid)
            
            if failed or new_msg or done_all:
                need_replan = True
            else:
                need_replan = False
            print(f"check if replan is needed for agent {aid}: need_replan: {need_replan} ;failed {failed}, new_msg {new_msg}, done_all {done_all}")
            logs_llm.append(f"check if replan is needed for agent {aid}: need_replan: {need_replan} ;failed {failed}, new_msg {new_msg}, done_all {done_all}")
            #  verify & replan
            if need_replan:
                verify_res = verify_actions(env, agent_id=aid)
                print(verify_res)
                logs_llm.append(f"verified res for agent {aid}: {verify_res}")
                if verify_res["need_replan"]:
                    subtask, actions, msg = get_agent_subtask(env, config, agent_id=aid, is_initial=False, info=verify_res['suggestion'])
                    print(f"new subtasks for  agent {aid}: {subtask}")
                    print(f"new actions for  agent {aid}: {actions}")
                    print(f"new message from agent {aid}: {msg}")
                    logs_llm.append(f"new subtasks for  agent {aid}: {subtask}")
                    logs_llm.append(f"new actions for  agent {aid}: {actions}")
                    logs_llm.append(f"new message from agent {aid}: {msg}")
                    # After replanning, update plan
                    env.upadate_decen_subtasks_by_agent(agent_id=aid, subtask = subtask, actions=actions)
                else:
                    # Get previous plan, and continuly work on previous plan
                    subtask, actions = env.get_curr_subtask_agent(aid)
            else: 
                subtask, actions = env.get_curr_subtask_agent(aid)
            
        
        with open(logs_llm_path, "a", encoding="utf-8") as f:
            for log in logs_llm:
                f.write(str(log) + "\n")
            logs_llm = []
        cnt += 1


    # Log final report and save video
    iscomplete, report = env.run_task_check()
    obj_status = env.get_all_object_status()
    print("Final Report: ", report)
    with open(logs_llm_path, "a", encoding="utf-8") as f:
        for log in logs_llm:
            f.write(str(log) + "\n")
        f.write("Final Report: " + str(report) + "\n")
        f.write("Success: " + str(iscomplete) + "\n")
        f.write(f"Total steps: {env.step_num}\n")
        # f.write(f"Total action steps: {env.action_step_num}\n")
        f.write("\n")
        for obj_id, status in obj_status.items():
            f.write(f"{obj_id}: {status}\n")

    env.save_to_video(delete_frames=delete_frames)
    env.close()
  

def batch_run(tasks, base_dir="config", start = 1, end=5, sleep_after=2.0, delete_frames=False, timeout=250):
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
                decen_main(test_id = r, config_path=str(cfg_path), delete_frames=delete_frames, timeout=timeout)
                time.sleep(sleep_after)

            print(f"==== Finished {cfg_path} ====")


if __name__ == "__main__":
    TASKS_1 = [
    # {
    #     "task_folder": "1_put_bread_lettuce_tomato_fridge",
    #     "task": "put bread, lettuce, and tomato in the fridge",
    #     "scenes": ["FloorPlan4", "FloorPlan5"] # "FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_put_computer_book_remotecontrol_sofa",
    #     "task": "put laptop, book and remote control on the sofa",
    #     "scenes": ["FloorPlan202"] #,"FloorPlan201", "FloorPlan202","FloorPlan203", "FloorPlan209", "FloorPlan224"
    # },
    # {
    #     "task_folder": "1_put_knife_bowl_mug_countertop",
    #     "task": "put knife, bowl, and mug on the counter top",
    #     "scenes": ["FloorPlan2"] #"FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_put_plate_mug_bowl_fridge",
    #     "task": "put plate, mug, and bowl in the fridge",
    #     "scenes": ["FloorPlan1"] #"FloorPlan1", "FloorPlan2",,"FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    {
        "task_folder": "1_put_remotecontrol_keys_watch_box",
        "task": "put remote control, keys, and watch in the box",
        "scenes": ["FloorPlan201"] # "FloorPlan201", "FloorPlan202", "FloorPlan203", ,"FloorPlan209", "FloorPlan215", "FloorPlan226", "FloorPlan228", "FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan207"
    },
    {
        "task_folder": "1_put_vase_tissuebox_remotecontrol_table",
        "task": "put vase, tissue box, and remote control on the side table1",
        "scenes": [ "FloorPlan201"] # "FloorPlan201", "FloorPlan219", "FloorPlan203", "FloorPlan216", "FloorPlan219"
    },

    # {
    #     "task_folder": "1_slice_bread_lettuce_tomato_egg",
    #     "task": "slice bread, lettuce, tomato, and egg with knife",
    #     "scenes": [ "FloorPlan1"] #"FloorPlan1", , "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_turn_off_faucet_light",
    #     "task": "turn off the sink faucet and turn off the light switch",
    #     "scenes": ["FloorPlan1"] #"FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "1_wash_bowl_mug_pot_pan",
    #     "task": "clean the bowl, mug, pot, and pan",
    #     "scenes": ["FloorPlan1"] #"FloorPlan3","FloorPlan1",  "FloorPlan2", "FloorPlan4", "FloorPlan5"
    # },
]

    TASKS_2 = [
        {
            "task_folder": "2_open_all_cabinets",
            "task": "open all the cabinets",
            "scenes": [ "FloorPlan1",]# "FloorPlan1", "FloorPlan6",   "FloorPlan7", "FloorPlan8", "FloorPlan10"
        },
        {
            "task_folder": "2_open_all_drawers",
            "task": "open all the drawers",
            "scenes": ["FloorPlan1"] # "FloorPlan1","FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9"
        },
        {
            "task_folder": "2_put_all_creditcards_remotecontrols_box",
            "task": "put all credit cards and remote controls in the box",
            "scenes": ["FloorPlan201"] #"FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan204", "FloorPlan205"
        },
        {
            "task_folder": "2_put_all_vases_countertop",
            "task": "put all the vases on the counter top",
            "scenes": ["FloorPlan1"] #, "FloorPlan5"
        },
        {
            "task_folder": "2_put_all_tomatoes_potatoes_fridge",
            "task": "put all tomatoes and potatoes in the fridge",
            "scenes": ["FloorPlan1"] # "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
        },
        
        {
            "task_folder": "2_turn_on_all_stove_knobs",
            "task": "turn on all the stove knobs",
            "scenes": ["FloorPlan1"] #"FloorPlan1", "FloorPlan2","FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan7", "FloorPlan8"
        },  
    ]

    TASKS_3 = [
        # {
        #     "task_folder": "3_clear_table_to_sofa",
        #     "task": "Put all readable objects on the sofa",
        #     "scenes": ["FloorPlan201"] #"FloorPlan201", "FloorPlan203", "FloorPlan204", "FloorPlan208", "FloorPlan223"
        # },
        # {
        #     "task_folder": "3_put_all_food_countertop",
        #     "task": "Put all food on the countertop",
        #     "scenes": [ "FloorPlan1"] #  "FloorPlan1", "FloorPlan2", "FloorPlan3","FloorPlan4","FloorPlan5"
        # },
        {
            "task_folder": "3_put_all_groceries_fridge",
            "task": "Put all groceries in the fridge",
            "scenes": [ "FloorPlan1"] #"FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
        },
        {
            "task_folder": "3_put_all_kitchenware_box",
            "task": "Put all kitchenware in the cardboard box",
            "scenes": ["FloorPlan201"]
        },
        {
            "task_folder": "3_put_all_school_supplies_sofa",
            "task": "Put all school supplies on the sofa",
            "scenes": ["FloorPlan201"] #"FloorPlan201", "FloorPlan202", "FloorPlan203","FloorPlan209", "FloorPlan212"
        },
        {
            "task_folder": "3_put_all_shakers_fridge",
            "task": "Put all shakers in the fridge",
            "scenes": [ "FloorPlan1"] #"FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
        },  
        {
            "task_folder": "3_put_all_shakers_tomato", # on countertop
            "task": "put all shakers and tomato on the counter top",
            "scenes": ["FloorPlan1"] # "FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
        },  
        {
            "task_folder": "3_put_all_silverware_drawer",
            "task": "Put all silverware in the drawer",
            "scenes": [ "FloorPlan2" ]  #"FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6"]
        },  
        {
            "task_folder": "3_put_all_tableware_countertop",
            "task": "Put all tableware on the countertop",
            "scenes": ["FloorPlan1"] #["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
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
    #     "scenes": ["FloorPlan201"] #"FloorPlan212" hen "FloorPlan201",  "FloorPlan202","FloorPlan203","FloorPlan209", 
    # },
    # {
    #     "task_folder": "4_clear_countertop_kitchen",
    #     "task": "Clear the countertop by placing items in their appropriate positions",
    #     "scenes": ["FloorPlan1"] # "FloorPlan1", "FloorPlan2", "FloorPlan30", "FloorPlan10", "FloorPlan6"
    # },
    # {
    #     "task_folder": "4_clear_floor_kitchen",
    #     "task": "Clear the floor by placing items at their appropriate positions",
    #     "scenes": ["FloorPlan1"]# "FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"
    # },
    # {
    #     "task_folder": "4_clear_table_kitchen",
    #     "task": "Clear the table by placing the items in their appropriate positions",
    #     "scenes": ["FloorPlan4"] #"FloorPlan4", "FloorPlan11", "FloorPlan15", "FloorPlan16", "FloorPlan17"
    # },
    
    # {
    #     "task_folder": "4_put_appropriate_storage",
    #     "task": "Place all utensils into their appropriate positions",
    #     "scenes": [  "FloorPlan2"] # "FloorPlan2",  "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6
    # }, 
    {
        "task_folder": "4_make_livingroom_dark",
        "task": "Make the living room dark",
        "scenes": ["FloorPlan201"] #"FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan204","FloorPlan205"
    }, 
]
    
    # batch_run(TASKS_1, base_dir="config", start=63, end=63, sleep_after=50, delete_frames=True)
    # batch_run(TASKS_2, base_dir="config", start=60, end=60, sleep_after=50, delete_frames=True)
    # batch_run(TASKS_3, base_dir="config", start=60, end=60, sleep_after=50, delete_frames=True)
    batch_run(TASKS_4, base_dir="config", start=61, end=61, sleep_after=50, delete_frames=True)


    # decen_main(test_id = 1, config_path="config/config.json", delete_frames=True, timeout=250)
