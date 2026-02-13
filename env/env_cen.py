import json
import os
from pathlib import Path
import cv2
import ai2thor.controller
from typing import Dict, List, Tuple, Any, Optional, Callable
import pickle
import numpy as np
from heapq import heappush, heappop
import time
from collections import deque, defaultdict
from thortils.navigation import get_shortest_path_to_object
# from thortils.utils import closest, closest_angles
from thortils.constants import H_ANGLES, V_ANGLES

import traceback
import math
import importlib.util
import re
from env_base import BaseEnv


def import_scene_initializer(task: str, floor_plan: str):
    file_path = os.path.join("Tasks", task, f"{floor_plan}.py")
    if not os.path.exists(file_path):
        return None

    spec = importlib.util.spec_from_file_location("SceneInitializer", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "SceneInitializer", None)

def import_task_checker(task_folder: str,):
    

    generic_path = os.path.join("Tasks", task_folder, "checker.py")
    if os.path.exists(generic_path):
        spec = importlib.util.spec_from_file_location("checker_module", generic_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "build_checker"):
            return getattr(module, "build_checker")

    return None

def closest_angles(values, query):
    """Returns the entry in `values` that is
    closest to `query` in unit circle angles"""
    values.append(360)
    return min(values, key=lambda v: abs(v-query)) % 360


class AI2ThorEnv_cen(BaseEnv):
    """Main AI2THOR environment for multi-agent tasks with global timer and frame saving."""
    def __init__(self, config_path: str = "config.json"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.num_agents = self.config["num_agents"] # curr max 5 agents
        self.agent_names = ["Alice", "Bob", "Charlie", "David", "Emma"][:self.num_agents]
        self.scene = self.config["scene"]
        self.task = self.config["task"]
        self.timeout = self.config["timeout"]
        self.model = self.config["model"]
        self.use_obs_summariser = self.config["use_obs_summariser"]
        self.use_act_summariser = self.config["use_act_summariser"]
        self.use_action_failure = self.config["use_action_failure"]
        self.use_shared_subtask = self.config["use_shared_subtask"]
        self.use_separate_subtask = self.config["use_separate_subtask"]
        self.use_shared_memory = self.config["use_shared_memory"]
        self.use_separate_memory = self.config["use_separate_memory"]
        self.use_plan = self.config["use_plan"]
        self.force_action = self.config["force_action"]
        self.overhead = self.config["overhead"]
        self.save_logs = self.config["save_logs"]
        self.logs = []
        
        self.controller = ai2thor.controller.Controller(width=1000, height=1000, gridSize=0.25)
        self.controller.reset(self.scene)
        
        self.subtasks = ["Initial subtask"] if self.use_shared_subtask else ["Initial subtask"] * self.num_agents
        self.memory = ["Nothing"] if self.use_shared_memory else ["Nothing"] * self.num_agents
        self.suggestion = [""]
        
        self.input_dict = {}
        self.step_num = [0] * self.num_agents

        self.open_subtasks = "None" if self.use_plan else None
        self.closed_subtasks = "None" if self.use_plan else None
        self.inventory = ["nothing"] * self.num_agents
        self.pending_high_level = defaultdict(deque) # {agent_id: [queue]} # eg. NavigateTo(Obj), PickUp(Obj)..
        self.pending_subtasks = defaultdict(deque) # {agent_id: [queue]} # description of subtasks
        self.action_queue = defaultdict(deque) # {agent_id: [queue] }for breaking down navigation steps or so
        self.current_hl = {} 
        self.cur_plan = {}
        self.action_step_num = 0
        self.subtask_success_history = {name: [] for name in self.agent_names}  # save the previous successful subtasks for each agent
        self.subtask_failure_reasons = {name: [] for name in self.agent_names} # save the previous failure subtasks and reason for each agent
        self.agent_failure_acts	= {name: [] for name in self.agent_names} # save 最近失敗過的 atomic action，通常用於偵測是否卡住
        self.nav_no_plan = {name: False for name in self.agent_names}   # 導航規劃為空的旗標
        self.last_check_reason = {name: None for name in self.agent_names}  # is_subtask_done 內部最近一次的判斷理由
        
        self.action_history = {name: [] for name in self.agent_names}
        self.action_success_history = {name: [] for name in self.agent_names}
        # self.agent_failure_acts = {name: [] for name in self.agent_names}
        # handle when openning objects such as fridge, the reachable positions is wrong
        self.mannual_block_pos = {} # {obj_name: [pos.]}

        self.all_obs_dict = {name: [] for name in self.agent_names}
        self.obs_summary_llm_cache_path = "summary_llm_cache.pkl"
        self.verbose = True
        self.skip_save_dir = False
        self.grid_size = 0.25
        # self.previous_object_ids = [None] * self.num_agents
        # self.previous_positions = [None] * self.num_agents
        self.start_time = None
        self.total_elapsed_time = 0.0
    
    def reset(self, task: str = None, test_case_id: str = None) -> str:
        """Reset the environment, start the global timer, and save initial frames before any actions."""
        self.task = task or self.config["task"]
        self.task_folder = self.config["task_folder"]
        self.controller.reset(self.scene)
        self.event = self.controller.step(
            {
                "action": "Initialize",
                "gridSize": self.grid_size,
                "renderObjectImage": True,
                "renderInstanceSegmentation": True,
                "agentCount": self.num_agents,
                "visibilityDistance": 40,
            }
        )
        scene_initializer = import_scene_initializer(self.task_folder, self.scene)
        if scene_initializer:
            if self.verbose:
                print(f"Preinitializing scene for task={self.task}, scene={self.scene}")
            self.event = scene_initializer().preinit(self.event, self.controller)


        # Task Checker（新）
        checker_factory = import_task_checker(self.task_folder)
        if checker_factory:
            # 讓 checker 能存取 env 的工具（get_object_status / get_readable_object_list）
            self.checker = checker_factory(env=self)
            print(f"[Checker] Loaded from task_folder='{self.task_folder}' for scene='{self.scene}'")
        else:
            self.checker = None
            print(f"[Checker] No checker found for task_folder='{self.task_folder}'")

        self.object_dict = {}
        self.step_num = [0] * self.num_agents
        self.inventory = ["nothing"] * self.num_agents
        self.subtasks = ["Initial subtask"] if self.use_shared_subtask else ["Initial subtask"] * self.num_agents
        self.memory = ["Nothing"] if self.use_shared_memory else ["Nothing"] * self.num_agents
        self.suggestion = [""]
        self.action_history = {name: [] for name in self.agent_names}
        self.action_success_history = {name: [] for name in self.agent_names}
        self.subtask_success_history = {name: [] for name in self.agent_names}  # save the previous successful subtasks for each agent
        self.subtask_failure_reasons = {name: [] for name in self.agent_names} 
        self.agent_failure_acts = {name: [] for name in self.agent_names}
        self.all_obs_dict = {name: [] for name in self.agent_names}
        self.nav_no_plan = {name: False for name in self.agent_names}
        self.last_check_reason = {name: None for name in self.agent_names}
        
        self.open_subtasks = "None" if self.use_plan else None
        self.closed_subtasks = "None" if self.use_plan else None
        self.pending_high_level = defaultdict(deque) # {agent_id: [queue]} # eg. NavigateTo(Obj), PickUp(Obj)..
        self.pending_subtasks = defaultdict(deque) # {agent_id: [queue]} # description of subtasks
        self.action_queue = defaultdict(deque) # {agent_id: [queue] }for breaking down navigation steps or so
        self.current_hl = {} 
        self.cur_plan = {}
        self.action_step_num = 0

        self.start_time = time.time()
        self.total_elapsed_time = 0.0
        self.input_dict["Task"] = task
        self.logs = []

        self.mannual_block_pos = {}

        # get third party camera properties: overhead view
        event = self.controller.step(action="GetMapViewCameraProperties")
        event = self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

        if not self.skip_save_dir:
            self.create_save_dirs(test_case_id, self.scene)
        
        for agent_id in range(self.num_agents):
            self.event = self.controller.step(
                dict(
                    action="Teleport",
                    position=dict(x=1.5 + agent_id * 0.5, y=0.9, z=-1.5),
                    rotation=dict(x=0, y=270, z=0),
                    agentId=agent_id
                )
            )

        self.update_object_dict()
        if not self.skip_save_dir:
            self.save_frame()
        
        return self.get_observations()
    
    def get_object_dict(self):
        return self.object_dict

    def get_observations(self) -> str:
        """Generate observation dictionary and return as string."""
        self.input_dict = {"Task": self.task, "Elapsed Time": f"{self.total_elapsed_time:.2f} seconds"}
        
        for agent_id, agent_name in enumerate(self.agent_names):
            obs, obs_list = self.generate_obs_text(agent_id, mode="mapping") # output: text, list or mapping
            self.input_dict[f"Agent {agent_id} {agent_name}'s observation"] = obs
            self.input_dict[f"{agent_name}'s state"] = self.get_agent_state(agent_id)
            self.input_dict[f"{agent_name}'s previous action"] = (
                "I have not taken any action yet" if not self.action_history[agent_name]
                else self.get_act_text(
                    self.action_history[agent_name][-1],
                    self.action_success_history[agent_name][-1],
                    agent_id
                )
            )
            if self.use_action_failure:
                self.input_dict[f"{agent_name}'s previous failures"] = self.get_act_failure_text(
                    self.agent_failure_acts[agent_name], agent_id
                )
            if self.use_shared_subtask:
                self.input_dict["Robots' subtasks"] = self.subtasks[0]
            elif self.use_separate_subtask:
                self.input_dict[f"{agent_name}'s subtask"] = self.subtasks[agent_id]
            if self.use_shared_memory:
                self.input_dict["Robots' combined memory"] = self.memory[0]
            elif self.use_separate_memory:
                self.input_dict[f"{agent_name}'s memory"] = self.memory[agent_id]
            if self.use_plan:
                self.input_dict["Robots' open subtasks"] = self.open_subtasks
                self.input_dict["Robots' completed subtasks"] = self.closed_subtasks
            self.all_obs_dict[agent_name] = obs_list
        return "\n".join(f"{k}: {v}" for k, v in self.input_dict.items())
    
    def get_object_in_view(self, agent_id: int) -> List[str]:
        """Get objects in the agent's view."""
        self.update_object_dict()
        detections = self.event.events[agent_id].instance_detections2D
        return list(detections.instance_masks.keys()) if detections else []
    


    def get_mapping_object_pos_in_view(self, agent_id: int) -> Dict[str, Dict[str, float]]:
        """
        以 'Name_Idx' 為鍵，回傳可見物件的 {'x','z'}。
        """
        self.update_object_dict()
        detections = self.event.events[agent_id].instance_detections2D
        if not detections:
            return {}

        

        id2xz = {}
        for obj in self.event.metadata.get("objects", []):
            pos = obj.get("position")
            if not pos:
                continue
            oid = obj.get("objectId")
            if oid:
                id2xz[oid] = (pos.get("x"), pos.get("z"))

        positions: Dict[str, Dict[str, float]] = {}

        for obj_id, mask in detections.instance_masks.items():
            if mask is None:
                continue
            if obj_id not in id2xz:
                continue

            obj_name, obj_uid = self.parse_object(obj_id)

            name_map = self.object_dict.setdefault(obj_name, {})
            idx = name_map.get(obj_uid)
            if idx is None:
                idx = len(name_map) + 1
                name_map[obj_uid] = idx

            key = f"{obj_name}_{idx}"
            x, z = id2xz[obj_id]
            if x is None or z is None:
                continue
            positions[key] = {"x": round(x, 2), "z": round(z, 2)}

        return positions



    def generate_obs_text(self, agent_id: int, prefix: str = "I see: ", mode="list") -> Tuple[str, List[str]]:
        """Generate observation text and list."""
        """
        mode: "list" | "mapping"
        - "list": 回傳 ['Tomato_1', 'Fridge_1', ...]
        - "mapping": 回傳 { 'Tomato_1': {'x': .., 'z': ..}, ... }
                    並產出一段以鍵名為主的摘要文字
        """
        if mode == "mapping":
            mapping = self.get_mapping_object_pos_in_view(agent_id)  # {Name_Idx: {'x','z'}}
            names = sorted(mapping.keys())
            obs_text = prefix + str(names)
            if self.use_obs_summariser:
                obs_text = prefix + str(self.summarise_obs(names))
            return obs_text, mapping

        # 預設行為：list
        objects_in_view = self.get_object_in_view(agent_id)
        obs_list = self.get_readable_object_list(objects_in_view)
        if self.use_obs_summariser:
            obs_list = self.summarise_obs(obs_list)
        obs_text = prefix + str(obs_list)
        return obs_text, obs_list
      
    
    def summarise_obs(self, obs_list: List[str]) -> List[str]:
        """Placeholder for observation summarization. (LLM) """
        return obs_list
    
    """
    available actions
    self.return_actions = ["Return"]
    self.move_actions = ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]
    self.rotate_actions = ["RotateRight", "RotateLeft"]
    self.look_actions = ["LookUp", "LookDown"]
    self.idle_actions = ["Done", "Idle"]
    self.object_interaction_actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "BreakObject", "CookObject", "SliceObject", "DirtyObject", "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject"]
    self.object_interaction_without_navigation  = ["DropHandObject", "ThrowObject"]
    """
       
    def get_navigation_step(self, action: str, agent_id: int) -> List[str]:
        object_name = action.split("(")[1].rstrip(")")
        # print(self.object_dict, object_name)
        # obj_id = self.convert_readable_object_to_id(object_name)
        try:
            obj_id = self.convert_readable_object_to_id(object_name)
        except ValueError as e:
            self._record_subtask_failure(agent_id, reason=f"object({object_name})-not-exist ({e})", at_action=action)
            self.agent_failure_acts[self.agent_names[agent_id]].append(action)
            return []

        cur_pos = self.get_agent_position_dict(agent_id)
        rot_meta = self.event.events[agent_id].metadata["agent"]["rotation"]
        cur_rot = (
            closest_angles(V_ANGLES, rot_meta["x"]),
            closest_angles(H_ANGLES, rot_meta["y"]),
            rot_meta["z"]
        )
        cur_pos_tuple = (cur_pos["x"], cur_pos["y"], cur_pos["z"])

        other_agents = [
            self.event.events[i].metadata["agent"]["position"]
            for i in range(self.num_agents)
            if i != agent_id
        ]
        # manual_block_list = [{"x": x, "y": y, "z": z} for positions in self.mannual_block_pos.values() for (x, y, z) in positions]
        manual_block_list = [{"x": pos["x"], "y": pos["y"], "z": pos["z"]} for positions in self.mannual_block_pos.values() for pos in positions]

        # print(f"other_agents position: {other_agents}")
        # print(f"manual_block_list position: {manual_block_list}")

        if other_agents:
            poses, plan = get_shortest_path_to_object(
                self.controller, obj_id, cur_pos_tuple, cur_rot, other_agent_position=other_agents, return_plan=True, cur_step=self.step_num[agent_id], isVisualize=False, agent_id=agent_id, mannual_block_pos=manual_block_list
            )
        else:
            poses, plan = get_shortest_path_to_object(
                self.controller, obj_id, cur_pos_tuple, cur_rot, return_plan=True, cur_step=self.step_num[agent_id], isVisualize=False, agent_id=agent_id, mannual_block_pos=manual_block_list
            )

        if not plan:
            obj_meta = next(
                (o for o in self.event.metadata["objects"] if o["objectId"] == obj_id),
                None
            )
            agent_pos = cur_pos
            obj_pos = obj_meta["position"]
            obj_base_name = object_name.split("_")[0]
            dist = ((agent_pos["x"] - obj_pos["x"]) ** 2 + (agent_pos["z"] - obj_pos["z"]) ** 2) ** 0.5
            print(f"[Navigation] Agent {self.agent_names[agent_id]} cannot find path to {object_name}. Distance: {dist:.2f} meters.")
            if dist < 1.0 and obj_base_name in self.small_objects:
                if obj_id not in self.get_object_in_view(agent_id):
                    return [f"LookDown(30)"]
                
            if dist < 1.0 and obj_base_name.startswith("Drawer"):
                print(f"{obj_id} detected within 1 meter but no path found.")
                # if obj_id not in self.get_object_in_view(agent_id):
                return [f"LookDown(30)"]
                
            self.nav_no_plan[self.agent_names[agent_id]] = True
            self._record_subtask_failure(agent_id, reason="no-path", at_action=action)
            return []
        micro_actions: List[str] = []
        xx, yy, zz = cur_rot
        align_initial_action = f"AlignOrientation({xx},{yy},{False})"
        micro_actions.append(align_initial_action)
        # print('align_initial_action: ',align_initial_action)
        
        for act_name, params in plan:
            action_str = self.convert_thortils_action((act_name, params))
            micro_actions.append(action_str)

        if poses and poses[-1] is not None:
            goal_pitch, goal_yaw = poses.pop()
            align_final_action = f"AlignOrientation({goal_pitch},{goal_yaw},{False})"
            micro_actions.append(align_final_action)
            # print('align_initial_action: ',align_initial_action)
        
        # print(f"nav_actions: {micro_actions}")
        return micro_actions

    # ---For one time planning- replan nav step for every step
    def step_decomp(self, actions: List[str], agent_id: int = None, isNeed2Add2Queue: bool = True):
        """break down the step into unit as define in Ai2Thor (MoveAhead, RotateRight etc.)"""
        res = []
        if agent_id is not None:
            # 分解特定agent的动作
            action = actions[agent_id]
            if action.startswith("NavigateTo"):
                obj = action.split("(")[1].rstrip(")")
                nav = f"NavigateTo({obj})"
                nav_steps = self.get_navigation_step(nav, agent_id)
                if isNeed2Add2Queue:
                    self.action_queue[agent_id].extend(nav_steps)
                res = nav_steps
                # nav_steps = self.get_navigation_step(action, agent_id)
                # self.action_queue[agent_id].extend(nav_steps)

            elif action.startswith(tuple(self.object_interaction_actions)):
                # obj = action.split("(")[1].rstrip(")")
                # nav = f"NavigateTo({obj})"
                # self.action_queue[agent_id].extend(self.get_navigation_step(nav, agent_id))
                if isNeed2Add2Queue:
                    self.action_queue[agent_id].append(action)
                res = action

            elif action in self.object_interaction_without_navigation:
                if isNeed2Add2Queue:
                    self.action_queue[agent_id].append(action)
                res = action

            elif action in self.idle_actions:
                if isNeed2Add2Queue:
                    self.action_queue[agent_id].append("Idle")
                res = "Idle"

            else:
                if isNeed2Add2Queue:
                    self.action_queue[agent_id].append(action)
                res = action
        else:
            # 分解所有agent的动作
            for agent_id, action in enumerate(actions):
                if action.startswith("NavigateTo"):
                    nav_steps = self.get_navigation_step(action, agent_id)
                    if isNeed2Add2Queue:
                        self.action_queue[agent_id].extend(nav_steps)
                    res.extend(nav_steps)
                    
                elif action.startswith(tuple(self.object_interaction_actions)):
                    obj = action.split("(")[1].rstrip(")")
                    nav = f"NavigateTo({obj})"
                    if isNeed2Add2Queue:
                        self.action_queue[agent_id].extend(self.get_navigation_step(nav, agent_id))
                        self.action_queue[agent_id].append(action)
                    res.append(action)

                elif action in self.object_interaction_without_navigation:
                    if isNeed2Add2Queue:
                        self.action_queue[agent_id].append(action)

                    res.append(action)

                elif action in self.idle_actions:
                    if isNeed2Add2Queue:
                        self.action_queue[agent_id].append("Idle")
                    res.append("Idle")

                else:
                    if isNeed2Add2Queue:
                        self.action_queue[agent_id].append(action)
                    res.append(action)
        return res
    
    
    def exe_step(self, actions:List[str]):
        """execute one step, each agent per step (can be IDLE)"""
        
        # self.step_decomp(actions)
        # print("curr action queue: ", self.action_queue)
        act_texts, act_successes = [], []
        any_terminal_fail = False 
        # print("Executing actions:", actions)
        for aid in range(self.num_agents):

            if self.current_hl[aid] and not self.action_queue[aid]:
                actions = ["Idle"] * self.num_agents
                actions[aid] = self.current_hl[aid]
                _ = self.step_decomp(actions, agent_id=aid)
            if self.save_logs:
                # self.logs.append(f"Executing action for agent {aid} ({self.agent_names[aid]}): {self.action_queue[aid]}")
                self.logs.append(f"""current high level task for agent {aid} ({self.agent_names[aid]}): {self.current_hl[aid]}""")
                self.logs.append(f"""remaining high level tasks for agent {aid} ({self.agent_names[aid]}): {self.pending_high_level[aid]}""")
            # print(f"Executing action for agent {aid} ({self.agent_names[aid]}")
            # print(f"""current high level task for agent {aid} ({self.agent_names[aid]}): {self.current_hl[aid]}""")
            # print(f"""remaining high level tasks for agent {aid} ({self.agent_names[aid]}): {self.pending_high_level[aid]}""")
            # print("******")
            # print(f"before exe_step: agent {aid} ({self.agent_names[aid]}) action queue: {self.action_queue[aid]}")

            # handle AlignOrientation separately
            if self.action_queue[aid] and self.action_queue[aid][0].startswith("AlignOrientation"):
                act = self.action_queue[aid].popleft()
                action_dict = self.parse_action(act, aid)
                if self.save_logs:
                    self.logs.append(f"Executing action for agent {aid} ({self.agent_names[aid]}): {self.action_queue[aid]}")
                self.event = self.controller.step(action_dict)
                success = self.event.events[aid].metadata["lastActionSuccess"]
            



            act = self.action_queue[aid].popleft() if self.action_queue[aid] else "Idle"
            # print(f"After exe_step: agent {aid} ({self.agent_names[aid]}) action queue: {self.action_queue[aid]}")
            # print("******")
            # print(act)


            if act != "Idle":
                if act.startswith('Rotate'):
                    action_dict = {"agentId": aid, "action": act, "degrees":30}
                    # action_dict = self.parse_action(act, aid)
                    for i in range(2):
                        self.event = self.controller.step(action_dict)
                        success = self.event.events[aid].metadata["lastActionSuccess"]
                        
                        self.save_last_frame(agent_id=aid, view="pov",
                                     filename=f"frame_{self.step_num[aid]}_{i+1}.png")
                        self.save_last_frame(view="overhead",
                                 filename=f"frame_{self.step_num[0]}_{i+1}.png")
                        if not success:
                            break
                    self.event = self.controller.step(action_dict)
                    if self.save_logs:
                        self.logs.append(f"Executing action for agent {aid} ({self.agent_names[aid]}): {action_dict}")
                    # print(f"Executing action for agent {aid} ({self.agent_names[aid]}): {action_dict}")
                    success = self.event.events[aid].metadata["lastActionSuccess"]
                else:
                    action_dict = self.parse_action(act, aid)
                    if self.save_logs:
                        self.logs.append(f"Executing action for agent {aid} ({self.agent_names[aid]}): {self.action_queue[aid]}")
                    # print(f"Executing action for agent {aid} ({self.agent_names[aid]}): {action_dict}")
                    self.event = self.controller.step(action_dict)
                    success = self.event.events[aid].metadata["lastActionSuccess"]
            else:
                success = True

            # TBD: handle blocking situation
            # by teleporting - start by testing small teleport distances in all directions, then keep increasing distance if nothing is found.
            if not success:
                err = self.event.events[aid].metadata.get("errorMessage") or "unknown-error"
                if "blocking" in err:
                    print("")


            self.step_num[aid] += 1
            if not success:
                if self.save_logs:
                    self.logs.append(f"Failed to Executing action for agent {aid} ({self.agent_names[aid]}): {action_dict}")
                    self.logs.append(f'errorMessage: {self.event.events[aid].metadata["errorMessage"] if self.event else "no-event"}')
                
                # print(f"Failed to Executing action for agent {aid} ({self.agent_names[aid]}): {action_dict}")
                # print(f'errorMessage: {self.event.events[aid].metadata["errorMessage"]}')
                err = self.event.events[aid].metadata.get("errorMessage") or "unknown-error"
                if "already" in err:
                    success = True
                    self.agent_failure_acts[self.agent_names[aid]] = []
                    if act.startswith("PickupObject"):
                        self.inventory[aid] = self.get_agent_object_held(aid)
                    elif act.startswith(("PutObject","DropHandObject")):
                        self.inventory[aid] = "nothing"
                else:
                    self._record_subtask_failure(aid, reason=f"failed-at: {act} ({err})", at_action=act)
                    self.agent_failure_acts[self.agent_names[aid]].append(act)
            else:
                self.agent_failure_acts[self.agent_names[aid]] = []
                if act.startswith("PickupObject"):
                    self.inventory[aid] = self.get_agent_object_held(aid)
                elif act.startswith(("PutObject","DropHandObject")):
                    self.inventory[aid] = "nothing"

            self.action_history[self.agent_names[aid]].append(act)
            self.action_success_history[self.agent_names[aid]].append(success)
            act_texts.append(self.get_act_text(act, success, aid))
            act_successes.append(success)

            sub = self.current_hl.get(aid)
            if sub:
                if self.save_logs:
                    self.logs.append(f'Checking if subtask: {sub} for agent {aid} ({self.agent_names[aid]}) is completed')
                    self.logs.append(f'previous success: {self.subtask_success_history[self.agent_names[aid]]}')
                # print(f'previous success: {self.subtask_success_history[self.agent_names[aid]]}')
                # print(f'check if subtask: {sub} is completed')
                if success and self.is_subtask_done(sub, aid):
                    if self.save_logs:
                        self.logs.append(f"Subtask {sub} for agent {aid} ({self.agent_names[aid]}) is done.")
                    print(f"Subtask {sub} for agent {aid} ({self.agent_names[aid]}) is done.")
                    self.subtask_success_history[self.agent_names[aid]].append(sub)
                    self.subtask_failure_reasons[self.agent_names[aid]] = []
                    self.current_hl[aid] = None
                    self.action_queue[aid].clear()
                    self.last_check_reason[self.agent_names[aid]] = None
                    self.nav_no_plan[self.agent_names[aid]] = False
                else:
                    last_reason = self.last_check_reason[self.agent_names[aid]]
                    
                    terminal = False
                    if last_reason and (last_reason == 'no-path' or 'not-exist' in last_reason) or self.nav_no_plan[self.agent_names[aid]]:
                        terminal = True
                    elif last_reason and last_reason.startswith("distance-too-far") and not self.action_queue[aid]:
                        last_reason += " (may be stuck because of unknown reason)"
                        # print(last_reason)
                        terminal = True
                  

                    if terminal:
                        self._record_subtask_failure(aid, reason=last_reason or "terminal-failure", at_action=sub)
                        self.current_hl[aid] = None
                        self.action_queue[aid].clear()
                        act_successes[aid] = False

                        self.last_check_reason[self.agent_names[aid]] = None  # 清空
                        self.nav_no_plan[self.agent_names[aid]] = False

                    else:
                        print(f'subtask: {sub} failed, errorMessage: {self.event.events[aid].metadata["errorMessage"] if self.event else ""}')
                    if self.save_logs:
                        self.logs.append(f'subtask: {sub} for agent {aid} ({self.agent_names[aid]}) failed, errorMessage: {self.event.events[aid].metadata["errorMessage"] if self.event else ""}')
                        self.logs.append(f'last reason: {last_reason}')
                    # print(f'subtask: {sub} failed, errorMessage: {last_reason}')
                # elif not success:
                #     self.action_queue[aid].clear()
                #     # replan TBD
                #     self.step_decomp([ sub if i==aid else "Idle"
                #                        for i in range(self.num_agents) ])

            # handle AlignOrientation separately
            if self.action_queue[aid] and self.action_queue[aid][0].startswith("AlignOrientation"):
                act = self.action_queue[aid].popleft()
                action_dict = self.parse_action(act, aid)
                if self.save_logs:
                    self.logs.append(f"Executing action for agent {aid} ({self.agent_names[aid]}): {self.action_queue[aid]}")
                self.event = self.controller.step(action_dict)
                success = self.event.events[aid].metadata["lastActionSuccess"]
            if (sub and not sub.startswith("NavigateTo")):  
                self.action_queue[aid].clear()
            self.logs.append(f"Current success subtask: {self.subtask_success_history}")

            # if not self.skip_save_dir:
            self.save_last_frame(agent_id=aid, view="pov",
                                     filename=f"frame_{self.step_num[aid]}.png")

        # if self.overhead and not self.skip_save_dir:
        self.save_last_frame(view="overhead",
                                 filename=f"frame_{self.step_num[0]}.png")

        return self.get_observations(), act_successes
    
   

    def action_loop(self, high_level_tasks: List[str]):
        """execute the actions from high level
        high_level_tasks: e.g.
          [
            [subtasks for agent_0],
            [subtasks for agent_2]
          ]
        """
        # init.
        for aid, tasks in enumerate(high_level_tasks):
            self.pending_high_level[aid] = deque(tasks)
            self.current_hl[aid] = None
            self.action_queue[aid].clear()
        
        if self.save_logs:
            self.logs.append(f"Initializing action loop with high level tasks:")
            self.logs.append(f"pending high level tasks: {self.pending_high_level}")
            self.logs.append(f"nxt task: {self.current_hl}")
            self.logs.append(f"current action queue: {self.action_queue}")
            self.logs.append("-------------------------")
        # print("Initializing action loop with high level tasks:")
        # print(f"pending high level tasks: {self.pending_high_level}")
        # print(f"nxt task: {self.current_hl}")
        # print(f"current action queue: {self.action_queue}")
        # print("-------------------------")

        history = []
        start_time = time.time()
        while True:
            for aid in range(self.num_agents):
                if not self.current_hl[aid] and self.pending_high_level[aid]:
                    nxt = self.pending_high_level[aid].popleft()
                    self.current_hl[aid] = nxt
                # add event manually
                # if self.step_num[aid] == 6:
                #     success, message = self.simulate_environment_event("break", "Mug_1")


            # break if not pending tasks
            if all(not self.pending_high_level[aid] for aid in range(self.num_agents)) \
               and all(self.current_hl[aid] is None for aid in range(self.num_agents)) \
               and all(not self.action_queue[aid] for aid in range(self.num_agents)):
                break
            
            # break if timeout
            elapsed = time.time() - start_time
            if self.timeout and elapsed > self.timeout:
                break
            if self.save_logs:
                self.logs.append("-------------------------")
                self.logs.append(f"pending high level tasks: {self.pending_high_level}")
                self.logs.append(f"nxt task: {self.current_hl}")
                self.logs.append(f"current action queue: {self.action_queue}")

            # print(f"pending high level tasks: {self.pending_high_level}")
            # print(f"nxt task: {self.current_hl}")
            # print(f"current action queue: {self.action_queue}")
            

            obs, succ = self.exe_step([])
            history.append((obs, succ))
            # debug
            if self.save_logs:
                self.logs.append(f"----------Step {self.step_num[0]}------------")
                self.logs.append(f"Step {self.step_num[0]}: {obs}")
                self.logs.append(f"Action success: {succ}")
            # print(f"----------Before {self.step_num[0]}------------")
            # print(f"pending high level tasks: {self.pending_high_level}")
            # print(f"nxt task: {self.current_hl}")
            # print(f"current action queue: {self.action_queue}")
            # print(f"----------After {self.step_num[0]}------------")

        if self.save_logs:
            self.logs.append("----------End-----------")
            filename = self.base_path / "logs.txt"
            with open(filename, "w", encoding="utf-8") as f:
                for log in self.logs:
                    f.write(str(log) + "\n")

        return history
    # ----

    def get_decomp_steps(self, pending_subtasks: Dict[int, List[str]]):
        """Set the action queue for each agent."""
        for aid, tasks in enumerate(pending_subtasks):
            self.pending_high_level[aid] = deque(tasks)
            # self.current_hl[aid] = None
            # self.action_queue[aid].clear()
            action = self.step_decomp(tasks, aid, isNeed2Add2Queue=False)
        return action

    def actions_decomp(self, actions: List[str]):
        '''
        從llm_c.py接收一系列動作，將其各自分解成可執行的原子動作
        Input:
        actions:
        [
            ['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)'], 
            ['NavigateTo(Lettuce_1)', 'PickupObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)']
        ]
        
        Output:
        [
        [[nav_step],[PickupObject]...],
        []
        ]

        '''
        res = []
        # 分解特定agent的动作
        for agent_id, agent_actions in enumerate(actions):
            res_agent = []
            for action in agent_actions:
                if action.startswith("NavigateTo"):
                    obj = action.split("(")[1].rstrip(")")
                    nav = f"NavigateTo({obj})"
                    nav_steps = self.get_navigation_step(nav, agent_id)
                    res_agent.append(nav_steps)
                    # nav_steps = self.get_navigation_step(action, agent_id)
                    # self.action_queue[agent_id].extend(nav_steps)
                elif action in self.idle_actions:
                    # self.action_queue[agent_id].append("Idle")
                    res_agent.append(["Idle"])
                else:
                    
                    # self.action_queue[agent_id].append(action)
                    res_agent.append([action])
            res.append(res_agent)
        return res
    
    def stepwise_action_loop(self, cur_plan):
        """
        cur_plan: list of dicts, each with:
            - subtask (str)
            - actions (List[str])
            - steps (List[List[str]])  # atomic per action

        """
        print("self.timeout: ", self.timeout)
        self.cur_plan = cur_plan
        if self.save_logs:
            self.logs.append(f"----Start a new plan----")
            self.logs.append(f"Current plan: {self.cur_plan}")


        pending_actions = []
        for aid, plan in enumerate(cur_plan):
            pending_actions.append(plan["actions"])

        # init.
        for aid, tasks in enumerate(pending_actions):
            self.pending_high_level[aid] = deque(tasks)
            self.current_hl[aid] = None
            self.action_queue[aid].clear()
        
        if self.save_logs:
            self.logs.append(f"Initializing action loop with high level tasks:")
            self.logs.append(f"pending high level tasks: {self.pending_high_level}")
            self.logs.append(f"nxt task: {self.current_hl}")
            self.logs.append(f"current action queue: {self.action_queue}")
            self.logs.append("-------------------------")
        # history = []
        start_time = time.time()
        while True:
            for aid in range(self.num_agents):
                if not self.current_hl[aid] and self.pending_high_level[aid]:
                    nxt = self.pending_high_level[aid].popleft()
                    self.current_hl[aid] = nxt
                # add event manually
                # if self.step_num[aid] == 6:
                #     success, message = self.simulate_environment_event("break", "Mug_1")

            # break if not pending tasks
            if all(not self.pending_high_level[aid] for aid in range(self.num_agents)) \
               and all(self.current_hl[aid] is None for aid in range(self.num_agents)) \
               and all(not self.action_queue[aid] for aid in range(self.num_agents)):
                break
            
            # break if timeout
            elapsed = time.time() - start_time
            
            if self.timeout and elapsed > self.timeout:
                if self.save_logs:
                    self.logs.append("**********Timeout!**********")
                # return False, self._gather_status()
                break

            if self.save_logs:
                self.logs.append("-------------------------")
                self.logs.append(f"pending high level tasks: {self.pending_high_level}")
                self.logs.append(f"nxt task: {self.current_hl}")
                self.logs.append(f"current action queue: {self.action_queue}")

            # execute actions
            obs, succ = self.exe_step([])
            # history.append((obs, succ))
            # debug
            if self.save_logs:
                self.logs.append(f"----------Step {self.step_num[0]}------------")
                self.logs.append(f"Step {self.step_num[0]}: {obs}")
                self.logs.append(f"Action success: {succ}")

            # # 若有 agent 失敗，立即跳出
            if not all(succ):
                break
            # start_time = time.time()
            
        if self.save_logs:
            self.logs.append("----------End-----------")
            filename = self.base_path / "logs.txt"
            with open(filename, "a", encoding="utf-8") as f:
                for log in self.logs:
                    f.write(str(log) + "\n")
        info = self._get_current_plan_status()
        # reset after each attempt
        self.subtask_success_history = {name: [] for name in self.agent_names}  # save the previous successful subtasks for each agent
        self.subtask_failure_reasons = {name: [] for name in self.agent_names} # save the previous failure subtasks and reason for each agent
        self.agent_failure_acts	= {name: [] for name in self.agent_names} # save 最近失敗過的 atomic action，通常用於偵測是否卡住
        self.nav_no_plan = {name: False for name in self.agent_names}   # 導航規劃為空的旗標
        self.last_check_reason = {name: None for name in self.agent_names}  # is_subtask_done 內部最近一次的判斷理由
        return all(succ), info
        
   
    def _get_current_plan_status(self):
        # 回傳目前任務狀態
        succ = []
        fail = []
        for aid in range(self.num_agents):
            if not self.current_hl[aid] and not self.subtask_failure_reasons[self.agent_names[aid]]:
                succ.append(self.cur_plan[aid]["subtask"])
            else:
                fail.append(self.cur_plan[aid]["subtask"])

        return {
            "step": self.action_step_num,
            "actions_success": self.subtask_success_history,
            "success_subtasks": succ,
            "failure_subtasks": fail,
            "subtask_failure_reasons": self.subtask_failure_reasons,
            "inventory": self.inventory.copy(),
            "failed_acts": self.agent_failure_acts,
        }

    def save_log(self):
        if self.save_logs:
            self.logs.append("----------End-----------")
            filename = self.base_path / "logs.txt"
            with open(filename, "a", encoding="utf-8") as f:
                for log in self.logs:
                    f.write(str(log) + "\n")
            self.logs = []  # clear logs after saving

    def _record_subtask_failure(self, agent_id: int, reason: str, at_action: str | None = None):
        """
        將當前 agent 所屬的「自然語言 subtask」記一筆失敗理由。
        - reason: "no-path" / "failed-at: MoveAhead" / "object-not-in-view" / "distance-too-far (1.73m)" / try/except 錯誤訊息 ...
        - at_action: 可填入目前的高階或低階動作字串，便於 LLM 重建脈絡
        """
        print(f'Recording subtask failure for agent {agent_id} ({self.agent_names[agent_id]}): {reason}')
        agent_name = self.agent_names[agent_id]
        nl_subtask = None
        if self.cur_plan and agent_id < len(self.cur_plan):
            nl_subtask = self.cur_plan[agent_id].get("subtask")  # 自然語言版本

        if nl_subtask is None:
            nl_subtask = self.current_hl.get(agent_id) or "Unknown-Subtask"

        payload = {"subtask": nl_subtask, "reason": reason}
        if at_action:
            payload["at_action"] = at_action

        self.subtask_failure_reasons[agent_name] = payload

        if at_action:
            self.agent_failure_acts[agent_name].append(at_action)



    def is_subtask_done(self, subtask: str, agent_id: int) -> bool:
        """
        check if a subtask is completed based on env.state
        - PickupObject(obj)
        - PutObject(obj)
        - OpenObject(obj)
        - CloseObject(obj)
        - ToggleObjectOn(obj)
        - ToggleObjectOff(obj)
        - BreakObject(obj)
        - SliceObject(obj)
        - FillObjectWithLiquid(obj)
        - EmptyLiquidFromObject(obj)
        - UseUpObject(obj)
        - DirtyObject(obj)
        - CleanObject(obj)

        True for :
        self.move_actions = ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]
        self.rotate_actions = ["RotateRight", "RotateLeft"]
        self.look_actions = ["LookUp", "LookDown"]
        self.idle_actions = ["Done", "Idle"]
        self.subtask_success_history
        """
        suc = False
        if subtask in self.idle_actions or subtask in self.move_actions or subtask in self.rotate_actions or subtask in self.look_actions or subtask in self.object_interaction_without_navigation: 
            suc = True
            # print('cam degree ', self.event.events[agent_id].metadata["agent"]["cameraHorizon"])
            # self.subtask_success_history[self.agent_names[agent_id]].append(subtask)
            return suc
        

        # subtask  "PickupObject(Tomato_1)"
        name, obj = subtask[:-1].split("(", 1)
        # print(f'checking action: {name} with obj: {obj}')
        if name in self.look_actions:
            suc = True
            # print('cam degree ', self.event.events[agent_id].metadata["agent"]["cameraHorizon"])
            # self.subtask_success_history[self.agent_names[agent_id]].append(subtask)
            # return suc
        
        elif name == "PickupObject":
            suc = self.inventory[agent_id] == obj
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-picked-up"
            # return self.inventory[agent_id] == obj

        elif name == "PutObject":
            if self.inventory[agent_id] != "nothing":
                suc = False
                if not suc:
                    self.last_check_reason[self.agent_names[agent_id]] = f"object({self.inventory[agent_id]})-not-put-down"
                return suc
            else: # TBD: need more test
                recep_status = self.get_object_status(obj)
                contains = recep_status.get("contains") or []
                prev_sub = ""
                for i in range(len(self.subtask_success_history[self.agent_names[agent_id]])-1, -1, -1):
                    a = self.subtask_success_history[self.agent_names[agent_id]][i]
                    if a.startswith("PickupObject"):
                        # check if the previous subtask is pickup
                        # e.g. suppose to be navigateTo() -> pickup() -> put()
                        prev_sub = a
                        break
                # prev_sub = self.subtask_success_history[self.agent_names[agent_id]][-2] # suppose to be navigateTo() -> pickup()
                if not prev_sub:
                    if not suc:
                        self.last_check_reason[self.agent_names[agent_id]] = "did-not-pickup-any-object-before-put"
                    return False
                prev_name, prev_obj = prev_sub[:-1].split("(", 1)
                prev_obj = prev_obj.split("_")[0]
                if self.save_logs:
                    self.logs.append(f"Previous action: {prev_name} with object: {prev_obj}")
                # print(f"Previous action: {prev_name} with object: {prev_obj}")
                
                for obj_in_container in contains:
                    if obj_in_container.startswith(prev_obj):
                        suc = True
                        break
                if self.save_logs:
                    self.logs.append(f"Checking if {prev_obj} is in the container: {contains}: {suc}")
                # print(f"Checking if {prev_obj} is in the container: {contains}: {suc}")
                if not suc:
                    self.last_check_reason[self.agent_names[agent_id]] = f"object({prev_obj})-not-put-into-container({obj})"

        elif name == "OpenObject":
            suc = self.get_object_status(obj).get("is_open", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-opened"
            # return self.get_object_status(obj).get("is_open", False)
        elif name == "CloseObject":
            suc = not self.get_object_status(obj).get("is_open", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-closed"
            # return not self.get_object_status(obj).get("is_open", False)

        elif name == "ToggleObjectOn":
            suc = self.get_object_status(obj).get("is_on", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-toggled-on"
            # return self.get_object_status(obj).get("is_on", False)
        elif name == "ToggleObjectOff":
            suc = not self.get_object_status(obj).get("is_on", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-toggled-off"
            # return not self.get_object_status(obj).get("is_on", False)

        elif name == "BreakObject":
            suc = self.get_object_status(obj).get("isBroken", False)
            if suc:
                self.update_object_dict()
                # print(self.object_dict)
            else:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-broken"
            # return self.get_object_status(obj).get("isBroken", False)
        elif name == "SliceObject":
            suc = self.get_object_status(obj).get("isSliced", False)
            if suc:
                self.update_object_dict()
                # print(self.object_dict)
            else:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-sliced"
            # print(self.get_all_objects())
            # return self.get_object_status(obj).get("isSliced", False)

        elif name == "FillObjectWithLiquid":
            suc = self.get_object_status(obj).get("isFilledWithLiquid", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-filled-with-liquid"
            # return self.get_object_status(obj).get("isFilledWithLiquid", False)
        elif name == "EmptyLiquidFromObject":
            suc = not self.get_object_status(obj).get("isFilledWithLiquid", True)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-emptied-from-liquid"
        elif name == "UseUpObject":
            suc = not self.get_object_status(obj).get("isUsedUp", True)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-used-up"
            # print(self.get_object_status(obj))
            # return not self.get_object_status(obj).get("isFilledWithLiquid", True)
        elif name == "DirtyObject":
            suc = self.get_object_status(obj).get("isDirty", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-dirty"
        elif name == "CleanObject":
            suc = not self.get_object_status(obj).get("isDirty", False)
            if not suc:
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-cleaned"

        elif name == "NavigateTo":
            # check if the agent is at the object and if the object is in view (by default visibilty: dist 1.5m)
            # obj_id = self.convert_readable_object_to_id(obj)
            try:
                obj_id = self.convert_readable_object_to_id(obj)
            except ValueError:
                # 記錄失敗並回報 False
                self.last_check_reason[self.agent_names[agent_id]] = f"object({obj})-not-exist"
                # self._record_subtask_failure(agent_id, reason=f"object({obj})-not-exist", at_action=subtask)
                return False

            agent_pos = self.get_agent_position_dict(agent_id)
            obj_metadata = next((o for o in self.event.metadata["objects"] if o["objectId"] == obj_id), None)
            # check distance -> in view -> center 
            agent_pos = self.get_agent_position_dict(agent_id)
            obj_metadata = next(obj for obj in self.event.metadata["objects"] if obj["objectId"] == obj_id)
            obj_pos = obj_metadata["position"]
            obj_name = obj.split("_")[0]
    
            dist = ((agent_pos["x"] - obj_pos["x"])**2 + (agent_pos["z"] - obj_pos["z"])**2)**0.5
            if self.save_logs:
                self.logs.append(f"Checking distance for object {obj_id} at {obj_pos} from agent {agent_id} at {agent_pos}: {dist:.2f}m")
            print(f"Checking distance for object {obj_id} at {obj_pos} from agent {agent_id} at {agent_pos}: {dist:.2f}m")
            if dist > 1.0 and dist < 1.5 and obj_name in self.large_receptacles and obj_id in self.get_object_in_view(agent_id):
                suc = True
                self.current_hl[agent_id] = None
                self.action_queue[agent_id].clear()
            elif agent_id > 0 and dist > 1.0 and dist < 1.5 and obj_name in self.large_receptacles and obj_id in self.get_object_in_view(agent_id):
                suc = True
                self.current_hl[agent_id] = None
                self.action_queue[agent_id].clear()
            # elif dist > 1.2:
            #     # error_type += f": distance-too-far ({dist:.2f}m)"
            #     if not self.action_queue[agent_id]:
            #         self.last_check_reason[self.agent_names[agent_id]] = f"no path to object {obj_id} with distance ({dist:.2f}m), may be block by other agent, should wait for one step"
            #     suc = False
            #     return suc 
            # elif obj_id in self.get_object_in_view(agent_id):
            #         #  and obj_name in self.receptacle_objects
            #         suc = True
            #         self.subtask_success_history[self.agent_names[agent_id]].append(subtask)
            #         return suc
            elif dist <= 1.0:
                if obj_id not in self.get_object_in_view(agent_id) and obj_name not in self.small_objects and not self.action_queue[agent_id]:
                    # not in view
                    self.last_check_reason[self.agent_names[agent_id]] = "object-not-in-view"
                    suc = False
                    return suc

                else:
                  
                    if obj_id in self.get_object_in_view(agent_id):
                        detections = self.event.events[agent_id].instance_detections2D
                        obj_bbox = detections[obj_id]
                        pitch_diff, act_pitch = self.estimate_pitch_to_center_object(obj_bbox, self.event.events[agent_id].frame.shape)
                        yaw_diff, act_yaw = self.estimate_yaw_to_center_object(obj_bbox, self.event.events[agent_id].frame.shape)
                        # print(f'obj_bbox: {obj_bbox}, pitch diff: {pitch_diff}, yaw_diff: {yaw_diff}')
                        suc = True 
                        p_degree = closest_angles(V_ANGLES, abs(pitch_diff))
                        if p_degree != 0:
                            # dist is close enough but cam degree is incorrect
                            # p_degree = closest_angles(V_ANGLES, abs(pitch_diff))
                            # print(f'need to change pitch to {p_degree}')
                            self.pending_high_level[agent_id].appendleft(act_pitch+"(" + str(p_degree) + ")")
                      
                    else:
                        if self.save_logs:
                            self.logs.append(f'in is_sub_task_done(): naivifate-to-object({obj})-failed')
                        # print(f'in is_sub_task_done(): naivifate-to-object({obj})-failed')
                        obj_in_view = self.get_mapping_object_pos_in_view(agent_id)
                        # print(f'agent {agent_id} can see: {obj_in_view}')
                        self.last_check_reason[self.agent_names[agent_id]] = f"naivifate-to-object({obj})-failed"
                        suc = False

        if suc:
            # self.subtask_success_history[self.agent_names[agent_id]].append(subtask)
            if name not in self.look_actions and name != "NavigateTo":
                # reset the pitch
                self.add_reset_pitch_subtask(agent_id)
            
        return suc
    
   
    def get_obj_in_containers(self):
        res = {}
        obj_list = self.get_all_objects()
        for obj in obj_list:
            obj_name, obj_id = obj.split("_")
            if obj_name in self.large_receptacles:
                recep_status = self.get_object_status(obj)
                contains = recep_status.get("contains") or []
                new_contains = []
                for id in contains:
                    readable = self.get_single_readable_object(id)
                    new_contains.append(readable)
                res[obj] = new_contains
        return res
    
    def get_pitch_reset_command(self, cur_deg):
        p_degree = closest_angles(V_ANGLES, abs(cur_deg))
        if self.save_logs:
            self.logs.append(f'Need to change pitch to {p_degree}')
        # print(f'Need to change pitch to {p_degree}')
        
        if p_degree > 0:
            # Currently looking down → need to look up
            return "LookUp(" + str(p_degree) + ")", p_degree
        elif p_degree < 0:
            # Currently looking up → need to look down
            return "LookDown(" + str(p_degree) + ")", p_degree
        else:
            return "", p_degree

        
    def add_reset_pitch_subtask(self, agent_id):
        cur_pitch = self.get_cur_cam_pitch(agent_id)
        command, _ = self.get_pitch_reset_command(cur_pitch)
        if self.save_logs:
            self.logs.append(f'Reset command: {command}')
        # print('Reset command:', command)
        if command:
            self.pending_high_level[agent_id].appendleft(command)



    def estimate_pitch_to_center_object(self, bbox, frame_shape=(1000, 1000, 3), vertical_fov_deg=60):
        '''
        calculate if the object is in center view, based on the from agent's 2d view
        '''
        height = frame_shape[0]
        y1, y2 = bbox[1], bbox[3]
        y_center = (y1 + y2) / 2
        y_norm = y_center / height
        delta_pitch = (y_norm - 0.5) * vertical_fov_deg
        if abs(delta_pitch) < 1:
            return 0, "centered"
        elif delta_pitch > 0:
            return delta_pitch, "LookDown"
        else:
            return -delta_pitch, "LookUp"
        
    def estimate_yaw_to_center_object(self, obj_bbox, frame_shape=(1000, 1000, 3)):
        """
        Estimate the horizontal angle needed to center the object.
        Returns (angle_diff, action_name), where action is RotateLeft or RotateRight.
        """
        x1, _, x2, _ = obj_bbox
        frame_width = frame_shape[1]
        bbox_center_x = (x1 + x2) / 2
        frame_center_x = frame_width / 2

        # offset: + if object is on right side, - if on left
        offset = bbox_center_x - frame_center_x

        # 假設最大視角為 90 度，估算角度差異（簡化線性推估）
        max_angle = 90
        angle_diff = (offset / (frame_width / 2)) * (max_angle / 2)

        if abs(angle_diff) < 30:  # 小於5度視為已對準
            return 0, ""

        action = "RotateRight" if angle_diff > 0 else "RotateLeft"
        degree = closest_angles(H_ANGLES, abs(angle_diff))
        return degree, action

    def run_task_check(self):
        if not getattr(self, "checker", None):
            return True, {"ok": True, "notes": ["no checker configured"]}
        ok, report = self.checker.check(self)
        if self.save_logs:
            print(f"[TaskCheck] ok={ok}\n{report}")
        return ok, report

    # def is_object_in_center_view(self, agent_pos, object_pos, agent_yaw_deg, agent_cam, threshold_deg=30) -> bool:
    #     """
    #     Return True if the object is within threshold degrees in both yaw and pitch from the agent's forward direction.
    #     agent_cam: camera pitch (0 = forward, >0 = looking down, <0 = looking up)
    #     """
    #     print(f'check if obj {object_pos} is in center view of {agent_pos} with yaw {agent_yaw_deg}, agent cam {agent_cam}')
    #     # 1. Compute direction vector from agent to object
    #     dx = object_pos['x'] - agent_pos['x']
    #     dy = object_pos['y'] - agent_pos['y']
    #     dz = object_pos['z'] - agent_pos['z']
    #     dir_vec = np.array([dx, dy, dz])
    #     dir_norm = np.linalg.norm(dir_vec)
    #     if dir_norm == 0:
    #         return True  # agent and object at same location
    #     dir_unit = dir_vec / dir_norm

    #     # 2. Convert to spherical angles: yaw and pitch to object
    #     yaw_obj_rad = math.atan2(dx, dz)  # yaw: horizontal
    #     pitch_obj_rad = math.asin(dy / dir_norm)  # pitch: vertical

    #     yaw_obj_deg = math.degrees(yaw_obj_rad)  # [-180, 180]
    #     pitch_obj_deg = -math.degrees(pitch_obj_rad)  # AI2-THOR: down is positive

    #     # 3. Normalize agent yaw to [-180, 180] and compute diffs
    #     yaw_diff = ((yaw_obj_deg - agent_yaw_deg + 180) % 360) - 180
    #     pitch_diff = pitch_obj_deg - agent_cam  # compare with camera horizon
    #     print(f'yaw diff: {yaw_diff}, pitch diff: {pitch_diff}')
    #     # 4. Check if both yaw and pitch are within threshold
    #     return abs(yaw_diff) < threshold_deg and abs(pitch_diff) < threshold_deg

    

    # unit of action: Pickup (include NavigateTo Object and pickup Object) 
    def step(self, actions: List[str]) -> Tuple[str, List[bool]]:
        """Execute actions for all agents and return observations and success flags."""
        act_successes, act_texts = [], []
        for agent_id, action in enumerate(actions):
            error_type = None
            if action.startswith("NavigateTo"):
                act_success, error_type = self.navigation_step(action, agent_id)
            elif action in ["Done", "Idle"]:
                act_success = True
            elif action.startswith(tuple(self.object_interaction_actions)):
                object_id = action.split("(")[1].rstrip(")")
                obj_id = self.convert_readable_object_to_id(object_id)
                nav_action = f"NavigateTo({object_id})"
                nav_success, nav_error = self.navigation_step(nav_action, agent_id)
                if self.save_logs:
                    self.logs.append(f"Navigation action for {action}: {nav_action}, success: {nav_success}, error: {nav_error}")
                print('nav_success: ', nav_success, nav_error)
                if not nav_success:
                    act_success = False
                    error_type = nav_error
                else:
                    action_dict = self.parse_action(action, agent_id)
                    self.event = self.controller.step(action_dict)
                    act_success = self.event.events[agent_id].metadata["lastActionSuccess"]
                    if not act_success:
                        error_type = "interaction-failed"
                        if obj_id not in self.get_object_in_view(agent_id):
                            error_type += ": object-not-in-view"
                        else:
                            agent_pos = self.get_agent_position_dict(agent_id)
                            obj_metadata = next(obj for obj in self.event.metadata["objects"] if obj["objectId"] == obj_id)
                            obj_pos = obj_metadata["position"]
                            dist = ((agent_pos["x"] - obj_pos["x"])**2 + (agent_pos["z"] - obj_pos["z"])**2)**0.5
                            if dist > 1.5:
                                error_type += f": distance-too-far ({dist:.2f}m)"
            else:
                action_dict = self.parse_action(action, agent_id)
                self.event = self.controller.step(action_dict)
                act_success = self.event.events[agent_id].metadata["lastActionSuccess"]
            
            self.step_num[agent_id] += 1
            if act_success:
                self.agent_failure_acts[self.agent_names[agent_id]] = []
                if action.startswith("PickupObject"):
                    self.inventory[agent_id] = self.get_agent_object_held(agent_id)
                elif action.startswith("PutObject") or action.startswith("DropHandObject"):
                    self.inventory[agent_id] = "nothing"
            else:
                self.agent_failure_acts[self.agent_names[agent_id]].append(action)
            
            self.action_history[self.agent_names[agent_id]].append(action)
            self.action_success_history[self.agent_names[agent_id]].append(act_success)
            act_texts.append(self.get_act_text(action, act_success, agent_id, error_type))
            act_successes.append(act_success)
        
        if not self.skip_save_dir:
            for agent_id in range(self.num_agents):
                self.save_last_frame(agent_id=agent_id, view="pov", filename=f"frame_{self.step_num[agent_id]}.png")
            if self.overhead:
                self.save_last_frame(view="overhead", filename=f"frame_{self.step_num[0]}.png")

        self.update_current_state(act_texts)
        return self.get_observations(), act_successes
   
    def navigation_step(self, action: str, agent_id: int) -> Tuple[bool, str]:
        if self.save_logs:
            self.logs.append(f"Agent {agent_id} performing NavigateTo: {action}")
        print(f"Agent {agent_id} performing NavigateTo: {action}")
        object_name = action.split("(")[1].rstrip(")")
        try:
            from thortils.constants import H_ANGLES, V_ANGLES
            # from thortils.utils import closest_angles

            obj_id = self.convert_readable_object_to_id(object_name)
            other_agents = [
                self.event.events[i].metadata["agent"]["position"]
                for i in range(self.num_agents) if i != agent_id
            ]
            cur_pos = self.get_agent_position_dict(agent_id)
            cur_rot = self.event.events[agent_id].metadata["agent"]["rotation"]
            cur_pos_tuple = (cur_pos["x"], cur_pos["y"], cur_pos["z"])
            cur_rot_tuple = (
                closest_angles(V_ANGLES, cur_rot["x"]),
                closest_angles(H_ANGLES, cur_rot["y"]),
                cur_rot["z"]
            )

            # 預先校正 pitch
            pitch = self.event.events[agent_id].metadata["agent"]["cameraHorizon"]
            pitch = closest_angles(V_ANGLES, pitch)
            if pitch != 0:
                self.event = self.controller.step({
                    "action": "LookUp" if pitch > 0 else "LookDown",
                    "agentId": agent_id,
                    "degrees": abs(pitch)
                })
                self.save_frame()

            # 規劃並執行路徑
            poses, plan = get_shortest_path_to_object(
                self.controller, obj_id,
                cur_pos_tuple, cur_rot_tuple, return_plan=True, cur_step=self.step_num[agent_id],
            )
            if self.save_logs:
                self.logs.append(f"Planned path to {obj_id}: {plan}")
            print('action plan', plan)
            if plan is None:
                return False, "no-path"

            for name, param in plan:
                if name in ["LookUp", "LookDown"]:
                    action_dict = {"action": name, "agentId": agent_id, "degrees": abs(param[2])}
                elif name in ["RotateLeft", "RotateRight"]:
                    action_dict = {"action": name, "agentId": agent_id, "degrees": abs(param[1])}
                else:
                    action_dict = {"action": name, "agentId": agent_id}
                self.event = self.controller.step(action_dict)
                self.step_num[agent_id] += 1
                self.save_frame()
                if not self.event.events[agent_id].metadata["lastActionSuccess"]:
                    return False, f"failed-at: {name}"

            # 最後根據 poses 補校正 yaw/pitch
            final_rot = poses[-1][1]
            cur_rot = self.event.events[agent_id].metadata["agent"]["rotation"]
            yaw_diff = ((final_rot["y"] - cur_rot["y"] + 180) % 360) - 180
            pitch_diff = final_rot["x"] - cur_rot["x"]

            while abs(yaw_diff) > 1:
                step = min(45, abs(yaw_diff))
                self.event = self.controller.step({
                    "action": "RotateRight" if yaw_diff > 0 else "RotateLeft",
                    "agentId": agent_id,
                    "degrees": step
                })
                yaw_diff -= step if yaw_diff > 0 else -step
                self.step_num[agent_id] += 1
                self.save_frame()

            while abs(pitch_diff) > 1:
                step = min(30, abs(pitch_diff))
                self.event = self.controller.step({
                    "action": "LookUp" if pitch_diff > 0 else "LookDown",
                    "agentId": agent_id,
                    "degrees": step
                })
                pitch_diff -= step if pitch_diff > 0 else -step
                self.step_num[agent_id] += 1
                self.save_frame()

            return True, None

        except Exception as e:
            if self.save_logs:
                self.logs.append("[EXCEPTION] navigation_step error:")
                self.logs.append(traceback.format_exc())
            print("[EXCEPTION] navigation_step error:")
            print(traceback.format_exc())
            return False, f"exception: {str(e)}"

   
    def convert_thortils_action(self, action: Tuple[str, Tuple]) -> str:
        """
        Convert thortils-style action to env action string, e.g. ('MoveAhead', ()) → "MoveAhead"
        or ('LookUp', (0,0,-30)) → "LookDown(30)"
        """
        act = action[0]
        params = action[1]

        if act == "MoveAhead":
            return "MoveAhead"
        elif act == "MoveRight":
            return "MoveRight"
        elif act == "MoveLeft":
            return "MoveLeft"
        elif act == "MoveBack":
            return "MoveBack"
        elif act == "RotateRight":
            return "RotateRight"
        elif act == "RotateLeft":
            return "RotateLeft"
        elif act == "LookUp":
            angle = params[2]
            return f"LookUp({abs(angle)})" if angle > 0 else f"LookDown({abs(angle)})"
        elif act == "LookDown":
            angle = params[2]
            return f"LookDown({abs(angle)})" if angle > 0 else f"LookUp({abs(angle)})"
        elif act == "AlignOrientation":
            pitch, yaw, z_flag = params
            return f"AlignOrientation({pitch},{yaw},{z_flag})"
        else:
            raise ValueError(f"Unknown thortils action: {action}")
    
    # --- From LLaMAR
    def update_subtask(self, subtask: str, agent_id: int = 0):
        """Update subtask for shared or per-agent usage."""
        if self.use_shared_subtask:
            self.subtasks[0] = subtask
        elif self.use_separate_subtask:
            self.subtasks[agent_id] = subtask
    
    
    
    def update_current_state(self, act_texts: List[str]):
        """Update the input dictionary with the current state."""
        self.new_all_obs = []
        for agent_id, agent_name in enumerate(self.agent_names):
            obs, obs_list = self.generate_obs_text(agent_id, mode="mapping")
            self.input_dict[f"{agent_name}'s observation"] = obs
            self.input_dict[f"{agent_name}'s state"] = self.get_agent_state(agent_id)
            self.input_dict[f"{agent_name}'s previous action"] = act_texts[agent_id]
            if self.use_action_failure:
                self.input_dict[f"{agent_name}'s previous failures"] = self.get_act_failure_text(
                    self.agent_failure_acts[agent_name], agent_id
                )
            if self.use_shared_subtask:
                self.input_dict["Robots' subtasks"] = self.subtasks[0]
            elif self.use_separate_subtask:
                self.input_dict[f"{agent_name}'s subtask"] = self.subtasks[agent_id]
            if self.use_shared_memory:
                self.input_dict["Robots' combined memory"] = self.memory[0]
            elif self.use_separate_memory:
                self.input_dict[f"{agent_name}'s memory"] = self.memory[agent_id]
            if self.use_plan:
                self.input_dict["Robots' open subtasks"] = self.open_subtasks
                self.input_dict["Robots' completed subtasks"] = self.closed_subtasks
            self.new_all_obs.append(obs_list)
        self.all_obs_dict = {self.agent_names[i]: self.new_all_obs[i] for i in range(self.num_agents)}
    # -----

    def update_plan(self, open_subtasks: List[str], closed_subtasks: List[str]):
        self.open_subtasks = open_subtasks
        self.closed_subtasks = closed_subtasks
        self.input_dict["Robots' open subtasks"] = self.open_subtasks
        self.input_dict["Robots' completed subtasks"] = self.closed_subtasks
    
    def update_memory(self, memory: str, suggestion="", agent_id: int = 0):
        """Update memory for shared or per-agent usage."""
        if self.use_shared_memory:
            self.memory[0] = memory
        elif self.use_separate_memory:
            self.memory[agent_id] = memory
            
        if suggestion:
            self.update_suggestion(suggestion)

    def update_suggestion(self, suggestion):
        """Update the suggestion for the agent."""
        self.suggestion = [suggestion]

    def _get_ceiling_image(self):
        """Capture an overhead image by toggling map view."""
        # event = self.controller.step(action="ToggleMapView")

        top_view_rgb = cv2.cvtColor(self.controller.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
        # f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
        # cv2.imwrite(f_name, top_view_rgb)

        # self.controller.step(action="ToggleMapView")
        return top_view_rgb
        # return event.cv2img
    
    def _write_image(self, pth: Path, img):
        """Write an image to the specified path."""
        cv2.imwrite(str(pth), img)
    


    def save_frame(self):
        """Save POV images for each agent and a single overhead image."""
        # if simulation:
        #     frame_num = "_" + str(self.simulation_step_num)
        # else:
        #     frame_num = ""
        
        for agent_id in range(self.num_agents):
            img = self.event.events[agent_id].cv2img
            pth = self.base_path / self.agent_names[agent_id] / "pov" / f"frame_{str(self.step_num[agent_id])}.png"
            self._write_image(pth, img)
        
        if self.overhead:
            img = self._get_ceiling_image()
            pth = self.base_path / "overhead" / f"frame_{str(self.step_num[0])}.png"
            self._write_image(pth, img)
        

    
    def save_last_frame(self, agent_id: int = None, view: str = "pov", filename: str = "last_frame.png"):
        """Save the frame from the last event for a specific agent or overhead view."""
        if not self.skip_save_dir:
            if view == "pov" and agent_id is not None:
                img = self.event.events[agent_id].cv2img
                pth = self.base_path / self.agent_names[agent_id] / "pov" / filename
                self._write_image(pth, img)
            elif view == "overhead":
                img = self._get_ceiling_image()
                pth = self.base_path / "overhead" / filename
                self._write_image(pth, img)
            else:
                raise ValueError("Invalid view or agent_id. Use 'pov' with a valid agent_id or 'overhead'.")
    
    def get_frame(self, agent_id: int = None, view: str = "pov") -> Path:
        """Get the path to the latest frame for the agent or overhead view."""
        if view == "pov" and agent_id is not None:
            image_path = self.base_path / self.agent_names[agent_id] / "pov" / f"frame_{self.step_num[agent_id]}.png"
        elif view == "overhead":
            image_path = self.base_path / "overhead" / f"frame_{self.step_num[0]}.png"
        else:
            raise ValueError("Invalid view or agent_id. Use 'pov' with a valid agent_id or 'overhead'.")
        return image_path
        
    
    def set_overhead(self, enable: bool):
        """Toggle overhead image capture."""
        self.overhead = enable
    
    def close(self):
        """Close the environment and stop the timer."""
        if self.controller is not None:
            self.controller.stop()
        self.start_time = None
    
    def get_all_objects(self) -> List[str]:
        """Return a list of all objects in the current scene in readable format."""
        object_ids = [obj["objectId"] for obj in self.event.metadata["objects"]]
        return self.get_readable_object_list(object_ids)
    
    def get_single_readable_object(self, object_id: str) -> str:
        """
        將單一 objectId 轉為可讀名稱 (如 Tomato_1) ，
        同時更新 self.object_dict 內部索引。
        """
        obj_name, obj_coord = self.parse_object(object_id)

        # 建立巢狀 dict：{ "Tomato": { "|-00.39|...": 1, ... } }
        if obj_name not in self.object_dict:
            self.object_dict[obj_name] = {}

        if obj_coord not in self.object_dict[obj_name]:

            next_idx = len(self.object_dict[obj_name]) + 1
            self.object_dict[obj_name][obj_coord] = next_idx

        idx = self.object_dict[obj_name][obj_coord]
        return f"{obj_name}_{idx}"


    def get_all_objects_with_ids(self) -> Dict[str, str]:
        """
        取得目前場景的所有物件，並回傳
        { 可讀名稱: 原始 objectId, ... } 的字典。
        """
        mapping: Dict[str, str] = {}

        # 逐一處理 event.metadata["objects"] 內的 objectId
        for obj in self.event.metadata["objects"]:
            obj_id = obj["objectId"]
            readable = self.get_single_readable_object(obj_id)
            mapping[readable] = obj_id

        return dict(sorted(mapping.items()))

    
    
    def simulate_environment_event(self, event_type: str, object_name: str, target_position: Dict[str, float] = None) -> Tuple[bool, str]:
        """Simulate an unexpected environment event: breaking or moving an object."""
        try:
            obj_id = self.convert_readable_object_to_id(object_name)
            obj_metadata = next((obj for obj in self.event.metadata["objects"] if obj["objectId"] == obj_id), None)
            if not obj_metadata:
                return False, f"Object {object_name} not found in the current scene."
            
            if event_type == "break":
                if not obj_metadata.get("breakable", False):
                    return False, f"Object {object_name} is not breakable."
                self.event = self.controller.step(
                    dict(action="BreakObject", objectId=obj_id, forceAction=True)
                )
                success = self.event.metadata["lastActionSuccess"]
                if not success:
                    return False, f"Failed to break {object_name}."
                return True, f"Object {object_name} has been broken."
            
            elif event_type == "move":
                if target_position is None:
                    return False, "Target position must be provided for 'move' event."
                if not all(key in target_position for key in ["x", "y", "z"]):
                    return False, "Target position must contain 'x', 'y', and 'z' coordinates."
                if obj_metadata.get("isPickedUp", False):
                    return False, f"Object {object_name} is currently picked up by an agent and cannot be moved."
                self.event = self.controller.step(
                    dict(action="TeleportObject", objectId=obj_id, position=target_position, rotation=obj_metadata["rotation"], forceAction=True)
                )
                success = self.event.metadata["lastActionSuccess"]
                if not success:
                    return False, f"Failed to move {object_name} to {target_position}."
                return True, f"Object {object_name} has been moved to {target_position}."
            
            else:
                return False, f"Unsupported event type: {event_type}. Use 'break' or 'move'."
        
        except ValueError as e:
            return False, f"Error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error during event simulation: {str(e)}"
        finally:
            # self.total_elapsed_time = time.time() - self.start_time
            if not self.skip_save_dir:
                self.save_frame()
                for aid in range(self.num_agents):
                    self.step_num[aid] += 1

    # LLM Input Methods from LLaMAR
    def get_planner_llm_input(self):
        """
        Returns the input to the subtask LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[0]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[1]}'s observation: list of objects the {agent_name[1]} is observing,
        Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        }}

        """
        # extract the agent_name's observations based on how many agents there are
        planner_llm__input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            planner_llm__input_feats.append(agent_name + "'s observation")
        planner_llm__input_feats.extend(
            ["Robots' open subtasks", "Robots' completed subtasks"]
        )
        return dict((k, self.input_dict[k]) for k in planner_llm__input_feats)

    def get_verifier_llm_input(self):
        """
        Returns the input to the verifier LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[i]}'s previous action: previous action of the {agent_name[0]},
        Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Robots' combined memory: description of robots' combined memory}}

        """
        # extract the agent_name's observations based on how many agents there are
        verifier_llm_input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            verifier_llm_input_feats.extend(
                [
                    agent_name + "'s observation",
                    agent_name + "'s state",
                    agent_name + "'s previous action",
                ]
            )
        verifier_llm_input_feats.extend(
            [
                "Robots' open subtasks",
                "Robots' completed subtasks",
                "Robots' combined memory",
            ]
        )
        return dict((k, self.input_dict[k]) for k in verifier_llm_input_feats)

    def get_action_llm_input(self, failure_module=False):
        """
        Returns the input to the subtask LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[i]}'s state: description of {agent_name[0]}'s state,
        {agent_name[i]}'s previous action: description of what {agent_name[0]} did in the previous time step and whether it was successful,
        {agent_name[i]}'s previous failures: if {agent_name[0]}'s few previous actions failed, description of what failed,
        Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
        Robots' combined memory: description of robot's combined memory}}
        """
        # extract the agent_name's observations based on how many agents there are
        # current additional info from failure ["failure reason"]
        llm_input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            llm_input_feats.extend(
                [
                    agent_name + "'s observation",
                    agent_name + "'s state",
                    agent_name + "'s previous action",
                    agent_name + "'s previous failures",
                ]
            )
        llm_input_feats.extend(
            [
                "Robots' open subtasks",
                "Robots' completed subtasks",
                # "Robots' subtasks",
                "Robots' combined memory",
            ]
        )

        if failure_module:
            # post action / failure module inputs
            # v0 - give failure reason (override environment-given), add logic for next action
            llm_input_feats.extend(
                    [
                        "failure reason",
                        # "logic for next action",
                    ]
            )
        return dict((k, self.input_dict[k]) for k in llm_input_feats)

    def get_failure_llm_input(self):
        """
        Returns the input to the subtask LLM
        Same input as action module currently, but at time t+1.

        ### INPUT FORMAT ###

        {{Task: description of the task the robots are supposed to do,
        {agent_name[i]}'s observation: list of objects the {agent_name[0]} is observing,
        {agent_name[i]}'s state: description of {agent_name[0]}'s state,
        {agent_name[i]}'s previous action: description of what {agent_name[0]} did in the previous time step and whether it was successful,
        {agent_name[i]}'s previous failures: if {agent_name[0]}'s few previous actions failed, description of what failed,
        Robots' open subtasks: list of subtasks  supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
        Robots' combined memory: description of robot's combined memory}}
        """
        # extract the agent_name's observations based on how many agents there are
        llm_input_feats = ["Task"]
        for i in range(self.num_agents):
            agent_name = self.agent_names[i]
            llm_input_feats.extend(
                [
                    agent_name + "'s observation",
                    agent_name + "'s state",
                    agent_name + "'s previous action",
                    agent_name + "'s previous failures",
                ]
            )
        llm_input_feats.extend(
            [
                "Robots' open subtasks",
                "Robots' completed subtasks",
                # "Robots' subtasks",
                "Robots' combined memory",
            ]
        )

        return dict((k, self.input_dict[k]) for k in llm_input_feats)
    
    # ----own
    def get_center_llm_input(self):
        obj_list = self.get_all_objects()
        contains_list = self.get_obj_in_containers()
        return {
            "Task": self.task,
            "Number of agents": self.num_agents,
            "Objects in environment": obj_list,
            "Objects in containers": contains_list,
        }
    
    def get_replan_llm_input(self):
        obj_list = self.get_all_objects()
        contains_list = self.get_obj_in_containers()
        return {
            "Task": self.task,
            "Number of agents": self.num_agents,
            "Robots' open subtasks": self.open_subtasks,
            "Robots' completed subtasks": self.closed_subtasks,
            "inventory": self.inventory,
            "Objects in environment": obj_list,
            "Objects in containers": contains_list,
        }
    
    def get_center_allocator_llm_input(self):
        """
        Returns the input to the subtask LLM
        ### INPUT FORMAT ###
        {{Task: description of the task the robots are supposed to do,
        Number of agents: number of agents in the environment,
        Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        inventory: list of objects in the robots' inventory,
        }}

        """
        return {
            "Task": self.task,
            "Number of agents": self.num_agents,
            "Robots' open subtasks": self.open_subtasks,
            "Robots' completed subtasks": self.closed_subtasks,
            "inventory": self.inventory,
        }
 

    def get_obs_llm_input(self, prev_info={}) -> Dict[str, Any]:
        """
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

        Returns the input to the subtask LLM

        ### Output FORMAT ###
        {{
        Task: description of the task the robots are supposed to do,
        Number of agents: number of agents in the environment,
        Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
        Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
        Objects in environment: list of all objects in the environment,
        {agent_name[i]}'s observation: list of objects and its postion the {agent_name[i]} is observing,
        {agent_name[i]}'s state: description of {agent_name[i]}'s state(position, facing direction and inventory),
        if info:
        {agent_name[i]}'s subtask_failure_reasons: description of what {agent_name[i]} did in the previous time step and whether it was successful and if it failed, why it failed,
        {agent_name[i]}'s previous failures: if {agent_name[i]}'s few previous actions failed, description of what failed,
        }}  
        """
        reachable_2d = self.get_cur_reachable_positions_2d(is_filter=True, mannual_block_pos=self.mannual_block_pos)
        reachable_2d = [(round(x, 2), round(z, 2)) for (x, z) in reachable_2d]
        contains_list = self.get_obj_in_containers()
        obj_list = self.get_all_objects()
        snap: Dict[str, Any] = {
            "Task": self.task,
            # "Step": int(self.step_num[0]),                         
            "Number of agents": self.num_agents,
            # "Reachable positions": reachable_2d,
            "Objects in environment": obj_list,  # list of all objects in the scene
            "Objects in containers": contains_list,
            # "inventory": list(self.inventory),                     # ['nothing', 'Mug_1', ...]
            "Robots' open subtasks": self.open_subtasks,
            "Robots' completed subtasks": self.closed_subtasks,
            "Robots' memory": self.memory[0],
        }

        for aid, name in enumerate(self.agent_names):
            # snap[f"{name}'s observation"]      = self.input_dict.get(f"{name}'s observation", "[]")
            snap[f"{name}'s observation"] = list(self.get_mapping_object_pos_in_view(aid).keys())
            snap[f"{name}'s state"]            = self.input_dict.get(f"{name}'s state", "")
            if prev_info:
                snap[f"{name}'s previous failures"] = self.input_dict.get(f"{name}'s previous failures", "None")
                subtask_failure_reasons = prev_info['subtask_failure_reasons'].get(name, "None")
                if subtask_failure_reasons:
                    reason_description = self.input_dict.get(f"{name}'s previous failures", "None") + f", because of {subtask_failure_reasons['reason']}"
                    snap[f"{name}'s subtask_failure_reasons"] = reason_description
                
        snap['suggestion'] = self.suggestion[0] if self.suggestion else "None"
            

        # if self.use_shared_subtask:
        #     snap["Robots' subtasks"] = self.subtasks[0]
        # elif self.use_separate_subtask:
        #     for aid, name in enumerate(self.agent_names):
        #         snap[f"{name}'s subtask"] = self.subtasks[aid]

        # if self.use_shared_memory:
        #     snap["Robots' combined memory"] = self.memory[0]
        # elif self.use_separate_memory:
        #     for aid, name in enumerate(self.agent_names):
        #         snap[f"{name}'s memory"] = self.memory[aid]

        return snap

    def get_memory_snapshot(self, recent_steps: int = 20, is_central=True) -> dict:
   
        # 每個 agent 的最近紀錄
        contains_list = self.get_obj_in_containers()
        obj_list = self.get_all_objects()
        agents = []
        for aid, name in enumerate(self.agent_names):
            acts = self.action_history.get(name, [])
            succs = self.action_success_history.get(name, [])
            fails = self.subtask_failure_reasons.get(name, {})
            # 取尾段
            recent_actions = acts[-recent_steps:]
            recent_success = succs[-recent_steps:]
            agents.append({
                "name": name,
                "state": self.get_agent_state(aid),
                "last_action": acts[-1] if acts else "Idle",
                "last_action_success": succs[-1] if succs else True,
                "recent_actions": recent_actions,
                "recent_success_flags": recent_success,
                "recent_fail_reasons": fails,  # 你目前存的是單筆 payload，這裡直接放
                "last_check_reason": self.last_check_reason.get(name),
                "visible_objects": list(self.get_mapping_object_pos_in_view(aid).keys()),
            })

        snapshot = {
            "task": self.task,
            "scene": self.scene,
            "step": max(self.step_num) if self.step_num else 0,
            "num_agents": self.num_agents,
            "agents": agents,
            "Objects in environment": obj_list if is_central else [],
            "Objects in containers": contains_list if is_central else [],
            "open_subtasks": list(self.open_subtasks or []),
            "closed_subtasks": list(self.closed_subtasks or []),
            "shared_memory": self.memory[0] if self.memory else "None",
            "suggestion": self.suggestion[0] if self.suggestion else "",
            "subtask_success_history": self.subtask_success_history,
            "subtask_failure_reasons": self.subtask_failure_reasons,
        }
        return snapshot
    

    def convert_to_dict_objprop(self, objs, obj_mass, obj_id):
        objs_dict = []
        for i, obj in enumerate(objs):
            obj_dict = {'name': obj, 'objectId': obj_id[i], 'mass': obj_mass[i]}
            # obj_dict = {'name': obj , 'mass' : 1.0}
            objs_dict.append(obj_dict)
        return objs_dict

    def get_all_objects_in_env(self):
        # get the list of all objects in the current scene
        obj_id = list([obj["objectId"] for obj in self.controller.last_event.metadata["objects"]])
        obj = list([obj["objectType"] for obj in self.controller.last_event.metadata["objects"]])
        obj_mass = list([obj["mass"] for obj in self.controller.last_event.metadata["objects"]])
        obj = self.convert_to_dict_objprop(obj, obj_mass, obj_id)
        return obj
    

if __name__ == "__main__":
    config_path = "config/config.json"
    env = AI2ThorEnv_cen(config_path)
    obs = env.reset()
    print("Initial Observations:\n", obs)
    print("All objects in scene:", env.get_all_objects())
    
    
    
    env.close()    