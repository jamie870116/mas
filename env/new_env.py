import json
import os
from pathlib import Path
import cv2
import ai2thor.controller
from typing import Dict, List, Tuple, Any
import pickle
import numpy as np
from heapq import heappush, heappop
import time
# import sys
# sys.path.append("/Users/apple/Desktop/UCSB/master_project/mas/thortils")
# from thortils.thortils.navigation import get_shortest_path_to_object
# from thortils.thortils.utils import closest, closest_angles


class BaseEnv:
    """Base class for AI2THOR environment utilities."""
    
    def __init__(self):
        self.controller = None
        self.event = None
        self.object_dict = {}  # {obj_name: {obj_id: num}}
        self.move_actions = ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]
        self.rotate_actions = ["RotateRight", "RotateLeft"]
        self.look_actions = ["LookUp", "LookDown"]
        self.object_interaction_actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff"]
    
    def random_spawn(self, seed: int = 0):
        """Randomly spawn objects in the environment."""
        self.controller.step(action="InitialRandomSpawn", randomSeed=seed)
    
    def check_cache(self, path: str) -> Dict:
        """Load or create a cache file."""
        path = Path(path)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return {}
    
    def create_save_dirs(self, test_case_id: str = None):
        """Create directories for saving images under a task-specific folder with test case subfolder."""
        self.base_path = Path("logs/" + self.task.replace(" ", "_"))
        if test_case_id:
            self.base_path = self.base_path / f"test_{test_case_id}"
        for agent_name in self.agent_names:
            (self.base_path / agent_name / "pov").mkdir(parents=True, exist_ok=True)
        (self.base_path / "overhead").mkdir(parents=True, exist_ok=True)
    
    def get_agent_state(self, agent_id: int) -> str:
        """Return a string describing the agent's state."""
        pos = self.get_agent_position(agent_id)
        rot = self.get_agent_rotation(agent_id)
        held = self.get_agent_object_held(agent_id)
        return f"I am at coordinates: {pos}, facing {rot}, holding {held}"
    
    def get_agent_position(self, agent_id: int) -> str:
        """Return a string describing the agent's position."""
        pos_dict = self.event.events[agent_id].metadata["agent"]["position"]
        return f"({pos_dict['x']:.2f}, {pos_dict['z']:.2f})"
    
    def get_agent_position_dict(self, agent_id: int) -> Dict[str, float]:
        """Return the agent's position as a dictionary."""
        return self.event.events[agent_id].metadata["agent"]["position"]
    
    def get_agent_rotation(self, agent_id: int) -> str:
        """Return a string describing the agent's rotation."""
        rot = self.event.events[agent_id].metadata["agent"]["rotation"]["y"]
        rot = int(np.round(rot)) % 360
        angles = [0, 90, 180, 270]
        closest_angle = min(angles, key=lambda x: abs(x - rot))
        if closest_angle == 0:
            return "north"
        elif closest_angle == 90:
            return "east"
        elif closest_angle == 180:
            return "south"
        elif closest_angle == 270:
            return "west"
        return f"{rot} degrees"
    
    def get_agent_object_held(self, agent_id: int) -> str:
        """Return a string describing the object held by the agent."""
        held_objs = self.event.events[agent_id].metadata["inventoryObjects"]
        if not held_objs:
            return "nothing"
        obj_id = held_objs[0]["objectId"]
        obj_name, obj_str_id = self.parse_object(obj_id)
        if obj_name not in self.object_dict:
            self.object_dict[obj_name] = {}
        if obj_str_id not in self.object_dict[obj_name]:
            self.object_dict[obj_name][obj_str_id] = len(self.object_dict[obj_name]) + 1
        return f"{obj_name}_{self.object_dict[obj_name][obj_str_id]}"
    
    def parse_object(self, object_str: str) -> Tuple[str, str]:
        """Parse object ID into name and coordinates."""
        obj_name = object_str.split("|")[0]
        obj_str_id = object_str[len(obj_name):]
        return obj_name, obj_str_id
    
    def get_readable_object_list(self, object_list: List[str]) -> List[str]:
        """Convert object IDs to readable format."""
        readable_list = []
        for obj in object_list:
            obj_name, obj_id = self.parse_object(obj)
            if obj_name not in self.object_dict:
                self.object_dict[obj_name] = {}
            if obj_id not in self.object_dict[obj_name]:
                self.object_dict[obj_name][obj_id] = len(self.object_dict[obj_name]) + 1
            readable_list.append(f"{obj_name}_{self.object_dict[obj_name][obj_id]}")
        return readable_list
    
    def convert_readable_object_to_id(self, object_name: str) -> str:
        """Convert readable object name to ID."""
        obj_name, obj_num = object_name.split("_")
        obj_num = int(obj_num)
        for obj_id, num in self.object_dict.get(obj_name, {}).items():
            if num == obj_num:
                return f"{obj_name}{obj_id}"
        raise ValueError(f"Object {object_name} not found in object_dict")
    
    def parse_action(self, action: str, agent_id: int) -> Dict:
        """Parse action string into AI2THOR-compatible dictionary."""
        action_dict = {"agentId": agent_id}
        if action in ["Done", "Idle"] + self.move_actions + self.rotate_actions + self.look_actions:
            action_dict["action"] = action
        elif action.startswith(tuple(self.object_interaction_actions)):
            action_name = action.split("(")[0]
            object_id = action.split("(")[1].rstrip(")")
            action_dict["action"] = action_name
            action_dict["objectId"] = self.convert_readable_object_to_id(object_id)
            if action_name == "PutObject" and "Fridge" in object_id:
                action_dict["forceAction"] = True
        elif action.startswith("DropHandObject"):
            action_dict["action"] = "DropHandObject"
        elif action.startswith("NavigateTo"):
            action_dict["action"] = "Pass"
        else:
            raise ValueError(f"Unsupported action: {action}")
        return action_dict
    
    def get_act_text(self, action: str, act_success: bool, agent_id: int, error_type: str = None) -> str:
        """Generate text describing the action outcome."""
        action_name = action.split("(")[0]
        if action_name in ["Move", "Rotate", "Look"]:
            direction = action.split("(")[1].rstrip(")") if "(" in action else action.replace("Move", "").replace("Rotate", "").replace("Look", "").lower()
            act_text = f"I {'moved' if action_name == 'Move' else 'rotated' if action_name == 'Rotate' else 'looked'} {direction}"
            act_text += " and was successful." if act_success else " but was unsuccessful."
        elif action_name in ["Done", "Idle"]:
            act_text = f"I was {'done' if action_name == 'Done' else 'idle'}."
        elif action_name in self.object_interaction_actions:
            object_id = action.split("(")[1].rstrip(")")
            action_verb = action_name.lower().replace("object", "")
            if action_name in ["ToggleObjectOn", "ToggleObjectOff"]:
                action_verb = "toggled " + ("on" if action_name == "ToggleObjectOn" else "off")
            act_text = f"I {action_verb} {object_id}"
            act_text += " and was successful." if act_success else " but was unsuccessful."
        elif action_name == "DropHandObject":
            act_text = f"I dropped the held object"
            act_text += " and was successful." if act_success else " but was unsuccessful."
        elif action_name == "NavigateTo":
            object_id = action.split("(")[1].rstrip(")")
            if act_success:
                act_text = f"I navigated to {object_id} and was successful."
            else:
                act_text = f"I tried to navigate to {object_id} but was unsuccessful."
                if error_type:
                    act_text += f" Reason: {error_type}."
        else:
            act_text = f"Performed {action} {'successfully' if act_success else 'unsuccessfully'}."
        return act_text
    
    def get_act_failure_text(self, actions: List[str], agent_id: int) -> str:
        """Generate text describing failed actions."""
        if not actions:
            return "None"
        failure_text = "Previously, I tried to "
        for i, action in enumerate(actions):
            action_name = action.split("(")[0]
            if action_name in ["Move", "Rotate", "Look"]:
                direction = action.split("(")[1].rstrip(")") if "(" in action else action.replace("Move", "").replace("Rotate", "").replace("Look", "").lower()
                act_text = f"{'move' if action_name == 'Move' else 'rotate' if action_name == 'Rotate' else 'look'} {direction}"
            elif action_name == "NavigateTo":
                object_id = action.split("(")[1].rstrip(")")
                act_text = f"navigate to {object_id}"
            elif action_name in self.object_interaction_actions:
                object_id = action.split("(")[1].rstrip(")")
                action_verb = action_name.lower().replace("object", "")
                if action_name in ["ToggleObjectOn", "ToggleObjectOff"]:
                    action_verb = "toggle " + ("on" if action_name == "ToggleObjectOn" else "off")
                act_text = f"{action_verb} {object_id}"
            elif action_name == "DropHandObject":
                act_text = "drop the held object"
            else:
                act_text = action.lower()
            failure_text += act_text
            if i < len(actions) - 2:
                failure_text += ", "
            elif i == len(actions) - 2:
                failure_text += " and "
        failure_text += " but was unsuccessful."
        return failure_text

class AI2ThorEnv(BaseEnv):
    """Main AI2THOR environment for multi-agent tasks with global timer and frame saving."""
    def __init__(self, config_path: str = "config.json"):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.num_agents = self.config["num_agents"]
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
        
        self.controller = ai2thor.controller.Controller(width=1000, height=1000)
        self.controller.reset(self.scene)
        
        self.agent_names = ["Alice", "Bob", "Charlie", "David", "Emma"][:self.num_agents]
        self.inventory = ["nothing"] * self.num_agents
        self.subtasks = ["Initial subtask"] if self.use_shared_subtask else ["Initial subtask"] * self.num_agents
        self.memory = ["Nothing"] if self.use_shared_memory else ["Nothing"] * self.num_agents
        self.open_subtasks = "None" if self.use_plan else None
        self.closed_subtasks = "None" if self.use_plan else None
        self.step_num = [0] * self.num_agents
        self.simulation_step_num = 1
        self.action_history = {name: [] for name in self.agent_names}
        self.action_success_history = {name: [] for name in self.agent_names}
        self.agent_failure_acts = {name: [] for name in self.agent_names}
        self.all_obs_dict = {name: [] for name in self.agent_names}
        self.obs_summary_llm_cache_path = "summary_llm_cache.pkl"
        self.verbose = True
        self.skip_save_dir = False
        self.grid_size = 0.25
        self.previous_object_ids = [None] * self.num_agents
        self.previous_positions = [None] * self.num_agents
        self.start_time = None
        self.total_elapsed_time = 0.0
    
    def reset(self, task: str = None, test_case_id: str = None) -> str:
        """Reset the environment, start the global timer, and save initial frames before any actions."""
        self.task = task or self.config["task"]
        self.controller.reset(self.scene)
        self.event = self.controller.step(
            {
                "action": "Initialize",
                "gridSize": self.grid_size,
                "renderObjectImage": True,
                "agentCount": self.num_agents,
                "visibilityDistance": 40
            }
        )
        self.object_dict = {}
        self.step_num = [0] * self.num_agents
        self.simulation_step_num = 1
        self.inventory = ["nothing"] * self.num_agents
        self.subtasks = ["Initial subtask"] if self.use_shared_subtask else ["Initial subtask"] * self.num_agents
        self.memory = ["Nothing"] if self.use_shared_memory else ["Nothing"] * self.num_agents
        self.action_history = {name: [] for name in self.agent_names}
        self.action_success_history = {name: [] for name in self.agent_names}
        self.agent_failure_acts = {name: [] for name in self.agent_names}
        self.all_obs_dict = {name: [] for name in self.agent_names}
        self.open_subtasks = "None" if self.use_plan else None
        self.closed_subtasks = "None" if self.use_plan else None
        self.previous_object_ids = [None] * self.num_agents
        self.previous_positions = [None] * self.num_agents
        self.start_time = time.time()
        self.total_elapsed_time = 0.0
        
        if not self.skip_save_dir:
            self.create_save_dirs(test_case_id)
        
        for agent_id in range(self.num_agents):
            self.event = self.controller.step(
                dict(
                    action="Teleport",
                    position=dict(x=1.5 + agent_id * 0.5, y=0.9, z=-1.5),
                    rotation=dict(x=0, y=270, z=0),
                    agentId=agent_id
                )
            )
        
        if not self.skip_save_dir:
            self.save_frame()
        
        return self.get_observations()
    
    def get_observations(self) -> str:
        """Generate observation dictionary and return as string."""
        self.input_dict = {"Task": self.task, "Elapsed Time": f"{self.total_elapsed_time:.2f} seconds"}
        
        for agent_id, agent_name in enumerate(self.agent_names):
            obs, obs_list = self.generate_obs_text(agent_id)
            self.input_dict[f"{agent_name}'s observation"] = obs
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
        detections = self.event.events[agent_id].instance_detections2D
        return list(detections.instance_masks.keys()) if detections else []
    
    def generate_obs_text(self, agent_id: int, prefix: str = "I see: ") -> Tuple[str, List[str]]:
        """Generate observation text and list."""
        objects_in_view = self.get_object_in_view(agent_id)
        obs_list = self.get_readable_object_list(objects_in_view)
        if self.use_obs_summariser:
            obs_list = self.summarise_obs(obs_list)
        obs_text = prefix + str(obs_list)
        return obs_text, obs_list
    
    def summarise_obs(self, obs_list: List[str]) -> List[str]:
        """Placeholder for observation summarization."""
        return obs_list
    
    # def step(self, actions: List[str]) -> Tuple[str, List[bool]]:
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
        
        self.total_elapsed_time = time.time() - self.start_time
        if not self.skip_save_dir:
            self.save_frame()
        
        self.update_current_state(act_texts)
        return self.get_observations(), act_successes
    
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
        """Execute navigation actions by finding a step-by-step path to the front of the target object."""
        object_id = action.split("(")[1].rstrip(")")
        try:
            obj_id = self.convert_readable_object_to_id(object_id)
            obj_metadata = next(obj for obj in self.event.metadata["objects"] if obj["objectId"] == obj_id)
            target_pos = obj_metadata["position"]
            target_rot = obj_metadata["rotation"]["y"] % 360
            current_pos = self.get_agent_position_dict(agent_id)
            start = (round(current_pos["x"] / self.grid_size), round(current_pos["z"] / self.grid_size))
            
            if (self.previous_object_ids[agent_id] == obj_id and
                self.previous_positions[agent_id] is not None and
                abs(current_pos["x"] - self.previous_positions[agent_id]["x"]) < self.grid_size and
                abs(current_pos["z"] - self.previous_positions[agent_id]["z"]) < self.grid_size):
                return True, None
            
            target_grid = (round(target_pos["x"] / self.grid_size), round(target_pos["z"] / self.grid_size))
            self.event = self.controller.step(dict(action="GetReachablePositions", agentId=agent_id))
            reachable_positions = self.event.metadata["actionReturn"]
            reachable_grid = {(round(pos["x"] / self.grid_size), round(pos["z"] / self.grid_size)) for pos in reachable_positions}
            
            front_offset = self.grid_size / 2
            if 0 <= target_rot < 90 or 270 <= target_rot < 360:
                front_pos = (target_grid[0], target_grid[1] + 1)
            elif 90 <= target_rot < 180:
                front_pos = (target_grid[0] + 1, target_grid[1])
            elif 180 <= target_rot < 270:
                front_pos = (target_grid[0], target_grid[1] - 1)
            else:
                front_pos = (target_grid[0] - 1, target_grid[1])
            
            if front_pos not in reachable_grid:
                max_distance = 2.0
                possible_positions = []
                for pos in reachable_positions:
                    grid_pos = (round(pos["x"] / self.grid_size), round(pos["z"] / self.grid_size))
                    if grid_pos == start:
                        continue
                    dist_to_target = ((grid_pos[0] - target_grid[0])**2 + (grid_pos[1] - target_grid[1])**2)**0.5
                    if dist_to_target <= max_distance:
                        possible_positions.append((dist_to_target, grid_pos))
                
                if not possible_positions:
                    min_dist = float('inf')
                    closest_pos = None
                    for pos in reachable_positions:
                        grid_pos = (round(pos["x"] / self.grid_size), round(pos["z"] / self.grid_size))
                        if grid_pos == start:
                            continue
                        dist = ((grid_pos[0] - target_grid[0])**2 + (grid_pos[1] - target_grid[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            closest_pos = grid_pos
                    if closest_pos is None:
                        return False, "no-reachable-position-found"
                    front_pos = closest_pos
                else:
                    possible_positions.sort(key=lambda x: x[0])
                    front_pos = possible_positions[0][1]
            
            path = self.a_star(start, front_pos, reachable_grid, agent_id)
            if not path:
                return False, "path-not-found"
            
            current_rotation = self.event.events[agent_id].metadata["agent"]["rotation"]["y"]
            rotation_step = 15
            max_iterations = 24
            for i in range(len(path) - 1):
                current = path[i]
                next_pos = path[i + 1]
                dx = next_pos[0] - current[0]
                dz = next_pos[1] - current[1]
                if dx == 0 and dz == 1:
                    target_angle = 0
                elif dx == 1 and dz == 0:
                    target_angle = 90
                elif dx == 0 and dz == -1:
                    target_angle = 180
                elif dx == -1 and dz == 0:
                    target_angle = 270
                else:
                    continue
                
                iteration = 0
                while abs(np.round(current_rotation, 2) - np.round(target_angle, 2)) > 5 and iteration < max_iterations:
                    angle_diff = (target_angle - current_rotation) % 360
                    rot_action = "RotateRight" if angle_diff <= 180 else "RotateLeft"
                    current_rotation = (current_rotation + rotation_step) % 360 if rot_action == "RotateRight" else (current_rotation - rotation_step) % 360
                    self.event = self.controller.step(dict(action=rot_action, agentId=agent_id, degrees=rotation_step))
                    self.step_num[agent_id] += 1
                    if not self.event.events[agent_id].metadata["lastActionSuccess"]:
                        return False, "rotation-failed"
                    if not self.skip_save_dir:
                        self.save_frame()
                    iteration += 1
                
                self.event = self.controller.step(dict(action="MoveAhead", agentId=agent_id))
                self.step_num[agent_id] += 1
                if not self.event.events[agent_id].metadata["lastActionSuccess"]:
                    return False, "move-failed"
                if not self.skip_save_dir:
                    self.save_frame()
            
            final_pos = self.get_agent_position_dict(agent_id)
            delta_x = target_pos["x"] - final_pos["x"]
            delta_z = target_pos["z"] - final_pos["z"]
            target_angle = np.degrees(np.arctan2(delta_x, delta_z)) % 360
            current_rotation = self.event.events[agent_id].metadata["agent"]["rotation"]["y"]
            angle_diff = (target_angle - current_rotation) % 360
            rot_action = "RotateLeft" if angle_diff > 180 else "RotateRight"
            total_degrees = min(angle_diff, 360 - angle_diff)
            if total_degrees > 5:
                self.event = self.controller.step(dict(action=rot_action, agentId=agent_id, degrees=total_degrees))
                self.step_num[agent_id] += 1
                if not self.event.events[agent_id].metadata["lastActionSuccess"]:
                    return False, "final-rotation-failed"
                if not self.skip_save_dir:
                    self.save_frame()
            
            self.previous_object_ids[agent_id] = obj_id
            self.previous_positions[agent_id] = current_pos.copy()
            
            final_pos = self.get_agent_position_dict(agent_id)
            final_rot = self.event.events[agent_id].metadata["agent"]["rotation"]["y"]
            dist = ((final_pos["x"] - target_pos["x"])**2 + (final_pos["z"] - target_pos["z"])**2)**0.5
            is_in_view = obj_id in self.get_object_in_view(agent_id)
            if dist > 1.5:
                return False, f"too-far-from-object: {dist:.2f} meters"
            
            if not is_in_view:
                for _ in range(4):
                    self.event = self.controller.step(dict(action="RotateRight", agentId=agent_id, degrees=90))
                    self.step_num[agent_id] += 1
                    if not self.event.events[agent_id].metadata["lastActionSuccess"]:
                        return False, "scan-rotation-failed"
                    if not self.skip_save_dir:
                        self.save_frame()
                    if obj_id in self.get_object_in_view(agent_id):
                        break
                if obj_id not in self.get_object_in_view(agent_id):
                    for look_action in ["LookUp", "LookDown"]:
                        self.event = self.controller.step(dict(action=look_action, agentId=agent_id, degrees=30))
                        self.step_num[agent_id] += 1
                        if not self.event.events[agent_id].metadata["lastActionSuccess"]:
                            return False, "vertical-scan-failed"
                        if not self.skip_save_dir:
                            self.save_frame()
                        if obj_id in self.get_object_in_view(agent_id):
                            break
                    if obj_id not in self.get_object_in_view(agent_id):
                        return False, "object-not-in-view-after-scan"
            
            return True, None
        except (KeyError, ValueError) as e:
            return False, "key"
        except StopIteration:
            return False, "object-not-found"
        except Exception as e:
            return False, f"unexpected-error: {str(e)}"
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], reachable_grid: set, agent_id: int) -> List[Tuple[int, int]]:
        """A* pathfinding algorithm to find a step-by-step path."""
        def heuristic(pos: Tuple[int, int]) -> float:
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        
        while open_set:
            current = heappop(open_set)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dz in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dz)
                if neighbor not in reachable_grid:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def update_subtask(self, subtask: str, agent_id: int = 0):
        """Update subtask for shared or per-agent usage."""
        if self.use_shared_subtask:
            self.subtasks[0] = subtask
        elif self.use_separate_subtask:
            self.subtasks[agent_id] = subtask
    
    def update_memory(self, memory: str, agent_id: int = 0):
        """Update memory for shared or per-agent usage."""
        if self.use_shared_memory:
            self.memory[0] = memory
        elif self.use_separate_memory:
            self.memory[agent_id] = memory
    
    def update_current_state(self, act_texts: List[str]):
        """Update the input dictionary with the current state."""
        self.new_all_obs = []
        for agent_id, agent_name in enumerate(self.agent_names):
            obs, obs_list = self.generate_obs_text(agent_id)
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
    
    def _get_ceiling_image(self):
        """Capture an overhead image by toggling map view."""
        event = self.controller.step(action="ToggleMapView")
        self.controller.step(action="ToggleMapView")
        return event.cv2img
    
    def _write_image(self, pth: Path, img):
        """Write an image to the specified path."""
        cv2.imwrite(str(pth), img)
    


    def save_frame(self, simulation: bool = False):
        """Save POV images for each agent and a single overhead image."""
        if simulation:
            frame_num = "_" + str(self.simulation_step_num)
        else:
            frame_num = ""
        
        for agent_id in range(self.num_agents):
            img = self.event.events[agent_id].cv2img
            pth = self.base_path / self.agent_names[agent_id] / "pov" / f"frame_{str(self.step_num[agent_id]) + frame_num}.png"
            self._write_image(pth, img)
        
        if self.overhead:
            img = self._get_ceiling_image()
            pth = self.base_path / "overhead" / f"frame_{str(self.step_num[0]) + frame_num}.png"
            self._write_image(pth, img)
        # current_time = self.total_elapsed_time
        # for agent_id in range(self.num_agents):
        #     img = self.event.events[agent_id].cv2img
        #     pth = self.base_path / self.agent_names[agent_id] / "pov" / f"frame_{current_time:.2f}.png"
        #     self._write_image(pth, img)
        
        # if self.overhead:
        #     img = self._get_ceiling_image()
        #     pth = self.base_path / "overhead" / f"frame_{current_time:.2f}.png"
        #     self._write_image(pth, img)

    
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
        # current_time = self.total_elapsed_time
        # if view == "pov":
        #     image_path = (
        #         self.base_path
        #         / self.agent_names[agent_id]
        #         / "pov"
        #         / f"frame_{current_time:.2f}.png"
        #     )
        # else:
        #     image_path = self.base_path / "overhead" / f"frame_{current_time:.2f}.png"
        # return image_path
    
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
    
    def get_object_status(self, object_name: str) -> Dict[str, Any]:
        """Return the status of a specific object given its readable name."""
        obj_id = self.convert_readable_object_to_id(object_name)
        for obj in self.event.metadata["objects"]:
            if obj["objectId"] == obj_id:
                status = {
                    "object_id": obj_id,
                    "name": obj["name"],
                    "position": obj["position"],
                    "rotation": obj["rotation"],
                    "is_open": obj.get("isOpen", False),
                    "is_on": obj.get("isToggled", False),
                    "is_picked_up": obj.get("isPickedUp", False),
                    "isSliced": obj.get("isSliced", False),
                    "isToggled": obj.get("isToggled", False),
                    "isBroken": obj.get("isBroken", False),
                    "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                    'contains': obj.get("receptacleObjectIds", None),
                }
                return status
        raise ValueError(f"Object {object_name} not found in the current scene.")
    
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
            self.total_elapsed_time = time.time() - self.start_time
            self.simulation_step_num += 1
            if not self.skip_save_dir:
                self.save_frame(simulation=True)
            self.simulation_step_num += 1

if __name__ == "__main__":
    config_path = "config/config.json"
    env = AI2ThorEnv(config_path)
    obs = env.reset()
    print("Initial Observations:\n", obs)
    print("All objects in scene:", env.get_all_objects())
    
    success, message = env.simulate_environment_event("break", "Mug_1")
    print(f"Break Event: {message}")
    # env.save_last_frame(agent_id=0, view="pov", filename="last_break_frame.png")
    
    target_pos = {"x": 2.0, "y": 0.9, "z": -1.0}
    success, message = env.simulate_environment_event("move", "Bowl_1", target_pos)
    print(f"Move Event: {message}")
    
    actions = ["MoveAhead", "MoveAhead"]
    obs, successes = env.step(actions)
    print("Step Observations:\n", obs)
    print("Action Successes:", successes)
    
    env.close()

# if __name__ == "__main__":
#     config_path = "config/config.json"
#     env = AI2ThorEnv(config_path)
#     obs = env.reset()
#     print("Initial Observations:\n", obs)
#     print('object in view of agent 1', env.get_object_in_view(0))
#     print("All objects in scene:", env.get_all_objects())
#     print("Before breaking: Status of Mug_1:", env.get_object_status("Mug_1"))
#     print("Before breaking: Status of Lettuce_1:", env.get_object_status("Lettuce_1"))

#     # Simulate breaking an object
#     success, message = env.simulate_environment_event("break", "Mug_1")
#     print(f"Break Event: {message}")
#     print("After breaking: Status of Mug_1:", env.get_object_status("Mug_1"))
#     print("After breaking: Status of Lettuce_1:", env.get_object_status("Lettuce_1"))

#     time.sleep(10)
#     # Simulate moving an object
#     target_pos = {"x": 2.0, "y": 0.9, "z": -1.0}
#     print("Before moving: Status of Bowl_1:", env.get_object_status("Bowl_1"))
#     success, message = env.simulate_environment_event("move", "Bowl_1", target_pos)
#     print(f"Move Event: {message}")
#     print("After moving: Status of Bowl_1:", env.get_object_status("Bowl_1"))

#     actions = ["MoveAhead", "MoveAhead"]
#     obs, successes = env.step(actions)
#     print("Step Observations:\n", obs)
#     print("Action Successes:", successes)
    
#     env.close()

    '''
Before breaking: Status of Mug_1: {'object_id': 'Mug|-01.76|+00.90|-00.62', 'name': 'Mug_e7fad100', 'position': {'x': -1.7607921361923218, 'y': 0.9000000357627869, 'z': -0.6206092834472656}, 'rotation': {'x': 2.6733881895779632e-05, 'y': 0.000685526872985065, 'z': -1.5666983017581515e-05}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': []}
Before breaking: Status of Lettuce_1: {'object_id': 'Lettuce|-01.81|+00.97|-00.94', 'name': 'Lettuce_2d8f3ab9', 'position': {'x': -1.8069753646850586, 'y': 0.9737617373466492, 'z': -0.9429945945739746}, 'rotation': {'x': 359.9818115234375, 'y': 256.7711181640625, 'z': 0.001349071622826159}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': None}

Break Event: Object Mug_1 has been broken.

After breaking: Status of Mug_1: {'object_id': 'Mug|-01.76|+00.90|-00.62', 'name': 'Mug_e7fad100', 'position': {'x': -1.7607921361923218, 'y': 0.9000000357627869, 'z': -0.6206092834472656}, 'rotation': {'x': 2.6733881895779632e-05, 'y': 0.000685526872985065, 'z': -1.5666983017581515e-05}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': True, 'isFilledWithLiquid': False, 'contains': []}
After breaking: Status of Lettuce_1: {'object_id': 'Lettuce|-01.81|+00.97|-00.94', 'name': 'Lettuce_2d8f3ab9', 'position': {'x': -1.798709511756897, 'y': 0.973779559135437, 'z': -0.9976263046264648}, 'rotation': {'x': 359.9550476074219, 'y': 262.6057434082031, 'z': 46.579410552978516}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': None}

Before moving: Status of Bowl_1: {'object_id': 'Bowl|+00.27|+01.10|-00.75', 'name': 'Bowl_208f368b', 'position': {'x': 0.2731873691082001, 'y': 1.1010208129882812, 'z': -0.7532863616943359}, 'rotation': {'x': -0.002493660431355238, 'y': 0.00026333334972150624, 'z': -0.0018902190495282412}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': []}
Move Event: Object Bowl_1 has been moved to {'x': 2.0, 'y': 0.9, 'z': -1.0}.
After moving: Status of Bowl_1: {'object_id': 'Bowl|+00.27|+01.10|-00.75', 'name': 'Bowl_208f368b', 'position': {'x': 2.0, 'y': 0.0010198839008808136, 'z': -1.000000238418579}, 'rotation': {'x': 3.404549352126196e-05, 'y': 0.0002766225952655077, 'z': -0.00011428898142185062}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': []}
    
    '''