
from pathlib import Path
import cv2
import ai2thor.controller
from typing import Dict, List, Tuple, Any
import pickle
import numpy as np
import re



class BaseEnv:
    """Base class for AI2THOR environment utilities."""  
    def __init__(self):
        self.controller = None
        self.event = None
        self.object_dict = {}  # {obj_name: {obj_id: num}}
        self.return_actions = ["Return"]
        self.move_actions = ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]
        self.rotate_actions = ["RotateRight", "RotateLeft"]
        self.look_actions = ["LookUp", "LookDown"]
        self.idle_actions = ["Done", "Idle"]
        self.object_interaction_actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "BreakObject", "CookObject", "SliceObject", "DirtyObject", "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject"]
        self.object_interaction_without_navigation  = ["DropHandObject", "ThrowObject"]
        self.receptacle_objects = [
            # Kitchen
            "Bowl", "Cabinet", "CoffeeMachine", "CounterTop", "Cup", "DiningTable",
            "Drawer", "Fridge", "GarbageCan", "Microwave", "Mug", "Pan", "Plate",
            "Pot", "Shelf", "SideTable", "Sink", "SinkBasin", "StoveBurner", "Toaster",

            # LivingRoom
            "ArmChair", "Box", "CoffeeTable", "CounterTop", "Cup", "Desk", "DiningTable",
            "Drawer", "Dresser", "GarbageCan", "Microwave", "Mug", "Plate",
            "Shelf", "SideTable", "Sofa", "TVStand",

            # Bedroom
            "ArmChair", "Bed", "Box", "Cup", "Desk", "Dresser", "Drawer", "GarbageCan",
            "LaundryHamper", "Mug", "Plate", "Shelf", "SideTable", "TVStand",

            # Bathroom
            "Bathtub", "BathtubBasin", "Cabinet", "CounterTop", "Drawer", "GarbageCan",
            "HandTowelHolder", "Shelf", "SideTable", "Sink", "SinkBasin", "Toilet",
            "ToiletPaperHanger", "TowelHolder"
        ]
        self.large_receptacles = ["Cabinet", "CounterTop",  "DiningTable",
            "Drawer", "Fridge", "GarbageCan", "Microwave", "Sofa",
             "Shelf", "SideTable", "SinkBasin","ArmChair", "Box","CoffeeTable", "Desk", "Dresser","Bathtub", "BathtubBasin", "Floor"]
        self.small_objects = [
            # kitchen
            "Apple", "Tomato","Potato", "Egg", "StoveKnob","Mug","Cup","SaltShaker", "Knife", "ButterKnife", "Fork", "Spoon", "GarbageCan", "Bowl", "Drawer",
            # living room
            "RemoteControl", "Newspaper", "KeyChain","Vase","TissueBox","LightSwitch","DeskLamp",  "Box",
            # bedroom
            "CD", "CellPhone", "Pencil", "Pen", "Watch", "AlarmClock", "CreditCard","TeddyBear",
            # Bathroom
            "Cloth", "Faucet", "DishSponge","PaperTowelRoll","SoapBar", "ToiletPaper"
        ]
        # only egg -> EggCracked; other Sliced
        self.sliceable_objects = ['Potato', 'Tomato', 'Bread', 'Lettuce', 'Egg', 'Apple'] 
        # These are the objects that is breakable and cannot be interacted or recovered after broken
        self.breakable_objects = ['Bowl','Bottle','Mug','Plate','Cup','Vase','ShowerDoor','Mirror','Statue','Window','WineBottle']
        self.dirtyable_objects = ['Bed','Bowl','Cloth','Cup','Mirror','Mug','Pan','Plate','Pot']
        self.usable_objects = ['PaperTowelRoll', 'SoapBottle', 'TissueBox', 'ToiletPaper']
    
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
    
    def create_save_dirs(self, test_case_id: str = None, scene: str = None):
        """Create directories for saving images under a task-specific folder with test case subfolder."""
        self.base_path = Path("logs/" + self.task.replace(" ", "_") + "/" + scene)
        if test_case_id:
            self.base_path = self.base_path / f"test_{test_case_id}"
        for agent_name in self.agent_names:
            (self.base_path / agent_name / "pov").mkdir(parents=True, exist_ok=True)
        (self.base_path / "overhead").mkdir(parents=True, exist_ok=True)
        event_log_path = self.base_path / "event.jsonl"
        with open(event_log_path, "w") as f:
            f.write("")  # Clear previous log
        for i in range(self.num_agents):
            event_log_path = self.base_path / f"event_{i}.jsonl"
            with open(event_log_path, "w") as f:
                f.write("")  # Clear previous log
    
    def extract_frame_indices(self, filename):
        """
        Extracts the primary and optional secondary frame index from filename like 'frame_0_2.png' or 'frame_1.png'.
        Returns a tuple (primary, secondary) for sorting.
        """
        match = re.match(r"frame_(\d+)(?:_(\d+))?\.png", filename)
        if match:
            primary = int(match.group(1))
            secondary = int(match.group(2)) if match.group(2) else 0
            return (primary, secondary)
        return (float('inf'), float('inf')) 

    def save_to_video(self, fps: int = 10, delete_frames: bool = False):
        """
        Convert saved frames into videos for each agent's POV and overhead view.
        
        Args:
            file_name (str): The task folder name (e.g., 'logs/Summary/put_remote_control,_keys,_and_watch_in_the_box/Floorplan201/test_{i}')
            fps (int): Frames per second for the output video (default: 30).
            delete_frames (bool): Whether to delete frame images after video creation.
        """
        task_path = self.base_path
       
        print(f"Resolved task path: {task_path}")
        
        if not task_path.exists() or not task_path.is_dir():
            raise ValueError(f"Task folder {task_path} does not exist or is not a directory.")
        
        # Find all subfolders (e.g., Alice/pov, Bob/pov, overhead)
        subfolders = []
        # Look for agent POV folders (e.g., Alice/pov)
        for agent_folder in task_path.iterdir():
            if agent_folder.is_dir() and (agent_folder / "pov").exists():
                subfolders.append(agent_folder / "pov")
            elif agent_folder.name == "overhead" and agent_folder.is_dir():
                subfolders.append(agent_folder)
        
        if not subfolders:
            raise ValueError(f"No valid subfolders (agent POV or overhead) found in {task_path}.")
        
        # Process each subfolder to create a video
        for subfolder in subfolders:
            # Collect all frame files in the subfolder
            # frame_files = sorted(
            #     [f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"],
            #     key=lambda x: int(re.search(r'frame_(\d+)\.png', x.name).group(1))
            # )
            frame_files = sorted(
                [f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"],
                key=lambda x: self.extract_frame_indices(x.name)
            )
            
            if not frame_files:
                print(f"No frames found in {subfolder}. Skipping video creation.")
                continue
            
            # Read the first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            height, width, _ = first_frame.shape
            
            # Define the output video path
            video_name = subfolder.name if subfolder.name == "overhead" else f"{subfolder.parent.name}_{subfolder.name}"
            video_path = task_path / f"{video_name}.mp4"
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            # Write each frame to the video
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                video_writer.write(frame)
            
            # Release the video writer
            video_writer.release()

            print(f"Video saved at {video_path}")
            if delete_frames:
                for frame_file in frame_files:
                    try:
                        frame_file.unlink()
                    except Exception as e:
                        print(f"Warning: could not delete {frame_file}: {e}")
                print(f"Deleted {len(frame_files)} frame images from {subfolder}")



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
    
    def get_cur_reachable_positions(self) -> List[Tuple[float, float]]:
        reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        return reachable_positions

    
    def get_cur_reachable_positions_2d(self, is_filter=False, mannual_block_pos={}) -> List[Tuple[float, float]]:
        reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        reachable_positions = [(p["x"], p["z"]) for p in reachable_positions_]

        if not is_filter:
            return reachable_positions

        # Helper
        def roundany(value: float, base: float = 0.25) -> float: return round(base * round(value / base), 3)
        def tuplize_xz(p): return (p['x'], p['z'])
        def round_pos(p): return tuple(roundany(x, self.grid_size) for x in p)
        def add(p, delta): return (p[0] + delta[0], p[1] + delta[1])
        sweep = [-1, 0, 1]

        blocked = set()
        for agent_id in range(self.num_agents):
            agent_pos_dict = self.get_agent_position_dict(agent_id)
            agent_pos_rounded = round_pos(tuplize_xz(agent_pos_dict))

            # Add agent position & its surrounding area
            for dx in sweep:
                for dz in sweep:
                    blocked.add(add(agent_pos_rounded, (dx * self.grid_size, dz * self.grid_size)))
                    
        if mannual_block_pos:
            for positions in mannual_block_pos.values():
                for pos in positions:
                    blocked.add(round_pos(tuplize_xz(pos)))

        filtered = []
        for pos in reachable_positions:
            rounded = round_pos(pos)
            if rounded not in blocked:
                filtered.append(pos)

        return filtered


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
    
    def get_cur_cam_pitch(self, agent_id):
        return self.event.events[agent_id].metadata["agent"]["cameraHorizon"]
    
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
    
    def update_object_dict(self):
        """
        Rebuild self.object_dict based on all objects currently in the scene.
        Format: {object_name: {unique_id: index}}
        """
        self.object_dict = {}
        object_ids = [obj["objectId"] for obj in self.event.metadata["objects"]]
        
        # print('object_ids', object_ids)
        for full_obj_id in object_ids:
            obj_name, obj_uid = self.parse_object(full_obj_id)

            if obj_name not in self.object_dict:
                self.object_dict[obj_name] = {}

            if obj_uid not in self.object_dict[obj_name]:
                isSliced = False
                isBroken = False
                if obj_name in self.sliceable_objects:
                    isSliced = self.get_object_status_byID(full_obj_id).get("isSliced", False)
                if obj_name in self.breakable_objects:
                    isBroken = self.get_object_status_byID(full_obj_id).get("isBroken", False)
                if isBroken:
                    continue
                if isSliced:
                    if obj_name != 'Egg':
                        if "Cracked" not in obj_uid:
                            continue
                    else:
                        if 'Sliced' not in obj_uid:
                            continue
                self.object_dict[obj_name][obj_uid] = len(self.object_dict[obj_name]) + 1


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
        readable_list.sort()
        return readable_list
    
    def convert_readable_object_to_id(self, object_name: str) -> str:
        """Convert readable object name to ID."""
        if "|" in object_name:
            # If the object name contains coordinates, we assume it's already in the correct format
            obj_name, obj_id = self.parse_object(object_name)
            if obj_id not in self.object_dict[obj_name]:
                ValueError(f"Object {object_name} not found in object_dict")
            return object_name
        else:
            obj_name, obj_num = object_name.split("_")
            if 'Sliced' in obj_name:
                obj_name = obj_name.replace("Sliced", "")

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
            action_dict["forceAction"] = True
        elif action.startswith(tuple(self.object_interaction_actions)):
            action_name = action.split("(")[0]
            object_id = action.split("(")[1].rstrip(")")
            action_dict["action"] = action_name
            action_dict["objectId"] = self.convert_readable_object_to_id(object_id)
            if action_name == "PutObject":
                action_dict["forceAction"] = True
            if action_name == "PickupObject":
                print('try to pick up ', object_id)
                action_dict["forceAction"] = True
            if action_name == "OpenObject" or action_name == "CloseObject":
                action_dict["forceAction"] = True
        elif action.startswith("DropHandObject"):
            action_dict["action"] = "DropHandObject"
            action_dict["forceAction"] = True
        elif action.startswith("ThrowObject"):
            action_dict["action"] = "ThrowObject"
            action_dict["moveMagnitude"] = 150.0
        elif action.startswith("NavigateTo"):
            action_dict["action"] = "Pass"
        elif action.startswith("Look"):
            action_name = action.split("(")[0]
            action_dict["action"] = action_name
            degrees = action.split("(")[1].rstrip(")")
            action_dict["degrees"] = int(degrees)
        elif action.startswith("AlignOrientation"):
            action_dict["agentId"] = agent_id
            pitch, yaw, z = action.split("(")[1].split(")")[0].split(",")
            agent_metadata = self.event.events[agent_id].metadata["agent"]
            horizon = agent_metadata["cameraHorizon"]

            action_dict["action"] = "TeleportFull"
            action_dict["rotation"] = dict(x=pitch, y=yaw, z=0)
            action_dict["position"] = agent_metadata["position"]
            z = eval(z) 
            if z:
                action_dict["horizon"] = 0
            else:
                action_dict["horizon"] = horizon

            action_dict["standing"] = True
            action_dict["forceAction"] = True
        else:
            raise ValueError(f"Unsupported action: {action}")
        return action_dict
    
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
                    "isUsedUp": obj.get("isUsedUp", True),
                    "isDirty": obj.get("isDirty", False),
                    "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                    'contains': obj.get("receptacleObjectIds", None),
                }
                
                # print('get status for ', object_name, status)
                return status
        raise ValueError(f"Object {object_name} not found in the current scene.")
    
    def get_all_object_status(self) -> Dict[str, Any]:
        """Return the status of all object"""
        res = {}
        for obj in self.event.metadata["objects"]:
            
            status = {
                "object_id": obj["objectId"],
                "name": obj["name"],
                "position": obj["position"],
                "rotation": obj["rotation"],
                "is_open": obj.get("isOpen", False),
                "is_on": obj.get("isToggled", False),
                "is_picked_up": obj.get("isPickedUp", False),
                "isSliced": obj.get("isSliced", False),
                "isToggled": obj.get("isToggled", False),
                "isBroken": obj.get("isBroken", False),
                "isUsedUp": obj.get("isUsedUp", True),
                "isDirty": obj.get("isDirty", False),
                "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                'contains': obj.get("receptacleObjectIds", None),
            }
            res[obj["objectId"]] = status
            # print('get status for ', object_name, status)
        return res
        # raise ValueError(f"Object {object_name} not found in the current scene.")
    
    def extract_frame_indices(self, filename):
        """
        Extracts the primary and optional secondary frame index from filename like 'frame_0_2.png' or 'frame_1.png'.
        Returns a tuple (primary, secondary) for sorting.
        """
        match = re.match(r"frame_(\d+)(?:_(\d+))?\.png", filename)
        if match:
            primary = int(match.group(1))
            secondary = int(match.group(2)) if match.group(2) else 0
            return (primary, secondary)
        return (float('inf'), float('inf')) 

    def save_to_video(self, fps: int = 10, delete_frames: bool = False):
        """
        Convert saved frames into videos for each agent's POV and overhead view.
        
        Args:
            file_name (str): The task folder name (e.g., 'logs/Summary/put_remote_control,_keys,_and_watch_in_the_box/Floorplan201/test_{i}')
            fps (int): Frames per second for the output video (default: 30).
            delete_frames (bool): Whether to delete frame images after video creation.
        """
        task_path = self.base_path
       
        print(f"Resolved task path: {task_path}")
        
        if not task_path.exists() or not task_path.is_dir():
            raise ValueError(f"Task folder {task_path} does not exist or is not a directory.")
        
        # Find all subfolders (e.g., Alice/pov, Bob/pov, overhead)
        subfolders = []
        # Look for agent POV folders (e.g., Alice/pov)
        for agent_folder in task_path.iterdir():
            if agent_folder.is_dir() and (agent_folder / "pov").exists():
                subfolders.append(agent_folder / "pov")
            elif agent_folder.name == "overhead" and agent_folder.is_dir():
                subfolders.append(agent_folder)
        
        if not subfolders:
            raise ValueError(f"No valid subfolders (agent POV or overhead) found in {task_path}.")
        
        # Process each subfolder to create a video
        for subfolder in subfolders:
            # Collect all frame files in the subfolder
            # frame_files = sorted(
            #     [f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"],
            #     key=lambda x: int(re.search(r'frame_(\d+)\.png', x.name).group(1))
            # )
            frame_files = sorted(
                [f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"],
                key=lambda x: self.extract_frame_indices(x.name)
            )
            
            if not frame_files:
                print(f"No frames found in {subfolder}. Skipping video creation.")
                continue
            
            # Read the first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            height, width, _ = first_frame.shape
            
            # Define the output video path
            video_name = subfolder.name if subfolder.name == "overhead" else f"{subfolder.parent.name}_{subfolder.name}"
            video_path = task_path / f"{video_name}.mp4"
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            # Write each frame to the video
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                video_writer.write(frame)
            
            # Release the video writer
            video_writer.release()

            print(f"Video saved at {video_path}")
            if delete_frames:
                for frame_file in frame_files:
                    try:
                        frame_file.unlink()
                    except Exception as e:
                        print(f"Warning: could not delete {frame_file}: {e}")
                print(f"Deleted {len(frame_files)} frame images from {subfolder}")




    def get_object_status_byID(self, obj_id: str) -> Dict[str, Any]:
        """Return the status of a specific object given its readable name."""
        # obj_id = self.convert_readable_object_to_id(object_name)
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
        raise ValueError(f"Object {obj_id} not found in the current scene.")

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

