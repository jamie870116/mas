"""
To check completion of subtasks in 1_put_pots_pans_stove_burner task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Plate)
    • PickupObject(Plate)
    • NavigateTo(Fridge) [with plate in inventory]
    • OpenObject(Fridge) [no need for objects in inventory, need to check only once]
    • PutObject(Fridge) [with plate in inventory]
    • NavigateTo(Mug)
    • PickupObject(Mug)
    • NavigateTo(Fridge) [with mug in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • PutObject(Fridge) [with mug in inventory]
    • NavigateTo(Bowl)
    • PickupObject(Bowl)
    • NavigateTo(Fridge) [with bowl in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • PutObject(Fridge) [with bowl in inventory]
    • CloseObject(Fridge)
Coverage:
    • Plate
    • Mug
    • Fridge
    • Bowl
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def make_drawers(n: int):
    return [f"Drawer_{i}" for i in range(1, n + 1)]

def build_checker(env=None):
    current_scene = env.scene if env else "FloorPlan1"
    if current_scene == "FloorPlan2":
        receptacle = make_drawers(13)  # 13 drawers in FloorPlan2
    elif current_scene == "FloorPlan3":
        receptacle = make_drawers(8)  # 8 drawers in FloorPlan3
    elif current_scene == "FloorPlan4":
        receptacle = make_drawers(6)  # 6 drawers in FloorPlan4
    elif current_scene == "FloorPlan5":
        receptacle = make_drawers(3)  # 3 drawers in FloorPlan5
    elif current_scene == "FloorPlan6":
        receptacle = make_drawers(4) # 4 drawers in FloorPlan6
    else:   
        receptacle = make_drawers(9)  # "FloorPlan1"

    receptacle.extend(["CounterTop_1", "CounterTop_2", "CounterTop_3"])
    
    required = ["Plate_1", "Mug_1", "Bowl_1"]

    cfg = {
        "is_multiple": True,
        "receptacle": receptacle,
        "recept_require_items": required,
        # "status_check": {"is_on": False},      
        # "status_require_items": [faucet],
    }
    
    return TaskConfigChecker.from_config(cfg)



# from AI2Thor.baselines.utils.checker import BaseChecker


# class Checker(BaseChecker):
#     def __init__(self) -> None:
#         subtasks = [
#             "NavigateTo(Plate)",
#             "PickupObject(Plate)",
#             "NavigateTo(Fridge, Plate)",
#             "PutObject(Fridge, Plate)",
#             "NavigateTo(Mug)",
#             "PickupObject(Mug)",
#             "NavigateTo(Fridge, Mug)",
#             "PutObject(Fridge, Mug)",
#             "NavigateTo(Bowl)",
#             "PickupObject(Bowl)",
#             "NavigateTo(Fridge, Bowl)",
#             "PutObject(Fridge, Bowl)",
#             "OpenObject(Fridge)",
#             "CloseObject(Fridge)",
#         ]
#         conditional_subtasks = [
#             "NavigateTo(Fridge, Plate)",
#             "PutObject(Fridge, Plate)",
#             "NavigateTo(Fridge, Mug)",
#             "PutObject(Fridge, Mug)",
#             "NavigateTo(Fridge, Bowl)",
#             "PutObject(Fridge, Bowl)",
#         ]
#         independent_subtasks = [
#             "NavigateTo(Plate)",
#             "PickupObject(Plate)",
#             "NavigateTo(Mug)",
#             "PickupObject(Mug)",
#             "NavigateTo(Bowl)",
#             "PickupObject(Bowl)",
#             "OpenObject(Fridge)",
#             "CloseObject(Fridge)",
#         ]
#         coverage = ["Plate", "Fridge", "Mug", "Bowl"]
#         interact_objects = ["Plate", "Mug", "Bowl"]
#         interact_receptacles = ["Fridge"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
