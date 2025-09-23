"""
To check completion of subtasks in 2_open_all_drawers
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Drawer)
    • OpenObject(Drawer) [no need for objects in inventory]
Coverage:
    • Drawer
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
        required = make_drawers(13)  # 13 drawers in FloorPlan2
    elif current_scene == "FloorPlan3":
        required = make_drawers(8)  # 8 drawers in FloorPlan3
    elif current_scene == "FloorPlan4":
        required = make_drawers(6)  # 6 drawers in FloorPlan4
    elif current_scene == "FloorPlan5":
        required = make_drawers(3)  # 3 drawers in FloorPlan5
    elif current_scene == "FloorPlan6":
        required = make_drawers(4) # 4 drawers in FloorPlan6
    elif current_scene == "FloorPlan7":
        required = make_drawers(4)
    elif current_scene == "FloorPlan8":
        required = make_drawers(8)
    elif current_scene == "FloorPlan9":
        required = make_drawers(11)
    else:   
        required = make_drawers(9)  # 9 drawers in Floorplan1


    cfg = {

        "status_check": [{"is_open": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)

# from AI2Thor.baselines.utils.checker import BaseChecker

# class Checker(BaseChecker):
#     def __init__(self) -> None:
#         subtasks = [
#             "NavigateTo(Drawer)",
#             "OpenObject(Drawer)",
#         ]
#         conditional_subtasks = [
#         ]

#         independent_subtasks = [
#             "NavigateTo(Drawer)",
#             "OpenObject(Drawer)",
#         ]
#         coverage = ["Drawer"]
#         interact_objects = ["Drawer"]
#         interact_receptacles = []

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
