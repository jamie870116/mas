"""
To check completion of subtasks
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(LightSwitch)
    • ToggleObjectOff(LightSwitch)

Coverage:
    • LightSwitch


"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):

    required = ["LightSwitch_1", "DeskLamp_1", "FloorLamp_1"]

    cfg = {
        # "receptacle": receptacle,
        # "recept_require_items": required,
        "status_check": [{"is_off": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)

# from AI2Thor.baselines.utils.checker import BaseChecker

# class Checker(BaseChecker):
#     def __init__(self) -> None:

#         subtasks = [
#         'NavigateTo(LightSwitch)', 'ToggleObjectOff(LightSwitch)'
#         ]

#         conditional_subtasks = []

#         independent_subtasks = subtasks

#         coverage = ["LightSwitch"]
#         interact_objects = coverage
#         interact_receptacles = []

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
