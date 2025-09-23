"""
To check completion of subtasks in 1_turn_off_faucet_light task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Faucet)
    • ToggleObjectOff(Faucet)
    • NavigateTo(LightSwitch)
    • ToggleObjectOff(LightSwitch)
Coverage:
    • Faucet
    • LightSwitch
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):

    required = ["LightSwitch_1", "Faucet_1"]

    cfg = {

        "status_check": [{"is_off": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)
# from AI2Thor.baselines.utils.checker import BaseChecker


# class Checker(BaseChecker):
#     def __init__(self) -> None:
#         subtasks = [
#             "NavigateTo(Faucet)",
#             "ToggleObjectOff(Faucet)",
#             "NavigateTo(LightSwitch)",
#             "ToggleObjectOff(LightSwitch)",
#         ]
#         conditional_subtasks = []
#         independent_subtasks = [
#             "NavigateTo(Faucet)",
#             "ToggleObjectOff(Faucet)",
#             "NavigateTo(LightSwitch)",
#             "ToggleObjectOff(LightSwitch)",
#         ]
#         coverage = ["Faucet", "LightSwitch"]
#         interact_objects = ["Faucet", "LightSwitch"]
#         interact_receptacles = []

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
