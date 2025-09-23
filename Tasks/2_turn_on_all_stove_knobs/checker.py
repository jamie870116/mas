"""
Subtasks:
    • NavigateTo(StoveKnob)
    • ToggleObjectOn(StoveKnob) 
Coverage:
    • StoveKnob 

Turn all the stoveknobs off before the pre-init
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    current_scene = env.scene if env else "FloorPlan1"
    if current_scene == "FloorPlan2":
        required = ["StoveKnob_1", "StoveKnob_2", "StoveKnob_3", "StoveKnob_4", "StoveKnob_5", "StoveKnob_6"]
    else:
        required = ["StoveKnob_1", "StoveKnob_2", "StoveKnob_3", "StoveKnob_4"]
    

    cfg = {

        "status_check": [{"is_on": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)
# from AI2Thor.baselines.utils.checker import BaseChecker

# class Checker(BaseChecker):
#     def __init__(self) -> None:
#         subtasks = [
#             "NavigateTo(StoveKnob)",
#             "ToggleObjectOn(StoveKnob)",
#         ]
#         conditional_subtasks = []
#         independent_subtasks = [
#             "NavigateTo(StoveKnob)",
#             "ToggleObjectOn(StoveKnob)",
#         ]
#         coverage = ["StoveKnob"] 
#         interact_objects = ["StoveKnob"] 
#         interact_receptacles = []

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
