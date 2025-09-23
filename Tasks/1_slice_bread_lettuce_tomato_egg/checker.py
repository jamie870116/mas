"""
To check completion of subtasks in 1_put_pots_pans_stove_burner task.
Also checks coverage of the task and success of the task.
Subtasks:
""
• NavigateTo(Bread)
• SliceObject(Bread)
• NavigateTo(Tomato)
• SliceObject(Tomato)
• NavigateTo(Egg)
• SliceObject(Egg)
• NavigateTo(Lettuce)
• SliceObject(Lettuce)""
Coverage:
    • Bread
    • Tomato
    • Egg
    • Lettuce
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):

    required = ["Bread_1", "Tomato_1", "Egg_1", "Lettuce_1"]

    cfg = {

        "status_check": [{"isSliced": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)

# from AI2Thor.baselines.utils.checker import BaseChecker


# class Checker(BaseChecker):
#     def __init__(self) -> None:
#         subtasks = [
#             "NavigateTo(Bread)",
#             "SliceObject(Bread)",
#             "NavigateTo(Tomato)",
#             "SliceObject(Tomato)",
#             "NavigateTo(Egg)",
#             "SliceObject(Egg)",
#             "NavigateTo(Lettuce)",
#             "SliceObject(Lettuce)",
#         ]
#         conditional_subtasks = []
#         independent_subtasks = [
#             "NavigateTo(Bread)",
#             "SliceObject(Bread)",
#             "NavigateTo(Tomato)",
#             "SliceObject(Tomato)",
#             "NavigateTo(Egg)",
#             "SliceObject(Egg)",
#             "NavigateTo(Lettuce)",
#             "SliceObject(Lettuce)",
#         ]
#         coverage = ["Bread", "Tomato", "Egg", "Lettuce"]
#         interact_objects = ["Bread", "Tomato", "Egg", "Lettuce"]
#         interact_receptacles = []

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
