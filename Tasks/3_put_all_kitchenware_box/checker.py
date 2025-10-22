"""
    To check completion of subtasks in 3_put_all_school_supplies_couch.
    Also checks coverage of the task and success of the task.
    Subtasks:
    • NavigateTo(Bowl)
    • PickObject(Bowl)
    • NavigateTo(Box) [with Bowl in inventory]
    • PutObject(Box) [with Bowl in inventory]
    • NavigateTo(Plate)
    • PickObject(Plate)
    • NavigateTo(Box) [with Plate in inventory]
    • PutObject(Box) [with Plate in inventory]
    Coverage:
        • Bowl
        • Pencil
        • Plate
        • Box
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
   
    receptacle = "Box_1"
    

    required = ["Bowl", "Plate"]

    cfg = {
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
#             "NavigateTo(Bowl)",
#             "PickObject(Bowl)",
#             "NavigateTo(Box, Bowl)",
#             "PutObject(Box, Bowl)",
#             "NavigateTo(Plate)",
#             "PickObject(Plate)",
#             "NavigateTo(Box, Plate)",
#             "PutObject(Box, Plate)",
#         ]
#         conditional_subtasks = [
#             "NavigateTo(Box, Bowl)",
#             "PutObject(Box, Bowl)",
#             "NavigateTo(Box, Plate)",
#             "PutObject(Box, Plate)",
#         ]
#         independent_subtasks = [
#             "NavigateTo(Bowl)",
#             "PickObject(Bowl)",
#             "NavigateTo(Plate)",
#             "PickObject(Plate)",
#         ]
#         coverage = ["Bowl", "Box", "Plate"]
#         interact_objects = ["Bowl", "Plate"]
#         interact_receptacles = ["Box"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
