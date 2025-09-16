"""
To check completion of subtasks in 1_wash_bowl_mug_pot_pan task.
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(Bowl)
• CleanObject(Bowl)
• NavigateTo(Mug)
• CleanObject(Mug)
• NavigateTo(Pan)
• CleanObject(Pan)
• NavigateTo(Pot)
• CleanObject(Pot)
Coverage:
    • Pot
    • Pan
    • Bowl
    • Mug
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    receptacle = ""

    required = ["Bowl_1", "Mug_1", "Pan_1", "Pot_1"]

    cfg = {
        # "receptacle": receptacle,
        # "recept_require_items": required,
        "status_check": [{"isDirty": False}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)

if __name__ == "__main__":
    checker = build_checker()
    print("Checker created:", checker)
    fake_env = None
    checker = build_checker(fake_env)
    result = checker.check(env=fake_env)  
    print("Check result:", result)
# from AI2Thor.baselines.utils.checker import BaseChecker


# class Checker(BaseChecker):
#     def __init__(self) -> None:
#         subtasks = [
#             "NavigateTo(Bowl)",
#             "CleanObject(Bowl)",
#             "NavigateTo(Mug)",
#             "CleanObject(Mug)",
#             "NavigateTo(Pan)",
#             "CleanObject(Pan)",
#             "NavigateTo(Pot)",
#             "CleanObject(Pot)",
#         ]
#         conditional_subtasks = []
#         independent_subtasks = [
#             "NavigateTo(Bowl)",
#             "CleanObject(Bowl)",
#             "NavigateTo(Mug)",
#             "CleanObject(Mug)",
#             "NavigateTo(Pan)",
#             "CleanObject(Pan)",
#             "NavigateTo(Pot)",
#             "CleanObject(Pot)",
#         ]
#         coverage = ["Bowl", "Mug", "Pan", "Pot"]
#         interact_objects = ["Bowl", "Mug", "Pan", "Pot"]
#         interact_receptacles = []

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
