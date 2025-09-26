"""
    To check completion of subtasks in 1_put_remotecontrol_keys_watch_box.
    Also checks coverage of the task and success of the task.
    Subtasks:
    • NavigateTo(Laptop)
    • PickObject(Laptop)
    • NavigateTo(Sofa) [with computer in inventory]
    • PutObject(Sofa) [with computer in inventory]
    • NavigateTo(Book)
    • PickObject(Book)
    • NavigateTo(Sofa) [with book in inventory]
    • PutObject(Sofa) [with book in inventory]
    • NavigateTo(RemoteControl)
    • PickObject(RemoteControl)
    • NavigateTo(Sofa) [with RemoteControl in inventory]
    • PutObject(Sofa) [with RemoteControl in inventory]
    Coverage:
        • Computer
        • RemoteControl
        • Book
        • Sofa
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    receptacle = "Sofa_1"

    required = ["Laptop", "RemoteControl", "Book"]

    cfg = {
        "receptacle": receptacle,
        "recept_require_items": required,
        # "status_check": {"is_on": False},      
        # "status_require_items": [faucet],
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
#             "NavigateTo(Laptop)",
#             "PickObject(Laptop)",
#             "NavigateTo(Sofa, Laptop)",
#             "PutObject(Sofa, Laptop)",
#             "NavigateTo(RemoteControl)",
#             "PickObject(RemoteControl)",
#             "NavigateTo(Sofa, RemoteControl)",
#             "PutObject(Sofa, RemoteControl)",
#             "NavigateTo(Book)",
#             "PickObject(Book)",
#             "NavigateTo(Sofa, Book)",
#             "PutObject(Sofa, Book)",
#         ]
#         conditional_subtasks = [
#             "NavigateTo(Sofa, Laptop)",
#             "PutObject(Sofa, Laptop)",
#             "NavigateTo(Sofa, RemoteControl)",
#             "PutObject(Sofa, RemoteControl)",
#             "NavigateTo(Sofa, Book)",
#             "PutObject(Sofa, Book)",
#         ]
#         independent_subtasks = [
#             "NavigateTo(Laptop)",
#             "PickObject(Laptop)",
#             "NavigateTo(RemoteControl)",
#             "PickObject(RemoteControl)",
#             "NavigateTo(Book)",
#             "PickObject(Book)",
#         ]
#         coverage = ["Laptop", "Sofa", "RemoteControl", "Book"]
#         interact_objects = ["Laptop", "RemoteControl", "Book"]
#         interact_receptacles = ["Sofa"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
