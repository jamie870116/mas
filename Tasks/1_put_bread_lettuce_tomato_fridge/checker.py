"""
To check completion of subtasks in 1_transport_bread_lettuce_tomato task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Bread)
    • PickupObject(Bread)
    • NavigateTo(Fridge) [with bread in inventory]
    • OpenObject(Fridge) [no need for objects in inventory, need to check only once]
    • PutObject(Fridge) [with bread in inventory]
    • NavigateTo(Tomato)
    • PickupObject(Tomato)
    • NavigateTo(Fridge) [with tomato in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • PutObject(Fridge) [with tomato in inventory]
    • NavigateTo(Lettuce)
    • PickupObject(Lettuce)
    • NavigateTo(Fridge) [with lettuce in inventory]
    • OpenObject(Fridge) [no need for objects in inventory]
    • PutObject(Fridge) [with lettuce in inventory]
    • CloseObject(Fridge)
Coverage:
    • Bread
    • Fridge
    • Tomato
    • Lettuce
"""


import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # Tasks/<task>/checker.py 上兩層到專案根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    receptacle = "Fridge_1"

    required = ["Bread", "Tomato", "Lettuce"]

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
#             "NavigateTo(Bread)",
#             "PickupObject(Bread)",
#             "NavigateTo(Fridge, Bread)",
#             "PutObject(Fridge, Bread)",
#             "NavigateTo(Tomato)",
#             "PickupObject(Tomato)",
#             "NavigateTo(Fridge, Tomato)",
#             "PutObject(Fridge, Tomato)",
#             "NavigateTo(Lettuce)",
#             "PickupObject(Lettuce)",
#             "NavigateTo(Fridge, Lettuce)",
#             "PutObject(Fridge, Lettuce)",
#             "OpenObject(Fridge)",
#             "CloseObject(Fridge)",
#         ]
#         conditional_subtasks = [
#             "NavigateTo(Fridge, Bread)",
#             "PutObject(Fridge, Bread)",
#             "NavigateTo(Fridge, Tomato)",
#             "PutObject(Fridge, Tomato)",
#             "NavigateTo(Fridge, Lettuce)",
#             "PutObject(Fridge, Lettuce)",
#         ]
#         independent_subtasks = [
#             "NavigateTo(Bread)",
#             "PickupObject(Bread)",
#             "NavigateTo(Tomato)",
#             "PickupObject(Tomato)",
#             "NavigateTo(Lettuce)",
#             "PickupObject(Lettuce)",
#             "OpenObject(Fridge)",
#             "CloseObject(Fridge)",
#         ]
#         coverage = ["Bread", "Fridge", "Tomato", "Lettuce"]
#         interact_objects = ["Bread", "Tomato", "Lettuce"]
#         interact_receptacles = ["Fridge"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
