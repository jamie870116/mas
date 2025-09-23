"""
To check completion of subtasks in 3_put_all_groceries_fridge task.
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
    • NavigateTo(Apple)
    • PickupObject(Apple)
    • NavigateTo(Fridge) [with apple in inventory]
    • OpenObject(Fridge) [no need for objects in inventory, need to check only once]
    • PutObject(Fridge) [with apple in inventory]
    • NavigateTo(Potato)
    • PickupObject(Potato)
    • NavigateTo(Fridge) [with potato in inventory]
    • OpenObject(Fridge) [no need for objects in inventory, need to check only once]
    • PutObject(Fridge) [with potato in inventory]
    • CloseObject(Fridge)
Coverage:
    • Bread
    • Fridge
    • Tomato
    • Apple
    • Potato
    • Lettuce
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    receptacle = ["CounterTop_1", "CounterTop_2", "CounterTop_3"]
    
    required = ["Bread_1", "Tomato_1", "Apple_1", "Potato_1", "Lettuce_1", "Egg_1"]

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
#             "NavigateTo(Apple)",
#             "PickupObject(Apple)",
#             "NavigateTo(Fridge, Apple)",
#             "PutObject(Fridge, Apple)",
#             "NavigateTo(Potato)",
#             "PickupObject(Potato)",
#             "NavigateTo(Fridge, Potato)",
#             "PutObject(Fridge, Potato)",
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
#             "NavigateTo(Fridge, Apple)",
#             "PutObject(Fridge, Apple)",
#             "NavigateTo(Fridge, Potato)",
#             "PutObject(Fridge, Potato)",
#         ]
#         independent_subtasks = [
#             "NavigateTo(Bread)",
#             "PickupObject(Bread)",
#             "NavigateTo(Tomato)",
#             "PickupObject(Tomato)",
#             "NavigateTo(Lettuce)",
#             "PickupObject(Lettuce)",
#             "NavigateTo(Apple)",
#             "PickupObject(Apple)",
#             "NavigateTo(Potato)",
#             "PickupObject(Potato)",
#             "OpenObject(Fridge)",
#             "CloseObject(Fridge)",
#         ]
#         coverage = ["Bread", "Fridge", "Tomato", "Lettuce", "Apple", "Potato"]
#         interact_objects = ["Bread", "Tomato", "Lettuce", "Apple", "Potato"]
#         interact_receptacles = ["Fridge"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
