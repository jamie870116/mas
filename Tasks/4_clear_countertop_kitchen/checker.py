"""
To check completion of subtasks in 4_clear_countertop_kitchen task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Tomato)
    • PickUpObject(Tomato)
    • NavigateTo(Fridge) [with Tomato in inventory]
    • PutObject(Fridge) [with Tomato in inventory]
    • NavigateTo(Apple)
    • PickUpObject(Apple)
    • NavigateTo(Fridge) [with Apple in inventory]
    • PutObject(Fridge) [with Apple in inventory]
    • NavigateTo(ButterKnife)
    • PickUpObject(ButterKnife)
    • NavigateTo(Drawer) [with ButterKnife in inventory]
    • OpenObject(Drawer) [with ButterKnife in inventory]
    • PutObject(Drawer) [with ButterKnife in inventory]
    • NavigateTo(Fork)
    • PickUpObject(Fork)
    • NavigateTo(Drawer) [with Fork in inventory]
    • OpenObject(Drawer) [with Fork in inventory]
    • PutObject(Drawer) [with Fork in inventory]
    • OpenObject(Fridge)
    • CloseObject(Fridge)

Coverage:
    • Apple
    • Tomato
    • Fork
    • ButterKnife
    • Fridge 
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
        receptacle = make_drawers(13)  # 13 drawers in FloorPlan2
    elif current_scene == "FloorPlan3":
        receptacle = make_drawers(8)  # 8 drawers in FloorPlan3
    elif current_scene == "FloorPlan4":
        receptacle = make_drawers(6)  # 6 drawers in FloorPlan4
    elif current_scene == "FloorPlan5":
        receptacle = make_drawers(3)  # 3 drawers in FloorPlan5
    elif current_scene == "FloorPlan6":
        receptacle = make_drawers(4) # 4 drawers in FloorPlan6
    else:   
        receptacle = make_drawers(9) 
    receptacle.append("Fridge_1")  # Add fridge as a receptacle
    required = ["ButterKnife_1", "Apple_1", "Tomato_1", "Fork_1"]

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
#         'NavigateTo(Tomato)',
#          'PickUpObject(Tomato)',
#          'NavigateTo(Fridge, Tomato)',
#          'PutObject(Fridge, Tomato)',
#          'NavigateTo(Apple)',
#          'PickUpObject(Apple)',
#          'NavigateTo(Fridge, Apple)',
#          'PutObject(Fridge, Apple)',
#          'NavigateTo(ButterKnife)',
#          'PickUpObject(ButterKnife)',
#          'NavigateTo(Drawer, Butterknife)',
#          'OpenObject(Drawer, Butterknife)',
#          'PutObject(Drawer, Butterknife)',
#          'NavigateTo(Fork)',
#          'PickUpObject(Fork)',
#          'NavigateTo(Drawer, Fork)',
#          'OpenObject(Drawer, Fork)',
#          'PutObject(Drawer, Fork)',
#          'OpenObject(Fridge)',
#          'CloseObject(Fridge)'
#         ]


#         conditional_subtasks = [
#         'NavigateTo(Fridge, Tomato)',
#          'PutObject(Fridge, Tomato)',
#          'NavigateTo(Fridge, Apple)',
#          'PutObject(Fridge, Apple)',
#          'NavigateTo(Drawer, Butterknife)',
#          'OpenObject(Drawer, Butterknife)',
#          'PutObject(Drawer, Butterknife)',
#          'NavigateTo(Drawer, Fork)',
#          'OpenObject(Drawer, Fork)',
#          'PutObject(Drawer, Fork)',
#         ]

#         independent_subtasks = [
#         'NavigateTo(Tomato)',
#          'PickUpObject(Tomato)',
#          'NavigateTo(Apple)',
#          'PickUpObject(Apple)',
#          'NavigateTo(ButterKnife)',
#          'PickUpObject(ButterKnife)',
#          'NavigateTo(Fork)',
#          'PickUpObject(Fork)',
#          'OpenObject(Fridge)',
#          'CloseObject(Fridge)'
#         ]

#         coverage = ["Fridge", "Apple", "Tomato", "Fork", "ButterKnife"]
#         interact_objects = coverage
#         interact_receptacles = ["Fridge"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
