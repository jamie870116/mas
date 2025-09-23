"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Tomato)
    • PickUpObject(Tomato)
    • NavigateTo(Fridge) [with Tomato in inventory]
    • PutObject(Fridge) [with Tomato in inventory]
    • NavigateTo(Lettuce)
    • PickUpObject(Lettuce)
    • NavigateTo(Fridge) [with Lettuce in inventory]
    • PutObject(Fridge) [with Lettuce in inventory]
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
    • NavigateTo(Spoon)
    • PickUpObject(Spoon)
    • NavigateTo(Drawer) [with Spoon in inventory]
    • OpenObject(Drawer) [with Spoon in inventory]
    • PutObject(Drawer) [with Spoon in inventory]
    • OpenObject(Fridge)
    • CloseObject(Fridge)

Coverage:
    • Butterknife
    • Spoon
    • Fork
    • Tomato
    • Lettuce
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
    if current_scene == "FloorPlan4":
        receptacle = make_drawers(6)  # 6 drawers in FloorPlan4
    elif current_scene == "FloorPlan11":
        receptacle = make_drawers(6)  # 6 drawers in FloorPlan11
    elif current_scene == "FloorPlan15":
        receptacle = make_drawers(12) # 12 drawers in FloorPlan15
    elif current_scene == "FloorPlan16":
        receptacle = make_drawers(5) # 5 drawers in FloorPlan16
    else:   
        receptacle = make_drawers(6) # 5 drawers in FloorPlan17
    receptacle.append("Fridge_1")  # Add fridge as a receptacle
    required = ["ButterKnife_1", "Spoon_1", "Fork_1", "Tomato_1", "Lettuce_1"]

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
#          'NavigateTo(Lettuce)',
#          'PickUpObject(Lettuce)',
#          'NavigateTo(Fridge, Lettuce)',
#          'PutObject(Fridge, Lettuce)',
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
#          'NavigateTo(Spoon)',
#          'PickUpObject(Spoon)',
#          'NavigateTo(Drawer, Spoon)',
#          'OpenObject(Drawer, Spoon)',
#          'PutObject(Drawer, Spoon)',
#          'OpenObject(Fridge)',
#          'CloseObject(Fridge)'
#         ]



#         conditional_subtasks = [
#         'NavigateTo(Fridge, Tomato)',
#          'PutObject(Fridge, Tomato)',
#          'NavigateTo(Fridge, Lettuce)',
#          'PutObject(Fridge, Lettuce)',
#          'NavigateTo(Drawer, Butterknife)',
#          'OpenObject(Drawer, Butterknife)',
#          'PutObject(Drawer, Butterknife)',
#          'NavigateTo(Drawer, Fork)',
#          'OpenObject(Drawer, Fork)',
#          'PutObject(Drawer, Fork)',
#          'NavigateTo(Drawer, Spoon)',
#          'OpenObject(Drawer, Spoon)',
#          'PutObject(Drawer, Spoon)'
#         ]

#         independent_subtasks = [
#         'NavigateTo(Tomato)',
#          'PickUpObject(Tomato)',
#          'NavigateTo(Lettuce)',
#          'PickUpObject(Lettuce)',
#          'NavigateTo(ButterKnife)',
#          'PickUpObject(ButterKnife)',
#          'NavigateTo(Fork)',
#          'PickUpObject(Fork)',
#          'NavigateTo(Spoon)',
#          'PickUpObject(Spoon)',
#          'OpenObject(Fridge)',
#          'CloseObject(Fridge)'
#         ]


#         # Butterknife, spoon, fork, tomato, lettuce, fridge
#         coverage = ["ButterKnife", "Spoon", "Fork", "Tomato", "Lettuce", "Fridge"]
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
