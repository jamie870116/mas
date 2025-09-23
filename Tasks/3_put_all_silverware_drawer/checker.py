"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(ButterKnife)
• PickUpObject(ButterKnife)
• NavigateTo(Drawer) [with ButterKnife in inventory]
• OpenObject(Drawer) [with ButterKnife in inventory]
• PutObject(Drawer) [with ButterKnife in inventory]
• NavigateTo(Knife)
• PickUpObject(Knife)
• NavigateTo(Drawer) [with Knife in inventory]
• OpenObject(Drawer) [with Knife in inventory]
• PutObject(Drawer) [with Knife in inventory]
• NavigateTo(Spatula)
• PickUpObject(Spatula)
• NavigateTo(Drawer) [with Spatula in inventory]
• OpenObject(Drawer) [with Spatula in inventory]
• PutObject(Drawer) [with Spatula in inventory]
• NavigateTo(Spoon)
• PickUpObject(Spoon)
• NavigateTo(Drawer) [with Spoon in inventory]
• OpenObject(Drawer) [with Spoon in inventory]
• PutObject(Drawer) [with Spoon in inventory]
• NavigateTo(Fork)
• PickUpObject(Fork)
• NavigateTo(Drawer) [with Fork in inventory]
• OpenObject(Drawer) [with Fork in inventory]
• PutObject(Drawer) [with Fork in inventory]

Coverage:
    • ButterKnife 
    • Knife
    • Spatula
    • Spoon
    • Fork
    • Ladle
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
    
    required = ["ButterKnife_1", "Knife_1", "Spatula_1", "Spoon_1", "Fork_1", "Ladle_1"]

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
#           'NavigateTo(ButterKnife)',
#          'PickUpObject(ButterKnife)',
#          'NavigateTo(Drawer, Butterknife)',
#          'OpenObject(Drawer, Butterknife)',
#          'PutObject(Drawer, Butterknife)',
#          'NavigateTo(Knife)',
#          'PickUpObject(Knife)',
#          'NavigateTo(Drawer, Knife)',
#          'OpenObject(Drawer, Knife)',
#          'PutObject(Drawer, Knife)',
#          'NavigateTo(Spatula)',
#          'PickUpObject(Spatula)',
#          'NavigateTo(Drawer, Spatula)',
#          'OpenObject(Drawer, Spatula)',
#          'PutObject(Drawer, Spatula)',
#          'NavigateTo(Spoon)',
#          'PickUpObject(Spoon)',
#          'NavigateTo(Drawer, Spoon)',
#          'OpenObject(Drawer, Spoon)',
#          'PutObject(Drawer, Spoon)',
#          'NavigateTo(Fork)',
#          'PickUpObject(Fork)',
#          'NavigateTo(Drawer, Fork)',
#          'OpenObject(Drawer, Fork)',
#          'PutObject(Drawer, Fork)'
#          ]



#         conditional_subtasks = [
#         'NavigateTo(Drawer, Butterknife)',
#          'OpenObject(Drawer, Butterknife)',
#          'PutObject(Drawer, Butterknife)',
#          'NavigateTo(Drawer, Knife)',
#          'OpenObject(Drawer, Knife)',
#          'PutObject(Drawer, Knife)',
#          'NavigateTo(Drawer, Spatula)',
#          'OpenObject(Drawer, Spatula)',
#          'PutObject(Drawer, Spatula)',
#          'NavigateTo(Drawer, Spoon)',
#          'OpenObject(Drawer, Spoon)',
#          'PutObject(Drawer, Spoon)',
#          'NavigateTo(Drawer, Fork)',
#          'OpenObject(Drawer, Fork)',
#          'PutObject(Drawer, Fork)'
#         ]

#         independent_subtasks = [
#         'NavigateTo(ButterKnife)',
#          'PickUpObject(ButterKnife)',
#          'NavigateTo(Knife)',
#          'PickUpObject(Knife)',
#          'NavigateTo(Spatula)',
#          'PickUpObject(Spatula)',
#          'NavigateTo(Spoon)',
#          'PickUpObject(Spoon)',
#          'NavigateTo(Fork)',
#          'PickUpObject(Fork)'
#         ]

#         coverage = ["ButterKnife", "Knife", "Spatula", "Spoon", "Fork", "Ladle", "Drawer"]
#         interact_objects = coverage
#         interact_receptacles = ["Drawer"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
