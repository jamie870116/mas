"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(KeyChain)
    • PickUpObject(KeyChain)
    • NavigateTo(Drawer) [with KeyChain in inventory]
    • OpenObject(Drawer) [with KeyChain in inventory]
    • PutObject(Drawer) [with KeyChain in inventory]
    • NavigateTo(Pencil)
    • PickUpObject(Pencil)
    • NavigateTo(Drawer) [with Pencil in inventory]
    • OpenObject(Drawer) [with Pencil in inventory]
    • PutObject(Drawer) [with Pencil in inventory]
    • NavigateTo(Pen)
    • PickUpObject(Pen)
    • NavigateTo(Drawer) [with Pen in inventory]
    • OpenObject(Drawer) [with Pen in inventory]
    • PutObject(Drawer) [with Pen in inventory]
    • NavigateTo(Book)
    • PickUpObject(Book)
    • NavigateTo(Drawer) [with Book in inventory]
    • OpenObject(Drawer) [with Book in inventory]
    • PutObject(Drawer) [with Book in inventory]
    • NavigateTo(Watch)
    • PickUpObject(Watch)
    • NavigateTo(Drawer) [with Watch in inventory]
    • OpenObject(Drawer) [with Watch in inventory]
    • PutObject(Drawer) [with Watch in inventory]

Coverage:
    • KeyChain
    • Pencil
    • Pen
    • Book
    • Watch
    • Drawer

202 don't have drawer
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
    if current_scene == "FloorPlan201":
        receptacle = make_drawers(2)  # 2 drawers in FloorPlan201
    elif current_scene == "FloorPlan202":
        receptacle = make_drawers(1)  # 0 drawers in FloorPlan202
    elif current_scene == "FloorPlan203":
        receptacle = make_drawers(7)  # 7 drawers in FloorPlan203
    elif current_scene == "FloorPlan209":
        receptacle = make_drawers(4)  # 4 drawers in FloorPlan209
    elif current_scene == "FloorPlan212":
        receptacle = make_drawers(1) # 1 drawers in FloorPlan212
    else:   
        receptacle = make_drawers(1) 
    receptacle.append("Shelf_1")  # Add fridge as a receptacle
    required = ["KeyChain_1", "Pencil_1", "Book_1", "Watch_1", "Pen_1"]

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
#         'NavigateTo(KeyChain)',
#          'PickUpObject(KeyChain)',
#          'NavigateTo(Drawer, Keychain)',
#          'OpenObject(Drawer, Keychain)',
#          'PutObject(Drawer, Keychain)',
#          'NavigateTo(Pencil)',
#          'PickUpObject(Pencil)',
#          'NavigateTo(Drawer, Pencil)',
#          'OpenObject(Drawer, Pencil)',
#          'PutObject(Drawer, Pencil)',
#          'NavigateTo(Pen)',
#          'PickUpObject(Pen)',
#          'NavigateTo(Drawer, Pen)',
#          'OpenObject(Drawer, Pen)',
#          'PutObject(Drawer, Pen)',
#          'NavigateTo(Book)',
#          'PickUpObject(Book)',
#          'NavigateTo(Drawer, Book)',
#          'OpenObject(Drawer, Book)',
#          'PutObject(Drawer, Book)',
#          'NavigateTo(Watch)',
#          'PickUpObject(Watch)',
#          'NavigateTo(Drawer, Watch)',
#          'OpenObject(Drawer, Watch)',
#          'PutObject(Drawer, Watch)'
#         ]

#         conditional_subtasks = [
#             'NavigateTo(Drawer, Keychain)',
#              'OpenObject(Drawer, Keychain)',
#              'PutObject(Drawer, Keychain)',
#              'NavigateTo(Drawer, Pencil)',
#              'OpenObject(Drawer, Pencil)',
#              'PutObject(Drawer, Pencil)',
#              'NavigateTo(Drawer, Pen)',
#              'OpenObject(Drawer, Pen)',
#              'PutObject(Drawer, Pen)',
#              'NavigateTo(Drawer, Book)',
#              'OpenObject(Drawer, Book)',
#              'PutObject(Drawer, Book)',
#              'NavigateTo(Drawer, Watch)',
#              'OpenObject(Drawer, Watch)',
#              'PutObject(Drawer, Watch)'
#             ]

#         independent_subtasks = [
#             'NavigateTo(KeyChain)',
#              'PickUpObject(KeyChain)',
#              'NavigateTo(Pencil)',
#              'PickUpObject(Pencil)',
#              'NavigateTo(Pen)',
#              'PickUpObject(Pen)',
#              'NavigateTo(Book)',
#              'PickUpObject(Book)',
#              'NavigateTo(Watch)',
#              'PickUpObject(Watch)'
#             ]

#         # keychain, pencil, pen, book, watch + (drawer)
#         coverage = ["KeyChain", "Pencil", "Pen", "Book", "Watch", "Drawer"]
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
