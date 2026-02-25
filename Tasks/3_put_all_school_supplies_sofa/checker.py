"""
To check completion of subtasks in 2_put_all_tomatoes_potatoes_fridge task.
Also checks coverage of the task and success of the task.
Subtasks:
• NavigateTo(Pen)
• PickUpObject(Pen)
• NavigateTo(Sofa) [with Pen in inventory]
• PutObject(Sofa) [with Pen in inventory]
• NavigateTo(Pencil)
• PickUpObject(Pencil)
• NavigateTo(Sofa) [with Pencil in inventory]
• PutObject(Sofa) [with Pencil in inventory]
• NavigateTo(Laptop)
• PickUpObject(Laptop)
• NavigateTo(Sofa) [with Laptop in inventory]
• PutObject(Sofa) [with Laptop in inventory]
• NavigateTo(Book)
• PickUpObject(Book)
• NavigateTo(Sofa) [with Book in inventory]
• PutObject(Sofa) [with Book in inventory]
• NavigateTo(CellPhone)
• PickUpObject(CellPhone)
• NavigateTo(Sofa) [with CellPhone in inventory]
• PutObject(Sofa) [with CellPhone in inventory]

Coverage:
    • Pen
    • Pencil
    • Laptop
    • Book
    • CellPhone
    • Sofa
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
   
    receptacle = "Sofa_1"
    
    
    current_scene = env.scene if env else "FloorPlan201"
    if current_scene == "FloorPlan201":
        required = ["Pen", "Pencil", "Laptop", "Book"]
    elif current_scene == "FloorPlan202":
        required = [ "Laptop", "Book"]
    elif current_scene == "FloorPlan203":
        required = ["Pencil", "Laptop", "Book", "CellPhone"]
    elif current_scene == "FloorPlan209":
        required = ["Pen", "Laptop", "Book"]
    else:  # 212
        required = ["Pen", "Pencil", "Laptop"]
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
#             'NavigateTo(Pen)',
#              'PickUpObject(Pen)',
#              'NavigateTo(Sofa, Pen)',
#              'PutObject(Sofa, Pen)',
#              'NavigateTo(Pencil)',
#              'PickUpObject(Pencil)',
#              'NavigateTo(Sofa, Pencil)',
#              'PutObject(Sofa, Pencil)',
#              'NavigateTo(Laptop)',
#              'PickUpObject(Laptop)',
#              'NavigateTo(Sofa, Laptop)',
#              'PutObject(Sofa, Laptop)',
#              'NavigateTo(Book)',
#              'PickUpObject(Book)',
#              'NavigateTo(Sofa, Book)',
#              'PutObject(Sofa, Book)',
#              'NavigateTo(CellPhone)',
#              'PickUpObject(CellPhone)',
#              'NavigateTo(Sofa, Cellphone)',
#              'PutObject(Sofa, Cellphone)'
#             ]

#         conditional_subtasks = [
#         'NavigateTo(Sofa, Pen)',
#          'PutObject(Sofa, Pen)',
#          'NavigateTo(Sofa, Pencil)',
#          'PutObject(Sofa, Pencil)',
#          'NavigateTo(Sofa, Laptop)',
#          'PutObject(Sofa, Laptop)',
#          'NavigateTo(Sofa, Book)',
#          'PutObject(Sofa, Book)',
#          'NavigateTo(Sofa, Cellphone)',
#          'PutObject(Sofa, Cellphone)'
#         ]


#         independent_subtasks = [
#             'NavigateTo(Pen)',
#              'PickUpObject(Pen)',
#              'NavigateTo(Pencil)',
#              'PickUpObject(Pencil)',
#              'NavigateTo(Laptop)',
#              'PickUpObject(Laptop)',
#              'NavigateTo(Book)',
#              'PickUpObject(Book)',
#              'NavigateTo(CellPhone)',
#              'PickUpObject(CellPhone)'
#             ]

#         coverage = ["Pen", "Pencil", "Laptop", "Book", "CellPhone", "Sofa"]
#         interact_objects = coverage
#         interact_receptacles = ["Sofa"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
