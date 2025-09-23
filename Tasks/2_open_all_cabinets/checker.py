"""
To check completion of subtasks in 2_open_all_drawers
Also checks coverage of the task and success of the task.
Subtasks:
    • NavigateTo(Cabinet)
    • OpenObject(Cabinet) [no need for objects in inventory]
Coverage:
    • Cabinet(floorplan1): 9/9
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker
def make_cabinet(n: int):
    return [f"Cabinet_{i}" for i in range(1, n + 1)]

def build_checker(env=None):
    current_scene = env.scene if env else "FloorPlan1"
    if current_scene == "FloorPlan1":
        required = make_cabinet(9)  # 9 cabinets in FloorPlan1,2
    elif current_scene == "FloorPlan6":
        required = make_cabinet(15)  # 15 cabinets in FloorPlan6
    elif current_scene == "FloorPlan7":   
        required = make_cabinet(13)  # 13 cabinets in FloorPlan7
    elif current_scene == "FloorPlan8":
        required = make_cabinet(17)
    elif current_scene == "FloorPlan9":
        required = make_cabinet(28)
    else:
        required = make_cabinet(6)  # 9 cabinets in other scenes 10

    cfg = {

        "status_check": [{"is_open": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)