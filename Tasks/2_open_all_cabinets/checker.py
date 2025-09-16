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

def build_checker(env=None):

    required = ["Cabinet_1", "Cabinet_2", "Cabinet_3", "Cabinet_4","Cabinet_5", "Cabinet_6", "Cabinet_7", "Cabinet_8","Cabinet_9"]

    cfg = {

        "status_check": [{"is_open": True}],      
        "status_require_items": [required],
    }
    return TaskConfigChecker.from_config(cfg)