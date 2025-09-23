"""
put knife bowl mug on countertop
flooplan3,4=>countertop1
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # Tasks/<task>/checker.py 上兩層到專案根
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    receptacle = ["CounterTop_1", "CounterTop_2", "CounterTop_3"]
    
    required = ["Bowl_1", "ButterKnife_1", "Mug_1"]

    cfg = {
        "is_multiple": True,
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