"""
    To check completion of subtasks in 1_put_remotecontrol_keys_watch_box.
    Also checks coverage of the task and success of the task.
    Subtasks:
    • NavigateTo(remotecontrol)
    • PickupObject(remotecontrol)
    • NavigateTo(Box) [with remotecontrol in inventory]
    • PutObject(Box) [with remotecontrol in inventory]
    • NavigateTo(KeyChain)
    • PickupObject(KeyChain)
    • NavigateTo(Box) [with KeyChain in inventory]
    • PutObject(Box) [with KeyChain in inventory]
    • NavigateTo(Watch)
    • PickupObject(Watch)
    • NavigateTo(Box) [with Watch in inventory]
    • PutObject(Box) [with Watch in inventory]
    Coverage:
        • Watch
        • RemoteControl
        • KeyChain
        • Box
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.task_config_checker import TaskConfigChecker

def build_checker(env=None):
    receptacle = "Box_1"

    required = ["Watch", "RemoteControl", "KeyChain"]

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
#             "NavigateTo(RemoteControl)",
#             "PickupObject(RemoteControl)",
#             "NavigateTo(Box, RemoteControl)",
#             "PutObject(Box, RemoteControl)",
#             "NavigateTo(KeyChain)",
#             "PickupObject(KeyChain)",
#             "NavigateTo(Box, KeyChain)",
#             "PutObject(Box, KeyChain)",
#             "NavigateTo(Watch)",
#             "PickupObject(Watch)",
#             "NavigateTo(Box, Watch)",
#             "PutObject(Box, Watch)",
#         ]
#         conditional_subtasks = [
#             "NavigateTo(Box, RemoteControl)",
#             "PutObject(Box, RemoteControl)",
#             "NavigateTo(Box, KeyChain)",
#             "PutObject(Box, KeyChain)",
#             "NavigateTo(Box, Watch)",
#             "PutObject(Box, Watch)",
#         ]
#         independent_subtasks = [
#             "NavigateTo(RemoteControl)",
#             "PickupObject(RemoteControl)",
#             "NavigateTo(KeyChain)",
#             "PickupObject(KeyChain)",
#             "NavigateTo(Watch)",
#             "PickupObject(Watch)",
#         ]
#         coverage = ["RemoteControl", "Box", "KeyChain", "Watch"]
#         interact_objects = ["RemoteControl", "KeyChain", "Watch"]
#         interact_receptacles = ["Box"]

#         super().__init__(
#             subtasks,
#             conditional_subtasks,
#             independent_subtasks,
#             coverage,
#             interact_objects,
#             interact_receptacles,
#         )
