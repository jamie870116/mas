# utils/scene_loader.py
import importlib.util
import os

def import_scene_initializer(task: str, floor_plan: str):
    file_path = os.path.join("Tasks", task, f"{floor_plan}.py")
    if not os.path.exists(file_path):
        return None

    spec = importlib.util.spec_from_file_location("SceneInitializer", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "SceneInitializer", None)