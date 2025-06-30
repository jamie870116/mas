# test_env_b_script.py

from env_b import AI2ThorEnv
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import save_to_video

def run_test(env, high_level_tasks, test_name, test_id, task_name=None):
    """
    Run a test case with multiple steps and print results.
      - high_level_tasks: List[List[str]]，每個 agent 的 high-level 
      - test_name: 
      - test_id:  測試識別字串，會傳給 reset()
    """
    print(f"\n=== {test_name} (Test ID: {test_id}) ===")
    obs = env.reset(test_case_id=test_id)
    # print("Initial Observations:\n", obs)

    # 用 action_loop 一次跑完所有 high-level tasks
    print("high_level_tasks: ", high_level_tasks)
    history = env.action_loop(high_level_tasks)
    for step_idx, (obs, succ) in enumerate(history, start=1):
        print(f"\n--- Step {step_idx} ---")
        # print("Observations:", obs)
        # print("Success flags:", succ)

    # 最後顯示各 agent 最新影格路徑
    for agent_id, name in enumerate(env.agent_names):
        print(f"{name} POV path:", env.get_frame(agent_id, "pov"))
    if env.overhead:
        print("Shared overhead path:", env.get_frame(view="overhead"))
    if task_name:
        print('logs/' + task_name.replace(" ", "_") + f"/test_{test_id}")
        save_to_video('logs/' + task_name.replace(" ", "_") + f"/test_{test_id}")
        

if __name__ == "__main__":
    # 初始化環境
    env = AI2ThorEnv("config/config.json")
    with open("config/config.json", "r") as f:
            config = json.load(f)
            task_name = config["task"]
    # # Test 1: Movement Actions
    # run_test(
    #     env,
    #     high_level_tasks=[["MoveAhead", "MoveBack"], ["MoveRight", "MoveLeft"]],
    #     test_name="Test 1: Movement (Alice / Bob)",
    #     test_id="1"
    # )

    # # Test 2: Rotation Actions
    # run_test(
    #     env,
    #     high_level_tasks=[["RotateRight", "RotateRight"], ["RotateLeft", "RotateLeft"]],
    #     test_name="Test 2: Rotation (Alice / Bob)",
    #     test_id="2"
    # )

    # # Test 3: Look Actions
    # run_test(
    #     env,
    #     high_level_tasks=[["LookUp", "LookDown"], ["LookDown", "LookUp"]],
    #     test_name="Test 3: Look (Alice / Bob)",
    #     test_id="3"
    # )

    # # Test 14: Object Interaction (Pickup -> Put)
    # obs = env.reset(test_case_id="14")
    # # 擷取當前可見的 Tomato / CounterTop
    # objs = env.get_readable_object_list(env.get_object_in_view(0))
    # tomato = next((o for o in objs if "Tomato" in o), "Tomato_1")
    # counter = next((o for o in objs if "CounterTop" in o), "CounterTop_1")

    # run_test(
    #     env,
    #     high_level_tasks=[[f"PickupObject({tomato})", f"PutObject({counter})"], ["Idle"]], # [[subtasks for agent_i], [...]]
    #     test_name=f"Test 4: Object Interaction ({tomato} → {counter})",
    #     test_id="14"
    # )

    # obs = env.reset(test_case_id="15")
    # objs = env.get_readable_object_list(env.get_object_in_view(0))
    # tomato = next((o for o in objs if "Tomato" in o), "Tomato_1")
    # counter = next((o for o in objs if "CounterTop" in o), "CounterTop_1")
    # bread = next((o for o in objs if "Bread" in o), "Bread_1")
    # run_test(
    #     env,
    #     high_level_tasks=[[f"PickupObject({tomato})", f"PutObject({counter})"], [f"PickupObject({bread})"]], # [[subtasks for agent_i], [...]]
    #     test_name="Test 15",
    #     test_id="15"
    # )

    obs = env.reset(test_case_id="16")
    objs = env.get_readable_object_list(env.get_object_in_view(0))
    tomato = next((o for o in objs if "Tomato" in o), "Tomato_1")
    counter = next((o for o in objs if "CounterTop" in o), "CounterTop_1")
    run_test(
        env,
        high_level_tasks=[[f"PickupObject({tomato})", f"PutObject({counter})"], ["Idle"]], # [[subtasks for agent_i], [...]]
        test_name="Test 16",
        test_id="16",
        task_name=task_name
    )
