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

    print("high_level_tasks: ", high_level_tasks)
    history = env.action_loop(high_level_tasks)
    for step_idx, (obs, succ) in enumerate(history, start=1):
        print(f"\n--- Step {step_idx} ---")
        # print("Observations:", obs)
        # print("Success flags:", succ)

    for agent_id, name in enumerate(env.agent_names):
        print(f"{name} POV path:", env.get_frame(agent_id, "pov"))
    if env.overhead:
        print("Shared overhead path:", env.get_frame(view="overhead"))

    if task_name:
        print('logs/' + task_name.replace(" ", "_") + f"/test_{test_id}")
        save_to_video('logs/' + task_name.replace(" ", "_") + f"/test_{test_id}")
        

if __name__ == "__main__":
    env = AI2ThorEnv("config/config.json")
    with open("config/config.json", "r") as f:
            config = json.load(f)
            task_name = config["task"]
    
    # obs = env.reset(test_case_id="0")
    # objs = env.get_readable_object_list(env.get_object_in_view(0))
    # run_test(
    #     env,
    #     high_level_tasks=[['NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)'], ['Idle']], # [[subtasks for agent_i], [...]]
    #     test_name="Test 0",
    #     test_id="0",
    #     task_name=task_name
    # )

    # obs = env.reset(test_case_id="17")
    # objs = env.get_readable_object_list(env.get_object_in_view(0))
    # tomato = next((o for o in objs if "Tomato" in o), "Tomato_1")
    # counter = next((o for o in objs if "CounterTop" in o), "CounterTop_1")
    # run_test(
    #     env,
    #     high_level_tasks=[[f"PickupObject({tomato})", f"PutObject({counter})"], ["Idle"]], # [[subtasks for agent_i], [...]]
    #     test_name="Test 17",
    #     test_id="17",
    #     task_name=task_name
    # )

    # obs = env.reset(test_case_id="18")
    # objs = env.get_readable_object_list(env.get_object_in_view(0))
    # # tomato = next((o for o in objs if "Tomato" in o), "Tomato_1")
    # # counter = next((o for o in objs if "CounterTop" in o), "CounterTop_1")
    # bread = next((o for o in objs if "Bread" in o), "Bread_1")
    # run_test(
    #     env,
    #     high_level_tasks=[["Idle"], [f"PickupObject({bread})"]], # [[subtasks for agent_i], [...]]
    #     test_name="Test 18",
    #     test_id="18",
    #     task_name=task_name
    # )

    # obs = env.reset(test_case_id="11")
    # objects_in_view_alice = env.get_readable_object_list(env.get_object_in_view(0))
    # tomato = next((obj for obj in objects_in_view_alice if "Tomato" in obj), "Tomato_1")
    # counter = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "CounterTop" in obj), "CounterTop_1")
    # cabinet = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "Cabinet" in obj), "Cabinet_1")
    # high_level_tasks = [
    #     [f"PickupObject({tomato})"],
    #     [f"OpenObject({cabinet})", f"CloseObject({cabinet})", "Idle"]
    # ]
    # run_test(
    #     env,
    #     high_level_tasks=high_level_tasks,
    #     test_name="Test 11",
    #     test_id=11,
    #     task_name = task_name,
    # )

    obs = env.reset(test_case_id="6")
    # objects_in_view_alice = env.get_readable_object_list(env.get_object_in_view(0))
    
    high_level_tasks = [
        ['NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'NavigateTo(Tomato_1)', 'PickupObject(Tomato_1)', 'NavigateTo(Fridge_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'], ['Idle']
    ]
    
    run_test(
        env,
        high_level_tasks=high_level_tasks,
        test_name="Test 6",
        test_id=6,
        task_name = task_name,
    )
