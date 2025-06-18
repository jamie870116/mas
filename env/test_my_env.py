import json
from env_b import AI2ThorEnv
import time
from typing import List

# test env_b.py
def run_test_new(
    env: AI2ThorEnv,
    composite_actions: List[List[str]],
    test_name: str,
    test_id: str,
    timeout_seconds: float = 60.0
):
    """
    Run a test case with timeout.

    composite_actions: 每个 agent 的复合动作列表
    timeout_seconds: 最长运行秒数，超时后自动退出
    """
    print(f"\n=== {test_name} (Test ID: {test_id}) ===")
    obs = env.reset(test_case_id=test_id)

    num_agents = env.num_agents
    done = [False] * num_agents
    start_time = time.time()

    step_count = 0
    while not all(done):
        # 超时检测
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"⚠️ Timeout reached ({elapsed:.1f}s), aborting test.")
            break

        # 构造本次 step 的动作
        actions = []
        for agent_id in range(num_agents):
            if not done[agent_id] and composite_actions[agent_id]:
                actions.append(composite_actions[agent_id][0])
            else:
                actions.append("Idle")

        # 执行一步
        step_count += 1
        print(f"\n--> Step {step_count}, elapsed {elapsed:.1f}s, actions: {actions}")
        obs, successes = env.step(actions)
        print("Observations:\n", obs)
        print("Success flags:", successes)

        # 更新完成标志：只有真正执行 PickupObject 成功才视为 done
        for agent_id, act in enumerate(actions):
            if act.startswith("PickupObject") and successes[agent_id]:
                done[agent_id] = True

    # 测试结束，输出最终帧路径
    for agent_id, name in enumerate(env.agent_names):
        print(f"{name} final POV:", env.get_frame(agent_id, "pov"))
    print("Overhead view:", env.get_frame(view="overhead"))


def run_test(env, actions, test_name, test_id):
    # test new_env.py
    """Run a test case with multiple steps and print results."""
    print(f"\n=== {test_name} (Test ID: {test_id}) ===")
    obs = env.reset(test_case_id=test_id)
    # print("Initial Observations:\n", obs)
    
    for i, action_step in enumerate(actions):
        print(f"\nStep {i + 1} Actions: {action_step}")
        obs, successes = env.step(action_step)
        print("Step Observations:\n", obs)
        print("Action Successes:", successes)

    

    for agent_id, name in enumerate(env.agent_names):
        pov_path = env.get_frame(agent_id, "pov")
        # print(f"{name} POV: {pov_path}")
    overhead_path = env.get_frame(view="overhead")
    # print(f"Shared Overhead: {overhead_path}")

if __name__ == "__main__":
    config_path = "config/config.json"
    env = AI2ThorEnv(config_path)
    # # Test env_b.py
    # obs = env.reset(test_case_id="12")
    # objects = env.get_readable_object_list(env.get_object_in_view(0))
    # tomato = next((o for o in objects if "Tomato" in o), "Tomato_1")
    # composite = [[f"PickupObject({tomato})"], []]

    # # 设定超时 30 秒
    # run_test_new(env, composite, "Test 4 with Timeout", "4", timeout_seconds=30.0)


    # Test 1: Movement Actions (Multi-step)
    # run_test(env, [["MoveAhead", "MoveBack"], ["MoveRight", "MoveLeft"]], 
    #          "Test 1: Movement Actions (Alice: MoveAhead + MoveBack, Bob: MoveRight + MoveLeft)", "1")
    
    # # Test 2: Rotation Actions
    # run_test(env, [["RotateRight", "RotateRight"], ["RotateLeft", "RotateLeft"]], 
    #          "Test 2: Rotation Actions (Alice: RotateRight + RotateRight, Bob: RotateLeft + RotateLeft)", "2")
    
    # # Test 3: Look Actions
    # run_test(env, [["LookUp", "LookDown"], ["LookDown", "LookUp"]], 
    #          "Test 3: Look Actions (Alice: LookUp + LookDown, Bob: LookDown + LookUp)", "3")
    
    # Test 4: Object Interaction (Pickup + Move + Put)
    obs = env.reset(test_case_id="4")
    objects_in_view_alice = env.get_readable_object_list(env.get_object_in_view(0))
    tomato = next((obj for obj in objects_in_view_alice if "Tomato" in obj), "Tomato_1")
    counter = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "CounterTop" in obj), "CounterTop_1")
    run_test(env, [[f"PickupObject({tomato})", "Idle"]], 
             f"Test 4: Object Interaction (Alice: PickupObject({tomato}) + MoveAhead + PutObject({counter}), Bob: Idle)", "4")
    
    # Test 5: DropHandObject
    # obs = env.reset(test_case_id="5")
    # run_test(env, [[f"PickupObject({tomato})", "Idle"], ["DropHandObject", "Idle"]], 
    #          f"Test 5: DropHandObject (Alice: PickupObject({tomato}) + DropHandObject, Bob: Idle)", "5")
    
    # # Test 6: OpenObject and CloseObject
    # obs = env.reset(test_case_id="6")
    # cabinet = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "Cabinet" in obj), "Cabinet_1")
    # run_test(env, [[f"OpenObject({cabinet})", "Idle"], [f"CloseObject({cabinet})", "Idle"]], 
    #          f"Test 6: OpenObject and CloseObject (Alice: OpenObject({cabinet}) + CloseObject({cabinet}), Bob: Idle)", "6")
    
    # # Test 10: Both Agents Performing Multi-Actions
    # run_test(env, [["MoveAhead", "RotateRight"], ["MoveRight", "RotateLeft"]], 
    #          "Test 10: Both Agents (Alice: MoveAhead + RotateRight, Bob: MoveRight + RotateLeft)", "10")

    # # Test 11: Both Agents Performing Multi-Actions (Open + Pickup, Close + Drop)
    # obs = env.reset(test_case_id="11")
    # objects_in_view_alice = env.get_readable_object_list(env.get_object_in_view(0))
    # tomato = next((obj for obj in objects_in_view_alice if "Tomato" in obj), "Tomato_1")
    # counter = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "CounterTop" in obj), "CounterTop_1")
    # cabinet = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "Cabinet" in obj), "Cabinet_1")
    # combined_actions = [
    #     [f"OpenObject({cabinet})", f"PickupObject({tomato})"],
    #     [f"CloseObject({cabinet})", "DropHandObject"]
    # ]
    # run_test(
    #     env,
    #     combined_actions,
    #     f"Test 11: Alice (OpenObject({cabinet}) + CloseObject({cabinet})), Bob (PickupObject({tomato}) + DropHandObject)",
    #     "11"
    # )

    env.close()

# from new_env.py
# if __name__ == "__main__":
#     config_path = "config/config.json"
#     env = AI2ThorEnv(config_path)
#     obs = env.reset()
#     print("Initial Observations:\n", obs)
#     print('object in view of agent 1', env.get_object_in_view(0))
#     print("All objects in scene:", env.get_all_objects())
#     print("Before breaking: Status of Mug_1:", env.get_object_status("Mug_1"))
#     print("Before breaking: Status of Lettuce_1:", env.get_object_status("Lettuce_1"))

#     # Simulate breaking an object
#     success, message = env.simulate_environment_event("break", "Mug_1")
#     print(f"Break Event: {message}")
#     print("After breaking: Status of Mug_1:", env.get_object_status("Mug_1"))
#     print("After breaking: Status of Lettuce_1:", env.get_object_status("Lettuce_1"))

#     time.sleep(10)
#     # Simulate moving an object
#     target_pos = {"x": 2.0, "y": 0.9, "z": -1.0}
#     print("Before moving: Status of Bowl_1:", env.get_object_status("Bowl_1"))
#     success, message = env.simulate_environment_event("move", "Bowl_1", target_pos)
#     print(f"Move Event: {message}")
#     print("After moving: Status of Bowl_1:", env.get_object_status("Bowl_1"))

#     actions = ["MoveAhead", "MoveAhead"]
#     obs, successes = env.step(actions)
#     print("Step Observations:\n", obs)
#     print("Action Successes:", successes)
    
#     env.close()

    '''
Before breaking: Status of Mug_1: {'object_id': 'Mug|-01.76|+00.90|-00.62', 'name': 'Mug_e7fad100', 'position': {'x': -1.7607921361923218, 'y': 0.9000000357627869, 'z': -0.6206092834472656}, 'rotation': {'x': 2.6733881895779632e-05, 'y': 0.000685526872985065, 'z': -1.5666983017581515e-05}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': []}
Before breaking: Status of Lettuce_1: {'object_id': 'Lettuce|-01.81|+00.97|-00.94', 'name': 'Lettuce_2d8f3ab9', 'position': {'x': -1.8069753646850586, 'y': 0.9737617373466492, 'z': -0.9429945945739746}, 'rotation': {'x': 359.9818115234375, 'y': 256.7711181640625, 'z': 0.001349071622826159}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': None}

Break Event: Object Mug_1 has been broken.

After breaking: Status of Mug_1: {'object_id': 'Mug|-01.76|+00.90|-00.62', 'name': 'Mug_e7fad100', 'position': {'x': -1.7607921361923218, 'y': 0.9000000357627869, 'z': -0.6206092834472656}, 'rotation': {'x': 2.6733881895779632e-05, 'y': 0.000685526872985065, 'z': -1.5666983017581515e-05}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': True, 'isFilledWithLiquid': False, 'contains': []}
After breaking: Status of Lettuce_1: {'object_id': 'Lettuce|-01.81|+00.97|-00.94', 'name': 'Lettuce_2d8f3ab9', 'position': {'x': -1.798709511756897, 'y': 0.973779559135437, 'z': -0.9976263046264648}, 'rotation': {'x': 359.9550476074219, 'y': 262.6057434082031, 'z': 46.579410552978516}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': None}

Before moving: Status of Bowl_1: {'object_id': 'Bowl|+00.27|+01.10|-00.75', 'name': 'Bowl_208f368b', 'position': {'x': 0.2731873691082001, 'y': 1.1010208129882812, 'z': -0.7532863616943359}, 'rotation': {'x': -0.002493660431355238, 'y': 0.00026333334972150624, 'z': -0.0018902190495282412}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': []}
Move Event: Object Bowl_1 has been moved to {'x': 2.0, 'y': 0.9, 'z': -1.0}.
After moving: Status of Bowl_1: {'object_id': 'Bowl|+00.27|+01.10|-00.75', 'name': 'Bowl_208f368b', 'position': {'x': 2.0, 'y': 0.0010198839008808136, 'z': -1.000000238418579}, 'rotation': {'x': 3.404549352126196e-05, 'y': 0.0002766225952655077, 'z': -0.00011428898142185062}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': []}
    
    '''