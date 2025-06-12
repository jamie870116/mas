import json
from new_env import AI2ThorEnv

def run_test(env, actions, test_name, test_id):
    """Run a test case with multiple steps and print results."""
    print(f"\n=== {test_name} (Test ID: {test_id}) ===")
    obs = env.reset(test_case_id=test_id)
    print("Initial Observations:\n", obs)
    
    for i, action_step in enumerate(actions):
        print(f"\nStep {i + 1} Actions: {action_step}")
        obs, successes = env.step(action_step)
        print("Step Observations:\n", obs)
        print("Action Successes:", successes)

    

    for agent_id, name in enumerate(env.agent_names):
        pov_path = env.get_frame(agent_id, "pov")
        print(f"{name} POV: {pov_path}")
    overhead_path = env.get_frame(view="overhead")
    print(f"Shared Overhead: {overhead_path}")

if __name__ == "__main__":
    config_path = "config/config.json"
    env = AI2ThorEnv(config_path)
    
    # Test 1: Movement Actions (Multi-step)
    run_test(env, [["MoveAhead", "MoveBack"], ["MoveRight", "MoveLeft"]], 
             "Test 1: Movement Actions (Alice: MoveAhead + MoveBack, Bob: MoveRight + MoveLeft)", "1")
    
    # Test 2: Rotation Actions
    run_test(env, [["RotateRight", "RotateRight"], ["RotateLeft", "RotateLeft"]], 
             "Test 2: Rotation Actions (Alice: RotateRight + RotateRight, Bob: RotateLeft + RotateLeft)", "2")
    
    # Test 3: Look Actions
    run_test(env, [["LookUp", "LookDown"], ["LookDown", "LookUp"]], 
             "Test 3: Look Actions (Alice: LookUp + LookDown, Bob: LookDown + LookUp)", "3")
    
    # Test 4: Object Interaction (Pickup + Move + Put)
    obs = env.reset(test_case_id="4")
    objects_in_view_alice = env.get_readable_object_list(env.get_object_in_view(0))
    tomato = next((obj for obj in objects_in_view_alice if "Tomato" in obj), "Tomato_1")
    counter = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "CounterTop" in obj), "CounterTop_1")
    run_test(env, [[f"PickupObject({tomato})", "Idle"], ["MoveAhead", "Idle"], [f"PutObject({counter})", "Idle"]], 
             f"Test 4: Object Interaction (Alice: PickupObject({tomato}) + MoveAhead + PutObject({counter}), Bob: Idle)", "4")
    
    # Test 5: DropHandObject
    obs = env.reset(test_case_id="5")
    run_test(env, [[f"PickupObject({tomato})", "Idle"], ["DropHandObject", "Idle"]], 
             f"Test 5: DropHandObject (Alice: PickupObject({tomato}) + DropHandObject, Bob: Idle)", "5")
    
    # Test 6: OpenObject and CloseObject
    obs = env.reset(test_case_id="6")
    cabinet = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "Cabinet" in obj), "Cabinet_1")
    run_test(env, [[f"OpenObject({cabinet})", "Idle"], [f"CloseObject({cabinet})", "Idle"]], 
             f"Test 6: OpenObject and CloseObject (Alice: OpenObject({cabinet}) + CloseObject({cabinet}), Bob: Idle)", "6")
    
    # Test 10: Both Agents Performing Multi-Actions
    run_test(env, [["MoveAhead", "RotateRight"], ["MoveRight", "RotateLeft"]], 
             "Test 10: Both Agents (Alice: MoveAhead + RotateRight, Bob: MoveRight + RotateLeft)", "10")

    # Test 11: Both Agents Performing Multi-Actions (Open + Pickup, Close + Drop)
    obs = env.reset(test_case_id="11")
    objects_in_view_alice = env.get_readable_object_list(env.get_object_in_view(0))
    tomato = next((obj for obj in objects_in_view_alice if "Tomato" in obj), "Tomato_1")
    counter = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "CounterTop" in obj), "CounterTop_1")
    cabinet = next((obj for obj in env.get_readable_object_list(env.get_object_in_view(0)) if "Cabinet" in obj), "Cabinet_1")
    combined_actions = [
        [f"OpenObject({cabinet})", f"PickupObject({tomato})"],
        [f"CloseObject({cabinet})", "DropHandObject"]
    ]
    run_test(
        env,
        combined_actions,
        f"Test 11: Alice (OpenObject({cabinet}) + CloseObject({cabinet})), Bob (PickupObject({tomato}) + DropHandObject)",
        "11"
    )

    env.close()
