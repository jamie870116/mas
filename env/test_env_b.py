# test_env_b_script.py
# from env_b import AI2ThorEnv
from env_cen import AI2ThorEnv_cen as AI2ThorEnv
import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import save_to_video

def run_plan(env, plan):
    # plan: list per-agent dicts with subtask/actions/steps
    ok, info = env.stepwise_action_loop(plan)
    print("OK:", ok)
    print("INFO:", info)
    return ok, info

def print_fail(info):
    print("\n=== Fail Snapshot ===")
    print("failure_subtasks:", info["failure_subtasks"])
    print("failed_acts:", info["failed_acts"])
    print("subtask_failure_reasons:", info["subtask_failure_reasons"])

def main():
    env = AI2ThorEnv("config/config.json")
    env.reset()

    # Case A: 物件不存在（觸發 try/except → 例外錯誤/failed-at）
    # 注意：此名稱必定轉不出 objectId，應該會進到你的 _record_subtask_failure
    plan_A = [
        {
            "subtask": "pick up the GhostApple and put it on the CounterTop",
            "actions": ["NavigateTo(GhostApple_1)", "PickupObject(GhostApple_1)"],
            "steps": [["NavigateTo(GhostApple_1)"], ["PickupObject(GhostApple_1)"]] # not used
        },
        {
            "subtask": "Idle",
            "actions": ["Idle"],
            "steps": [["Idle"]]
        }
    ]
    okA, infoA = run_plan(env, plan_A)
    llm_input = env.get_obs_llm_input(prev_info=infoA)
    print_fail(infoA)
    print("LLM input for Case A:\n", llm_input)

    # Case B: 導航規劃為空（no-path）
    # 將某個可見物件移到極端位置，或使用你已知會產生無路徑的目標（如被障礙封死）
    # 下面示例：把 Tomato_1 挪到不可達座標（這個座標是否不可達依據場景，必要時自己調整）
    # 先找場景中確實有的 Tomato，若無則改別種常見物件。
    # objs = env.get_all_objects()
    # target_name = next((o for o in objs if o.startswith("Tomato_")), None)
    # if target_name:
    #     success, msg = env.simulate_environment_event(
    #         "move", target_name, {"x": 50.0, "y": 0.9, "z": 50.0}
    #     )
    #     print("Move for no-path simulation:", success, msg)

    #     plan_B = [
    #         {
    #             "subtask": f"navigate to {target_name}",
    #             "actions": [f"NavigateTo({target_name})"],
    #             "steps": [[f"NavigateTo({target_name})"]]
    #         },
    #         {
    #             "subtask": "Idle",
    #             "actions": ["Idle"],
    #             "steps": [["Idle"]]
    #         }
    #     ]
    #     okB, infoB = run_plan(env, plan_B)
    #     print_fail(infoB)

    # Case C: object-not-in-view / distance-too-far
    # 直接下達「PutObject(CounterTop_1)」而不先拾取，或故意在距離>1m時檢查 NavigateTo 判定失敗。
    # 這裡用 NavigateTo 近但不夠近（必要時調整 agent 初始位置，或先 MoveAhead 一兩步再測）
    # 若場景有 Fridge_1，就試著 NavigateTo 它一次，觀察 is_subtask_done 的分支理由
    # target2 = next((o for o in env.get_all_objects() if o.startswith("Fridge_")), None)
    # if target2:
    #     plan_C = [
    #         {
    #             "subtask": f"navigate to {target2} (close but not centered)",
    #             "actions": [f"NavigateTo({target2})"],
    #             "steps": [[f"NavigateTo({target2})"]]
    #         },
    #         {"subtask": "Idle","actions": ["Idle"],"steps": [["Idle"]]}
    #     ]
    #     okC, infoC = run_plan(env, plan_C)
    #     print_fail(infoC)

    env.close()

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
    # main()
    '''
    - test all the actions
        -self.move_actions = ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]
        -self.rotate_actions = ["RotateRight", "RotateLeft"]
        -self.look_actions = ["LookUp", "LookDown"]
        -self.idle_actions = ["Done", "Idle"]
        -self.object_interaction_actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "BreakObject", "CookObject", "SliceObject", "DirtyObject", "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject"]
        -self.object_interaction_without_navigation  = ["DropHandObject", "ThrowObject"]

    - obj in floorplan1
    ['Apple_1', 'Book_1', 'Bottle_1', 'Bowl_1', 'Bread_1', 'ButterKnife_1', 'Cabinet_1', 'Cabinet_2', 'Cabinet_3', 'Cabinet_4', 'Cabinet_5', 'Cabinet_6', 'Cabinet_7', 'Cabinet_8', 'Cabinet_9', 
    'CoffeeMachine_1', 'CounterTop_1', 'CounterTop_2', 'CounterTop_3', 'CreditCard_1', 'Cup_1', 'DishSponge_1', 'Drawer_1', 'Drawer_2', 'Drawer_3', 'Drawer_4', 'Drawer_5', 'Drawer_6', 'Drawer_7', 'Drawer_8', 'Drawer_9', 
    'Egg_1', 'Faucet_1', 'Floor_1', 'Fork_1', 'Fridge_1', 'GarbageCan_1', 'HousePlant_1', 'Kettle_1', 'Knife_1', 'Lettuce_1', 'LightSwitch_1', 'Microwave_1', 'Mug_1', 'Pan_1', 'PaperTowelRoll_1', 'PepperShaker_1', 'Plate_1', 
    'Pot_1', 'Potato_1', 'SaltShaker_1', 'Shelf_1', 'Shelf_2', 'Shelf_3', 'ShelvingUnit_1', 'Sink_1', 'Sink_2', 'SoapBottle_1', 'Spatula_1', 'Spoon_1', 'Statue_1', 'Stool_1', 'Stool_2', 'StoveBurner_1', 'StoveBurner_2', 'StoveBurner_3', 
    'StoveBurner_4', 'StoveKnob_1', 'StoveKnob_2', 'StoveKnob_3', 'StoveKnob_4', 'Toaster_1', 'Tomato_1', 'Vase_1', 'Vase_2', 'Window_2', 'WineBottle_1']
    
    may need the mapping of Apple_1 -> position and id

    '''

    env = AI2ThorEnv("config/config.json")
    with open("config/config.json", "r") as f:
            config = json.load(f)
            task_name = config["task"]
    

    obs = env.reset(test_case_id="18")
    # input_llm = env.get_obs_llm_input()
    # print("LLM input:\n", input_llm)
    # reach_pos = env.get_cur_reachable_positions_2d()
    # print("Initial reachable positions:", reach_pos)
    # # list of [x,y]
    # # obs = env.get_observations()
    
    get_object_dict = env.get_object_dict()
    print("Object dictionary:", get_object_dict)
    # for i in range(2):
    #     state = env.get_agent_state(i)
    #     view = env.get_object_in_view(i)
    #     mapping = env.get_mapping_object_pos_in_view(i)
    #     print(f"Agent {i} ({env.agent_names[i]}) observation: I see:", mapping) # list of object ids in view
    #     print(f"Agent {i} ({env.agent_names[i]}) state: {state}") # I am at coordinates: (2.00, -1.50), facing west, holding nothing
    #     print(f"Agent {i} ({env.agent_names[i]}) can see object:", view) # list of object ids in view
    # objs = env.get_all_objects()
    # # print(objs)
    # high_level_tasks = [
    #     #  ["RotateRight"], ["Idle"]
    #     ['NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'NavigateTo(Bread_1)', 'PickupObject(Bread_1)', 'NavigateTo(Fridge_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'],  ['Idle']
    # ]
    # high_level_tasks = [
    #     ['Idle']
    #     # ["NavigateTo(Bread_1)", "PickupObject(Bread_1)", "NavigateTo(CounterTop_1)", "PutObject(CounterTop_1)"]
    #     #  ['NavigateTo(ButterKnife_1)', 'PickupObject(ButterKnife_1)', 'NavigateTo(Lettuce_1)', 'SliceObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)', 'NavigateTo(Lettuce_2)', 'PickupObject(Lettuce_2)', 'NavigateTo(Pan_1)', 'PutObject(Pan_1)', 'PickupObject(Pan_1)', 'NavigateTo(StoveBurner_1)', 'PutObject(StoveBurner_1)','NavigateTo(StoveKnob_1)', 'ToggleObjectOn(StoveKnob_1)','NavigateTo(StoveKnob_1)','ToggleObjectOff(StoveKnob_1)'], ['Idle']
    # ]
    
    # run_test(
    #     env,
    #     high_level_tasks=high_level_tasks,
    #     test_name="Test 10",
    #     test_id=10,
    #     task_name = task_name,
    # )
    # input_llm = env.get_obs_llm_input()
    # print("LLM input:\n", input_llm)
    # reach_pos = env.get_cur_reachable_positions_2d()
    # print("Initial reachable positions:", reach_pos)
    # # list of [x,y]
    # # obs = env.get_observations()
    
    # print("Initial Observations:", obs)
    # for i in range(2):
    #     state = env.get_agent_state(i)
    #     view = env.get_object_in_view(i)
    #     mapping = env.get_mapping_object_pos_in_view(i)
    #     print(f"Agent {i} ({env.agent_names[i]}) observation: I see:", mapping) # list of object ids in view
    #     print(f"Agent {i} ({env.agent_names[i]}) state: {state}") # I am at coordinates: (2.00, -1.50), facing west, holding nothing
    #     print(f"Agent {i} ({env.agent_names[i]}) can see object:", view) # list of object ids in view
    # objs = env.get_all_objects()
    env.close()   
    # objs = env.get_readable_object_list(env.get_object_in_view(0))
    # print(objs)
    # high_level_tasks = [
    #     ['NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'NavigateTo(Mug_1)','PickupObject(Mug_1)', 'NavigateTo(Fridge_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'], ['Idle']
    # ]
    
    # run_test(
    #     env,
    #     high_level_tasks=high_level_tasks,
    #     test_name="Test 2",
    #     test_id=2,
    #     task_name = task_name,
    # )
    # before_reachable_position = [(-0.75, -1.25), (-0.75, -1.5), (-0.75, -1.75), (0.75, -1.25), (0.75, -1.5), (0.75, -1.75), (2.0, 2.0), (1.5, 2.0), (-1.0, -1.0), (-0.75, 1.25), (-0.75, 1.5), (-0.75, 1.75), (0.75, 1.25), (0.75, 1.5), (0.75, 1.75), (-1.25, -1.0), (-1.0, 0.5), (-1.0, 0.75), (-1.0, 0.25), (-1.0, 0.0), (-1.0, -0.25), (-1.0, -0.5), (-1.0, -0.75), (-1.25, 0.75), (-1.25, 0.5), (-1.25, 0.25), (-1.25, 0.0), (-1.25, -0.25), (-1.25, -0.5), (-1.25, -0.75), (0.5, -1.25), (0.5, -1.5), (0.5, -1.75), (1.25, -1.25), (1.25, -1.5), (1.25, -1.75), (-1.0, 2.0), (1.75, -1.0), (1.75, -2.0), (0.0, 2.0), (0.0, 2.25), (-1.25, 2.0), (0.5, 1.25), (0.5, 1.5), (0.5, 1.75), (1.5, -1.25), (1.5, -1.5), (1.5, -1.75), (1.75, 0.75), (1.75, 0.5), (1.75, 0.25), (1.75, 0.0), (1.75, -0.25), (1.75, -0.75), (1.75, -0.5), (0.25, 2.0), (-0.5, -1.25), (-0.5, -1.5), (-0.5, -1.75), (1.5, 1.25), (1.5, 1.5), (1.5, 1.0), (1.5, 1.75), (1.75, 2.0), (-0.25, -1.25), (-0.25, -1.5), (-0.25, -1.75), (-0.5, 1.25), (-0.5, 1.5), (-0.5, 1.75), (1.0, -1.0), (-1.0, -1.5), (-1.0, -1.75), (-1.0, -1.25), (-1.5, 2.0), (-0.75, 2.0), (-0.25, 1.25), (-0.25, 1.5), (-0.25, 1.75), (2.0, -0.75), (2.0, -0.5), (1.0, 0.0), (1.0, -0.25), (1.25, 1.25), (1.25, 1.5), (1.25, 1.0), (1.25, 1.75), (0.25, -1.25), (0.25, -1.5), (0.25, -1.75), (1.0, 2.0), (2.0, 1.25), (2.0, 1.5), (2.0, 1.0), (2.0, 1.75), (0.5, 2.0), (1.75, 1.25), (1.75, 1.5), (1.75, 1.0), (1.75, 1.75), (0.75, 2.0), (0.0, -1.25), (0.0, -1.5), (0.0, -1.75), (-1.25, -1.25), (-1.25, -1.5), (-1.25, -1.75), (-1.5, 1.75), (1.25, -1.0), (-1.0, 1.0), (-1.0, 1.25), (-1.0, 1.5), (-1.0, 1.75), (-0.5, 2.0), (1.0, -1.25), (0.0, 1.25), (0.0, 1.5), (0.0, 1.75), (1.0, -1.5), (1.0, -1.75), (2.0, -1.0), (2.0, -2.0), (-1.25, 1.0), (-1.25, 1.25), (-1.25, 1.5), (-1.25, 1.75), (1.5, -1.0), (1.5, -2.0), (1.25, 0.75), (1.25, 0.5), (1.25, 0.25), (1.25, 0.0), (1.25, -0.25), (1.25, -0.75), (1.25, -0.5), (-0.25, 2.0), (-0.25, 2.25), (0.25, 1.25), (0.25, 1.5), (0.25, 1.75), (1.0, 1.25), (1.0, 1.5), (1.0, 1.75), (2.0, 0.75), (2.0, 0.5), (2.0, 0.25), (2.0, 0.0), (2.0, -0.25), (1.5, 0.75), (1.5, 0.5), (1.5, 0.25), (1.5, 0.0), (1.5, -0.25), (1.5, -0.75), (1.5, -0.5), (1.25, 2.0)]
    # #  after opening fridge
    # final_reachable_position = [(-0.75, -1.25), (-0.75, -1.5), (-0.75, -1.75), (0.75, -1.25), (0.75, -1.5), (0.75, -1.75), (2.0, 2.0), (1.5, 2.0), (-1.0, -1.0), (-0.75, 1.75), (0.75, 1.25), (0.75, 1.5), (0.75, 1.75), (-1.25, -1.0), (-1.0, -0.75), (-1.0, -0.5), (-1.0, -0.25), (-1.0, 0.0), (-1.25, -0.75), (0.5, -1.25), (0.5, -1.5), (0.5, -1.75), (-1.25, -0.5), (-1.25, -0.25), (-1.25, 0.0), (1.25, -1.25), (1.25, -1.5), (1.25, -1.75), (-1.0, 2.0), (1.75, -1.0), (1.75, -2.0), (0.0, 2.0), (0.0, 2.25), (-1.25, 2.0), (0.5, 1.25), (0.5, 1.5), (0.5, 1.75), (1.5, -1.25), (1.5, -1.5), (1.5, -1.75), (1.75, 0.75), (1.75, 0.5), (1.75, 0.25), (1.75, 0.0), (1.75, -0.25), (1.75, -0.5), (1.75, -0.75), (0.25, 2.0), (-0.5, -1.25), (-0.5, -1.5), (-0.5, -1.75), (1.5, 1.25), (1.5, 1.5), (1.5, 1.0), (1.5, 1.75), (1.75, 2.0), (-0.25, -1.25), (-0.25, -1.5), (-0.25, -1.75), (-0.5, 1.25), (-0.5, 1.5), (-0.5, 1.75), (1.0, -1.0), (-1.0, -1.25), (-1.0, -1.5), (-1.0, -1.75), (-1.5, 2.0), (-0.75, 2.0), (-0.25, 1.25), (-0.25, 1.5), (-0.25, 1.75), (2.0, -0.5), (2.0, -0.75), (1.0, 0.0), (1.0, -0.25), (1.25, 1.25), (1.25, 1.5), (1.25, 1.0), (1.25, 1.75), (0.25, -1.25), (0.25, -1.5), (0.25, -1.75), (1.0, 2.0), (2.0, 1.0), (2.0, 1.75), (2.0, 1.25), (2.0, 1.5), (0.5, 2.0), (1.75, 1.5), (1.75, 1.0), (1.75, 1.75), (1.75, 1.25), (0.75, 2.0), (0.0, -1.25), (0.0, -1.5), (0.0, -1.75), (-1.25, -1.25), (-1.25, -1.5), (-1.25, -1.75), (-1.5, 1.75), (1.25, -1.0), (-1.0, 1.0), (-1.0, 1.75), (-0.5, 2.0), (1.0, -1.25), (0.0, 1.25), (0.0, 1.5), (0.0, 1.75), (2.0, -1.0), (1.0, -1.5), (1.0, -1.75), (2.0, -2.0), (-1.25, 1.0), (-1.25, 1.75), (1.5, -1.0), (1.5, -2.0), (1.25, 0.75), (1.25, 0.5), (1.25, 0.25), (1.25, 0.0), (1.25, -0.25), (1.25, -0.5), (1.25, -0.75), (-0.25, 2.0), (-0.25, 2.25), (0.25, 1.25), (0.25, 1.5), (0.25, 1.75), (1.0, 1.25), (1.0, 1.5), (1.0, 1.75), (2.0, 0.75), (2.0, 0.5), (2.0, 0.25), (2.0, 0.0), (2.0, -0.25), (1.5, 0.75), (1.5, 0.5), (1.5, 0.25), (1.5, 0.0), (1.5, -0.25), (1.5, -0.5), (1.5, -0.75), (1.25, 2.0)]


    # import matplotlib.pyplot as plt

    # # 提取 x, z 座標
    # before_x = [x for x, z in before_reachable_position]
    # before_z = [z for x, z in before_reachable_position]

    # after_x = [x for x, z in final_reachable_position]
    # after_z = [z for x, z in final_reachable_position]

    # # 建立圖表與子圖
    # fig, axs = plt.subplots(1, 2, figsize=(16, 7))  # 一列兩欄

    # # 畫 before 圖
    # axs[0].scatter(before_x, before_z, c='blue', s=50, alpha=0.7, label='Before')
    # axs[0].set_title("Before Opening Fridge")
    # axs[0].set_xlabel("X")
    # axs[0].set_ylabel("Z")
    # axs[0].grid(True)
    # axs[0].axis('equal')
    # axs[0].legend()

    # # 畫 after 圖
    # axs[1].scatter(after_x, after_z, c='green', s=50, alpha=0.7, label='After')
    # axs[1].set_title("After Opening Fridge")
    # axs[1].set_xlabel("X")
    # axs[1].set_ylabel("Z")
    # axs[1].grid(True)
    # axs[1].axis('equal')
    # axs[1].legend()

    # # 顯示圖表
    # plt.tight_layout()
    # plt.show()


'''
新增的 reachable positions:
(-1.5, 0.900999903678894, 0.75)
(-1.5, 0.900999903678894, 1.5)
- x :-1.5,-1.25,-1
Before openning Object Fridge_1 status: {'object_id': 'Fridge|-02.10|+00.00|+01.07', 'name': 'Fridge_e92350c6', 'position': {'x': -2.0969998836517334, 'y': 0.0, 'z': 1.0720000267028809}, 'rotation': {'x': -0.0, 'y': 89.9999771118164, 'z': -0.0}, 'is_open': False, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': ['Egg|-02.04|+00.81|+01.24']}
Executing action for agent 0 (Alice): {'agentId': 0, 'action': 'OpenObject', 'objectId': 'Fridge|-02.10|+00.00|+01.07'}
After openning Object Fridge_1 status: {'object_id': 'Fridge|-02.10|+00.00|+01.07', 'name': 'Fridge_e92350c6', 'position': {'x': -2.0969998836517334, 'y': 0.0, 'z': 1.0720000267028809}, 'rotation': {'x': -0.0, 'y': 89.9999771118164, 'z': -0.0}, 'is_open': True, 'is_on': False, 'is_picked_up': False, 'isSliced': False, 'isToggled': False, 'isBroken': False, 'isFilledWithLiquid': False, 'contains': ['Egg|-02.04|+00.81|+01.24']}
'''