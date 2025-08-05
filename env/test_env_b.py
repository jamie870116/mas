# test_env_b_script.py
# from env_b import AI2ThorEnv
from env_cen import AI2ThorEnv_cen as AI2ThorEnv
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
    
    '''

    env = AI2ThorEnv("config/config.json")
    with open("config/config.json", "r") as f:
            config = json.load(f)
            task_name = config["task"]
    

    obs = env.reset(test_case_id="17")
    # objs = env.get_all_objects()
    # # print(objs)
    high_level_tasks = [
        #  ["RotateRight"], ["Idle"]
        ['NavigateTo(Fridge_1)', 'OpenObject(Fridge_1)', 'NavigateTo(Bread_1)', 'PickupObject(Bread_1)', 'NavigateTo(Fridge_1)', 'PutObject(Fridge_1)', 'CloseObject(Fridge_1)'],  ['Idle']
    ]
    # high_level_tasks = [
    #     ['NavigateTo(Pan_1)', 'PickupObject(Pan_1)', 'NavigateTo(StoveBurner_1)', 'PutObject(StoveBurner_1)','NavigateTo(StoveKnob_1)', 'ToggleObjectOn(StoveKnob_1)','NavigateTo(StoveKnob_1)','ToggleObjectOff(StoveKnob_1)'], ['Idle']

    #     #  ['NavigateTo(ButterKnife_1)', 'PickupObject(ButterKnife_1)', 'NavigateTo(Lettuce_1)', 'SliceObject(Lettuce_1)', 'NavigateTo(CounterTop_1)', 'PutObject(CounterTop_1)', 'NavigateTo(Lettuce_2)', 'PickupObject(Lettuce_2)', 'NavigateTo(Pan_1)', 'PutObject(Pan_1)', 'PickupObject(Pan_1)', 'NavigateTo(StoveBurner_1)', 'PutObject(StoveBurner_1)','NavigateTo(StoveKnob_1)', 'ToggleObjectOn(StoveKnob_1)','NavigateTo(StoveKnob_1)','ToggleObjectOff(StoveKnob_1)'], ['Idle']
    # ]
    
    run_test(
        env,
        high_level_tasks=high_level_tasks,
        test_name="Test 17",
        test_id=17,
        task_name = task_name,
    )

    # env.close()
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