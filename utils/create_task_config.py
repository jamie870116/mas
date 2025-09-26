import json
from pathlib import Path

BASE_CONFIG = {
    "num_agents": 2,
    "scene": "",
    "task": "",
    "timeout": 30,
    "model": "gpt-4.1-2025-04-14",
    "use_obs_summariser": False,
    "use_act_summariser": False,
    "use_action_failure": True,
    "use_shared_subtask": True,
    "use_separate_subtask": False,
    "use_shared_memory": True,
    "use_separate_memory": False,
    "use_plan": True,
    "force_action": False,
    "temperature": 0.7,
    "overhead": True,
    "test_id": 1,
    "save_logs": True,
    "task_folder": ""
}


TASKS_1 = [
    {
        "task_folder": "1_put_bread_lettuce_tomato_fridge",
        "task": "put bread, lettuce, and tomato in the fridge",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "1_put_computer_book_remotecontrol_sofa",
        "task": "put labtop, book and remote control on the sofa",
        "scenes": ["FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan209", "FloorPlan224"]
    },
    {
        "task_folder": "1_put_knife_bowl_mug_countertop",
        "task": "put knife, bowl, and mug on the counter top",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "1_put_plate_mug_bowl_fridge",
        "task": "put plate, mug, and bowl in the fridge",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "1_put_remotecontrol_keys_watch_box",
        "task": "put remote control, keys, and watch in the box",
        "scenes": ["FloorPlan201", "FloorPlan202", "FloorPlan203", "FloorPlan207","FloorPlan209", "FloorPlan215", "FloorPlan226", "FloorPlan228"]
    },
    {
        "task_folder": "1_put_vase_tissuebox_remotecontrol_table",
        "task": "put vase, tissue box, and remote control on the dinning table",
        "scenes": ["FloorPlan201", "FloorPlan203", "FloorPlan216", "FloorPlan219", "FloorPlan229"]
    },
    {
        "task_folder": "1_slice_bread_lettuce_tomato_egg",
        "task": "slice bread, lettuce, tomato, and egg with knife",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "1_turn_off_faucet_light",
        "task": "turn off the sink faucet and turn off the light switch",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "1_wash_bowl_mug_pot_pan",
        "task": "clean the bowl, mug, pot, and pan",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
]

TASKS_2 = [
    {
        "task_folder": "2_open_all_cabinets",
        "task": "open all the cabinets",
        "scenes": ["FloorPlan1", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9", "FloorPlan10"]
    },
    {
        "task_folder": "2_open_all_drawers",
        "task": "open all the drawers",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9"]
    },
    {
        "task_folder": "2_put_all_creditcards_remotecontrols_box",
        "task": "put all credit cards and remote controls in the box",
        "scenes": ["FloorPlan201", "FloorPlan203","FloorPlan204", "FloorPlan205"]
    },
    {
        "task_folder": "2_put_all_tomatoes_potatoes_fridge",
        "task": "put all tomatoes and potatoes in the fridge",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "2_put_all_vases_countertop",
        "task": "put all the vases on the counter top",
        "scenes": ["FloorPlan1", "FloorPlan5"]
    },
    {
        "task_folder": "2_turn_on_all_stove_knobs",
        "task": "turn on all the stove knobs",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "FloorPlan9"]
    },  
]

TASKS_3 = [
    {
        "task_folder": "3_clear_table_to_sofa",
        "task": "Put all readable objects on the sofa",
        "scenes": ["FloorPlan201", "FloorPlan203", "FloorPlan204", "FloorPlan208", "FloorPlan223"]
    },
    {
        "task_folder": "3_put_all_food_countertop",
        "task": "Put all food on the countertop",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "3_put_all_groceries_fridge",
        "task": "Put all groceries in the fridge",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "3_put_all_kitchenware_box",
        "task": "Put all kitchenware in the cardboard box",
        "scenes": ["FloorPlan201"]
    },
    {
        "task_folder": "3_put_all_school_supplies_sofa",
        "task": "Put all school supplies on the sofa",
        "scenes": ["FloorPlan201", "FloorPlan202", "FloorPlan203","FloorPlan209", "FloorPlan212"]
    },
    {
        "task_folder": "3_put_all_shakers_fridge",
        "task": "Put all shakers in the fridge",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },  
    {
        "task_folder": "3_put_all_shakers_tomato", # on countertop
        "task": "put all shakers and tomato on the counter top",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },  
    {
        "task_folder": "3_put_all_silverware_drawer",
        "task": "Put all silverware in the drawer",
        "scenes": [ "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6"]
    },  
    {
        "task_folder": "3_put_all_tableware_countertop",
        "task": "Put all tableware on the countertop",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },  
    # {
    #     "task_folder": "3_transport_groceries",
    #     "task": "put_all_food_countertops",
    #     "scenes": ["FloorPlan1"]
    # },  
    
]

TASKS_4 = [
    {
        "task_folder": "4_clear_couch_livingroom",
        "task": "Clear the couch by placing the items in other appropriate positions ",
        "scenes": ["FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan209", "FloorPlan212"]
    },
    {
        "task_folder": "4_clear_countertop_kitchen",
        "task": "Clear the countertop by placing items in their appropriate positions",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan30", "FloorPlan10", "FloorPlan6"]
    },
    {
        "task_folder": "4_clear_floor_kitchen",
        "task": "Clear the floor by placing items at their appropriate positions",
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5"]
    },
    {
        "task_folder": "4_clear_table_kitchen",
        "task": "Clear the table by placing the items in their appropriate positions",
        "scenes": ["FloorPlan4", "FloorPlan11", "FloorPlan15", "FloorPlan16", "FloorPlan17"]
    },
    {
        "task_folder": "4_make_livingroom_dark",
        "task": "Make the living room dark",
        "scenes": ["FloorPlan201", "FloorPlan202","FloorPlan203","FloorPlan204", "FloorPlan205"]
    },
    {
        "task_folder": "4_put_appropriate_storage",
        "task": "Place all utensils into their appropriate positions",
        "scenes": ["FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6"]
    },  
]


def generate_configs(task_list, base_dir="config"):
    for task_id, task in enumerate(task_list, start=1):
        task_folder = task["task_folder"]
        task_name = task["task"]
        for scene in task["scenes"]:
            cfg = dict(BASE_CONFIG)
            cfg["scene"] = scene
            cfg["task"] = task_name
            cfg["task_folder"] = task_folder
            cfg["test_id"] = task_id

            # config/{task_folder}/{scene}/config.json
            script_dir = Path(__file__).parent  # /mas/utils
            cfg_path = (script_dir / ".." / base_dir / task_folder/ scene / "config.json").resolve() 
            cfg_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

            print(f"Generated {cfg_path}")

if __name__ == "__main__":
    generate_configs(TASKS_1)
    generate_configs(TASKS_2)
    generate_configs(TASKS_3)
    generate_configs(TASKS_4)