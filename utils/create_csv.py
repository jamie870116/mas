import argparse
import json
from pathlib import Path
import re
import ast
from typing import Dict, Any, List, Tuple
import importlib.util
import csv

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
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "1_put_computer_book_remotecontrol_sofa",
        "task": "put labtop, book and remote control on the sofa",
        "scenes": ["floorplan201", "floorplan202", "floorplan203", "floorplan209", "floorplan224"]
    },
    {
        "task_folder": "1_put_knife_bowl_mug_countertop",
        "task": "put knife, bowl, and mug on the counter top",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "1_put_plate_mug_bowl_fridge",
        "task": "put plate, mug, and bowl in the fridge",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "1_put_remotecontrol_keys_watch_box",
        "task": "put remote control, keys, and watch in the box",
        "scenes": ["floorplan201", "floorplan202", "floorplan203", "floorplan207","floorplan209", "floorplan215", "floorplan226", "floorplan228"]
    },
    {
        "task_folder": "1_put_vase_tissuebox_remotecontrol_table",
        "task": "put vase, tissue box, and remote control on the dinning table",
        "scenes": ["floorplan201", "floorplan203", "floorplan216", "floorplan219", "floorplan229"]
    },
    {
        "task_folder": "1_slice_bread_lettuce_tomato_egg",
        "task": "slice bread, lettuce, tomato, and egg with knife",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "1_turn_off_faucet_light",
        "task": "turn off the sink faucet and turn off the light switch",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "1_wash_bowl_mug_pot_pan",
        "task": "clean the bowl, mug, pot, and pan",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
]

TASKS_2 = [
    # {
    #     "task_folder": "2_open_all_cabinets",
    #     "task": "open all the cabinets",
    #     "scenes": ["floorplan1", "floorplan6", "floorplan7", "floorplan8", "floorplan9", "floorplan10"]
    # },
    # {
    #     "task_folder": "2_open_all_drawers",
    #     "task": "open all the drawers",
    #     "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5", "floorplan6", "floorplan7", "floorplan8", "floorplan9"]
    # },
    # {
    #     "task_folder": "2_put_all_creditcards_remotecontrols_box",
    #     "task": "put all credit cards and remote controls in the box",
    #     "scenes": ["floorplan201", "floorplan203","floorplan204", "floorplan205"]
    # },
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
        "scenes": ["FloorPlan1", "FloorPlan2", "FloorPlan3", "FloorPlan4", "FloorPlan5", "FloorPlan6", "FloorPlan7", "FloorPlan8", "floorplan9"]
    },  
]

TASKS_3 = [
    {
        "task_folder": "3_clear_table_to_sofa",
        "task": "Put all readable objects on the sofa",
        "scenes": ["floorplan201", "floorplan203", "floorplan204", "floorplan208", "floorplan223"]
    },
    {
        "task_folder": "3_put_all_food_countertop",
        "task": "Put all food on the countertop",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "3_put_all_groceries_fridge",
        "task": "Put all groceries in the fridge",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "3_put_all_kitchenware_box",
        "task": "Put all kitchenware in the cardboard box",
        "scenes": ["floorplan201"]
    },
    {
        "task_folder": "3_put_all_school_supplies_sofa",
        "task": "Put all school supplies on the sofa",
        "scenes": ["floorplan201", "floorplan202", "floorplan203","floorplan209", "floorplan212"]
    },
    {
        "task_folder": "3_put_all_shakers_fridge",
        "task": "Put all shakers in the fridge",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },  
    {
        "task_folder": "3_put_all_shakers_tomato", # on countertop
        "task": "put all shakers and tomato on the counter top",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },  
    {
        "task_folder": "3_put_all_silverware_drawer",
        "task": "Put all silverware in the drawer",
        "scenes": [ "floorplan2", "floorplan3", "floorplan4", "floorplan5", "floorplan6"]
    },  
    {
        "task_folder": "3_put_all_tableware_countertop",
        "task": "Put all tableware on the countertop",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },  
    # {
    #     "task_folder": "3_transport_groceries",
    #     "task": "put_all_food_countertops",
    #     "scenes": ["floorplan1"]
    # },  
    
]

TASKS_4 = [
    {
        "task_folder": "4_clear_couch_livingroom",
        "task": "Clear the couch by placing the items in other appropriate positions ",
        "scenes": ["floorplan201", "floorplan202","floorplan203","floorplan209", "floorplan212"]
    },
    {
        "task_folder": "4_clear_countertop_kitchen",
        "task": "Clear the countertop by placing items in their appropriate positions",
        "scenes": ["floorplan1", "floorplan2", "floorplan30", "floorplan10", "floorplan6"]
    },
    {
        "task_folder": "4_clear_floor_kitchen",
        "task": "Clear the floor by placing items at their appropriate positions",
        "scenes": ["floorplan1", "floorplan2", "floorplan3", "floorplan4", "floorplan5"]
    },
    {
        "task_folder": "4_clear_table_kitchen",
        "task": "Clear the table by placing the items in their appropriate positions",
        "scenes": ["floorplan4", "floorplan11", "floorplan15", "floorplan16", "floorplan17"]
    },
    {
        "task_folder": "4_make_livingroom_dark",
        "task": "Make the living room dark",
        "scenes": ["floorplan201", "floorplan202","floorplan203","floorplan204", "floorplan205"]
    },
    {
        "task_folder": "4_put_appropriate_storage",
        "task": "Place all utensils into their appropriate positions",
        "scenes": ["floorplan2", "floorplan3", "floorplan4", "floorplan5", "floorplan6"]
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





def parse_log_file(log_path: str) -> Tuple[dict, bool, List[int], Dict[str, dict]]:
    """
    讀取單一 log_llm.txt，回傳：
    - final_report: dict
    - success: bool
    - steps: list[int]
    - objects: Dict[str, dict]，key 為 'Apple|-01.65|+00.81|+00.07' 這種 object_id 前綴
    """
    script_dir = Path(__file__).parent.parent
    print(script_dir)
    text = Path(script_dir / log_path).read_text(encoding="utf-8")

    # 1. Final Report
    m_report = re.search(r"Final Report:\s*(\{.*?\})\s*Success:", text, re.DOTALL)
    if not m_report:
        raise ValueError(f"Cannot find Final Report in {log_path}")
    report_str = m_report.group(1)
    final_report = ast.literal_eval(report_str)

    # 2. Success: True / False
    m_success = re.search(r"Success:\s*(True|False)", text)
    if not m_success:
        raise ValueError(f"Cannot find Success in {log_path}")
    success = m_success.group(1) == "True"

    # 3. Total steps: [32, 32]
    m_steps = re.search(r"Total steps:\s*(\[[^\]]*\])", text)
    if not m_steps:
        raise ValueError(f"Cannot find Total steps in {log_path}")
    steps_str = m_steps.group(1)
    steps = ast.literal_eval(steps_str)  # -> list[int]

    # 4. 物件列表：從 Total steps 那行之後開始掃
    lines = text.splitlines()
    objects: Dict[str, dict] = {}

    # 找到 "Total steps" 那一行的 index
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Total steps:"):
            start_idx = i + 1
            break

    # 從下一行開始 parse，形式大致為：
    # Apple|-01.65|+00.81|+00.07: {'object_id': 'Apple|-01.65|+00.81|+00.07', ...}
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        # 只處理看起來像 "XXX: { ... }" 的行
        m_obj = re.match(r"^([^:]+):\s*(\{.*\})\s*$", line)
        if not m_obj:
            continue
        obj_key = m_obj.group(1).strip()
        obj_dict_str = m_obj.group(2)
        try:
            obj_dict = ast.literal_eval(obj_dict_str)
        except Exception:
            # 如果有少數行不是標準 dict，就略過
            continue
        objects[obj_key] = obj_dict

    return final_report, success, steps, objects

BASE_DIR = Path(__file__).parent.parent  # 專案根目錄可以依需要調整


def load_checker_module(task_folder: str):
    """
    動態載入 Task/{task_folder}/checker.py
    假設 checker.py 裡面有一個函式：
        compute_transport_rate(final_report, objects) -> float
    之後你可以再擴充其他介面，例如 compute_metrics(...)
    """
    checker_path = BASE_DIR / "Task" / task_folder / "checker.py"
    if not checker_path.exists():
        raise FileNotFoundError(f"Checker file not found: {checker_path}")

    module_name = f"checker_{task_folder}"
    spec = importlib.util.spec_from_file_location(module_name, checker_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def compute_transport_rate(final_report: dict) -> float:
    """
    根據 final_report 計算 Transport Rate (TR)。
    支援三種情況：
    1) 純狀態檢查（isDirty 等）
    2) 多目的地搬運（containment.mode == 'multiple'）
    3) 單一目的地搬運（只有 found_counts / missing）
    """

    # =============== 1. 純狀態檢查型 ===============
    # 例如：
    # 'status_checks': [{'isDirty': False}],
    # 'status_targets': [['Bowl_1', 'Mug_1', 'Pan_1', 'Pot_1']],
    # 'status_results': [{'target': 'Bowl_1', 'ok': True}, ...]
    if final_report.get("status_checks"):
        # 展開所有 target
        targets = [
            t
            for group in final_report.get("status_targets", [])
            for t in group
        ]
        n_targets = len(targets)
        if n_targets == 0:
            return 0.0

        # 把 status_results 轉成 {target -> ok}
        ok_map = {
            res.get("target"): res.get("ok", False)
            for res in final_report.get("status_results", [])
        }
        n_ok = sum(1 for t in targets if ok_map.get(t, False))
        return n_ok / n_targets

    # =============== 2. 搬運型（containment） ===============
    containment = final_report.get("containment") or {}
    required_counts = containment.get("required_counts")  # 可能存在，也可能不存在
    found_counts = containment.get("found_counts", {})
    missing = containment.get("missing", {})

    # 2-1. 你第二種形式：有 required_counts 的情況（多目的地）
    # Final Report 範例：
    # 'containment': {
    #   'mode': 'multiple',
    #   'required_counts': {'Tomato': 1, 'Apple': 1},
    #   'found_counts': {..., 'Apple': 1, 'Tomato': 1},
    #   'missing': {}
    # }
    if required_counts:
        total_required = sum(required_counts.values())
        if total_required == 0:
            return 0.0

        # 用 required_counts - missing 計算，比直接信任 found_counts 更穩健
        total_missing = sum(missing.get(k, 0) for k in required_counts.keys())
        total_found = max(total_required - total_missing, 0)
        return total_found / total_required

    # 2-2. 你第三種形式：單一目的地且沒有 required_counts
    # Final Report 範例：
    # 'receptacle': 'Fridge_1',
    # 'required_items': ['Bread', 'Tomato', 'Lettuce'],
    # 'containment': {'found_counts': {'Apple': 1, 'Bread': 1},
    #                 'missing': {'Tomato': 1, 'Lettuce': 1}}
    required_items = final_report.get("required_items", [])
    if required_items and missing:
        total_required = len(required_items)
        if total_required == 0:
            return 0.0
        total_missing = sum(missing.values())
        total_found = max(total_required - total_missing, 0)
        return total_found / total_required


    if "ok" in final_report:
        return 1.0 if final_report["ok"] else 0.0

    return 0.0

def evaluate_tasks(tasks: List[Dict[str, Any]], method="", taskset="") -> List[Dict[str, Any]]:

    all_results = []

    for task_cfg in tasks:
        task_folder = task_cfg["task_folder"]
        scenes = task_cfg["scenes"]

  

        for scene in scenes:
            # 1. 讀 config/{task_folder}/{scene}/config.json
            config_path = BASE_DIR / "config" / task_folder / scene / "config.json"
            if not config_path.exists():
                print(f"[WARN] config not found: {config_path}")
                continue

            config = json.loads(config_path.read_text(encoding="utf-8"))
            task_str = config["task"]  # e.g. "Clear the couch by placing ..."
            task_log_file = task_str.replace(" ", "_")  # e.g. "Clear_the_couch_by_..."

            # 2. logs/{task_log_file}/{scene}/test_*/log_llm.txt
            if method:
                if taskset:
                    logs_root = BASE_DIR / "logs" / method / taskset / task_log_file / scene
                else:
                    logs_root = BASE_DIR / "logs" / method / task_log_file / scene

            else:                
                logs_root = BASE_DIR / "logs" / task_log_file / scene
            
            if not logs_root.exists():
                print(f"[WARN] logs root not found: {logs_root}")
                continue

            # 找出所有 test_* 目錄
            for test_dir in sorted(logs_root.glob("test_*")):
                if not test_dir.is_dir():
                    continue

                log_file = test_dir / "logs_llm.txt"
                if not log_file.exists():
                    alt_log_file = test_dir / "log_llm.txt"
                    if alt_log_file.exists():
                        log_file = alt_log_file
                    else:
                        print(f"[WARN] log file not found in {test_dir}")
                        continue

                try:
                    final_report, success, steps, objects = parse_log_file(str(log_file))
                except Exception as e:
                    print(f"[ERROR] Failed to parse {log_file}: {e}")
                    continue

    
                tr = compute_transport_rate(final_report)
                result = {
                    "task_folder": task_folder,
                    "scene": scene,
                    "test_id": test_dir.name,  # e.g. "test_0001"
                    "success": success,
                    "steps": steps[0],
                    "max_step": max(steps) if steps else None,
                    "transport_rate": tr,
                }
                all_results.append(result)

                # 簡單印一下，也方便 debug
                print(
                    f"[RESULT] {task_folder} | {scene} | {test_dir.name} | "
                    f"success={success} | TR={tr}  | Steps={steps[0]}"
                )

    return all_results

from typing import List, Dict, Any

def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        print("No results to summarize.")
        return {}

    n = len(results)

    avg_success = sum(1 if r["success"] else 0 for r in results) / n
    avg_tr = sum(r["transport_rate"] for r in results) / n
    avg_steps = sum(r["steps"] for r in results if r["steps"] is not None) / n

    summary = {
        "num_tasks": n,
        "avg_success_rate": avg_success,
        "avg_transport_rate": avg_tr,
        "avg_steps": avg_steps,
    }

    print("\n===== OVERALL SUMMARY =====")
    print(f"Total Tasks        : {n}")
    print(f"Avg Success Rate   : {avg_success:.4f}")
    print(f"Avg Transport Rate : {avg_tr:.4f}")
    print(f"Avg Steps          : {avg_steps:.2f}")
    print("===========================\n")

    return summary

def save_results_to_csv(results, csv_path):
    """
    將 list[dict] 結構的結果輸出成 CSV。
    會自動從 dict keys 決定欄位。
    """
    if not results:
        print("No results to save.")
        return

    # CSV 欄位名稱來自 result dict 的 keys
    fieldnames = list(results[0].keys())
    csv_path = csv_path if csv_path.endswith(".csv") else f"{csv_path}.csv"
    csv_path = Path(BASE_DIR/ "logs" / csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV to: {csv_path}")

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--method_dir", default="", choices=["", "summary", "log_cen", "decen"])
    args.add_argument("--taskset", default="ALL", choices=["ALL","TASKS_1","TASKS_2","TASKS_3","TASKS_4"])
    args.add_argument("--output_csv", default="results.csv")
    args.add_argument("--sum", action="store_true")
    args = args.parse_args()
    if args.taskset == "ALL":
        selected = []
        for k in ["TASKS_1","TASKS_2","TASKS_3","TASKS_4"]:
            selected.extend(globals()[k])
    else:
        selected = globals()[args.taskset]
    taskset_name = args.taskset if args.taskset != "ALL" else ""
    res = evaluate_tasks(selected, method=args.method_dir, taskset=taskset_name)
    if args.sum:
        summarize_results(res)
    save_results_to_csv(res, args.output_csv)


if __name__ == "__main__":
    main()
    
    # results_1 = evaluate_tasks(TASKS_1)
    # sum1 = summarize_results(results_1)
    # save_results_to_csv(results_1, 'cen_log_r_task1.csv')
    # results_2 = evaluate_tasks(TASKS_2)
    # save_results_to_csv(results_2, 'cen_summary_r_task2.csv')
    # results_3 = evaluate_tasks(TASKS_3)
    # save_results_to_csv(results_3, 'cen_summary_r_task3.csv')
    # results_4 = evaluate_tasks(TASKS_4)
    # save_results_to_csv(results_4, 'cen_log_r_task4.csv')
    # sum4 = summarize_results(results_4)
   