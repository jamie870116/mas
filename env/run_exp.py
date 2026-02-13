# File name: run_exp.py
# Run the experiment automatically for all tasks

import os
import time

import argparse, json, sys
from pathlib import Path
from datetime import datetime
import importlib

from tasks import TASKS_ALL

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def flatten_tasks(tasks_list):
    """
    tasks_list: list of {"task_folder":..., "task":..., "scenes":[...]}
    return: list of dicts:
      {"task_folder":..., "task":..., "scene":..., "config_path":...}
    """
    out = []
    for t in tasks_list:
        for scene in t["scenes"]:
            out.append({
                "task_folder": t["task_folder"],
                "task": t.get("task", ""),
                "scene": scene,
            })
    return out

def load_state(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done": {}, "updated_at": None}

def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["summary", "log", "decen"])
    ap.add_argument("--taskset", default="ALL", choices=["ALL","TASKS_1","TASKS_2","TASKS_3","TASKS_4"])
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--base_dir", default="config")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=1)
    ap.add_argument("--sleep_after", type=float, default=5.0)
    ap.add_argument("--delete_frames", action="store_true")
    ap.add_argument("--timeout", type=int, default=350)

    args = ap.parse_args()

    # 1) select tasks
    if args.taskset == "ALL":
        selected = []
        for k in ["TASKS_1","TASKS_2","TASKS_3","TASKS_4"]:
            selected.extend(TASKS_ALL[k])
    else:
        selected = TASKS_ALL[args.taskset]

    # 2) flatten into configs
    configs = flatten_tasks(selected)

    # 3) checkpoint
    state_path = Path(".runs") / f"{args.method}_{args.taskset}.json"
    state = load_state(state_path)

    # 找下一段未完成 index（以 index 當 key，簡單穩定）
    done = state["done"]  # dict of {"idx": {"time":...}}
    next_indices = []
    for i in range(len(configs)):
        if str(i) not in done:
            next_indices.append(i)
        if len(next_indices) >= args.chunk:
            break

    if not next_indices:
        print(f"[DONE] All configs finished for {args.method} {args.taskset}.")
        sys.exit(0)

    # 4) build a chunk task list that batch_run can accept (no changes in original)
    #    Each config becomes a task item with a single scene.
    chunk_tasks = []
    for idx in next_indices:
        c = configs[idx]
        chunk_tasks.append({
            "task_folder": c["task_folder"],
            "task": c["task"],
            "scenes": [c["scene"]],
        })

    print(f"[CHUNK] method={args.method} taskset={args.taskset} indices={next_indices}")
    for idx in next_indices:
        c = configs[idx]
        print(f"  - idx={idx} {c['task_folder']} {c['scene']}")

    # 5) import method module and call batch_run
    module_map = {
        "summary": "llm_cm",
        "log": "llm_log3",
        "decen": "llm_decen",
    }
    mod = importlib.import_module(module_map[args.method])

    if not hasattr(mod, "batch_run"):
        raise RuntimeError(f"{module_map[args.method]} has no batch_run()")
    print(f"[START] Running {args.method} on {len(chunk_tasks)} configs from {args.base_dir} with indices {next_indices}...")
    try:
        mod.batch_run(
            chunk_tasks,
            base_dir=args.base_dir,
            start=args.start,
            end=args.end,
            sleep_after=args.sleep_after,
            delete_frames=args.delete_frames,
            timeout=args.timeout,
        )
        # success: mark done
        for idx in next_indices:
            done[str(idx)] = {"time": now()}
        save_state(state_path, state)
        print(f"[CHUNK DONE] wrote checkpoint {state_path}")

    except Exception as e:
        # failure: do not mark done; exit non-zero so bash can log it and continue/retry
        print(f"[CHUNK ERROR] {repr(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
    # selected = []
    # for k in ["TASKS_1","TASKS_2","TASKS_3","TASKS_4"]:
    #     selected.extend(TASKS_ALL[k])
    # config = flatten_tasks(selected)
    # print(config)

   
    