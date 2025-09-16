# checkers/task_config_checker.py
from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Tuple, Iterable, Optional, Union

StatusChecks = Union[Dict[str, Any], List[Tuple[str, Any]], List[List[Any]]]

class TaskConfigChecker:
    def __init__(
        self,
        receptacle: Optional[str] = None,
        recept_require_items: Optional[List[str]] = None,
        status_checks: Optional[StatusChecks] = None,
        status_require_items: Optional[List[str]] = None,
    ):
        self.receptacle = receptacle
        self.need_items = list(recept_require_items or [])
        self.status_checks = list(status_checks or {})
        self.status_targets = list(status_require_items or [])
        # print(f"Initialized TaskConfigChecker with receptacle={self.receptacle}, need_items={self.need_items}, status_checks={self.status_checks}, status_targets={self.status_targets}")

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TaskConfigChecker":
        return cls(
            receptacle=cfg.get("receptacle"),
            recept_require_items=cfg.get("recept_require_items", []),
            status_checks=cfg.get("status_check"),
            status_require_items=cfg.get("status_require_items", []),
        )

    def check(self, env) -> Tuple[bool, Dict[str, Any]]:
        report: Dict[str, Any] = {
            "receptacle": self.receptacle,
            "required_items": self.need_items,
            "status_checks": self.status_checks,
            "status_targets": self.status_targets,
            "containment": {},
            "status_results": [],
            "ok_containment": True,
            "ok_status": True,
            "ok": True,
        }
        ok_cont = self._check_containment(env, report)
        ok_stat = self._check_status(env, report)
        report["ok_containment"] = ok_cont
        report["ok_status"] = ok_stat
        report["ok"] = ok_cont and ok_stat
        return report["ok"], report

    # ---------- internal ----------
    def _check_containment(self, env, report: Dict[str, Any]) -> bool:
        if not (self.receptacle and self.need_items):
            return True
        try:
            rec_stat = env.get_object_status(self.receptacle)
        except Exception as e:
            report["containment"] = {"error": f"receptacle not found: {e}"}
            return False
        ids = rec_stat.get("contains") or []
        readable = env.get_readable_object_list(ids) if ids else []
        counts = Counter(self._base(n) for n in readable)
        missing = {}
        for item in self.need_items:
            base = self._base(item)
            if counts.get(base, 0) < 1:
                missing[base] = 1
        report["containment"] = {"found_counts": dict(counts), "missing": missing}
        return len(missing) == 0

    # def _check_status(self, env, report: Dict[str, Any]) -> bool:
    #     if not (self.status_checks and self.status_targets):
    #         return True
    #     results, all_ok = [], True
    #     for target in self.status_targets:
    #         try:
    #             st = env.get_object_status(target)
    #         except Exception as e:
    #             results.append({"target": target, "ok": False, "error": str(e)})
    #             all_ok = False
    #             continue
    #         t_ok, detail = self._eval_checks(st, self.status_checks)
    #         results.append({"target": target, "ok": t_ok, "detail": detail})
    #         all_ok = all_ok and t_ok
    #     report["status_results"] = results
    #     return all_ok
    
    def _check_status(self, env, report: Dict[str, Any]) -> bool:
        """
        假設：
        - len(self.status_checks) == len(self.status_targets)
        - 每個 self.status_checks[i] 僅包含一個狀態鍵（dict 單鍵，或 ('key', value)）
        - self.status_targets[i] 是要檢查的一組目標（list[str] 或 str）
        """
        if not (self.status_checks and self.status_targets):
            return True
        if len(self.status_checks) != len(self.status_targets):
            raise ValueError("len(status_check) 必須等於 len(status_require_items)")

        results, all_ok = [], True

        for targets, check_for in zip(self.status_targets, self.status_checks):
            # print(f"Checking targets: {targets} for conditions: {check_for}")
            # ---- normalize targets 為 list[str] ----
            if isinstance(targets, str):
                targets = [targets]
            elif not isinstance(targets, list):
                targets = list(targets)

            # ---- normalize check_for 為 List[Tuple[str, Any]] ----
            if isinstance(check_for, tuple):
                checks = [check_for]                       # 例：("isDirty", False)
            elif isinstance(check_for, dict):
                if len(check_for) != 1:
                    raise ValueError("only one key allowed in each status_check(dict) ")
                checks = list(check_for.items())           # 例：[("isDirty", False)]
            elif isinstance(check_for, list):
                checks = check_for                         # 已是 list of (k,v)
            else:
                raise ValueError("status_check required dict、(k,v) tuple 或 list of pairs")

            for target in targets:
                try:
                    st = env.get_object_status(target)
                except Exception as e:
                    results.append({"target": target, "ok": False, "error": str(e)})
                    all_ok = False
                    continue

                t_ok, detail = self._eval_checks(st, checks)
                results.append({"target": target, "ok": t_ok, "detail": detail})
                all_ok = all_ok and t_ok

        report["status_results"] = results
        return all_ok

    @staticmethod
    def _base(name: str) -> str:
        return name.split("_")[0] if "_" in name else name


    @staticmethod
    def _eval_checks(obj_status: Dict[str, Any], checks: Iterable[Tuple[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        detail, ok_all = {}, True
        for name, expected in checks:
            if name == "is_closed":
                actual = (obj_status.get("is_open", False) is False)
            elif name == "is_open":
                actual = bool(obj_status.get("is_open", False))
            elif name == "is_on":
                actual = (obj_status.get("isToggled", False) is True)
            elif name == "is_off":
                actual = (obj_status.get("isToggled", False) is False)
            elif name == "isDirty":
                actual = bool(obj_status.get("isDirty", False))

            else:
                actual = obj_status.get(name, None)
            ok = (actual == expected)
            detail[name] = {"expected": expected, "actual": actual, "ok": ok}
            ok_all = ok_all and ok
        return ok_all, detail