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
        is_multiple: bool = False,
    ):
        self.receptacle = receptacle
        self.need_items = list(recept_require_items or [])
        self.status_checks = list(status_checks or {})
        self.status_targets = list(status_require_items or [])
        self.is_multiple = bool(is_multiple)   

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "TaskConfigChecker":
        return cls(
            receptacle=cfg.get("receptacle"),
            recept_require_items=cfg.get("recept_require_items", []),
            status_checks=cfg.get("status_check"),
            status_require_items=cfg.get("status_require_items", []),
            is_multiple=cfg.get("is_multiple", False),
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
        
        
        if not self.is_multiple:
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
        else:
            recepts = self.receptacle if isinstance(self.receptacle, list) else [self.receptacle]
            found_ids: List[str] = []
            found_readable_by_rec: Dict[str, List[str]] = {}
            rec_errors: Dict[str, str] = {}

            for r in recepts:
                try:
                    rs = env.get_object_status(r)  # r: str
                    ids = rs.get("contains") or []
                    readables = env.get_readable_object_list(ids) if ids else []
                    found_ids.extend(ids or [])
                    found_readable_by_rec[r] = readables
                except Exception as e:
                    rec_errors[r] = str(e)

            all_readables = [n for lst in found_readable_by_rec.values() for n in lst]
            found_counts = Counter(self._base(n) for n in all_readables)

            required_counts = Counter(self._base(x) for x in self.need_items)

            missing: Dict[str, int] = {}
            for base, need_cnt in required_counts.items():
                have = found_counts.get(base, 0)
                if have < need_cnt:
                    missing[base] = need_cnt - have

            report["containment"] = {
                "mode": "multiple",
                "receptacles": recepts,
                "per_receptacle_found": found_readable_by_rec,
                "found_counts": dict(found_counts),
                "required_counts": dict(required_counts),
                "missing": missing,
                **({"errors": rec_errors} if rec_errors else {}),
            }
            return len(missing) == 0

    
    def _check_status(self, env, report: Dict[str, Any]) -> bool:

        if not (self.status_checks and self.status_targets):
            return True
        if len(self.status_checks) != len(self.status_targets):
            raise ValueError("len(status_check) 必須等於 len(status_require_items)")

        results, all_ok = [], True

        for targets, check_for in zip(self.status_targets, self.status_checks):

            if isinstance(targets, str):
                targets = [targets]
            elif not isinstance(targets, list):
                targets = list(targets)

            if isinstance(check_for, tuple):
                checks = [check_for]                       
            elif isinstance(check_for, dict):
                if len(check_for) != 1:
                    raise ValueError("only one key allowed in each status_check(dict) ")
                checks = list(check_for.items())           
            elif isinstance(check_for, list):
                checks = check_for                        
            else:
                raise ValueError("status_check required dict、(k,v) tuple 或 list of pairs")

            for target in targets:
                try:
                    st = env.get_object_status(target)
                except Exception as e:
                    results.append({"target": target, "ok": False, "error": str(e)})
                    all_ok = False
                    continue
                print(f"Checking status for {target}: {st} against {checks}")
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
            elif name == "isSliced":
                actual = bool(obj_status.get("isSliced", False))
                obj_name = obj_status.get("name", "")
                if any(tag in obj_name for tag in ["Slice", "Sliced", "Cracked"]):
                    actual = True
            else:
                actual = obj_status.get(name, None)
            ok = (actual == expected)
            detail[name] = {"expected": expected, "actual": actual, "ok": ok}
            ok_all = ok_all and ok
        return ok_all, detail