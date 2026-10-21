#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ocr_run.py 多进程改造（1.3.0）的离线单测。

为什么单独一个文件：`sandbox/ocr_run.py` 是**下发给沙箱执行**的脚本，它的
ProcessPoolExecutor 分支没法在本地跑真子进程（子进程里没有 paddle）。所以
这里只锁三件能在本地证伪的事：

  1. `_plan_workers` 的约束（内存/CPU/上限）—— 算错会 OOM 或超订阅；
  2. `_predict_page` 的失败隔离 —— 单个坏页不能拖垮整批；
  3. 结果**按页序重组** —— 多进程返回乱序，不排序会让 JSON 里页码错位，
     引用"第 N 页"就全错。

真·多进程吞吐与时序在沙箱里实测（见 1.3.0 实测记录），本地不断言耗时。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER_PY = ROOT / "sandbox" / "ocr_run.py"


def _load_runner():
    """把 sandbox/ocr_run.py 当普通模块加载（不依赖沙箱路径）。"""
    spec = importlib.util.spec_from_file_location("ocr_run_under_test", RUNNER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ocr_run_under_test"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class _FakeEngine:
    """按页图文件名决定行为：含 'bad' 的页抛异常，其余返回固定文本。"""

    def __init__(self, texts_per_page=3):
        self.texts_per_page = texts_per_page
        self.calls = []

    def predict(self, path):
        self.calls.append(path)
        if "bad" in Path(path).name:
            raise RuntimeError("模拟单页推理失败: numpy 维度错误")
        return [{"rec_texts": ["t%d" % i for i in range(self.texts_per_page)],
                 "rec_scores": [0.99] * self.texts_per_page}]


# ────────────────────────── 1. worker 规划 ──────────────────────────


def test_plan_workers_respects_memory_budget():
    """4 核 / 8GB（当前生产模板）：内存只够 3 个 worker（3 × 1800MB < 6.4GB 预算）。"""
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=4, mem_mb=7952, requested=0)
    assert workers == 3, (workers, threads)
    assert threads == 1, (workers, threads)
    assert workers * mod.SAFE_MEM_PER_WORKER_MB <= int(7952 * 0.8), "worker 数超内存预算"


def test_plan_workers_respects_cpu_and_cap():
    """32 核 / 40GB：内存允许 18 个，被 SAFE_MAX_WORKERS=16 截住，每 worker 2 线程。"""
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=32, mem_mb=40960, requested=0)
    assert workers == mod.SAFE_MAX_WORKERS, (workers, threads)
    assert threads >= 1, "每 worker 至少 1 线程"
    assert workers * threads <= 32, "超订阅：进程×线程数超过核数"


def test_plan_workers_survives_unknown_memory():
    """拿不到 MemTotal 时不能算出 0 个 worker（pid 1 容器常见 /proc 不可读）。"""
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=8, mem_mb=None, requested=0)
    assert workers == 8, (workers, threads)
    assert threads == 1, (workers, threads)


def test_plan_workers_explicit_request_wins():
    """--workers 1 必须能强制退回串行（OOM 后的降级重跑路径）。"""
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=32, mem_mb=40960, requested=1)
    assert workers == 1, (workers, threads)
    assert threads == 32, "单进程时应把全部核给它的 OMP"


# ────────────────────── 2. 单页失败隔离 ──────────────────────


def test_predict_page_swallows_single_page_error():
    """坏页返回 ok=False + error，**不抛异常**——抛了就是整批失败。"""
    mod = _load_runner()
    mod._ENGINE = _FakeEngine()
    page_no, rec, elapsed = mod._predict_page((7, "/tmp/_ocr_x_bad_p7.png"))
    assert page_no == 7
    assert rec["ok"] is False, rec
    assert "RuntimeError" in rec["error"], rec
    assert rec["texts"] == [], rec
    assert elapsed >= 0


def test_predict_page_collects_texts():
    mod = _load_runner()
    mod._ENGINE = _FakeEngine(texts_per_page=4)
    page_no, rec, _ = mod._predict_page((3, "/tmp/_ocr_x_p3.png"))
    assert rec["ok"] is True, rec
    assert len(rec["texts"]) == 4, rec
    assert len(rec["scores"]) == 4, rec
    # _FakeEngine 不返回坐标键 ⇒ polys 为空是正确行为（_collect 只认 rec_polys/dt_polys）
    assert rec["polys"] == [], rec


# ────────────────────── 3. 按页序重组 + 进度行 ──────────────────────


def _capture_progress(mod):
    """替换 _progress，收集进度行文本。"""
    lines = []
    mod._progress = lambda msg: lines.append(mod.PROGRESS_PREFIX + msg)
    return lines


def test_emit_page_carries_absolute_done_counter():
    """进度行必须带 done=<绝对计数>，格式与 MCP 解析器一致。"""
    mod = _load_runner()
    lines = _capture_progress(mod)
    mod._emit_page({"ok": True, "texts": ["a", "b"]}, 37, 262, 70.2, 12)
    assert len(lines) == 1, lines
    line = lines[0]
    assert line == "[ocr_run] page 37/262 ok texts=2 elapsed=70.2s done=12", line
    # 与 MCP 常量同源
    assert "@PROGRESS_PREFIX@" not in line
    assert line.startswith(mod.PROGRESS_PREFIX + "page ")


def test_emit_page_marks_failed_page_but_still_counts():
    """失败页也要计入 done，否则 MCP 的 pages_ocred 永远停在上一次成功页。"""
    mod = _load_runner()
    lines = _capture_progress(mod)
    mod._emit_page({"ok": False, "texts": [], "error": "X"}, 5, 10, 0.0, 5)
    assert "fail" in lines[0], lines
    assert lines[0].endswith("done=5"), lines


def test_serial_results_are_page_ordered():
    """串行路径基线：页序即输入序（多进程要与之对齐，否则引用错页）。"""
    mod = _load_runner()
    texts = _capture_progress(mod)
    mod._ENGINE = _FakeEngine()
    mod._engine_construct = lambda kw, threads=None: None  # 跳过真构造
    
    images = [Path("/tmp/_ocr_x_p%d.png" % i) for i in (1, 2, 3, 4)]
    results = mod._run_serial(mod.SAFE_KW, images, len(images))
    assert [r["page"] for r in results] == [1, 2, 3, 4], results
    assert [ln.split()[2] for ln in texts] == [
        "1/4", "2/4", "3/4", "4/4",
    ], texts
    # done 单调递增
    dones = [int(ln.rsplit("done=", 1)[1]) for ln in texts]
    assert dones == [1, 2, 3, 4], dones


def test_serial_continues_after_bad_page():
    """串行路径遇坏页必须继续跑完 —— 一页失败不能吃掉后面 200 页。"""
    mod = _load_runner()
    _capture_progress(mod)
    mod._ENGINE = _FakeEngine()
    mod._engine_construct = lambda kw, threads=None: None  # 跳过真构造
    
    images = [Path("/tmp/_ocr_x_p%d.png" % i) for i in (1, 2, 3)]
    images.insert(1, Path("/tmp/_ocr_x_bad_p9.png"))
    results = mod._run_serial(mod.SAFE_KW, images, len(images))
    assert len(results) == 4, results
    assert [r["ok"] for r in results] == [True, False, True, True], results
