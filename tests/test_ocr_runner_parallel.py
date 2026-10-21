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


def test_plan_workers_falls_back_to_serial_on_4_cores():
    """4 核 / 8GB（**当前生产模板**）：必须退回串行。

    实测（同一批页、同一次运行内）：4 核上 w=4×1 线程 = 0.66x、w=3×1 线程 = 0.71x，
    都**比串行慢**。所以这里必须是 (1, 4)——上线后绝不比旧版慢，这是回归闸门。
    """
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=4, mem_mb=7952, requested=0)
    assert (workers, threads) == (1, 4), (workers, threads)


def test_plan_workers_falls_back_to_serial_on_8_cores():
    """8 核 / 16GB：2 个进程摊不平单线程的 5x 代价，实测 0.99x ⇒ 也退回串行。"""
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=8, mem_mb=16002, requested=0)
    assert (workers, threads) == (1, 8), (workers, threads)


def test_plan_workers_uses_multiprocess_on_16_cores():
    """16 核 / 32GB：4 进程 × 4 线程 = 实测 1.87x（w=16×1 线程 2.06x 但吃 16 份内存）。"""
    mod = _load_runner()
    workers, threads = mod._plan_workers(cpu=16, mem_mb=32104, requested=0)
    assert (workers, threads) == (4, 4), (workers, threads)


def test_plan_workers_never_leaves_cores_idle():
    """不变量：**进程×线程不得少于核数**。

    旧版在 4 核/8GB 上算出 (3, 1)=3 线程 < 4 核，第 4 个核全程闲着自己还慢 42%
    （内存闸门把 workers 压到 3，而 threads=cpu//workers=1），这是那次回归的直接机理。
    """
    mod = _load_runner()
    for cpu in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64):
        for mem in (2048, 4096, 7952, 16002, 32104, 40960, None):
            workers, threads = mod._plan_workers(cpu=cpu, mem_mb=mem, requested=0)
            assert workers * threads >= cpu, (cpu, mem, workers, threads)


def test_plan_workers_respects_memory_budget():
    """内存紧时必须压住进程数（超配的后果是 OOM killer 直接 SIGKILL worker）。"""
    mod = _load_runner()
    for cpu in (4, 8, 16, 32):
        for mem in (2048, 4096, 7952, 16002, 32104):
            workers, threads = mod._plan_workers(cpu=cpu, mem_mb=mem, requested=0)
            if workers > 1:
                assert workers * mod.SAFE_MEM_PER_WORKER_MB <= int(mem * 0.8), \
                    (cpu, mem, workers, threads)


def test_plan_workers_parallel_never_below_min_workers():
    """要么串行，要么进程数 >= 阈值——不存在"2 个进程"这种既慢又费内存的中间态。"""
    mod = _load_runner()
    for cpu in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        for mem in (2048, 7952, 16002, 32104, 40960, None):
            workers, _threads = mod._plan_workers(cpu=cpu, mem_mb=mem, requested=0)
            assert workers == 1 or workers >= mod.SAFE_MIN_PARALLEL_WORKERS, \
                (cpu, mem, workers)


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
