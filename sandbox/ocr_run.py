#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""沙箱侧 PaddleOCR 唯一推荐入口。

为什么需要这个封装（别绕过它自己写 PaddleOCR(...)）：
    PaddlePaddle 3.3.1 的 oneDNN 后端在 PP-OCRv6 det 模型上会抛
        NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
        not support [pir::ArrayAttribute<pir::DoubleAttribute>]
        (at .../onednn/onednn_instruction.cc:116)
    实测（2026-10-14，模板 ocr-20g / tpl-ed72fc866af84d0dad57e84a，4 vCPU Xeon w9-3595X）：
        C0 默认                                  -> FAIL
        C7 纯环境变量 FLAGS_use_mkldnn=0          -> FAIL（PaddleX 会显式 enable_mkldnn，覆盖该 FLAG）
        C2 纯环境变量 FLAGS_enable_pir_api=0      -> FAIL
        C1 构造参数 enable_mkldnn=False           -> PASS（103 段文字，真实表单页）
    即：唯一有效开关是**构造参数** enable_mkldnn=False，环境变量改不动。
    本封装把它设为默认，调用方不需要知道这个坑。

    PDF 实测（3 页扫描件型 PDF，1240x1753 @150dpi，无文本层）：
        A1 3.x + enable_mkldnn=False + 逐页渲染   -> PASS  3/3 页，231 段，89.7s
        A2 3.x 默认参数（mkldnn 开）              -> FAIL  与生产同一条报错
        A3 3.x 直接把 PDF 喂 predict()            -> PASS  232 段（原生支持，但丢分页归属）
        B1 2.x 默认参数（其 argparse 默认即 False）-> PASS  3/3 页，233 段，14.0s
        交叉核对：2.x vs 3.x 字符一致率 0.961；3.x vs tesseract 0.870
    注：3.x 原生也能吃 PDF，但返回的是合并结果、没有页码。本封装坚持**逐页渲染**，
    就是为了在输出里保住「第 N 页」，便于引用。

为什么 SAFE_KW 里还要限 det 边长（别删）：
    PaddlePaddle 3.3.1 这套默认**不限制 det 输入边长**，单页推理峰值随像素近似线性增长
    （实测 ≈1000–1170 MB/Mpx）。一个 A3 横向折页（2481x1684 = 4.18 Mpx）就要 ~3.9GB，
    在 4GB 沙箱上必被 OOM 杀掉（rc=-9）。这跟 PDF 总大小/页数无关 —— 只跟**单页像素**有关，
    所以小 PDF 过得去、真实手册（夹 A3 折页）必死。
    加 text_det_limit_side_len=1500 / type=max 后：峰值 3925MB -> 2127MB，识别段数不变。
        （960 会严重漏检：同一张图只剩 3 段。1500 保精度，别往下调。）
    详见 ocr_run.py 里 SAFE_DET_LIMIT 的实测表。

用法:
    python /opt/ocr/ocr_run.py <图片或PDF路径> [更多路径...] [--json] [--lang ch]
    python /opt/ocr/ocr_run.py page.png                 # 纯文本，一行一段
    python /opt/ocr/ocr_run.py a.pdf b.png --json       # 结构化 JSON

退出码:
    0 = 成功且取到文字；2 = 跑通但 0 段文字；1 = 异常（stderr 有完整 traceback）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time as _time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

# 版本标记（沙箱内用 `grep __version__ /opt/ocr/ocr_run.py` 就能确认部署的是哪一版）
#   1.0.0  初版：enable_mkldnn=False + 三个模块 False
#   1.0.1  修 --dpi 未下传的 bug；默认 dpi 200 -> 150
#   1.1.0  方案A：加 det 边长限制 1500（大页面不再 OOM）；SAFE_DPI_MAX 150 -> 200
#   1.2.0  增量进度：渲染与逐页推理都往 stderr 打 `[ocr_run] ...` 进度行。
#          ★ 用途：MCP 的 ocr_pdf_to_text 靠它算真实 pages_ocred。此前
#          result.json 只在整批跑完才落地，轮询期间统计恒 0 ⇒ 模型误判
#          "22 分钟零进展"（2026-10-20 会话 add0229b 事故）。进度行必须
#          逐页 flush：stderr 被重定向到文件，缓冲住就等于没进度。
#   1.2.1  首行 start 标记前移到**最前**（argparse 之前、paddleocr import 之前）。
#          ★ 用途：MCP 侧的启动可见性探针靠 err.log 是否有内容判断"进程真
#          起来了"。原 env 行在 `from paddleocr import PaddleOCR` 之后，paddle
#          导入 5~15s，15s 探针窗口会误判成"没启动"。start 行 <1s 落地。
#   1.3.0  多进程并行（--workers）。逐页串行 → ProcessPoolExecutor。
#          ★ 为什么：单页 70s（扫描件）× 262 页 = 5.1 小时，而沙箱此前只给
#          4 核 —— 串行时**一页的卷积只能吃满 4 核**，加核无效。必须多进程。
#          ★ 进度行必须带 done=<绝对计数>：多进程下页号乱序，而 MCP 侧
#          err.log 取的是 **tail**，靠"页号集合大小"统计会严重低估（尾部 20
#          行只含 20 个页号 ⇒ 报 20 页，实际 262）。done 单调递增，取最后一
#          行即当前值，与串行/乱序/tail 截断都无关。
#          ★ 不改 mkldnn：仍走 PaddleOCR 3.x + enable_mkldnn=False（oneDNN
#          在 paddle 3.3.1 上必崩，见上）。提速只靠并行，不靠换引擎。
#   1.3.1  **修 1.3.0 的默认配比回归**：auto 原来按"一核一进程、每进程 1 线程"
#          算，在生产 4 核/8GB 模板上实测比串行**慢 34%（0.66x）**、4 核 auto
#          也慢 42%（0.71x）——因为每进程 1 线程时单页慢 5 倍以上，4 个进程
#          摊不平。改为"每进程保留 4 线程"（cpu//4），且 workers 不足 4 个
#          就**退回串行**。16 核模板上 auto 取 4×4 线程 = 1.87x（实测）。
#          实测表见 _plan_workers 上方注释；教训：**小样本（2 页）测并行度
#          会把"任务数不足"误读成"并行有效"**。
__version__ = "1.3.1"

#: 进度行前缀。MCP 侧 `_OCR_PROGRESS_MARK` 必须与此一致（有单测锁定）。
PROGRESS_PREFIX = "[ocr_run] "


def _progress(msg: str) -> None:
    """往 stderr 打一行进度并立即 flush（重定向到文件时不能靠缓冲）。"""
    print(PROGRESS_PREFIX + msg, file=sys.stderr, flush=True)

# 构造参数基线：与镜像预热时**完全一致**（外加 det 边长限制，见 SAFE_DET_LIMIT）。
# 这三个 *_orientation/unwarping* 关掉有两个理由：
#   1) 镜像只烘了 det/rec 两个模型，开着的模块缺模型会触发联网下载（慢且依赖出网）；
#   2) 默认表格页不需要矫正/行向分类，关了更快。
SAFE_KW = dict(
    lang="ch",
    enable_mkldnn=False,              # ← 关键：绕开 paddle 3.3.1 oneDNN 缺陷
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # det 输入最长边上限：不设它，大页面(>2.5Mpx)推理必 OOM。见下方 SAFE_DET_LIMIT。
    text_det_limit_side_len=1500,
    text_det_limit_type="max",
)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ── det 输入边长上限（方案 A，2026-10-16 实测）────────────────────────────
# 同一张 2481x1684（A3 横向）图片，只改这一个参数，独立子进程采样峰值内存：
#   原样（默认，无限制）              rc=-9 OOM   27.7s  peak=3925MB  ← 撞满 3925MB 沙箱
#   + text_det_limit_side_len=960     rc=0         7.4s  peak=1364MB  但 texts=3（严重漏检）
#   + text_det_limit_side_len=1500    rc=0        24.8s  peak=2127MB  texts=28（与基线一致）
# 结论：1500 是"内存减半 + 精度不损"的甜点；960 太快而把字都漏掉了，**别往下调**。
# type="max" 的语义 = 把最长边缩到 side_len（不是放大），所以小图不受影响。
SAFE_DET_LIMIT = 1500

# ── PDF 渲染 DPI（实测边界，别轻易调高）──────────────────────────────────
# 2026-10-14，模板 ocr-20g-v2（4 vCPU / 4GB 内存），3 页 A4 扫描件：
#   100 -> PASS  74s
#   150 -> PASS  89s   ← 默认；231 段，与 A1 实测一致
#   200 -> FAIL  RC=137 OOM（66s 被杀）    300 -> 同因
# 200dpi 的 A4 渲染 = 1653x2339，单页推理峰值就吃穿 4GB 沙箱的 3.6GB 可用内存。
# 注意：曾经有个 bug 让 --dpi 从未生效（_expand 没把 dpi 传下去），
#      表面上"改 dpi 没用、全部 OOM"，实为恒用 200。已修（2026-10-14）。
# 2026-10-16 起 SAFE_DET_LIMIT 把推理峰值跟页面尺寸**解耦**了（det 自己会缩），
# 所以 200dpi 在 ≥8GB 模板上重新变得可行（实测见当日验证记录）；4GB 模板仍建议 150。
SAFE_DPI_DEFAULT = 150
SAFE_DPI_MAX = 200

# ── 多进程并行（1.3.0）──────────────────────────────────────────────────
# 每 worker 常驻内存预算。**1.3.0 沙箱实测（4 vCPU / 7952MB，A4@150dpi 密集文字页）**：
#   1 进程 mem_lo 6366MB、2 进程 5536MB、4 进程 4179MB
#   ⇒ 每个增量进程实占 **约 700~1400MB**（单页推理峰值 2127MB 是 A3 横向的最坏
#   情况，A4 远低于此）。取 1800MB 兼顾"最坏页也不 OOM"与"别把 worker 数算少了"。
# 超配的后果是 OOM killer 直接 SIGKILL worker（Python 捕获不到），整批作废。
SAFE_MEM_PER_WORKER_MB = 1800
#: worker 数硬上限。内存约束才是真正的闸门（进程数 × 1.8GB），这条只是防手滑。
SAFE_MAX_WORKERS = 16

#: 每 worker 的推理线程数下限。
SAFE_MIN_THREADS_PER_WORKER = 1

# ── 关键实测结论（决定下面的默认配比，改之前先看）────────────────────
# ★ 2026-10-21 三档模板实测（**同一台机、同一批 4 页 ×N、150dpi，运行内对照**）。
#   上一版这里写的是"一核一进程"，依据是一次**只用 2 页**的实验（w=4 时其实
#   只有 2 个进程在跑）——样本太小，结论是错的，现以这组数据为准：
#
#   模板          单进程(全线程)   多进程最佳         相对该机串行
#   4 核 / 8GB    19.96s/页       w=3×1t 0.71x       w=4×1t **0.66x（倒亏）**
#   8 核 / 16GB   18.3s/页        w=2×4t 0.99x       w=4×2t 0.74x
#   16 核 / 32GB  18.8s/页        w=16×1t 2.06x      w=4×4t 1.87x（auto 取这档）
#
# 三条硬结论：
#   1) **单进程在 ~4 线程就饱和**：4 核 19.96s/页、8 核 18.3s/页、16 核 18.8s/页
#      几乎一样 ⇒ 加核不会让单进程变快，只能靠多进程把闲置的核捡回来；
#   2) **每进程 1 线程时单页要慢 5 倍以上**（4 核实测 107s vs 19.96s）⇒ 进程太少
#      就摊不平这个代价：4 核只能起 4 个进程 ⇒ 净亏 34%；8 核 2 进程 ⇒ 打平；
#      16 核 4~16 个进程才转为正收益；
#   3) 因此**默认配比按"每进程保留 4 线程"来算**：workers = cpu // 4，且当
#      workers < SAFE_MIN_PARALLEL_WORKERS 时**直接退回串行**——绝不能出现
#      "上线后比单进程还慢"这种回归（4 核模板上就是 34%）。
SAFE_THREADS_SATURATION = 4
#: 进程并行的最小进程数（低于此值退回串行）。依据见结论 2 的实测表。
SAFE_MIN_PARALLEL_WORKERS = 4


def _plan_workers(cpu: int | None, mem_mb: int | None, requested: int = 0) -> tuple[int, int]:
    """决定 (worker 数, 每 worker 线程数)。

    显式 `--workers` 一律照办（排障/A-B 用），只按 cpu 推线程数。

    auto 的约束按优先级：
      1. 每进程保留 SAFE_THREADS_SATURATION 个线程 ⇒ workers <= cpu // 4；
      2. 内存：workers * SAFE_MEM_PER_WORKER_MB <= 可用内存 * 0.8；
      3. 上限：SAFE_MAX_WORKERS；
      4. **兜底：workers < SAFE_MIN_PARALLEL_WORKERS 就退回串行**（workers=1、
         threads=cpu）。这条是防回归的闸门，别删。
    """
    cpu = cpu or 1

    def threads_for(workers: int) -> int:
        # ★ 向上取整：向下取整时"内存闸门压了 workers"会让总线程数少于核数
        #   （cpu=32/mem=16GB ⇒ 7 进程×4 线程=28 < 32），白白闲 4 个核。
        #   宁可轻微超订阅（35/32）也不能闲核——实测闲核就是净亏。
        return max(SAFE_MIN_THREADS_PER_WORKER, -(-cpu // workers))

    if requested > 0:
        return requested, threads_for(requested)

    workers = max(1, cpu // SAFE_THREADS_SATURATION)
    if mem_mb:
        mem_workers = max(1, int(mem_mb * 0.8) // SAFE_MEM_PER_WORKER_MB)
        workers = min(workers, mem_workers)
    workers = max(1, min(workers, SAFE_MAX_WORKERS))
    if workers < SAFE_MIN_PARALLEL_WORKERS:
        # 进程太少摊不平"单线程每页慢 5x"，退回串行（见上面实测表）
        return 1, cpu
    return workers, threads_for(workers)


#: worker 进程内的 PaddleOCR 单例。用 initializer 构造一次、复用 N 页 ——
#: PaddleOCR 构造（加载 det/rec 模型）要 5~15s，每页重建等于把并行收益吃光。
_ENGINE = None


def _engine_construct(kw: dict, threads: int | None = None) -> None:
    """构造 PaddleOCR 单例。子进程 initializer 与串行路径共用这一处。

    threads 只在这里设环境变量：必须在 `import paddle` **之前**，否则 OMP 已
    初始化，改环境变量无效。
    """
    global _ENGINE
    if threads:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[name] = str(threads)
    from paddleocr import PaddleOCR

    _ENGINE = PaddleOCR(**kw)


def _predict_page(task: tuple[int, str]) -> tuple[int, dict, float]:
    """worker 入口：OCR 一页，返回 (页号, 结果记录, 耗时秒)。

    ★ 单页异常必须在本函数内吃掉：一个坏页（图片截断/超大/OOM 前的
      MemoryError）不能拖垮整批。真正的 SIGKILL 捕获不到，由主进程的
      BrokenProcessPool 分支兜底。
    """
    page_no, path_str = task
    t0 = _time.monotonic()
    rec = {"page": page_no, "input": path_str, "ok": False, "error": "",
           "texts": [], "scores": [], "polys": []}
    try:
        res = _ENGINE.predict(path_str)
        acc = {"texts": [], "scores": [], "polys": []}
        _collect(res, acc)
        rec.update(ok=True, **acc)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return page_no, rec, _time.monotonic() - t0


def _worker_init(kw: dict, threads: int) -> None:
    """子进程 initializer：只构造引擎，不碰输入路径。"""
    try:
        _engine_construct(kw, threads)
    except Exception:  # noqa: BLE001
        # 构造失败必须让父进程看得见，否则会表现为"worker 起来了但每页都失败"
        print("[ocr_run] worker 构造 PaddleOCR 失败 (pid=%d):" % os.getpid(),
              file=sys.stderr, flush=True)
        traceback.print_exc()
        raise



def _mem_total_mb():
    """读 /proc/meminfo 的 MemTotal；拿不到就返回 None（不阻断流程）。"""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("MemTotal:"):
                    return int(ln.split()[1]) // 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def _collect(o, out, depth=0):
    """从 OCRResult（dict 子类）里挖 rec_texts / rec_scores / 坐标框。"""
    if depth > 6:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("rec_texts", "texts") and isinstance(v, (list, tuple)):
                out["texts"].extend([str(x) for x in v])
            elif k in ("rec_scores", "scores") and isinstance(v, (list, tuple)):
                out["scores"].extend([float(x) for x in v])
            elif k in ("rec_polys", "dt_polys") and isinstance(v, (list, tuple)):
                out["polys"].extend([_as_list(x) for x in v])
            elif isinstance(v, dict):
                _collect(v, out, depth + 1)
            elif isinstance(v, (list, tuple)) and len(v) < 80:
                _collect(v, out, depth + 1)
    elif isinstance(o, (list, tuple)) and len(o) < 80:
        for x in o:
            _collect(x, out, depth + 1)


def _as_list(x):
    try:
        return [[int(p[0]), int(p[1])] for p in x]
    except Exception:  # noqa: BLE001
        return []


def _pdf_to_pngs(path: Path, dpi: int = SAFE_DPI_DEFAULT):
    """PDF -> 逐页 PNG；优先 pypdfium2，回退 pdf2image，都没有就明确报错。"""
    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(str(path))
        total = len(pdf)
        _progress("render start file=%s total=%d dpi=%d" % (path.name, total, dpi))
        out = []
        for i in range(total):
            img = pdf[i].render(scale=dpi / 72).to_pil()
            p = Path("/tmp") / f"_ocr_{path.stem}_p{i + 1}.png"
            img.save(p)
            out.append(p)
            # 每 10 页 + 末页打一次：渲染 262 页要几分钟，全程静默会让调用方
            # 以为进程死了（MCP 侧据 err.log 是否有内容判断"在渲染"）。
            if (i + 1) % 10 == 0 or i + 1 == total:
                _progress("render %d/%d" % (i + 1, total))
        return out
    except ImportError:
        pass
    try:
        from pdf2image import convert_from_path  # type: ignore

        _progress("render start file=%s via=pdf2image dpi=%d" % (path.name, dpi))
        out = []
        for i, img in enumerate(convert_from_path(str(path), dpi=dpi), 1):
            p = Path("/tmp") / f"_ocr_{path.stem}_p{i}.png"
            img.save(p)
            out.append(p)
        _progress("render %d/%d" % (len(out), len(out)))
        return out
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "输入是 PDF，但沙箱里既没有 pypdfium2 也没有 pdf2image。"
            "请先 `pip install pypdfium2`，或自行把 PDF 渲染成 PNG 后再调用。"
        ) from e


def _expand(paths, dpi: int = SAFE_DPI_DEFAULT):
    out = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"输入不存在: {p}")
        suf = p.suffix.lower()
        if suf == ".pdf":
            out.extend(_pdf_to_pngs(p, dpi))
        elif suf in IMAGE_EXT:
            out.append(p)
        else:
            raise ValueError(f"不支持的后缀 {suf}（支持图片: {sorted(IMAGE_EXT)}，以及 .pdf）")
    return out


def _emit_page(rec: dict, page_no: int, total: int, elapsed: float, done: int) -> None:
    """打一行逐页进度。**主进程独占此函数**——多进程下 worker 不打进度行，
    统一由收集端发，避免多个子进程交错写同一行。

    格式（MCP `main.py::_parse_ocr_progress` 解析，双端有单测锁定）：
        `[ocr_run] page <页号>/<总页数> <ok|fail> texts=<k> elapsed=<s>s done=<已完成页数>`
    `done` 是绝对计数，**单调递增**；页号在多进程下是乱序的。MCP 只认 done，
    页号仅供人看（定位是哪页出问题）。
    """
    _progress(
        "page %d/%d %s texts=%d elapsed=%.1fs done=%d"
        % (page_no, total, "ok" if rec["ok"] else "fail",
           len(rec["texts"]), elapsed, done)
    )


def _run_serial(kw: dict, inputs: list, total: int) -> list[dict]:
    """单进程串行（--workers 1）。保留作 A/B 基线与降级路径。"""
    try:
        _engine_construct(kw, None)
    except Exception:  # noqa: BLE001
        print("[ocr_run] PaddleOCR 构造失败:", file=sys.stderr)
        traceback.print_exc()
        raise
    results = []
    for page_no, img in enumerate(inputs, 1):
        page_no, rec, elapsed = _predict_page((page_no, str(img)))
        results.append(rec)
        _emit_page(rec, page_no, total, elapsed, len(results))
    return results


def _run_parallel(kw: dict, inputs: list, total: int, workers: int,
                  threads: int) -> tuple[list[dict], bool]:
    """多进程并行。返回 (按页序排好的 results, 是否发生 worker 崩溃)。

    为什么用 ProcessPoolExecutor 而不是线程：Paddle 的推理是 C++ 里跑满了
    GIL 的重计算，线程池拿不到并行；只有多进程能真正摊到多核。
    """
    results: list[dict] = []
    done = 0
    broken = False
    tasks = [(i, str(img)) for i, img in enumerate(inputs, 1)]
    cursor = 0
    pending: dict = {}

    def _fail(page_no: int, exc) -> None:
        nonlocal done
        msg = "%s: %s" % (type(exc).__name__, str(exc).splitlines()[0] if str(exc) else "")
        rec = {"page": page_no, "input": "", "ok": False, "error": msg,
               "texts": [], "scores": [], "polys": []}
        done += 1
        results.append(rec)
        _emit_page(rec, page_no, total, 0.0, done)

    def _ok(fut, page_no: int) -> None:
        nonlocal done
        try:
            _, rec, elapsed = fut.result()
        except BaseException as exc:  # noqa: BLE001
            _fail(page_no, exc)
            return
        done += 1
        results.append(rec)
        _emit_page(rec, page_no, total, elapsed, done)

    # 窗口 = workers（不放大：每多一个在飞任务就多压一份页图内存）
    window = max(2, workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(kw, threads)) as pool:
        while cursor < len(tasks) or pending:
            while not broken and cursor < len(tasks) and len(pending) < window:
                page_no, path_str = tasks[cursor]
                cursor += 1
                try:
                    pending[pool.submit(_predict_page, (page_no, path_str))] = page_no
                except BaseException as exc:  # noqa: BLE001
                    broken = True
                    _fail(page_no, exc)
            if pending:
                finished, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for fut in finished:
                    page_no = pending.pop(fut)
                    exc = fut.exception()
                    # BrokenProcessPool = worker 被 OOM killer 杀了（SIGKILL
                    # 捕获不到），此时 pool 已不可用，后续 submit 必抛。
                    if isinstance(exc, BrokenProcessPool):
                        broken = True
                    _ok(fut, page_no)
            if broken:
                for fut, page_no in list(pending.items()):
                    fut.cancel()
                    _fail(page_no, BrokenProcessPool("worker 进程异常退出"))
                pending.clear()
                for page_no, _p in tasks[cursor:]:
                    _fail(page_no, BrokenProcessPool("worker 进程异常退出"))
                cursor = len(tasks)
                break

    return sorted(results, key=lambda r: r["page"]), broken


def main(argv=None) -> int:
    # ★ 第一句就打 start 行：MCP 侧启动可见性探针只认 err.log 有没有内容，
    #   而 `from paddleocr import PaddleOCR` 要 5~15s。放在 import 之后会让
    #   探针把"正在 import"误判成"没起来"（1.2.1）。
    print(PROGRESS_PREFIX + "start pid=%d argv=%s"
          % (os.getpid(), " ".join(sys.argv[1:])[:300]), file=sys.stderr, flush=True)
    ap = argparse.ArgumentParser(description="沙箱侧 OCR（已内置 enable_mkldnn=False + det 边长限制）")
    ap.add_argument("paths", nargs="+", help="图片或 PDF 路径")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON（默认输出纯文本）")
    ap.add_argument("--lang", default=SAFE_KW["lang"], help="识别语言（默认 ch）")
    ap.add_argument("--dpi", type=int, default=SAFE_DPI_DEFAULT,
                    help="PDF 渲染 DPI（默认 %d）。>%d 在 4GB 沙箱上实测会被 OOM 杀掉，"
                         "确需调高请先确认内存。" % (SAFE_DPI_DEFAULT, SAFE_DPI_MAX))
    ap.add_argument("--det-limit", type=int, default=SAFE_DET_LIMIT,
                    help="det 输入最长边上限（默认 %d）。设 0 = 不限制，"
                         "大页面(>2.5Mpx)会直接把沙箱内存吃满被杀 —— 除非你确认内存充足，否则别关。"
                         % SAFE_DET_LIMIT)
    ap.add_argument("--allow-mkldnn", action="store_true",
                    help="显式打开 oneDNN（默认关闭）。除非你确认 paddle 已修复，否则别用。")
    ap.add_argument("--workers", type=int, default=0,
                    help="并行进程数。0=自动（按 CPU 与内存推算，见 _plan_workers）；"
                         "1=退回逐页串行（A/B 基线，或 worker 被 OOM 杀掉后的降级重跑）。")
    args = ap.parse_args(argv)

    # 只查模块在不在，**不真 import**：paddle 一 import 就会建 OMP 线程池，
    # fork 出来的 worker 会继承一个已初始化的线程池状态（且线程数已固定，
    # 后面的 OMP_NUM_THREADS 设置失效）。真正的 import 放在 worker 内 /
    # _engine_construct 里。
    if importlib.util.find_spec("paddleocr") is None:
        print("[ocr_run] 找不到 paddleocr 模块，镜像未正确预装？", file=sys.stderr)
        return 1

    kw = dict(SAFE_KW)
    kw["lang"] = args.lang
    if args.allow_mkldnn:
        kw["enable_mkldnn"] = True
    if args.det_limit > 0:
        kw["text_det_limit_side_len"] = args.det_limit
        kw["text_det_limit_type"] = "max"
    else:
        kw.pop("text_det_limit_side_len", None)
        kw.pop("text_det_limit_type", None)

    mem = _mem_total_mb()
    print("[ocr_run] env MemTotal=%sMB dpi=%d det_limit=%s mkldnn=%s"
          % (mem if mem else "?", args.dpi, args.det_limit if args.det_limit > 0 else "OFF",
             kw.get("enable_mkldnn")), file=sys.stderr)

    if args.dpi > SAFE_DPI_MAX:
        print("[ocr_run] 警告：--dpi %d 超过实测安全上限 %d，在 4GB 沙箱上很可能被 OOM 杀掉"
              % (args.dpi, SAFE_DPI_MAX), file=sys.stderr)
    if args.det_limit <= 0 and mem is not None and mem < 6000:
        print("[ocr_run] 警告：det 边长限制已关闭，而本沙箱只有 %dMB 内存。"
              "超 2.5Mpx 的页面极可能 OOM 被杀。" % mem, file=sys.stderr)
    elif args.det_limit > 0 and args.det_limit < 1200:
        print("[ocr_run] 警告：--det-limit %d 偏低，实测会把长文本行的字漏掉（960 时只剩 3 段）。"
              % args.det_limit, file=sys.stderr)


    try:
        inputs = _expand(args.paths, args.dpi)
    except Exception as e:  # noqa: BLE001
        print(f"[ocr_run] 输入处理失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    total_pages = len(inputs)
    workers, threads = _plan_workers(os.cpu_count(), mem, args.workers)
    # 日志必须点名并行配置：出问题时要能一眼看出"几个进程 × 几线程"，
    # 而不是回来翻命令行。
    _progress("parallel workers=%d threads_per_worker=%d cpu=%s mem=%sMB pages=%d"
              % (workers, threads, os.cpu_count(), mem if mem else "?", total_pages))
    _progress("ocr start pages=%d" % total_pages)

    try:
        if workers <= 1:
            results = _run_serial(kw, inputs, total_pages)
            broken = False
        else:
            results, broken = _run_parallel(kw, inputs, total_pages, workers, threads)
    except Exception:  # noqa: BLE001
        print("[ocr_run] OCR 执行失败:", file=sys.stderr)
        traceback.print_exc()
        return 1

    total_text = sum(len(r["texts"]) for r in results)
    if broken:
        print("[ocr_run] 有 worker 异常退出（多为内存不足被 OOM killer 杀掉）："
              "失败页已单独标记，建议用 --workers %d 重跑缺页。"
              % max(1, workers // 2), file=sys.stderr)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for rec in results:
            for t in rec["texts"]:
                print(t)

    if broken or any(r["error"] for r in results):
        return 1
    return 0 if total_text > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
