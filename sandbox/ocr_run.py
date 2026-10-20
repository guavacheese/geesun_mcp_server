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
import json
import os
import sys
import time as _time
import traceback
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
__version__ = "1.2.0"

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


def main(argv=None) -> int:
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
    args = ap.parse_args(argv)

    try:
        from paddleocr import PaddleOCR
    except Exception:  # noqa: BLE001
        print("[ocr_run] 无法 import paddleocr，镜像未正确预装？", file=sys.stderr)
        traceback.print_exc()
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

    try:
        ocr = PaddleOCR(**kw)
    except Exception:  # noqa: BLE001
        print("[ocr_run] PaddleOCR 构造失败:", file=sys.stderr)
        traceback.print_exc()
        return 1

    results = []
    total_text = 0
    total_pages = len(inputs)
    _progress("ocr start pages=%d" % total_pages)
    for page_no, img in enumerate(inputs, 1):
        page_t0 = _time.monotonic()
        rec = {"page": page_no, "input": str(img), "ok": False, "error": "",
               "texts": [], "scores": [], "polys": []}
        try:
            res = ocr.predict(str(img))
            acc = {"texts": [], "scores": [], "polys": []}
            _collect(res, acc)
            rec.update(ok=True, **acc)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            print(f"[ocr_run] {img} 推理失败: {rec['error']}", file=sys.stderr)
            traceback.print_exc()
        total_text += len(rec["texts"])
        results.append(rec)
        # 逐页进度行：MCP 的 ocr_pdf_to_text 解析它算真实 pages_ocred
        # （格式 `[ocr_run] page <i>/<n> <ok|fail> texts=<k> elapsed=<s>s`，
        # 解析器见 main.py::_parse_ocr_progress，两边有单测锁定）。
        _progress(
            "page %d/%d %s texts=%d elapsed=%.1fs"
            % (page_no, total_pages, "ok" if rec["ok"] else "fail",
               len(rec["texts"]), _time.monotonic() - page_t0)
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for rec in results:
            for t in rec["texts"]:
                print(t)

    if any(r["error"] for r in results):
        return 1
    return 0 if total_text > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
