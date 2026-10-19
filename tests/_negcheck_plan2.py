# -*- coding: utf-8 -*-
"""方案 2 anchor 的负向验证：逐个变体必须让对应断言变红。

用法：python tests/_negcheck_plan2.py
只读 main.py 的文本做内存变体，不落盘、不改仓库。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import test_ocr_pdf_to_text as T  # noqa: E402

SRC = T.MAIN_PY.read_text(encoding="utf-8")


class _P:
    """伪装成 pathlib.Path，让 _src() 拿到变体文本。"""

    def __init__(self, text):
        self._text = text

    def read_text(self, encoding=None):
        return self._text


def run_variant(name, mutated, test_name):
    old = T.MAIN_PY
    T.MAIN_PY = _P(mutated)
    try:
        getattr(T, test_name)()
        print("  [%s] -> PASS（anchor 失效！）" % test_name)
        return False
    except AssertionError as e:
        first = str(e).splitlines()[0][:100]
        print("  [%s] -> RED OK: %s" % (test_name, first))
        return True
    finally:
        T.MAIN_PY = old


def mutate(old, new):
    assert old in SRC, "变体锚点不在源码里：%r" % old[:60]
    return SRC.replace(old, new, 1)


VARIANTS = [
    (
        "负向1: pages 默认值回到 all",
        mutate('pages: str = "20",', 'pages: str = "all",'),
        "test_default_pages_is_not_all",
    ),
    (
        "负向2: 判活退回 kill -0（PID 文件方案）",
        mutate(
            "    done_code = await _poll_done()\n    if done_code is None:",
            "    done_code = await _poll_done()\n"
            "    st = await _sandbox_run(sandbox_id, \"kill -0 $(cat /tmp/p) 2>/dev/null && echo ALIVE || echo DEAD\", timeout=30)\n"
            "    if done_code is None:",
        ),
        "test_background_launch_uses_done_flag_not_pid_liveness",
    ),
    (
        "负向3: 后台化退回同步等（去掉 nohup &）",
        mutate(
            '"nohup bash -c \'python3 %s %s --json --dpi %d > %s 2> %s; echo $? > %s\' "',
            '"bash -c \'python3 %s %s --json --dpi %d > %s 2> %s; echo $? > %s\' "',
        ),
        "test_background_launch_uses_done_flag_not_pid_liveness",
    ),
    (
        "负向4: _OCR_WAIT_MAX 放大到 3600（旧预算思维）",
        mutate("_OCR_WAIT_MAX = 300", "_OCR_WAIT_MAX = 3600"),
        "test_wait_seconds_clamped_to_mcp_client_hard_limit",
    ),
    (
        "负向5: 去掉 wait_seconds 钳制",
        mutate(
            "wait_seconds = max(0, min(_OCR_WAIT_MAX, int(wait_seconds)))",
            "wait_seconds = int(wait_seconds)",
        ),
        "test_wait_seconds_clamped_to_mcp_client_hard_limit",
    ),
    (
        "负向6: 去掉指纹比对直接复用结果",
        mutate(
            "if prev_fp == fingerprint and \"OK\" in (has_result.get(\"stdout\") or \"\"):",
            "if \"OK\" in (has_result.get(\"stdout\") or \"\"):",
        ),
        "test_fingerprint_guard_before_result_reuse",
    ),
    (
        "负向7: 轮询循环中和 deadline 判定（永不退出）",
        mutate(
            "        if _time.time() >= deadline:\n",
            "        if False and _time.time() >= deadline:\n",
        ),
        "test_wait_loop_has_deadline_exit",
    ),
]


def main():
    ok = 0
    for name, mutated, test_name in VARIANTS:
        print("===", name, "===")
        if run_variant(name, mutated, test_name):
            ok += 1
    print()
    print("结果：%d/%d 个变体被 anchor 抓住" % (ok, len(VARIANTS)))
    return 0 if ok == len(VARIANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
