# -*- coding: utf-8 -*-
"""负向验证：分批（方案 B）的每处语义单独回退，确认对应单测变红。

做法：改 main.py → 跑指定测试 → 无论成败都还原原文件。
任何一条"回退后测试仍绿"都说明该测试没有真正锁住行为（假绿）。

覆盖的 5 条语义：
  ① 中间批次不收口（回退成"第一批完成就 completed"= 静默截断）
  ② 产物按批隔离（回退成清场清根目录 = 删掉前序所有批）
  ③ 失败批显式上报（回退成静默跳过 = 交付残缺而不告知）
  ④ 分批状态按指纹重开（回退成不校验 = 页范围错位）
  ⑤ batch_pages 上限钳制（回退成不钳制 = 分批退化成一次性全量）
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MAIN = ROOT / "main.py"
PY = r"C:/Users/GY24428/.workbuddy/binaries/python/envs/cubesandbox-verify/Scripts/python.exe"

CASES = [
    (
        "① 中间批次直接收口（静默截断）",
        """    total = state.get("batch_count") or 0
    nxt = _batch_current_index(state)
    if nxt is None:""",
        """    total = state.get("batch_count") or 0
    nxt = _batch_current_index(state)
    if True:""",
        "test_batch_mode_advances_one_batch_per_call",
    ),
    (
        "② 各批共用根产物（前序批被覆盖）",
        '        out_json = "%s/result.json" % batch_dir',
        '        out_json = "%s/result.json" % _OCR_WORKDIR',
        "test_batch_artifacts_are_isolated_per_batch",
    ),
    (
        "③ 失败批静默跳过",
        """    for item in summary["batches_failed"]:
        warnings.append(""",
        """    for item in []:
        warnings.append(""",
        "test_batch_failure_is_skipped_and_reported",
    ),
    (
        "④ 分批状态不校验指纹",
        '        if isinstance(cand, dict) and cand.get("fingerprint") == fingerprint:',
        '        if isinstance(cand, dict) and True:',
        "test_batch_state_restarts_when_fingerprint_changes",
    ),
    (
        "⑤ batch_pages 不钳制",
        "    batch_pages = max(1, min(batch_pages, _OCR_BATCH_PAGES_MAX))",
        "    batch_pages = max(1, batch_pages)",
        "test_batch_pages_param_is_declared_and_clamped",
    ),
]

fails = 0
for label, old, new, test in CASES:
    backup = MAIN.with_suffix(".py.negbak")
    shutil.copy2(MAIN, backup)
    try:
        text = MAIN.read_text(encoding="utf-8")
        if old not in text:
            print("[SKIP] %s：锚点未命中\n  %r" % (label, old[:100]))
            fails += 1
            continue
        MAIN.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
        r = subprocess.run(
            [PY, "-m", "pytest", "tests/test_ocr_pdf_to_text.py::%s" % test, "-q"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        if r.returncode == 0:
            print("[FAIL] %s：回退后 %s 仍然通过（该断言锁不住行为）" % (label, test))
            fails += 1
        else:
            print("[OK]   %s：%s 如期变红" % (label, test))
    finally:
        shutil.copy2(backup, MAIN)
        backup.unlink()

print()
if fails:
    print("负向验证未通过：%d 条" % fails)
    sys.exit(1)
print("负向验证通过：%d/%d 条回退均被捕获" % (len(CASES), len(CASES)))
