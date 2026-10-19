# -*- coding: utf-8 -*-
"""阻塞 IO 护栏（blocking-IO guard）—— 2026-10-19 653d8515 事故的回归防线。

事故根因：`async def` 工具里直接跑同步 e2b 调用（connect / files.write /
commands.run / 大文件 read）会把 MCP 单事件循环堵满整个命令时长。OCR 预算
封顶 3600s ⇒ 循环堵 60 分钟 ⇒ 客户端 SSE 300s 读超时 ⇒ 重连请求干排队
55 分钟 ⇒ 工具响应永远送不回 ⇒ agent 侧 tools/call future 永不 resolve ⇒
run 永久 "running"（僵尸 38 小时）。

参照 deer-flow 的两层防御（backend/docs/BLOCKING_IO_DETECTION.md）：
  静态层 —— AST 扫描 async 函数体里的同步 e2b 调用（本文件 test_*_static）；
  运行时层 —— 摘出真实函数 + mock e2b，验证循环在"慢同步调用"期间仍自由
            （本文件 test_*_runtime，等价 deer-flow 的 Blockbuster anchor）。

测试环境不装 fastmcp / e2b_code_interpreter（见 test_ocr_pdf_to_text.py 的
同样理由），因此被测函数用 AST 摘出后单独 exec，依赖名由本文件注入。
"""
import ast
import asyncio
import pathlib
import time

import pytest

MAIN_PY = pathlib.Path(__file__).resolve().parent.parent / "main.py"

# 事故相关的同步阻塞调用（e2b SDK 无 async 客户端，这些全是同步 HTTP/IO）
_BLOCKING_CALLS = (
    "E2BSandbox.connect",
    "sb.files.write",
    "sb.files.read",
    "sb.commands.run",
)
# async 直接体里同样禁止的同步文件 IO（大文件 read/write 堵循环与 e2b 同罪；
# 这 7 个受检函数的所有文件 IO 都应下沉 to_thread，不会误报）
_BLOCKING_FILE_IO = ("open",)


def _src() -> str:
    return MAIN_PY.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_src())


def _load_async_func(name: str):
    """从 main.py 摘出顶层 async 函数并编译（不执行模块顶层代码）。"""
    tree = _tree()
    node = None
    for n in tree.body:
        if isinstance(n, ast.AsyncFunctionDef) and n.name == name:
            node = n
            break
    assert node is not None, f"main.py 中找不到顶层 async 函数 {name}"
    mod = ast.Module(body=[node], type_ignores=[])
    ns: dict = _runtime_namespace()
    exec(compile(mod, str(MAIN_PY), "exec"), ns)  # noqa: S102
    return ns[name]


def _dotted(node: ast.AST) -> str | None:
    """把 a.b.c 属性链还原成字符串（仅限 Name/Attribute 链）。"""
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


# ─────────────────────────── 静态层（AST） ───────────────────────────

@pytest.mark.parametrize(
    "func_name",
    [
        "copy_script_to_sandbox",
        "upload_to_sandbox",
        "decrypt_and_upload_to_sandbox",
        "download_from_sandbox",
        "_sandbox_write",
        "_sandbox_run",
        "_sandbox_read",
    ],
)
def test_no_blocking_e2b_call_on_async_body(func_name):
    """含同步 e2b 调用的下沉函数，只允许经 asyncio.to_thread 被调用。

    事故形态：`async def` 里直跑同步 e2b（connect / files.write /
    commands.run / 大文件 read）把 MCP 单事件循环堵满整个命令时长。

    判定方式（比"直接体里有没有 connect"更强——后者对
    `return _sync()` 这种本地同步函数直调无感，负向验证实测漏过）：
      1. 收集 async 函数内所有同步 FunctionDef（to_thread 下沉区）；
      2. 若某个下沉函数体内含阻塞调用，则它在整个 async 函数内的每一次
         引用都必须是 asyncio.to_thread 的第一个参数；
      3. 此外，async 函数直接体（不含同步子树）里不得再出现阻塞调用。
    """
    tree = _tree()
    node = None
    for n in tree.body:
        if isinstance(n, ast.AsyncFunctionDef) and n.name == func_name:
            node = n
            break
    assert node is not None

    # ── 1. 下沉函数集合 + 各自是否含阻塞调用 ──
    sync_defs: dict[str, bool] = {}  # name -> has_blocking_call
    for sub in ast.walk(node):
        if isinstance(sub, ast.FunctionDef) and not isinstance(sub, ast.AsyncFunctionDef):
            has_blocking = any(
                isinstance(c, ast.Call) and _dotted(c.func) in _BLOCKING_CALLS
                for c in ast.walk(sub)
            )
            # 同步读文件（open().read()）同样算阻塞：大文件会把循环堵数秒
            if not has_blocking:
                has_blocking = any(
                    isinstance(c, ast.Call)
                    and _dotted(c.func) in ("open",)
                    for c in ast.walk(sub)
                )
            sync_defs[sub.name] = has_blocking

    violations: list[str] = []

    # ── 2. 每个含阻塞调用的下沉函数，其所有引用必须是 to_thread 的首参 ──
    to_thread_first_args = {
        c.args[0].id
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and _dotted(c.func) == "asyncio.to_thread"
        and c.args
        and isinstance(c.args[0], ast.Name)
    }

    for sub in ast.walk(node):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
            continue
        callee = sub.func.id
        if callee in sync_defs and sync_defs[callee] and callee not in to_thread_first_args:
            violations.append(
                f"行 {sub.lineno}: 含阻塞调用的 {callee}() 被直接调用——"
                "必须 await asyncio.to_thread({callee})".format(callee=callee)
            )

    # ── 3. async 直接体（剥掉同步子树）里不得有阻塞调用 ──
    def strip_sync_defs(body):
        out = []
        for stmt in body:
            if isinstance(stmt, ast.FunctionDef) and not isinstance(stmt, ast.AsyncFunctionDef):
                continue
            out.append(stmt)
        return out

    for stmt in strip_sync_defs(node.body):
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                dotted = _dotted(sub.func)
                if dotted in _BLOCKING_CALLS:
                    violations.append(f"行 {sub.lineno}: async 直接体直跑 {dotted}")
                elif dotted in _BLOCKING_FILE_IO:
                    violations.append(
                        f"行 {sub.lineno}: async 直接体直跑 {dotted}()——文件 IO 未下沉"
                    )

    assert not violations, (
        f"{func_name} 存在事件循环阻塞形态：\n  " + "\n  ".join(violations) + "\n"
        "请把同步段包进 asyncio.to_thread(_sync) 后再提交（653d8515 事故回归防线）。"
    )


def test_async_helpers_delegate_via_to_thread():
    """三个 helper 必须显式经过 asyncio.to_thread（防换一种阻塞写法绕过静态扫描）。"""
    tree = _tree()
    for name in ("_sandbox_write", "_sandbox_run", "_sandbox_read"):
        node = next(
            (n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name),
            None,
        )
        assert node is not None
        calls = [
            _dotted(sub.func)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call)
        ]
        assert "asyncio.to_thread" in calls, (
            f"{name} 没有走 asyncio.to_thread——事件循环自由无从保证"
        )


def test_large_file_reads_are_offloaded():
    """上传/解密链路上的全量文件 read 必须下沉（256MiB 文件同步 read 同样堵循环）。"""
    tree = _tree()
    checks = {
        "upload_to_sandbox": ["_read_content"],
        "_decrypt_file_internal": ["_read_all"],
    }
    for func_name, sync_helpers in checks.items():
        node = next(
            (n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == func_name),
            None,
        )
        assert node is not None
        # 同步下沉函数可能定义在 try 块等嵌套结构里，用 walk 全收集（排除 async 嵌套函数）
        inner_sync_names = {
            sub.name
            for sub in ast.walk(node)
            if isinstance(sub, ast.FunctionDef) and not isinstance(sub, ast.AsyncFunctionDef)
        }
        for helper in sync_helpers:
            assert helper in inner_sync_names, (
                f"{func_name} 缺少同步读文件下沉函数 {helper}——全量 read 仍在 async 体上直跑"
            )
        to_thread_args = []
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and _dotted(sub.func) == "asyncio.to_thread"
                and sub.args
                and isinstance(sub.args[0], ast.Name)
            ):
                to_thread_args.append(sub.args[0].id)
        for helper in sync_helpers:
            assert helper in to_thread_args, (
                f"{func_name} 的 {helper} 没有通过 asyncio.to_thread 调用"
            )


# ─────────────────────────── 运行时层（anchor） ───────────────────────────

def _runtime_namespace() -> dict:
    """exec 摘出函数时注入的依赖（mock e2b + 真 asyncio）。"""

    class _Files:
        def __init__(self, delay):
            self._delay = delay

        def write(self, path, content):
            time.sleep(self._delay)  # 同步 IO：直跑会堵循环
            return None

        def read(self, path, format=None):
            time.sleep(self._delay)
            return b"data"

    class _Commands:
        def __init__(self, delay):
            self._delay = delay

        def run(self, cmd, timeout=None):
            time.sleep(self._delay)

            class _Res:
                exit_code = 0
                stdout = "ok"
                stderr = ""

            return _Res()

    class _Sandbox:
        def __init__(self):
            self.files = _Files(BLOCK_DELAY)
            self.commands = _Commands(BLOCK_DELAY)

    class _E2BSandboxMock:
        @staticmethod
        def connect(sandbox_id):
            time.sleep(BLOCK_DELAY)  # connect 是同步 HTTP
            return _Sandbox()

    return {
        "asyncio": asyncio,
        "E2BSandbox": _E2BSandboxMock,
        "_e2b_env": lambda: None,
    }


#: 每次同步调用阻塞时长。取 0.3s：ticker 每 0.02s 一个 tick，
#: 循环自由 ⇒ ~15 个 tick；被堵 ⇒ 0.3s 内 0~1 个，判定余量充足。
BLOCK_DELAY = 0.3


@pytest.mark.parametrize(
    "func_name, call",
    [
        ("_sandbox_run", lambda f: f("sid-x", "echo hi")),
        ("_sandbox_write", lambda f: f("sid-x", "/tmp/x", b"data")),
        ("_sandbox_read", lambda f: f("sid-x", "/tmp/x")),
    ],
)
def test_event_loop_stays_free_during_slow_e2b_calls(func_name, call):
    """慢同步 e2b 调用期间事件循环必须保持自由（653d8515 的运行时判据）。

    若有人把 asyncio.to_thread 拆掉改成直跑，ticker 在 BLOCK_DELAY 内
    几乎收不到 tick，本测试失败——这就是 deer-flow Blockbuster anchor 的
    轻量等价物。
    """
    func = _load_async_func(func_name)

    async def _body():
        stop = asyncio.Event()
        ticks: list = []

        async def ticker():
            while not stop.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        t = asyncio.create_task(ticker())
        try:
            await call(func)
        finally:
            stop.set()
            await t
        return ticks

    ticks = asyncio.run(_body())

    assert len(ticks) >= 5, (
        f"{func_name} 执行期间循环被堵：0.3s 阻塞里只收到 {len(ticks)} 个 tick"
        "（预期 ≥5）——同步 e2b 调用又回到了事件循环上。"
    )


def test_helper_still_returns_correct_shape():
    """to_thread 改造不得破坏返回结构（_sandbox_run 的 exit_code/stdout 解析保持）。"""
    func = _load_async_func("_sandbox_run")
    r = asyncio.run(func("sid-x", "echo hi", timeout=60))
    assert r == {"success": True, "stdout": "ok", "stderr": ""}
