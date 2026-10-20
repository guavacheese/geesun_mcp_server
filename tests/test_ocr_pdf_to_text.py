# -*- coding: utf-8 -*-
"""ocr_pdf_to_text 的纯函数与接线断言。

为什么用 AST 提取而不是 `import main`：main.py 顶层 import fastmcp /
e2b_code_interpreter，测试环境不装这些重依赖。纯函数用 AST 摘出来单独编译即可，
接线断言直接在源码 AST 上做 —— 后者比 import 更严格（能验证装饰器与调用点）。
"""
import ast
import json
import pathlib
import re
import types

import pytest

MAIN_PY = pathlib.Path(__file__).resolve().parent.parent / "main.py"


def _src() -> str:
    return MAIN_PY.read_text(encoding="utf-8")


def _load_func(name: str):
    """从 main.py 里摘出顶层函数并编译成可调用对象（不执行模块顶层代码）。"""
    tree = ast.parse(_src())
    node = None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            node = n
            break
    assert node is not None, f"main.py 中找不到顶层函数 {name}"
    mod = ast.Module(body=[node], type_ignores=[])
    # 只 exec 函数体，模块级 import 不会执行 —— 这里补上被测函数用到的名字
    ns: dict = {"json": json, "os": __import__("os"), "re": __import__("re")}
    exec(compile(mod, str(MAIN_PY), "exec"), ns)  # noqa: S102
    return ns[name]


# ─────────────────────────── 纯函数语义 ───────────────────────────

parse = _load_func("_parse_page_spec")


@pytest.mark.parametrize(
    "spec, count, expect, err_contains",
    [
        ("all", 262, None, None),
        ("", 262, None, None),
        ("*", 262, None, None),
        ("3", 262, [2], None),
        ("1-5", 262, [0, 1, 2, 3, 4], None),
        ("3,7,9", 262, [2, 6, 8], None),
        ("1-3,8", 262, [0, 1, 2, 7], None),
        (" 5 , 2 ", 262, [1, 4], None),          # 空格 + 乱序 → 去重升序
        ("5,5,5", 262, [4], None),               # 去重
        ("2-2", 262, [1], None),                 # 单元素范围
        ("0", 262, None, "从 1 开始"),
        ("263", 262, None, "超出总页数"),
        ("5-3", 262, None, "起点大于终点"),
        ("abc", 262, None, "页号无法解析"),
        ("1-a", 262, None, "页范围无法解析"),
        (",", 262, None, "页选择为空"),
    ],
)
def test_parse_page_spec(spec, count, expect, err_contains):
    pages, err = parse(spec, count)
    if expect is None and err_contains is None:
        assert pages is None and err is None
        return
    if err_contains:
        assert pages is None and err_contains in (err or "")
        return
    assert err is None, err
    assert pages == expect


def test_parse_page_spec_page_count_unknown_allows_any():
    """page_count=0（拿不到页数）时不做越界判定，但仍然归一化。"""
    pages, err = parse("999", 0)
    assert err is None and pages == [998]


# ─────────────────────────── 接线断言 ───────────────────────────
# 教训来源：fcb3c06 那次改逻辑时调用点没跟着改，7 条用例里 6 条照绿，
# 只有 AST 断言抓到。凡"定义在 A、必须被 B 调用"的契约，一律配 AST 断言。


def _func_node(tree, name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def test_tool_is_registered_as_mcp_tool():
    tree = ast.parse(_src())
    node = _func_node(tree, "ocr_pdf_to_text")
    assert node is not None, "ocr_pdf_to_text 未定义"
    decs = []
    for d in node.decorator_list:
        if isinstance(d, ast.Call):
            decs.append(ast.unparse(d.func))
        else:
            decs.append(ast.unparse(d))
    assert "mcp.tool" in decs, f"缺少 @mcp.tool() 装饰器，实际: {decs}（不进 tool list 模型就看不到）"


def test_tool_calls_sandbox_ocr_entry_not_own_impl():
    """必须复用 /opt/ocr/ocr_run.py，不得自己实现推理。"""
    src = _src()
    tree = ast.parse(src)
    node = _func_node(tree, "ocr_pdf_to_text")
    body = ast.get_source_segment(src, node)
    assert "_OCR_ENTRY" in body, "工具未引用 _OCR_ENTRY（必须走沙箱封装）"
    assert "upload_to_sandbox" in body, "工具未复用既有上传链路（DLP 解密在它里面）"


def test_no_direct_ocr_library_import_in_tool():
    """工具体内不得 import paddleocr / pytesseract —— 那正是四次事故的写法。"""
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    body = ast.get_source_segment(src, node)
    for banned in ("import paddleocr", "from paddleocr", "import pytesseract", "from pytesseract"):
        assert banned not in body, f"工具体内出现 {banned!r}，必须交给沙箱封装"


def test_entry_and_workdir_constants_present():
    """执行入口 = **MCP 下发到工作目录的副本**；镜像内那份只作模板锚点。

    2026-10-20 改造：入口从 `/opt/ocr/ocr_run.py`（镜像烘焙）改为
    `/home/user/_ocr/ocr_run.py`（每次启动任务前由 MCP 下发）。理由：
      1) 「加一行逐页进度」不值得重建 OCR 镜像（要 ctl 建模板 + 过
         artifact digest 缓存，慢且易踩坑）；
      2) 两处独立演进必然漂移，漂移表现就是「OCR 在跑但 pages_ocred 恒 0」。
    镜像内路径必须单独命名（_OCR_IMAGE_ENTRY），S2 预检用它判断
    「这是不是一个 OCR 模板」，不得再当执行入口。
    """
    src = _src()
    assert re.search(r'^_OCR_WORKDIR\s*=\s*"/home/user/_ocr"', src, re.M)
    assert re.search(
        r'^_OCR_ENTRY\s*=\s*"%s/ocr_run\.py"\s*%\s*_OCR_WORKDIR', src, re.M
    ), "_OCR_ENTRY 必须指向下发副本（'%s/ocr_run.py' % _OCR_WORKDIR）"
    assert re.search(
        r'^_OCR_IMAGE_ENTRY\s*=\s*"/opt/ocr/ocr_run\.py"', src, re.M
    ), "镜像内路径必须单独命名（_OCR_IMAGE_ENTRY），只作 S2 预检锚点"
    assert re.search(r"^_OCR_RUNNER_LOCAL\s*=", src, re.M), "缺少本地 payload 路径"


def test_precheck_echoes_ok():
    """precheck 命令必须带 `&& echo OK`。

    血泪来源（2026-10-17 实测）：判据读的是 stdout 里有没有 OK，而 grep 命中后
    输出的是 `__version__ = "1.1.0"`，不含 OK。漏掉 echo 会让「文件明明在」
    被误报成「沙箱内缺少 ocr_run.py（模板过旧）」，白白多跑一轮构建部署。
    """
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    body = ast.get_source_segment(src, node)
    assert "&& echo OK" in body, "precheck 缺少 `&& echo OK`，判定将恒为失败"


def test_first_json_value_tolerates_polluted_stdout():
    """真实事故复刻：paddleocr 往 stdout 打了 `<Response [404]>`，正文 JSON 在后面。"""
    f = _load_func("_first_json_value")
    polluted = (
        "<Response [404]>\n<Response [404]>\n"
        '[{"page": 1, "input": "x.png", "ok": true, "error": "", "texts": ["CST"],'
        ' "scores": [0.99], "polys": []}]\n'
    )
    got = f(polluted)
    assert isinstance(got, list) and got[0]["page"] == 1
    assert got[0]["texts"] == ["CST"]


def test_first_json_value_edge_cases():
    f = _load_func("_first_json_value")
    assert f("") is None
    assert f(None) is None
    assert f("no json here") is None
    assert f('{"a": 1}') == {"a": 1}
    # 纯标量数组不是结果集，按设计跳过（`[404]` 就是这种噪声）
    assert f("log line\n[1, 2]\ntrailing garbage") is None
    # 前面有 `{` 但不成 JSON → 继续往后找真正的结果数组
    assert f('{not json\n[{"page": 1}]') == [{"page": 1}]
    # 混合场景：噪声 `[404]` 在前，真结果在后
    assert f("[404] then [{\"page\": 2}]") == [{"page": 2}]


def test_first_json_value_used_by_tool():
    """接线：结果收集必须调 _first_json_value，而不是裸 json.loads。

    2026-10-19 submit-and-poll 改造后，收集逻辑搬到 _ocr_collect（三条路径
    汇流：轮询完成 / 指纹复用 / 续查时进程已结束），锚点随之更新——断言的是
    「收集函数走容错解析」这个不变量，不是某个具体函数名。
    """
    src = _src()
    node = _func_node(ast.parse(src), "_ocr_collect")
    body = ast.get_source_segment(src, node)
    assert "_first_json_value" in body, "解析必须走容错函数（第三方库会污染 stdout）"


def test_return_contract_fields():
    """返回值必须带 summary 字段且**不含全文**字段（262 页全文会撑爆上下文）。

    2026-10-19 submit-and-poll 改造后返回结构分两处：completed 契约在
    _ocr_collect，running 契约在 ocr_pdf_to_text——并集检查。
    """
    src = _src()
    tree = ast.parse(src)
    body = ""
    for name in ("ocr_pdf_to_text", "_ocr_collect"):
        node = _func_node(tree, name)
        body += ast.get_source_segment(src, node) or ""
    for key in ("pages_total", "pages_ocred", "pages_failed", "total_texts",
                "payload_json", "payload_txt", "preview", "warnings"):
        assert f'"{key}"' in body, f"返回契约缺 {key}"
    assert '"status"' in body, "submit-and-poll 契约缺 status 字段"
    assert '"progress_hint"' in body, "running 分支缺 progress_hint"
    assert '"text"' not in body.replace('"texts"', ""), "返回值不应内联全文（会撑爆上下文）"


# ────────────────── submit-and-poll 契约（2026-10-19）──────────────────


def test_default_pages_is_not_all():
    """pages 默认值不得是 "all"。

    事故 653d8515：模型没传 pages ⇒ 262 页全量 OCR ≈110 分钟（30s/页实测），
    而旧预算 min(3600, n*25) 封顶 60 分钟必砍。默认改 20 页后，模型即使
    完全不传参也落在"一次调用能做完"的区间内。
    """
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    seg = ast.get_source_segment(src, node)
    m = re.search(r"pages:\s*str\s*=\s*\"([^\"]+)\"", seg)
    assert m, "找不到 pages 参数声明"
    assert m.group(1) != "all", (
        f'pages 默认值又变回 "all"——全量 OCR 会重演 653d8515 的预算绞杀'
    )


def test_file_path_optional_with_either_or_guard():
    """file_path 必须可选（与 sandbox_pdf_path 二选一），且有 either/or 校验。

    2026-10-19 e2e 实锤：docstring 明说"PDF 已在沙箱里就直接给沙箱路径"，
    但 file_path 是必填位——模型照文档只传 sandbox_pdf_path，被 pydantic
    missing_argument 打回（ToolException 回灌自纠，浪费整轮）。契约必须
    与文档一致：要么 file_path（自动解密上传），要么 sandbox_pdf_path。
    """
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    seg = ast.get_source_segment(src, node)
    m = re.search(r"file_path:\s*str\s*\|\s*None\s*=\s*None", seg)
    assert m, (
        "file_path 必须声明为 'str | None = None'——否则只传 sandbox_pdf_path "
        "的调用会被 schema 校验拒绝（missing_argument）"
    )
    # either/or 校验必须在函数体内，且两个分支都判
    assert re.search(r"not sandbox_pdf_path and not file_path", seg), (
        "缺少 either/or 校验：file_path 与 sandbox_pdf_path 都为空时必须报错"
    )


def _code_only(seg: str) -> str:
    """剥掉注释行——断言"代码里不许出现 X"时，解释为什么不许用 X 的
    注释本身不该被判违规（test_background_launch_uses_done_flag 踩过）。
    """
    return "\n".join(
        l for l in seg.splitlines() if not l.strip().startswith("#")
    )


def test_background_launch_uses_done_flag_not_pid_liveness():
    """判活必须用 done.flag 文件，不得用 PID/kill -0。

    2026-10-19 bgtest4 探针实锤：e2b 每次 execute 是独立 PID namespace，
    跨 execute 会话 kill -0 对已死进程恒报 ALIVE（自然退出 5s 后仍"活"），
    且镜像里 ps/pgrep 都没装。文件系统才是跨 namespace 的可靠信号源。
    """
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    seg = _code_only(ast.get_source_segment(src, node))
    assert "done.flag" in seg, "后台任务必须写 done.flag 作为完成信号"
    assert "kill -0" not in seg, (
        "kill -0 判活已证实不可靠（PID namespace 隔离），改用文件信号"
    )
    assert "nohup" in seg, "OCR 必须 nohup 后台化，否则 commands.run 会同步等到底"
    # 启动命令必须把 shell 立即返回（& 后台化），否则退化成同步等待
    assert re.search(r"&\s*echo\s+BG_STARTED", seg), (
        "后台启动命令必须以 '& echo BG_STARTED' 收尾，确保秒级返回"
    )


def test_wait_seconds_clamped_to_mcp_client_hard_limit():
    """wait_seconds 必须钳制，且上限等于 MCP 客户端的 httpx read 超时（300s）。

    mcp SDK streamablehttp_client 用同一个 httpx.AsyncClient 发 POST，
    timeout=httpx.Timeout(t, read=sse_read_timeout)，sse_read_timeout
    默认 60*5=300s（streamable_http.py:104/711）——工具等更久没人收，
    POST 到 300s 就被客户端 ReadTimeout 掐断。上限必须与之对齐。
    """
    src = _src()
    m = re.search(r"^_OCR_WAIT_MAX\s*=\s*(\d+)", src, re.M)
    assert m, "找不到 _OCR_WAIT_MAX 常量"
    assert int(m.group(1)) == 300, (
        f"_OCR_WAIT_MAX={m.group(1)} 与 MCP 客户端 300s read 超时不一致——"
        "等更久会被客户端凭空掐断，且无法把结果交回去"
    )
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    seg = ast.get_source_segment(src, node)
    assert "max(0, min(_OCR_WAIT_MAX, int(wait_seconds)))" in seg, (
        "wait_seconds 必须钳制到 [0, _OCR_WAIT_MAX]"
    )


def test_fingerprint_guard_before_result_reuse():
    """复用已有结果前必须比对参数指纹（src/pages/dpi）。

    没有指纹门禁，pages="1-5" 之后用 pages="10-20" 续调会读到第一批的
    产物——语义错误比浪费一次 OCR 严重得多。
    """
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    seg = ast.get_source_segment(src, node)
    assert "fingerprint" in seg, "必须构造参数指纹"
    assert "prev_fp == fingerprint" in seg, (
        "复用结果前必须比对指纹相等，否则会返回别的页范围/DPI 的产物"
    )
    # 指纹必须含 pages 与 dpi，只含路径不够
    fp_line = next(
        (l for l in seg.splitlines() if "fingerprint =" in l and "prev_fp" not in l),
        "",
    )
    assert "pages" in fp_line and "dpi" in fp_line, (
        f"指纹必须含 pages 与 dpi：{fp_line.strip()}"
    )


def test_wait_loop_has_deadline_exit():
    """轮询循环必须有 deadline 出口（返回 status=running 让调用方续查）。

    没有出口就是新的同步等——把 653d8515 的 60 分钟原样搬进轮询循环。
    AST 语义断言（不是字符串存在性）：轮询循环里必须有一个 if，其条件
    是**裸的**时间比较（不是 `False and x >= deadline` 这种被中和的写法），
    且体内 return running 态。字符串版锚点曾被"中和判定"变体骗过。
    """
    src = _src()
    tree = ast.parse(src)
    node = _func_node(tree, "ocr_pdf_to_text")
    assert node is not None

    guarded = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.While):
            continue
        for stmt in sub.body:
            if not isinstance(stmt, ast.If):
                continue
            # 条件必须是裸 Compare：BoolOp 包裹意味着可能被常量短路中和
            if not isinstance(stmt.test, ast.Compare):
                continue
            cmp_src = ast.get_source_segment(src, stmt.test) or ""
            if "deadline" not in cmp_src:
                continue
            body_src = ast.get_source_segment(src, stmt) or ""
            assert '"status": "running"' in body_src, (
                "deadline 出口必须返回 running 态让调用方续查，否则工具又变成同步等"
            )
            guarded.append(cmp_src.strip())

    assert guarded, (
        "轮询循环里找不到以 deadline 比较为条件的出口——"
        "工具会无限期等待，重演 653d8515 的挂死形态"
    )


# ────────── 陈旧 done.flag / 指纹错配回归（2026-10-19 e2e T4 实锤）──────────


async def _stub_ensure_runner(sandbox_id):
    """默认桩：「下发已成功」。把下发动作与 OCR 状态机解耦——
    真实下发的字节级验证见 test_ensure_ocr_runner_writes_repo_payload。"""
    return {"success": True, "version": "1.2.0-test", "error": None}


def _load_ocr_module():
    """把 ocr_pdf_to_text 及其依赖（_ocr_collect/_fail/_parse_page_spec/常量）
    摘进同一命名空间，返回可 await 调用的真函数——不 import main（重依赖）。"""
    src = _src()
    tree = ast.parse(src)
    want_names = {"ocr_pdf_to_text", "_ocr_collect", "_fail", "_parse_page_spec",
                  "_first_json_value", "_parse_ocr_progress"}
    body = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in want_names]
    assert len(body) == len(want_names), "依赖函数不全：%s" % [n.name for n in body]
    for n in body:
        n.decorator_list = []  # 剥掉 @mcp.tool() 等装饰器——ns 里没有 mcp
    # 模块级常量按需注入（_OCR_ENTRY/_OCR_WORKDIR/_OCR_DPI_*/_OCR_WAIT_MAX/
    # _OCR_POLL_INTERVAL/_OCR_HELPER_PY/_OCR_PREVIEW_CHARS）
    consts = {}
    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        # 兼容 `A, B = 1, 2` 元组赋值（_OCR_DPI_MIN/_OCR_DPI_MAX）
        targets = n.targets[0].elts if isinstance(n.targets[0], ast.Tuple) else [n.targets[0]]
        values = (n.value.elts if isinstance(n.value, ast.Tuple) else [n.value] * len(targets))
        for tgt, val in zip(targets, values):
            if isinstance(tgt, ast.Name) and tgt.id.startswith("_OCR_"):
                try:
                    consts[tgt.id] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    pass  # _OCR_HELPER_PY 等 literal_eval 吃不下就跳过
    ns = {"json": json, "os": __import__("os"), "re": __import__("re"), **consts}
    # 派生常量（`"%s/ocr_run.py" % _OCR_WORKDIR` 这类 BinOp，literal_eval 吃不下）
    # 用与 main.py 相同的表达式补，避免测试常量与生产脱钩。
    workdir = ns.get("_OCR_WORKDIR", "/home/user/_ocr")
    ns["_OCR_ENTRY"] = "%s/ocr_run.py" % workdir
    ns["_OCR_IMAGE_ENTRY"] = "/opt/ocr/ocr_run.py"
    # 下发动作在单测里用 stub 顶替（真实下发的测试见
    # test_ocr_runner_shipped_*，那里 mock 掉 _sandbox_write）。
    ns["_ensure_ocr_runner"] = _stub_ensure_runner
    mod = ast.Module(body=body, type_ignores=[])
    exec(compile(mod, str(MAIN_PY), "exec"), ns)  # noqa: S102
    return ns


class _FakeSandbox:
    """按命令内容分发的假沙箱，重放 T4 现场：

    done.flag 里留着上一次（pages=1-2）任务的退出码 0，ocr.meta 里是旧指纹，
    result.json 是旧任务的两页产物。本次调用 pages="1"。
    """

    def __init__(self):
        self.commands: list[str] = []
        self.meta = "/home/user/_ocr/e2e.pdf|1-2|150"  # 旧指纹
        self.done = "0"                                 # 陈旧完成信号
        self.result_json = json.dumps(
            [{"page": 1, "ok": True, "texts": ["OLD-1"]},
             {"page": 2, "ok": True, "texts": ["OLD-2"]}]
        )
        self.fresh_result = json.dumps([{"page": 1, "ok": True, "texts": ["NEW-1"]}])

    async def run(self, sandbox_id, cmd, timeout=None):
        self.commands.append(cmd)
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:  # S2 环境预检
            return {"success": True, "stdout": '__version__ = "1.1.0"\nOK\n', "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):  # 页数探测
            return {"success": True, "stdout": "2\n", "stderr": ""}
        if "_page.py /home/user/_ocr/e2e.pdf 0" in cmd:  # 按页拆分
            return {"success": True, "stdout": "SPLIT_OK\n", "stderr": ""}
        if cmd.startswith("cat ") and "done.flag" in cmd:
            return {"success": True, "stdout": self.done + "\n", "stderr": ""}
        if cmd.startswith("cat ") and "ocr.meta" in cmd:
            return {"success": True, "stdout": self.meta + "\n", "stderr": ""}
        if cmd.startswith("test -f ") and "result.json" in cmd:
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "BG_STARTED" in cmd:  # 后台启动：新任务秒完
            self.meta = "/home/user/_ocr/e2e.pdf|1|150"  # meta 先写后启动
            return {"success": True, "stdout": "BG_STARTED\n", "stderr": ""}
        return {"success": True, "stdout": "", "stderr": ""}

    async def write(self, sandbox_id, path, data):
        if path.endswith("ocr.meta"):
            self.meta = data.decode("utf-8")
        return {"success": True, "error": None}

    async def read(self, sandbox_id, path):
        # 启动后才换新产物：没收旧 flag 就直接读的（修复前）会拿到两页旧数据
        fresh = any("BG_STARTED" in c for c in self.commands)
        return {"success": True, "data": (self.fresh_result if fresh else self.result_json).encode(),
                "error": None}


def test_stale_done_flag_with_old_fingerprint_forces_relaunch():
    """done.flag 在但指纹不匹配 ⇒ 必须作废重跑，不得收旧产物。

    2026-10-19 e2e T4：pages="1" 收到 pages="1-2" 任务留下的 done.flag=0，
    旧代码跳过指纹判定直接收集 ⇒ 读到 2 条旧记录，want=[0] 映射时
    IndexError: list index out of range（ToolException 回灌，整轮报废）。
    """
    import asyncio as _asyncio

    ns = _load_ocr_module()
    sb = _FakeSandbox()
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["upload_to_sandbox"] = None  # 本路径不走上传
    ns["_resolve_host_path"] = None
    ns["_upload_precheck"] = None

    r = _asyncio.run(ns["ocr_pdf_to_text"](
        sandbox_id="sbx-test",
        sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
        pages="1",
        dpi=150,
        wait_seconds=3,
    ))

    assert any("BG_STARTED" in c for c in sb.commands), (
        "陈旧 done.flag + 旧指纹时必须重新启动后台任务（BG_STARTED）——"
        "否则收的是上一次参数组合的产物"
    )
    assert r["success"] is True, r
    assert r["status"] == "completed", r
    assert r["pages_ocred"] == 1, "应只收新任务（pages=1）的 1 页结果，实际：%s" % r
    assert r["total_texts"] == 1, r
    assert "NEW-1" in r["preview"], "预览应来自新任务产物：%s" % r["preview"][:120]


def test_ocr_collect_mapping_never_index_errors():
    """_ocr_collect 的页号映射对短 want/长 recs 不得 IndexError（纵深防御）。"""
    import asyncio as _asyncio

    ns = _load_ocr_module()
    recs = json.dumps([{"page": 1, "ok": True, "texts": ["A"]},
                       {"page": 2, "ok": True, "texts": ["B"]}])

    async def _read(sandbox_id, path):
        return {"success": True, "data": recs.encode(), "error": None}

    async def _write(sandbox_id, path, data):
        return {"success": True, "error": None}

    ns["_sandbox_read"] = _read
    ns["_sandbox_write"] = _write
    r = _asyncio.run(ns["_ocr_collect"](
        "sbx", "/home/user/_ocr/result.json", "/home/user/_ocr/err.log",
        2, [0], "1", 150, 0.0, [], 1,
    ))
    assert r["success"] is True, r
    assert "===== page 1 =====" in r["preview"] and "===== page 2 =====" in r["preview"]


def test_fingerprint_check_precedes_done_flag_branch():
    """AST 顺序锚点：读指纹必须发生在 done.flag 分支判定之前。

    防止"把 _read_fingerprint 挪回 if done_code is None 里"的回归——
    那正是 T4 IndexError 的结构性根因（flag 在就永不验指纹）。
    """
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    seg = _code_only(ast.get_source_segment(src, node))
    i_fp = seg.find("prev_fp = await _read_fingerprint()")
    i_branch = seg.find("if done_code is None:")
    assert i_fp != -1, "找不到 prev_fp 读取"
    assert i_branch != -1, "找不到 done.flag 分支"
    assert i_fp < i_branch, (
        "指纹读取被挪进了 done.flag 分支——完成信号存在时将永不校验指纹，"
        "重演 T4 收旧产物 IndexError"
    )


# ────────── P0/P1（2026-10-20 add0229b）：后台启动无效 + 进度恒 0 ──────────

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNNER_PY = REPO_ROOT / "sandbox" / "ocr_run.py"


def test_cleanup_and_launch_are_separate_commands():
    """清场与启动必须是**两条独立的 commands.run**。

    2026-10-20 add0229b 实锤：合成一条 `A && B && C & echo D` 时 bash 会把
    整个 and-list 后台化（等价于 `{ A && B && C; } & echo D`）。外层子 shell
    继承了 commands.run 的 stdout 管道，且要等 C（nohup 前台执行）跑完才
    退出 ⇒ commands.run 卡满 timeout 抛 TimeoutException，e2b 收会话时把
    python3 连锅端；而 A/B 里的 rm -f 早已执行成功 ⇒ 旧产物被清空、新产物
    永不生成。外部表现 = 续查 5 次 / 22 分钟 / pages_ocred 恒 0。

    沙箱 A/B（_bg_ab2_inner.py，真实 OCR 命令）：`&&` 链 25s 卡满被 Timeout
    杀；拆开后 0.0s 返回 BG_STARTED，python3 存活并在真跑。
    """
    src = _src()
    seg = _code_only(ast.get_source_segment(src, _func_node(ast.parse(src), "ocr_pdf_to_text")))
    assert "mkdir -p %s; rm -f %s %s %s" in seg, (
        "清场命令必须用 `;` 分隔（mkdir ...; rm -f ...）——用 `&&` 会把后续命令"
        "串成 and-list 一起被后台化，commands.run 卡到超时并杀死 OCR"
    )
    launch_lines = [ln for ln in seg.splitlines() if "nohup bash -c" in ln]
    assert launch_lines, "找不到 nohup 启动命令"
    for ln in launch_lines:
        assert "mkdir" not in ln and "rm -f" not in ln, (
            "启动命令里不得再合并 mkdir/rm —— 那正是 and-list 后台化的来源"
        )


def test_launch_visibility_probe_exists():
    """必须有「启动可见性」判据：既没 BG_STARTED 也没 err.log 内容 ⇒ 显式失败。

    否则就是 add0229b 的静默：工具连回 5 次 status=running + pages_ocred=0，
    模型判定卡死 → 换策略解析文本层 → 最终被循环检测硬停。
    """
    src = _src()
    seg = _code_only(ast.get_source_segment(src, _func_node(ast.parse(src), "ocr_pdf_to_text")))
    assert "BG_STARTED" in seg
    assert "_OCR_LAUNCH_PROBE_TRIES" in seg, "缺少启动探针轮次常量"
    assert 'return _fail(\n                    "launch"' in seg or '"launch",' in seg, \
        "启动失败必须返回 step=launch 的显式错误"
    assert "wc -c <" in seg, "探针必须检查 err.log 是否有内容（进程落首行的证据）"


class _NoLaunchSandbox:
    """后台任务从未起来的现场：launch 命令 stdout 空（BG_STARTED 丢失），
    之后既无 done.flag 也无 err.log 内容——add0229b 的 22 分钟空等起点。"""

    def __init__(self):
        self.commands: list[str] = []

    async def run(self, sandbox_id, cmd, timeout=None):
        self.commands.append(cmd)
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):  # S1 定位校验
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:  # S2 环境预检
            return {"success": True, "stdout": '__version__ = "1.1.0"\nOK\n', "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):  # S3 页数探测
            return {"success": True, "stdout": "5\n", "stderr": ""}
        if "nohup bash -c" in cmd:  # 启动：BG_STARTED 丢失 + 无 stderr
            return {"success": True, "stdout": "", "stderr": ""}
        if "wc -c <" in cmd:  # err.log 大小探针：恒 0
            return {"success": True, "stdout": "0\n", "stderr": ""}
        return {"success": True, "stdout": "", "stderr": ""}

    async def write(self, sandbox_id, path, data):
        return {"success": True, "error": None}

    async def read(self, sandbox_id, path):
        return {"success": False, "data": b"", "error": "not found"}


def test_launch_failure_is_reported_not_silently_running():
    """启动不起来必须显式失败，不得静默进入 running 让模型空等。"""
    import asyncio as _asyncio

    ns = _load_ocr_module()
    sb = _NoLaunchSandbox()
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None
    ns["_OCR_LAUNCH_PROBE_TRIES"] = 2  # 别让单测真等 10s
    ns["_OCR_LAUNCH_PROBE_INTERVAL"] = 0

    r = _asyncio.run(ns["ocr_pdf_to_text"](
        sandbox_id="sbx-test",
        sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
        pages="all",
        dpi=150,
        wait_seconds=1,
    ))
    assert r["success"] is False, r
    assert r["step"] == "launch", "启动失败必须报 step=launch，实际：%s" % r
    assert "未能启动" in r["error"], r["error"]


class _ProgressSandbox:
    """OCR 真在跑的现场：BG_STARTED 正常，err.log 里已有逐页进度行。"""

    def __init__(self, err_text: str):
        self.err_text = err_text
        self.commands: list[str] = []

    async def run(self, sandbox_id, cmd, timeout=None):
        self.commands.append(cmd)
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):  # S1 定位校验
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:
            return {"success": True, "stdout": '__version__ = "1.1.0"\nOK\n', "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "20\n", "stderr": ""}
        if "nohup bash -c" in cmd:
            return {"success": True, "stdout": "BG_STARTED\n", "stderr": ""}
        if cmd.startswith("tail -n"):
            return {"success": True, "stdout": self.err_text, "stderr": ""}
        return {"success": True, "stdout": "", "stderr": ""}

    async def write(self, sandbox_id, path, data):
        return {"success": True, "error": None}

    async def read(self, sandbox_id, path):
        return {"success": False, "data": b"", "error": "not found"}


def test_running_result_carries_sandbox_path_and_real_progress():
    """running 返回必须带沙箱路径 + 真实进度（不是恒 0）。

    add0229b：模型 `ls /home/user` 没看到 `_ocr/` 子目录 ⇒ 结论"PDF 没上传"
    ⇒ 重传 268MB。返回值里给 sandbox_path/sandbox_dir、提示里写"不要重新上传"
    就能避免这类"合理误判"。
    """
    import asyncio as _asyncio

    err = "\n".join([
        "[ocr_run] env MemTotal=7952MB dpi=150 det_limit=1500 mkldnn=False",
        "[ocr_run] render start file=e2e.pdf total=20 dpi=150",
        "[ocr_run] render 20/20",
        "[ocr_run] ocr start pages=20",
        "[ocr_run] page 7/20 ok texts=48 elapsed=31.2s",
    ])
    ns = _load_ocr_module()
    sb = _ProgressSandbox(err)
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None

    r = _asyncio.run(ns["ocr_pdf_to_text"](
        sandbox_id="sbx-test",
        sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
        pages="all",
        dpi=150,
        wait_seconds=0,  # 立即进 running 分支
    ))
    assert r["status"] == "running", r
    assert r["pages_ocred"] == 7, "进度必须来自 err.log 逐页行，实际：%s" % r["pages_ocred"]
    assert r["pages_ocred_total"] == 20, r
    assert r["sandbox_path"] == "/home/user/_ocr/e2e.pdf", r
    assert r["sandbox_dir"] == "/home/user/_ocr", r
    assert "正在逐页 OCR" in r["progress_hint"], r["progress_hint"]
    assert "不要重新上传" in r["progress_hint"], r["progress_hint"]


def test_parse_ocr_progress_reads_page_lines():
    """进度解析：取最后一行的 i/n；渲染阶段不计入页进度。"""
    ns = _load_ocr_module()
    parse = ns["_parse_ocr_progress"]
    err = "\n".join([
        "[ocr_run] env MemTotal=7952MB dpi=150 det_limit=1500 mkldnn=False",
        "[ocr_run] render start file=x.pdf total=262 dpi=150",
        "[ocr_run] render 10/262",
        "[ocr_run] render 262/262",
        "[ocr_run] ocr start pages=262",
        "[ocr_run] page 1/262 ok texts=48 elapsed=31.2s",
        "[ocr_run] page 2/262 ok texts=0 elapsed=29.8s",
    ])
    assert parse(err) == (2, 262), parse(err)
    # 只有渲染进度 ⇒ 页进度仍 0（total 未知），4b 据此显示"正在渲染"
    assert parse("[ocr_run] render 30/262") == (0, None)
    assert parse("") == (0, None)


def test_progress_mark_matches_sandbox_script():
    """MCP 的进度标记必须与沙箱脚本实际打印的前缀一致（双端契约）。

    两边分别是 main.py::_OCR_PROGRESS_MARK 与 sandbox/ocr_run.py 的
    PROGRESS_PREFIX + "page i/n"。改任何一边而忘另一边 ⇒ pages_ocred 恒 0
    且**不报错**——这类静默退化只能靠契约测试锁住。
    """
    ns = _load_ocr_module()
    mark = ns["_OCR_PROGRESS_MARK"]
    payload = RUNNER_PY.read_text("utf-8")
    m = re.search(r'^PROGRESS_PREFIX\s*=\s*"([^"]*)"', payload, re.M)
    assert m, "sandbox/ocr_run.py 里找不到 PROGRESS_PREFIX"
    prefix = m.group(1)
    assert mark == prefix + "page ", (
        "MCP 标记 %r 与脚本前缀 %r 不匹配（脚本打的是 '%spage i/n'）"
        % (mark, prefix, prefix)
    )
    assert '"page %d/%d %s texts=%d elapsed=%.1fs"' in payload, "逐页进度行格式变了"
    assert re.search(r'__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"', payload), "缺少版本号"


def test_ensure_ocr_runner_writes_repo_payload():
    """下发内容必须与仓库 sandbox/ocr_run.py **逐字节一致**，且版本可回显。"""
    import asyncio as _asyncio

    src = _src()
    node = _func_node(ast.parse(src), "_ensure_ocr_runner")
    assert node is not None, "_ensure_ocr_runner 未定义"
    node.decorator_list = []
    ns = {
        "_OCR_ENTRY": "/home/user/_ocr/ocr_run.py",
        "_OCR_RUNNER_LOCAL": RUNNER_PY,
        "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MAIN_PY), "exec"), ns)  # noqa: S102
    captured = {}

    async def _write(sid, path, data):
        captured["path"] = path
        captured["data"] = data
        return {"success": True, "error": None}

    ns["_sandbox_write"] = _write
    r = _asyncio.run(ns["_ensure_ocr_runner"]("sbx"))
    assert r["success"] is True, r
    assert r["version"].startswith("1.2"), "下发版本应 >= 1.2.0（带逐页进度），实际 %r" % r["version"]
    assert captured["path"] == "/home/user/_ocr/ocr_run.py"
    assert captured["data"] == RUNNER_PY.read_bytes(), "下发字节与仓库文件不一致"
