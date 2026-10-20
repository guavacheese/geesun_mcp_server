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
    launch = _launch_src()
    assert "nohup" in launch, "OCR 必须 nohup 后台化，否则 commands.run 会同步等到底"
    # 启动命令必须把 shell 立即返回（& 后台化），否则退化成同步等待
    assert re.search(r"&\s*echo\s+BG_STARTED", launch), (
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
                  "_first_json_value", "_parse_ocr_progress", "_ocr_task_age_seconds",
                  "_bump_silent_streak", "_build_ocr_launch_cmd",
                  # 分批（方案 B，2026-10-20）
                  "_page_span_text", "_plan_ocr_batches", "_batch_state_new",
                  "_batch_current_index", "_batch_summary", "_batch_state_load",
                  "_batch_state_save", "_batch_concat_texts", "_ocr_batch_finish",
                  "_ocr_batch_settle"}
    body = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in want_names]
    assert len(body) == len(want_names), "依赖函数不全：%s" % [n.name for n in body]
    for n in body:
        n.decorator_list = []  # 剥掉 @mcp.tool() 等装饰器——ns 里没有 mcp
    # 模块级常量按需注入（_OCR_ENTRY/_OCR_WORKDIR/_OCR_DPI_*/_OCR_WAIT_MAX/
    # _OCR_POLL_INTERVAL/_OCR_HELPER_PY/_OCR_PREVIEW_CHARS）
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.AnnAssign):
            # `_OCR_SILENT_STREAK: dict[str, int] = {}` 这类带注解的常量
            if (
                isinstance(n.target, ast.Name)
                and n.target.id.startswith("_OCR_")
                and n.value is not None
            ):
                try:
                    consts[n.target.id] = ast.literal_eval(n.value)
                except (ValueError, SyntaxError):
                    pass
            continue
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
    ns = {"json": json, "os": __import__("os"), "re": __import__("re"),
          # _build_ocr_launch_cmd 需要 shlex.quote（P0-1 引号修复）
          "shlex": __import__("shlex"),
          # logger 桩：被测函数只做日志输出，不能因缺名而崩
          "logger": types.SimpleNamespace(
              info=lambda *a, **k: None,
              warning=lambda *a, **k: None,
              error=lambda *a, **k: None,
          ),
          **consts}
    # 派生常量（`"%s/ocr_run.py" % _OCR_WORKDIR` 这类 BinOp，literal_eval 吃不下）
    # 用与 main.py 相同的表达式补，避免测试常量与生产脱钩。
    workdir = ns.get("_OCR_WORKDIR", "/home/user/_ocr")
    ns["_OCR_ENTRY"] = "%s/ocr_run.py" % workdir
    ns["_OCR_IMAGE_ENTRY"] = "/opt/ocr/ocr_run.py"
    # 分批常量同样是 BinOp（`"%s/batches" % _OCR_WORKDIR`），literal_eval 吃不下
    ns["_OCR_BATCH_DIR"] = "%s/batches" % workdir
    ns["_OCR_BATCH_STATE"] = "%s/batches/state.json" % workdir
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
        if "LINKED" in cmd or "COPIED" in cmd:  # P0-3 输入路径归一化
            return {"success": True, "stdout": "LINKED\n", "stderr": ""}
        if "wc -c <" in cmd:  # P0-2 启动可见性探针：err.log 有 start 行
            return {"success": True, "stdout": "48\n", "stderr": ""}
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


def _launch_src() -> str:
    """`_build_ocr_launch_cmd` 的源码（启动命令的唯一构造点）。"""
    src = _src()
    return _code_only(
        ast.get_source_segment(src, _func_node(ast.parse(src), "_build_ocr_launch_cmd"))
    )


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
    launch_lines = [
        ln for ln in _launch_src().splitlines() if "nohup bash -c" in ln
    ]
    assert launch_lines, "找不到 nohup 启动命令"
    for ln in launch_lines:
        assert "mkdir" not in ln and "rm -f" not in ln, (
            "启动命令里不得再合并 mkdir/rm —— 那正是 and-list 后台化的来源"
        )
    # 调用点不得把 mkdir/rm 拼进启动命令
    call = [ln for ln in seg.splitlines() if "_build_ocr_launch_cmd(" in ln]
    assert call, "启动命令必须经由 _build_ocr_launch_cmd 构造"
    assert not any("rm -f" in ln for ln in call), "启动调用点不得携带 rm -f"


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
        if "LINKED" in cmd or "COPIED" in cmd:  # P0-3 输入路径归一化
            return {"success": True, "stdout": "LINKED\n", "stderr": ""}
        if "wc -c <" in cmd:  # P0-2 启动可见性探针：err.log 有 start 行
            return {"success": True, "stdout": "48\n", "stderr": ""}
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):  # S1 定位校验
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:
            return {"success": True, "stdout": '__version__ = "1.1.0"\nOK\n', "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "20\n", "stderr": ""}
        if "SPLIT_OK" in cmd:  # 按页拆分：末尾用 test -s 做真实成功判据
            return {"success": True, "stdout": "SPLIT_OK\n", "stderr": ""}
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


class _ResumeSandbox:
    """续查现场（add0229b 的真实形态）：meta 已是本次指纹、done.flag 没有、
    result.json 被 shell 重定向创建成 0 字节，err.log 仍在增长（任务在跑）。

    age 控制 mtime 距今秒数：<= _OCR_LIVENESS_SEC 表示在跑；-1 表示没有 err.log。
    """

    def __init__(self, age: int, err_text: str = "", result_nonempty: bool = False):
        self.age = age
        self.err_text = err_text
        self.result_nonempty = result_nonempty
        self.commands: list[str] = []

    async def run(self, sandbox_id, cmd, timeout=None):
        self.commands.append(cmd)
        if "LINKED" in cmd or "COPIED" in cmd:  # P0-3 输入路径归一化
            return {"success": True, "stdout": "LINKED\n", "stderr": ""}
        if "wc -c <" in cmd:  # P0-2 启动可见性探针
            return {"success": True, "stdout": "48\n", "stderr": ""}
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:
            return {"success": True, "stdout": '__version__ = "1.1.0"\nOK\n', "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "20\n", "stderr": ""}
        if "stat -c %Y" in cmd:  # 判活探针
            return {"success": True, "stdout": "AGE=%d\n" % self.age, "stderr": ""}
        if cmd.startswith("test -s") and cmd.endswith("result.json && echo OK"):
            return {"success": True, "stdout": ("OK\n" if self.result_nonempty else ""), "stderr": ""}
        if "nohup bash -c" in cmd:
            return {"success": True, "stdout": "BG_STARTED\n", "stderr": ""}
        if cmd.startswith("tail -n"):
            return {"success": True, "stdout": self.err_text, "stderr": ""}
        return {"success": True, "stdout": "", "stderr": ""}

    async def write(self, sandbox_id, path, data):
        return {"success": True, "error": None}

    async def read(self, sandbox_id, path):
        return {"success": False, "data": b"", "error": "not found"}


def _resume_ns(sb):
    ns = _load_ocr_module()
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None
    # meta 预置为本次指纹（src=沙箱路径, pages=all, dpi=150）
    ns["_read_fingerprint_stub"] = None
    return ns


def test_inflight_task_is_not_restarted_on_resume():
    """同指纹任务仍在推进时，续查**不得重启**（否则 unlink 正在写的产物）。

    add0229b 的续查形态：模型每 ~300s 用完全相同的参数再调一次。
    旧实现只看 result.json 是否存在——而 `python3 ... > result.json` 在启动
    瞬间就创建了 0 字节文件，于是"正在跑"被判成"有产物"，要么复用空文件
    报 read 失败，要么清场重启把正在写的 inode unlink 掉、结果全丢。
    正确判据：err.log 的 mtime 新鲜 ⇒ 在跑 ⇒ 不清场、不启动、直接续等。
    """
    import asyncio as _asyncio

    err = "\n".join([
        "[ocr_run] ocr start pages=20",
        "[ocr_run] page 12/20 ok texts=40 elapsed=28.1s",
    ])
    sb = _ResumeSandbox(age=12, err_text=err)

    async def _run():
        ns = _resume_ns(sb)
        # 让 _read_fingerprint 返回与本次一致的指纹
        orig_poll = sb.run

        async def _patched(sandbox_id, cmd, timeout=None):
            if cmd.startswith("cat ") and "ocr.meta" in cmd:
                return {"success": True,
                        "stdout": "/home/user/_ocr/e2e.pdf|all|150\n", "stderr": ""}
            return await orig_poll(sandbox_id, cmd, timeout)

        ns["_sandbox_run"] = _patched
        return await ns["ocr_pdf_to_text"](
            sandbox_id="sbx-test",
            sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
            pages="all",
            dpi=150,
            wait_seconds=0,
        )

    r = _asyncio.run(_run())
    assert not any("nohup bash -c" in c for c in sb.commands), (
        "任务在跑时不得重新启动（会 unlink 正在写入的 result.json）"
    )
    assert r["status"] == "running", r
    assert r["pages_ocred"] == 12, "续查必须给出真实进度，实际：%s" % r.get("pages_ocred")


def test_empty_result_json_is_not_reused():
    """result.json 存在但为空（任务在跑/被杀）时不得当成"已有产物"复用。"""
    import asyncio as _asyncio

    sb = _ResumeSandbox(age=-1, result_nonempty=False)

    async def _run():
        ns = _resume_ns(sb)
        orig_poll = sb.run

        async def _patched(sandbox_id, cmd, timeout=None):
            if cmd.startswith("cat ") and "ocr.meta" in cmd:
                return {"success": True,
                        "stdout": "/home/user/_ocr/e2e.pdf|all|150\n", "stderr": ""}
            return await orig_poll(sandbox_id, cmd, timeout)

        ns["_sandbox_run"] = _patched
        return await ns["ocr_pdf_to_text"](
            sandbox_id="sbx-test",
            sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
            pages="all",
            dpi=150,
            wait_seconds=0,
        )

    r = _asyncio.run(_run())
    assert any("nohup bash -c" in c for c in sb.commands), (
        "err.log 无活动且 result.json 为空 ⇒ 必须重新启动任务"
    )
    assert r["status"] == "running", r


def test_stale_errlog_allows_restart():
    """err.log 陈旧（SIGKILL 残留）超过判活窗口 ⇒ 允许重启。

    否则 OOM 被杀的进程会留下 err.log，让任务永远显示"在跑"，再也重启不了。
    """
    import asyncio as _asyncio

    sb = _ResumeSandbox(age=9999, err_text="[ocr_run] ocr start pages=20")

    async def _run():
        ns = _resume_ns(sb)
        orig_poll = sb.run

        async def _patched(sandbox_id, cmd, timeout=None):
            if cmd.startswith("cat ") and "ocr.meta" in cmd:
                return {"success": True,
                        "stdout": "/home/user/_ocr/e2e.pdf|all|150\n", "stderr": ""}
            return await orig_poll(sandbox_id, cmd, timeout)

        ns["_sandbox_run"] = _patched
        return await ns["ocr_pdf_to_text"](
            sandbox_id="sbx-test",
            sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
            pages="all",
            dpi=150,
            wait_seconds=0,
        )

    _asyncio.run(_run())
    assert any("nohup bash -c" in c for c in sb.commands), "陈旧 err.log 必须允许重启"


def test_result_reuse_requires_nonempty_file():
    """复用判定必须 `test -s`（非空），不能 `test -f`。

    `python3 ... > result.json` 在命令启动瞬间就创建了 0 字节文件，用
    `test -f` 会把"正在跑"误判成"有产物"⇒ 复用空文件 ⇒ _fail("read")
    （2026-10-20 e2e T2/T3 实锤：T1 用 wait_seconds=5 提前返回后 T2 立刻续查）。
    """
    src = _src()
    seg = _code_only(ast.get_source_segment(src, _func_node(ast.parse(src), "ocr_pdf_to_text")))
    assert "test -s %s && echo OK" in seg, "复用判定必须用 test -s（非空文件）"
    assert 'test -f %s && echo OK" % shlex.quote(out_json)' not in seg, (
        "不得用 test -f 判 result.json——空文件也会通过，把进行中的任务当成已完成"
    )


def test_liveness_gate_precedes_restart():
    """判活闸门必须存在且先于启动：err.log mtime 新鲜 ⇒ 不重启。

    否则续查会清场重启，把正在写的 result.json unlink 掉（结果全丢）。
    """
    src = _src()
    seg = _code_only(ast.get_source_segment(src, _func_node(ast.parse(src), "ocr_pdf_to_text")))
    assert "_ocr_task_age_seconds" in seg, "缺少 mtime 判活探针"
    assert "_OCR_LIVENESS_SEC" in seg, "缺少判活窗口常量"
    i_age = seg.find("_ocr_task_age_seconds")
    i_launch = seg.find("_build_ocr_launch_cmd(")
    assert i_age != -1 and i_launch != -1 and i_age < i_launch, (
        "判活必须在启动命令之前完成（否则重启已在进行）"
    )


# ────────── P0-1/P0-2/P0-3 + P1（2026-10-20 add0229b 第二轮）──────────


def test_launch_cmd_roundtrip_with_special_chars():
    """启动命令必须对**含空格/中文/引号/$** 的路径安全（P0-1）。

    现场（67 上 A/B）：`nohup bash -c 'python3 <shlex.quote(pdf)> ...'` 里外层
    手写单引号与 shlex.quote 生成的内层单引号嵌套 ⇒ bash 在第一个内层单引号
    处截断命令串。实测 argv：
        无空格 long.pdf       → [ocr_run.py, long.pdf, --json, --dpi, 150]  产物齐全
        含空格 康视达 CST...pdf → [ocr_run.py, "/home/user/_ocr/康视达"]    产物全无
    本测试用 shlex.split 做两次往返（外层 shell 切一次、内层 bash -c 再切一次），
    等价于 bash 的解析结果，锁住"整串 quote"这个修法。
    """
    import shlex

    ns = _load_ocr_module()
    build = ns["_build_ocr_launch_cmd"]
    entry = "/home/user/_ocr/ocr_run.py"
    outs = ("/home/user/_ocr/result.json", "/home/user/_ocr/err.log", "/home/user/_ocr/done.flag")
    for name in (
        "康视达 CST2026手册.pdf.sub.pdf",   # 空格 + 中文（add0229b 现场）
        "康视达_CST2026手册.pdf",           # 下划线（84433a68 现场，旧写法侥幸能跑）
        "a'b.pdf",                          # 单引号
        "a$USER b.pdf",                     # $ 展开
        "a\"b c.pdf",                       # 双引号
    ):
        pdf = "/home/user/_ocr/" + name
        cmd = build(entry, pdf, 150, *outs)
        outer = shlex.split(cmd)
        assert outer[:2] == ["nohup", "bash"], outer
        assert "-c" in outer, "必须把整条内层命令作为 bash -c 的单个参数"
        inner = outer[outer.index("-c") + 1]
        argv = shlex.split(inner)
        assert argv[:5] == ["python3", entry, pdf, "--json", "--dpi"], (
            "路径 %r 未完整还原，实际 argv=%r" % (name, argv[:6])
        )
        # 重定向与退出码回写必须还在（旧 bug 里它们全被截断丢弃）
        # 注：shlex 不把 `;` 当分隔符，故 err.log 后面可能粘着 ';'
        assert ">" in argv and outs[0] in argv, argv
        assert "2>" in argv and any(t.rstrip(";") == outs[1] for t in argv), argv
        assert outs[2] in argv and argv[-1] == outs[2], (
            "done.flag 回写必须是最后一段，实际：%r" % (argv[-4:],)
        )


def test_legacy_handwired_single_quotes_still_break_spaces():
    """反向锚点：证明旧写法确实会碎（防止有人把修法回退成手写单引号）。

    这条断言不是在测生产代码，而是把根因钉在测试里——以后再看到
    `bash -c 'python3 ...'` 这种手写引号，这里会立刻提醒它为什么不行。
    """
    import shlex

    entry = "/home/user/_ocr/ocr_run.py"
    pdf = "/home/user/_ocr/康视达 CST2026手册.pdf.sub.pdf"
    legacy = (
        "nohup bash -c 'python3 %s %s --json --dpi 150 > %s 2> %s; echo $? > %s' "
        "</dev/null >/dev/null 2>&1 & echo BG_STARTED"
        % (shlex.quote(entry), shlex.quote(pdf),
           shlex.quote("/home/user/_ocr/result.json"),
           shlex.quote("/home/user/_ocr/err.log"),
           shlex.quote("/home/user/_ocr/done.flag"))
    )
    inner = shlex.split(legacy)[shlex.split(legacy).index("-c") + 1]
    argv = shlex.split(inner)
    assert argv[2] != pdf, "旧写法居然还原对了？那 P0-1 的前提不成立，需重新取证"
    assert argv[2] == "/home/user/_ocr/康视达", argv


class _BgStartedNoErrlogSandbox:
    """BG_STARTED 有、err.log 永远没有内容的现场（P0-2 的核心回归点）。

    missing=True  → err.log 从未被创建（启动命令没跑到底）
    missing=False → err.log 已创建但始终 0 字节（python3 没执行)
    """

    def __init__(self, missing: bool):
        self.missing = missing
        self.commands: list[str] = []

    async def run(self, sandbox_id, cmd, timeout=None):
        self.commands.append(cmd)
        if "LINKED" in cmd or "COPIED" in cmd:
            return {"success": True, "stdout": "LINKED\n", "stderr": ""}
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:
            return {"success": True, "stdout": '__version__ = "1.2.1"\nOK\n', "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "5\n", "stderr": ""}
        if "nohup bash -c" in cmd:  # 启动命令"成功"，BG_STARTED 照常回
            return {"success": True, "stdout": "BG_STARTED\n", "stderr": ""}
        if "wc -c <" in cmd:  # 关键：BG_STARTED 有，但 err.log 始终无内容
            return {"success": True, "stdout": "MISSING\n" if self.missing else "0\n",
                    "stderr": ""}
        if cmd.startswith("cat ") and "done.flag" in cmd:
            return {"success": True, "stdout": "", "stderr": ""}
        return {"success": True, "stdout": "", "stderr": ""}

    async def write(self, sandbox_id, path, data):
        return {"success": True, "error": None}

    async def read(self, sandbox_id, path):
        return {"success": False, "data": b"", "error": "not found"}


def _run_launch_probe(sb, tries=2):
    import asyncio as _asyncio

    ns = _load_ocr_module()
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None
    ns["_OCR_LAUNCH_PROBE_TRIES"] = tries
    ns["_OCR_LAUNCH_PROBE_INTERVAL"] = 0
    return _asyncio.run(ns["ocr_pdf_to_text"](
        sandbox_id="sbx-test",
        sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
        pages="all",
        dpi=150,
        wait_seconds=1,
    ))


def test_launch_probe_not_short_circuited_by_bg_started():
    """BG_STARTED 不能短路启动验证（P0-2，add0229b 二轮的直接死因）。

    旧代码 `if done is None and "BG_STARTED" not in stdout:` 才做探针 ——
    引号截断事故里 BG_STARTED 照常输出（echo 在截断之前），于是探针被跳过，
    工具连续 11 次回 running/pages_ocred=0，模型空等 22 分钟。
    """
    sb = _BgStartedNoErrlogSandbox(missing=True)
    r = _run_launch_probe(sb)
    assert r["success"] is False, r
    assert r["step"] == "launch", r
    assert "未能启动" in r["error"], r["error"]
    assert "始终未被创建" in r["error"], "必须指出 err.log 根本没被创建：%s" % r["error"]
    assert "BG_STARTED=收到" in r["error"], (
        "必须点明 BG_STARTED 已收到但仍失败（否则排查会再次误判成'启动成功'）"
    )


def test_launch_probe_flags_empty_errlog_separately():
    """err.log 存在但为空 ⇒ 文案必须与"不存在"区分（两种故障修法不同）。"""
    sb = _BgStartedNoErrlogSandbox(missing=False)
    r = _run_launch_probe(sb)
    assert r["success"] is False and r["step"] == "launch", r
    assert "已创建但始终为空" in r["error"], r["error"]


def test_batch_page_semantics_are_consistent():
    """pages_* 一律指"本批"，文档总页数单独给 pdf_total_pages（P1-1）。

    add0229b：同一份返回里 pages_total=262（文档总页数）与
    pages_ocred_total=10（本批页数）并存，模型据此算错剩余时间。
    """
    import asyncio as _asyncio

    err = "[ocr_run] page 2/3 ok texts=10 elapsed=30.0s"
    ns = _load_ocr_module()
    sb = _ProgressSandbox(err)  # 文档共 20 页
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None

    r = _asyncio.run(ns["ocr_pdf_to_text"](
        sandbox_id="sbx-test",
        sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
        pages="1-3",
        dpi=150,
        wait_seconds=0,
    ))
    assert r["status"] == "running", r
    assert r["pages_total"] == 3, r
    assert r["pdf_total_pages"] == 20, r
    assert r["pages_ocred_total"] == 3, r
    assert r["pages_ocred"] == 2, r


def test_silent_streak_escalates_to_suspected_dead():
    """连续零 stderr ⇒ 提示里要出现"第 N 次 / 疑似已静默退出 / 建议按失败处理"（P1-2）。

    三态时代"无输出"和"正在渲染"混在一起，模型到第 5 次续查才自己发现异常。
    """
    import asyncio as _asyncio

    ns = _load_ocr_module()
    sb = _ProgressSandbox("")  # err.log 空 ⇒ 第四态
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None
    ns["_OCR_SILENT_STREAK"].clear()

    def _call():
        return _asyncio.run(ns["ocr_pdf_to_text"](
            sandbox_id="sbx-test",
            sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
            pages="all",
            dpi=150,
            wait_seconds=0,
        ))

    first = _call()
    assert "第 1 次" in first["progress_hint"], first["progress_hint"]
    assert "建议按失败处理" not in first["progress_hint"], "第一次不该劝放弃"
    second = _call()
    assert "第 2 次" in second["progress_hint"], second["progress_hint"]
    assert "建议按失败处理" in second["progress_hint"], second["progress_hint"]


def test_ocr_runner_prints_start_marker_before_heavy_import():
    """沙箱侧必须在 paddleocr import **之前**打 start 行（P0-2 的前提）。

    启动探针只认 err.log 有没有内容；`from paddleocr import PaddleOCR` 要
    5~15s，start 行若留在 env 行位置（import 之后），15s 探针窗口会把
    "正在 import"误判成"没启动"，反而制造假失败。
    """
    tree = ast.parse(RUNNER_PY.read_text("utf-8"))
    main_fn = _func_node(tree, "main")
    body = [n for n in main_fn.body]
    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), (
        "main() 的第一句必须是 print(start)，实际：%s" % ast.dump(first)[:120]
    )
    print_src = ast.get_source_segment(RUNNER_PY.read_text("utf-8"), first) or ""
    assert "start" in print_src and "PROGRESS_PREFIX" in print_src, print_src
    assert "flush=True" in print_src, "stderr 重定向到文件，必须 flush"


def test_mcp_logger_is_not_swallowed():
    """本模块 logger 必须自带 handler 且 level=INFO（P1-3）。

    add0229b 排查时 MCP 日志里一条 `[OCR]` 都没有：`getLogger(__name__)` 没有
    handler，root 默认 WARNING ⇒ 所有 logger.info 被 lastResort 丢弃，只有
    `[DLP]`（warning 级）可见。缺这条，下次还得靠猜。
    """
    src = _src()
    assert "logger.addHandler(" in src, "必须给模块 logger 挂 handler，否则 info 全被丢"
    assert "logger.setLevel(logging.INFO)" in src, "必须显式抬到 INFO"
    assert "logging.StreamHandler(sys.stderr)" in src, (
        "handler 必须写 stderr——stdio 传输下 stdout 是协议流，污染即断连"
    )


def test_module_level_imports_cover_runtime_deps():
    """摘出的函数依赖的模块必须在 main.py **模块级**导入过。

    为什么需要这条：`_load_ocr_module` 会把 json/os/re/shlex/logger 注入命名
    空间，于是"main.py 漏了 import shlex"在单测里完全看不见 —— 1.0.13 就这样
    把 `name 'shlex' is not defined` 带到了生产，e2e 第一次调用即炸。函数内的
    局部 `import shlex` 也救不了：`_build_ocr_launch_cmd` 是模块级函数。
    """
    tree = ast.parse(_src())
    imported = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported |= {a.asname or a.name for a in n.names}
    for mod in ("json", "os", "re", "shlex"):
        assert mod in imported, (
            "main.py 模块级缺少 `import %s`——函数内局部 import 会被 AST 提取器绕过，"
            "单测照样绿、生产直接 NameError" % mod
        )
    assigned = {
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    assert "logger" in assigned, "main.py 模块级缺少 logger 赋值"


def test_completed_result_has_same_page_fields_as_running():
    """completed 与 running 必须共用同一套页数字段（e2e 实测的缺口）。

    1.0.14 的 e2e：completed 返回 pages_ocred=3、pages_total=3，却没有
    pages_ocred_total（running 有）——同名口径缺键，模型两边对不上。
    """
    src = _src()
    node = _func_node(ast.parse(src), "_ocr_collect")
    seg = ast.get_source_segment(src, node)
    for key in ("pages_total", "pdf_total_pages", "pages_ocred", "pages_ocred_total"):
        assert '"%s"' % key in seg, "completed 返回缺少字段 %s" % key


# ────────── 自动分批（方案 B，2026-10-20）──────────
# 要守的不是"分批算法对不对"（纯函数直接测），而是三条产品语义：
#   ① 分批 ≠ 截断：一页都不许丢，总页数不设上限；
#   ② 一次调用只推进一批，**绝不是"第一批完成就 completed"**（那是静默截断）；
#   ③ 某批失败只影响该批，且必须在 warnings 里显式列出（用户要知道缺了哪段）。


def test_page_span_text_folds_continuous_ranges():
    """页范围文本：连续折叠成 a-b，非连续列段，超限折叠。"""
    ns = _load_ocr_module()
    span = ns["_page_span_text"]
    assert span([81, 82, 83, 84]) == "81-84"
    assert span([1]) == "1"
    assert span([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"
    assert span([]) == "空"
    # 超过 limit 段 ⇒ 折叠并带上总页数，避免把 262 个页号糊进提示
    many = list(range(1, 30, 2))  # 1,3,5,...,29 = 15 段
    text = span(many, limit=3)
    assert text.startswith("1,3,5") and "等 15 页" in text, text


def test_plan_ocr_batches_covers_every_page_exactly_once():
    """分批必须覆盖全部页、不重不漏，且**不设总页数上限**（分批≠截断）。

    262 页 / 40 ⇒ 7 批（40×6 + 22）。1000 页 / 40 ⇒ 25 批。
    这是"总页数不设限"的直接证明：批数随页数增长，页集合恒等于全集。
    """
    ns = _load_ocr_module()
    plan = ns["_plan_ocr_batches"]
    for total, size in ((262, 40), (1000, 40), (100, 40), (81, 40), (40, 40), (39, 40)):
        groups = plan(list(range(total)), size)
        flat = [p for g in groups for p in g]
        assert flat == list(range(total)), (
            "总页数 %d、批大小 %d 时页集合与全集不一致（丢页或重复）" % (total, size)
        )
        assert len(flat) == len(set(flat)), "存在重复页"
        assert all(len(g) <= size for g in groups), "有批次超过 batch_pages"
        assert len(groups) == (total + size - 1) // size, (
            "批数不对：%d 页 / %d ⇒ %d 批，实际 %d" % (total, size, (total + size - 1) // size, len(groups))
        )
    # 不分批的两条路径
    assert plan(list(range(20)), 40) == [list(range(20))]
    assert plan(list(range(20)), 0) == [list(range(20))]


def test_batch_current_index_skips_finished_and_failed():
    """当前批 = 第一个 pending/running；失败批不自动重试（坏 PDF 会无限重试）。"""
    ns = _load_ocr_module()
    cur = ns["_batch_current_index"]
    st = ns["_batch_state_new"]("fp", [[0, 1], [2, 3], [4]], 2, 0.0)
    assert cur(st) == 0
    st["batches"][0]["status"] = "done"
    st["batches"][1]["status"] = "failed"
    assert cur(st) == 2, "失败批应被跳过，直接推进到下一批"
    st["batches"][2]["status"] = "done"
    assert cur(st) is None, "全部结束必须返回 None"
    st["batches"][0]["status"] = "running"
    assert cur(st) == 0, "running 必须被接管（进程可能仍在跑，由判活闸门决定续等还是重启）"


def test_batch_summary_counts_done_only():
    """汇总只统计 done 批（failed 的 ocred 不计入进度，否则模型会高估完成度）。"""
    ns = _load_ocr_module()
    st = ns["_batch_state_new"]("fp", [[0, 1], [2], [3, 4]], 2, 0.0)
    st["batches"][0].update(status="done", ocred=2, texts=10)
    st["batches"][1].update(status="failed", error="exit=1")
    st["batches"][2].update(status="done", ocred=2, texts=7)
    s = ns["_batch_summary"](st)
    assert s["batches_done"] == 2, s
    assert s["pages_ocred_doc"] == 4, s
    assert s["total_texts_doc"] == 17, s
    assert len(s["batches_failed"]) == 1, s
    assert s["batches_failed"][0]["batch"] == 2, s
    assert s["batches_failed"][0]["pages"] == "3", s
    assert s["batches_failed"][0]["error"] == "exit=1", s


class _BatchSandbox:
    """模拟"分批推进"的沙箱：跨多次工具调用保持状态。

    · ``launched``  已收到 nohup 启动命令的批次目录号（模拟进程已跑完）
    · ``fail_batches``  这些批启动后"秒退 exit=1 且不产 result.json"
    · state.json / ocr.meta / result.txt 通过 write 落库，read 时回取——
      这正是 MCP 侧"无状态、进度只能存沙箱"的真实形态。
    """

    def __init__(self, page_count: int, batch_pages: int, fail_batches=()):
        self.page_count = page_count
        self.batch_pages = batch_pages
        self.fail_batches = set(fail_batches)
        self.launched: set[int] = set()
        self.commands: list[str] = []
        self.state_bytes: bytes | None = None
        self.metas: dict[int, str] = {}
        self.texts: dict[int, str] = {}
        # 每批页数（最后一批可能不满）。键 0 = 根路径（单批模式，不分批）。
        n_batch = (page_count + batch_pages - 1) // batch_pages
        self.pages_by_idx = {
            i + 1: min(batch_pages, page_count - i * batch_pages) for i in range(n_batch)
        }
        self.pages_by_idx[0] = page_count

    @staticmethod
    def _bidx(cmd: str) -> int | None:
        m = re.search(r"/batches/b(\d{3})/", cmd)
        return int(m.group(1)) if m else None

    async def run(self, sandbox_id, cmd, timeout=None):
        self.commands.append(cmd)
        idx = self._bidx(cmd)
        if "LINKED" in cmd or "COPIED" in cmd:
            return {"success": True, "stdout": "LINKED\n", "stderr": ""}
        if "wc -c <" in cmd:  # 启动可见性探针：err.log 有 start 行
            return {"success": True, "stdout": "48\n", "stderr": ""}
        if cmd.startswith("test -f /home/user/_ocr/e2e.pdf"):
            return {"success": True, "stdout": "OK\n", "stderr": ""}
        if "grep -m1" in cmd:
            return {"success": True, "stdout": '__version__ = "1.2.1"\nOK\n', "stderr": ""}
        if cmd.startswith("head -c"):  # 收口时的 preview
            return {"success": True, "stdout": "PREVIEW-ALL", "stderr": ""}
        if cmd.startswith("cat ") and "result.txt" in cmd and ">" in cmd:  # 合并全文
            return {"success": True, "stdout": "", "stderr": ""}
        if cmd.startswith("cat ") and "done.flag" in cmd:
            key = idx or 0
            if key in self.launched:
                return {"success": True, "stdout": "0\n" if key not in self.fail_batches else "1\n",
                        "stderr": ""}
            return {"success": True, "stdout": "", "stderr": ""}
        if cmd.startswith("cat ") and "ocr.meta" in cmd:
            return {"success": True, "stdout": (self.metas.get(idx or 0, "") + "\n"), "stderr": ""}
        if cmd.startswith("test -s ") and "result.json" in cmd:
            key = idx or 0
            if key in self.launched and key not in self.fail_batches:
                return {"success": True, "stdout": "OK\n", "stderr": ""}
            return {"success": True, "stdout": "", "stderr": ""}
        if "stat -c %Y" in cmd:  # 判活：err.log 新鲜
            return {"success": True, "stdout": "AGE=5\n", "stderr": ""}
        if "nohup bash -c" in cmd:
            self.launched.add(idx or 0)
            return {"success": True, "stdout": "BG_STARTED\n", "stderr": ""}
        if "mkdir -p" in cmd and "rm -f" in cmd:  # 清场（只清本批目录）
            return {"success": True, "stdout": "", "stderr": ""}
        if "mkdir -p" in cmd and "_page.py" in cmd:  # 拆分本批子 PDF
            # 页号段只到第一个 ';' 为止——后面还有 mv/test -s，不能算进页数
            m = re.search(r"_page\.py\s+\S+([^;]*)", cmd)
            picked = len(re.findall(r"\d+", m.group(1) if m else ""))
            if picked:
                self.pages_by_idx[idx or 0] = picked
            return {"success": True, "stdout": "SPLIT_OK\n", "stderr": ""}
        if cmd.endswith("_page.py /home/user/_ocr/e2e.pdf"):  # 页数探测
            return {"success": True, "stdout": "%d\n" % self.page_count, "stderr": ""}
        if cmd.startswith("tail -n"):
            return {"success": True, "stdout": "[ocr_run] page 1/%d ok texts=1" % self.batch_pages,
                    "stderr": ""}
        return {"success": True, "stdout": "", "stderr": ""}

    async def write(self, sandbox_id, path, data):
        idx = self._bidx(path)
        if path.endswith("state.json"):
            self.state_bytes = data
        elif path.endswith("ocr.meta"):
            self.metas[idx or 0] = data.decode("utf-8")
        elif path.endswith("result.txt"):
            self.texts[idx or 0] = data.decode("utf-8")
        return {"success": True, "error": None}

    async def read(self, sandbox_id, path):
        if path.endswith("state.json"):
            if self.state_bytes is None:
                return {"success": False, "data": b"", "error": "not found"}
            return {"success": True, "data": self.state_bytes, "error": None}
        if path.endswith("result.json"):
            key = self._bidx(path) or 0
            if key not in self.launched or key in self.fail_batches:
                return {"success": False, "data": b"", "error": "not found"}
            recs = [
                {"page": i + 1, "ok": True, "texts": ["B%d-P%d" % (key, i + 1)]}
                for i in range(self.pages_by_idx.get(key, 0))
            ]
            return {"success": True, "data": json.dumps(recs).encode(), "error": None}
        return {"success": False, "data": b"", "error": "not found"}


def _batch_env(page_count=100, batch_pages=40, fail_batches=()):
    import asyncio as _asyncio

    ns = _load_ocr_module()
    sb = _BatchSandbox(page_count, batch_pages, fail_batches)
    ns["_sandbox_run"] = sb.run
    ns["_sandbox_write"] = sb.write
    ns["_sandbox_read"] = sb.read
    ns["_is_valid_sandbox_id"] = lambda sid: True
    ns["_upload_precheck"] = None
    ns["_resolve_host_path"] = None

    def _call(pages="all", dpi=150, wait=0):
        return _asyncio.run(ns["ocr_pdf_to_text"](
            sandbox_id="sbx-test",
            sandbox_pdf_path="/home/user/_ocr/e2e.pdf",
            pages=pages,
            dpi=dpi,
            wait_seconds=wait,
            batch_pages=batch_pages,
        ))

    return ns, sb, _call


def test_batch_mode_advances_one_batch_per_call():
    """一次调用只推进一批，且**中间批次绝不返回 completed**（防静默截断）。

    100 页 / 40 ⇒ 3 批。旧行为会是"第一批跑完就 completed"，用户只拿到 40 页
    却以为拿到全本——这比失败更糟。必须：每批完成 ⇒ running + 明确的
    "第 N 批尚未启动，同参再调一次"，最后一颗批完成才 completed。
    """
    _ns, _sb, call = _batch_env(page_count=100, batch_pages=40)

    r1 = call()
    assert r1["status"] == "running", "第 1 批完成后不得返回 completed：%s" % r1
    assert r1["batch"] == "2/3", r1
    assert r1["batch_count"] == 3, r1
    assert r1["batch_pages"] == "41-80", r1
    assert r1["batches_done"] == 1, r1
    assert r1["pages_ocred_doc"] == 40, r1
    assert r1["pdf_total_pages"] == 100, r1
    assert "不要重新上传" in r1["progress_hint"], r1["progress_hint"]
    assert "尚未启动" in r1["progress_hint"], r1["progress_hint"]

    r2 = call()
    assert r2["status"] == "running", r2
    assert r2["batch"] == "3/3", r2
    assert r2["batches_done"] == 2, r2
    assert r2["pages_ocred_doc"] == 80, r2

    r3 = call()
    assert r3["status"] == "completed", "最后一批完成才允许 completed：%s" % r3
    assert r3["batch"] == "3/3", r3
    assert r3["success"] is True, r3
    assert r3["pdf_total_pages"] == 100, r3
    # 一页不丢：40 + 40 + 20
    assert r3["pages_ocred_doc"] == 100, r3
    assert r3["pages_total"] == 100, r3
    assert r3["preview"] == "PREVIEW-ALL", r3
    assert r3["batches_failed"] == [], r3

    # 第 4 次调用：已收口 ⇒ 直接返回 completed，不得再启动进程
    n_launch = sum(1 for c in _sb.commands if "nohup bash -c" in c)
    r4 = call()
    assert r4["status"] == "completed", r4
    assert sum(1 for c in _sb.commands if "nohup bash -c" in c) == n_launch, (
        "全部批次完成后再次调用，不得重新启动任何进程"
    )


def test_batch_artifacts_are_isolated_per_batch():
    """每批产物落在自己的 batches/bNNN/ 下，清场只清本批（否则前序批全丢）。"""
    _ns, sb, call = _batch_env(page_count=100, batch_pages=40)
    call()
    joined = "\n".join(sb.commands)
    assert "/batches/b001/result.json" in joined, "第 1 批产物应落在 batches/b001/ 下"
    assert "/batches/b001/done.flag" in joined, joined[:400]
    # 清场命令里不得出现根目录产物（那会删掉别的批）
    preps = [c for c in sb.commands if "mkdir -p" in c and "rm -f" in c]
    assert preps, "找不到清场命令"
    for p in preps:
        rm_part = p.split("rm -f", 1)[1]
        assert "/batches/b" in rm_part, "清场目标不在批目录内：%s" % p
        assert "/home/user/_ocr/result.json" not in rm_part, (
            "清场删到了根目录 result.json（会踩别的批次/单批任务）：%s" % p
        )
    # 每批一个子 PDF，避免第 2 批把第 1 批的子 PDF 覆盖掉
    assert "/batches/b001/input.sub.pdf" in joined, joined[:600]


def test_batch_failure_is_skipped_and_reported():
    """某批失败：不阻断其它批，但必须在 warnings 里点名（缺哪段要能补救）。

    这是"失败只丢当前批"的产品含义：第 2 批（41-80 页）挂了，用户仍然拿到
    第 1、3 批的文本，并且**知道** 41-80 缺失、可以用 pages="41-80" 补跑。
    静默跳过等于交付残缺而不告知。
    """
    _ns, _sb, call = _batch_env(page_count=100, batch_pages=40, fail_batches=(2,))

    r1 = call()
    assert r1["status"] == "running" and r1["batch"] == "2/3", r1

    r2 = call()  # 第 2 批失败
    assert r2["status"] == "running", "单批失败不得中断整轮：%s" % r2
    assert r2["batch"] == "3/3", r2
    assert r2["batches_done"] == 1, r2
    assert any("第 2 批" in w and "失败" in w for w in r2["warnings"]), r2["warnings"]

    r3 = call()  # 第 3 批完成 ⇒ 收口
    assert r3["status"] == "completed", r3
    assert r3["success"] is True, "第 1、3 批有文本，整体仍应算成功：%s" % r3
    assert r3["pages_ocred_doc"] == 60, "40 + 20，失败批不计入：%s" % r3
    assert len(r3["batches_failed"]) == 1, r3
    assert r3["batches_failed"][0]["batch"] == 2, r3
    assert r3["batches_failed"][0]["pages"] == "41-80", r3
    assert any("41-80" in w for w in r3["warnings"]), (
        "收口时必须告诉用户哪几页缺失、怎么补：%s" % r3["warnings"]
    )


def test_batch_state_restarts_when_fingerprint_changes():
    """参数变了（dpi）⇒ 分批状态必须重开，不得沿用旧批次进度。

    指纹含「源 PDF + dpi + 批大小 + 目标页集合」，任一变化都是新任务。
    沿用旧 state 的后果：第 2 批的产物被当成第 1 批交付，页范围错位。
    """
    _ns, sb, call = _batch_env(page_count=100, batch_pages=40)
    r1 = call()
    assert r1["batches_done"] == 1, r1

    r2 = call(dpi=200)
    # 重开后只完成"新 state 的第 1 批"；若沿用旧 state，这里会是第 2 批完成（=2）
    assert r2["batches_done"] == 1, (
        "dpi 变了应重开分批状态（只完成新第 1 批），实际 batches_done=%s，batch=%s"
        % (r2.get("batches_done"), r2.get("batch"))
    )
    state = json.loads(sb.state_bytes.decode("utf-8"))
    assert "dpi=200" in state["fingerprint"], state["fingerprint"]
    assert state["batch_count"] == 3, state


def test_batch_mode_requires_explicit_opt_in_via_page_count():
    """目标页数 ≤ batch_pages ⇒ 走原单批路径，返回里不得出现批次字段。"""
    _ns, _sb, call = _batch_env(page_count=20, batch_pages=40)
    r = call()
    assert r["status"] == "completed", r
    assert "batch" not in r, "单批路径不应出现 batch 字段：%s" % r
    assert r["pages_ocred"] == 20, r


def test_batch_pages_param_is_declared_and_clamped():
    """batch_pages 必须在 schema 里（默认 0=自动），且被钳到 <=_OCR_BATCH_PAGES_MAX。"""
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    args = [a.arg for a in node.args.args]
    assert "batch_pages" in args, "ocr_pdf_to_text 缺少 batch_pages 参数（模型无法控制批大小）"
    defaults = dict(zip(args[-len(node.args.defaults):], node.args.defaults))
    assert isinstance(defaults.get("batch_pages"), ast.Constant), defaults
    assert defaults["batch_pages"].value == 0, "batch_pages 默认必须是 0（自动）"
    assert "_OCR_BATCH_PAGES_MAX" in src and "min(batch_pages, _OCR_BATCH_PAGES_MAX)" in src, (
        "batch_pages 必须被 _OCR_BATCH_PAGES_MAX 钳制，否则模型传 9999 就把分批退化成"
        "一次性全量（正是本改动要消灭的形态）"
    )


def test_batch_split_moves_subpdf_into_batch_dir():
    """批模式的子 PDF 必须真的搬进 batches/bNNN/（生产 e2e 实测的缺陷）。

    沙箱侧 `_page.py` 固定把子 PDF 写成 `src + ".sub.pdf"`（helper 的既有约定）。
    分批时若只把"喂给 ocr_run.py 的路径"改成 batches/bNNN/input.sub.pdf，文件
    根本不存在 —— 2026-10-20 生产 e2e 三批全部：
        FileNotFoundError: 输入不存在: /home/user/_ocr/batches/b001/input.sub.pdf
    而单测当时是绿的：假沙箱对任何含 `_page.py` 的拆分命令都回 SPLIT_OK，
    **比实现更宽容**。所以这里直接对命令串断言"拆完要搬、搬完要验"。

    ★ 通用教训：凡是"外部工具的输出位置由它的既有约定决定"的步骤，测试都
      必须断言**目标位置**，不能只断言"命令返回成功"。
    """
    _ns, sb, call = _batch_env(page_count=100, batch_pages=40)
    call()
    splits = [c for c in sb.commands if "_page.py" in c and "mkdir -p" in c]
    assert splits, "找不到拆分命令"
    cmd = splits[0]
    assert "mv -f" in cmd, "批模式必须在拆分后把子 PDF 搬到批目录：%s" % cmd
    assert "/batches/b001/input.sub.pdf" in cmd, cmd
    assert "/home/user/_ocr/e2e.pdf.sub.pdf" in cmd, "mv 的源应是 helper 的既有输出名：%s" % cmd
    assert "test -s" in cmd and "SPLIT_OK" in cmd, (
        "拆分成功判据必须是 test -s 目标文件（只看 rc 会把'没搬过去'当成功）：%s" % cmd
    )


def test_single_batch_path_does_not_move_subpdf():
    """单批路径保持历史行为：子 PDF 留在根目录 e2e.pdf.sub.pdf，不 mv。"""
    _ns, sb, call = _batch_env(page_count=20, batch_pages=40)
    call(pages="1-10")  # 显式页范围才需要拆子 PDF（all + 20 页不拆）
    splits = [c for c in sb.commands if "_page.py" in c and "mkdir -p" in c]
    assert splits, "显式 pages 应触发拆分"
    joined = "\n".join(splits)
    assert "mv -f" not in joined, "单批路径不应引入 mv（改历史行为=引入回归风险）"
    assert "/home/user/_ocr/e2e.pdf.sub.pdf" in joined, joined


class _SplitSilentSandbox(_BatchSandbox):
    """拆分命令 rc 成功但**目标文件不存在**：模拟"拆出来了但没搬过去"。"""

    async def run(self, sandbox_id, cmd, timeout=None):
        if "_page.py" in cmd and "mkdir -p" in cmd:
            self.commands.append(cmd)
            return {"success": True, "stdout": "", "stderr": ""}  # 没有 SPLIT_OK
        return await super().run(sandbox_id, cmd, timeout)


def test_split_without_target_must_fail_not_silently_ocr():
    """"拆了但目标不存在"必须显式 _fail("split")，不能带着坏路径去启动 OCR。

    这条正是 e2e 事故的"如果不修会怎样"：旧代码只看 commands.run 的 rc，
    于是启动了一个指向不存在文件的 OCR 进程，三批各烧 20 秒探针后全失败。
    """
    import asyncio as _asyncio

    ns = _load_ocr_module()
    sb = _SplitSilentSandbox(100, 40)
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
        wait_seconds=0,
        batch_pages=40,
    ))
    assert r["success"] is False, r
    assert r["step"] == "split", r
    assert "input.sub.pdf" in r["error"], r["error"]
    assert not any("nohup bash -c" in c for c in sb.commands), (
        "拆分失败后不得启动 OCR 进程（否则白烧一轮探针才报错）"
    )
