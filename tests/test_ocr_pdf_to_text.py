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
    src = _src()
    assert re.search(r'^_OCR_ENTRY\s*=\s*"/opt/ocr/ocr_run\.py"', src, re.M), \
        "_OCR_ENTRY 必须指向 /opt/ocr/ocr_run.py（镜像内唯一封装）"
    assert re.search(r"^_OCR_WORKDIR\s*=\s*", src, re.M)


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
