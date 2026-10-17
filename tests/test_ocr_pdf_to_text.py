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
    """接线：工具必须调 _first_json_value，而不是裸 json.loads。"""
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    body = ast.get_source_segment(src, node)
    assert "_first_json_value" in body, "解析必须走容错函数（第三方库会污染 stdout）"


def test_return_contract_fields():
    """返回值必须带 summary 字段且**不含全文**字段（262 页全文会撑爆上下文）。"""
    src = _src()
    node = _func_node(ast.parse(src), "ocr_pdf_to_text")
    body = ast.get_source_segment(src, node)
    for key in ("pages_total", "pages_ocred", "pages_failed", "total_texts",
                "payload_json", "payload_txt", "preview", "warnings"):
        assert f'"{key}"' in body, f"返回契约缺 {key}"
    assert '"text"' not in body.replace('"texts"', ""), "返回值不应内联全文（会撑爆上下文）"
