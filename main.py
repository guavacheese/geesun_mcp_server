import base64
import os
import io
import re
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Optional
import httpx
from e2b_code_interpreter import Sandbox as E2BSandbox

load_dotenv()

decrypt_api_url = os.getenv("DECRYPT_API_URL")
mcp = FastMCP("decrypt-file")

# v3.1 护栏：公司 DLP 加密文件头魔数。
# 判断"文件是否加密"不靠扩展名（txt/py 也可能被加密），靠文件头特征。
DLP_HEADER_SIGNATURES: tuple[bytes, ...] = (b"%TSD-Header",)


@mcp.tool()
async def copy_script_to_sandbox(
    script_name: str,
    sandbox_id: str,
    sandbox_path: str | None = None,
    skill_name: str | None = None,
) -> dict:
    """
    将 skill 脚本直传沙箱，不走 LLM 上下文。

    自动搜索所有 skill 来源目录（__system__ → __agent__ → 所有 __user_*__），
    无需指定 skill_source。

    用法（无需猜测路径）：
    - skill_name 通常【不必提供】——工具会按 script_name 在所有 skill
      目录中全局搜索，唯一命中直接使用；
    - 若同名脚本存在于多个 skill（返回 ambiguous + candidates 候选清单），
      结合当前任务从候选清单中选定 skill_name 后重试一次即可；
    - 仅当你明确脚本在哪个 skill 下时，才可显式传入 skill_name 加速定位。

    Args:
        script_name: 脚本文件名（如 extract_structure.py）
        sandbox_id: 沙箱 ID
        sandbox_path: 沙箱目标路径（如 /home/user/extract_structure.py），
                      不传则自动取 /home/user/{script_name}
        skill_name: （可选）脚本所在技能名；不传则全局搜索唯一命中即用，
                    多命中时按返回的 candidates 消歧后重试
    """
    if sandbox_path is None:
        sandbox_path = f"/home/user/{script_name}"
    agent_workspace = os.environ.get("AGENT_WORKSPACE", "")
    if not agent_workspace:
        return {"success": False, "error": "AGENT_WORKSPACE 未设置"}

    skills_base = f"{agent_workspace}/skills"

    # 收集所有 skill 来源目录：__system__ → __agent__ → 所有 __user_*__
    source_dirs = ["__system__", "__agent__"]
    if os.path.isdir(skills_base):
        for entry in sorted(os.listdir(skills_base)):
            if entry.startswith("__user_") and os.path.isdir(f"{skills_base}/{entry}"):
                source_dirs.append(entry)

    # ① 指定了 skill_name：仅在该 skill 下精确查找（最快路径）
    found_path = None
    if skill_name:
        for source in source_dirs:
            test_path = f"{skills_base}/{source}/{skill_name}/scripts/{script_name}"
            if os.path.isfile(test_path):
                found_path = test_path
                break

    # ② 未指定或指定 skill 下未找到 → 全局收集全部命中（不静默取首个）
    if not found_path:
        matches = []  # (source, skill_dir, path)
        for source in source_dirs:
            src = f"{skills_base}/{source}"
            if not os.path.isdir(src):
                continue
            for skill_dir in sorted(os.listdir(src)):
                candidate = f"{src}/{skill_dir}/scripts/{script_name}"
                if os.path.isfile(candidate):
                    matches.append((source, skill_dir, candidate))

        if len(matches) == 1:
            _source, skill_name, found_path = matches[0]
        elif len(matches) > 1:
            # 多命中：不静默选错，返回候选清单让 agent 基于事实决策
            return {
                "success": False,
                "ambiguous": True,
                "script_name": script_name,
                "candidates": [
                    {
                        "skill_name": skill_dir,
                        "source": source,
                        "path": path,
                        "size": os.path.getsize(path),
                    }
                    for source, skill_dir, path in matches
                ],
                "hint": "同名脚本存在于多个 skill，请结合当前任务指定 skill_name 后重试",
            }

    if not found_path:
        return {
            "success": False,
            "error": f"在所有 skill 目录中均未找到 {script_name}，"
            f"已搜索: {', '.join(source_dirs)}",
        }

    with open(found_path, "rb") as f:
        content = f.read()

    # E2B API 连接配置
    os.environ.setdefault("E2B_API_URL", os.environ.get("E2B_API_URL", ""))
    os.environ.setdefault("E2B_API_KEY", os.environ.get("E2B_API_KEY", ""))
    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        os.environ.setdefault("SSL_CERT_FILE", ssl_cert)

    sb = E2BSandbox.connect(sandbox_id)
    try:
        sb.files.write(sandbox_path, content)
    except Exception as e:
        return {
            "success": False,
            "sandbox_path": sandbox_path,
            "size": 0,
            "error": str(e),
        }

    return {"success": True, "sandbox_path": sandbox_path, "size": len(content)}


@mcp.tool()
async def upload_to_sandbox(
    file_path: str,
    remote_path: str,
    sandbox_id: str,
) -> dict:
    """
    上传宿主机的文件到沙箱（不解密），不经过 LLM 上下文。

    适用于 XML、TXT 等不需要解密的文件。
    文件从宿主机直传沙箱，不经过 LLM 上下文。

    Args:
        file_path: 虚拟路径（/uploads/{user_id}/{session_id}/{filename}）
        remote_path: 沙箱内的目标路径（如 /home/user/data.xml）
        sandbox_id: 目标沙箱 ID

    Returns:
        {"success": bool, "sandbox_path": str | None, "size": int, "error": str | None}
    """
    # 路径转换：虚拟路径 → 物理路径
    if file_path.startswith("/uploads/"):
        upload_root = os.environ.get("UPLOAD_ROOT", "")
        if upload_root:
            relative_path = file_path[len("/uploads/") :]
            file_path = f"{upload_root}/{relative_path}"

    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception as e:
        return {
            "success": False,
            "sandbox_path": None,
            "size": 0,
            "error": f"读取文件失败: {str(e)}",
        }

    # ── v3.1 护栏：DLP 加密文件（%TSD-Header 文件头）不得从明文通道上传 ──
    # upload_to_sandbox 不解密，密文进沙箱后 AI 无法解析，纯浪费步数。
    # 判断靠文件头（不靠扩展名）：txt/py 等文本类文件也可能被公司 DLP 加密。
    if any(content.startswith(sig) for sig in DLP_HEADER_SIGNATURES):
        return {
            "success": False,
            "sandbox_path": None,
            "size": 0,
            "error": (
                f"检测到 '{file_path}' 是公司 DLP 加密文件（文件头 {DLP_HEADER_SIGNATURES[0].decode(errors='replace')}），"
                "upload_to_sandbox 不解密，密文进沙箱后无法解析。"
                "请改用 decrypt_and_upload_to_sandbox 解密后上传到沙箱。"
            ),
        }

    # E2B API 连接配置
    os.environ.setdefault("E2B_API_URL", os.environ.get("E2B_API_URL", ""))
    os.environ.setdefault("E2B_API_KEY", os.environ.get("E2B_API_KEY", ""))
    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        os.environ.setdefault("SSL_CERT_FILE", ssl_cert)

    sb = E2BSandbox.connect(sandbox_id)
    try:
        sb.files.write(remote_path, content)
        return {
            "success": True,
            "sandbox_path": remote_path,
            "size": len(content),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "sandbox_path": None,
            "size": 0,
            "error": str(e),
        }


@mcp.tool()
async def decrypt_and_upload_to_sandbox(
    file_path: str,
    remote_path: str,
    sandbox_id: str,
) -> dict:
    """
    【唯一方式】读取被公司加密的文件：解密并直接上传到 CubeSandbox 沙箱内。

    公司 DLP 会加密 PDF / Excel / Word，以及 txt、py 等任意文本类文件
    （判断是否加密看文件头 %TSD-Header，不看扩展名）。
    本工具是读取加密文件的唯一正确入口；upload_to_sandbox 会拒绝密文。

    解密后的明文会写入沙箱文件系统，不经过 LLM 上下文，不落地宿主机磁盘。

    Args:
        file_path: 虚拟路径（/uploads/{user_id}/{session_id}/{filename}）
        remote_path: 沙箱内的目标路径（如 /home/user/data.xlsx）
        sandbox_id: 目标沙箱 ID，从聊天上下文的"沙箱 ID"字段获取

    Returns:
        {"success": bool, "sandbox_path": str | None, "size": int, "error": str | None}
    """
    # 路径转换：虚拟路径 → 物理路径
    if file_path.startswith("/uploads/"):
        # /uploads/{user_id}/{session_id}/{filename}
        # → /mnt/d/workspace/geesun_agent/data/uploads/{user_id}/{session_id}/{filename}
        upload_root = os.environ.get("UPLOAD_ROOT", "")
        if upload_root:
            # 去掉 /uploads/ 前缀，拼接物理路径
            relative_path = file_path[len("/uploads/") :]
            file_path = f"{upload_root}/{relative_path}"

    # 1. 解密到内存
    result = await _decrypt_file_internal(file_path)
    if not result["success"]:
        return {
            "success": False,
            "sandbox_path": None,
            "size": 0,
            "error": result["error"],
        }

    # E2B API 配置已通过 load_dotenv() 从 .env 加载到 os.environ
    os.environ.setdefault("E2B_API_URL", os.environ.get("E2B_API_URL", ""))
    os.environ.setdefault("E2B_API_KEY", os.environ.get("E2B_API_KEY", ""))
    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        os.environ.setdefault("SSL_CERT_FILE", ssl_cert)

    sb = E2BSandbox.connect(sandbox_id)
    try:
        sb.files.write(remote_path, result["data"])
        return {
            "success": True,
            "sandbox_path": remote_path,
            "size": len(result["data"]),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "sandbox_path": None,
            "size": 0,
            "error": str(e),
        }
    # ☝️ 注意：不调 sb.kill()，沙箱由 Agent 管理
    #    connect 创建的对象会在函数返回后被 Python GC 回收
    #    对 Agent 的沙箱无影响，文件已成功写入


async def _decrypt_file_internal(file_path: str) -> dict:
    """内部解密函数（供其他函数调用）"""
    try:
        # 读取原始加密文件
        with open(file_path, "rb") as f:
            file_content = f.read()

        # 调用解密API
        files = {
            "file": (
                os.path.basename(file_path),
                file_content,
            )
        }

        # response = requests.post(
        #     decrypt_api_url,
        #     files=files,
        #     timeout=30,
        # )
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                decrypt_api_url,
                files=files,
            )

        if response.status_code != 200:
            return {
                "success": False,
                "data": None,
                "error": f"解密API返回错误{response.status_code}",
                "size": 0,
            }
        return {
            "success": True,
            "data": response.content,
            "error": None,
            "size": len(response.content),
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e),
            "size": 0,
        }


# @mcp.tool()
# async def decrypt_to_tempfile(
#     file_path: str,
#     output_path: str,
# ) -> dict:
#     """
#     解密文件并写入指定路径（/tmp下），供同机其他进程直接使用。
#     注意：调用方用完后负责清理临时文件。
#     :return: {"success": bool, "output_path": str, "error": str}
#     """

#     result = await _decrypt_file_internal(file_path)
#     if not result["success"]:
#         return {
#             "success": False,
#             "output_path": None,
#             "error": result["error"],
#         }

#     with open(output_path, "wb") as f:
#         f.write(result["data"])

#     return {
#         "success": True,
#         "output_path": output_path,
#         "error": None,
#     }


def _is_valid_sandbox_id(sandbox_id: str) -> bool:
    """检查 sandbox_id 是否格式有效（32位hex或36位含横线UUID，且不含中文等占位文本）"""
    sid = sandbox_id.strip()
    if not sid:
        return False
    # 标准 UUID 格式 8-4-4-4-12（含横线）
    if re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", sid, re.I
    ):
        return True
    # 32 位 hex（无横线）
    if re.match(r"^[0-9a-f]{32}$", sid, re.I):
        return True
    return False


@mcp.tool()
async def download_from_sandbox(
    sandbox_id: str,
    sandbox_path: str,
    host_path: str,
) -> dict:
    """
    从沙箱下载文件到宿主机，不经过 LLM 上下文。

    沙箱内的文件（如审计报告）通过本工具直写宿主机磁盘，
    不走 LLM 上下文，避免大内容撑爆 token。

    容错：如果文件已存在于 host_path 则直接返回成功（跳过沙箱连接），
    适用于 AI 先用 write_file 写入 /reports/ 后又误调本工具的场景。

    Args:
        sandbox_id: 沙箱 ID
        sandbox_path: 沙箱内的文件路径（如 /home/user/plc_audit_report_xxx.md）
        host_path: 宿主机目标路径。
                   如果以 /reports/ 开头则自动转换为物理路径（$REPORT_ROOT/...），
                   否则作为物理路径直接使用。

    ⚠️ 本工具【没有】file_path 参数（那是 upload_to_sandbox /
    decrypt_and_upload_to_sandbox 的参数）！参数只有 sandbox_id +
    sandbox_path + host_path 三个。
    ⚠️ 若目标文件是 write_file 刚写入 /reports/ 的交付物：它已在宿主机上，
    【不要】调用本工具——直接交付即可。仅当文件在沙箱 /home/user/ 下、
    由 execute 脚本生成时才需要本工具拉回。

    Returns:
        {"success": bool, "host_path": str | None, "size": int, "error": str | None}
    """
    # 路径转换：虚拟 /reports/ 路径 → 物理路径
    if host_path.startswith("/reports/"):
        report_root = os.environ.get("REPORT_ROOT", "")
        if report_root:
            relative_path = host_path[len("/reports/") :]
            host_path = f"{report_root}/{relative_path}"

    # ─── 容错1：文件已在宿主机上，跳过沙箱下载 ───
    # 场景：AI 先用 write_file 写到 /reports/，文件已落盘，然后又调本工具
    if os.path.isfile(host_path):
        file_size = os.path.getsize(host_path)
        return {
            "success": True,
            "host_path": host_path,
            "size": file_size,
            "error": None,
        }

    # ─── 容错2：sandbox_id 无效时，不尝试 E2B 连接 ───
    # 场景：create_sandbox 失败（资源不足/超时等）导致 sandbox_id 为空或占位文本
    if not _is_valid_sandbox_id(sandbox_id):
        return {
            "success": False,
            "host_path": None,
            "size": 0,
            "error": f"sandbox_id 无效或为空: '{sandbox_id}'，且宿主机文件不存在: {host_path}",
        }

    # E2B API 连接配置
    os.environ.setdefault("E2B_API_URL", os.environ.get("E2B_API_URL", ""))
    os.environ.setdefault("E2B_API_KEY", os.environ.get("E2B_API_KEY", ""))
    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        os.environ.setdefault("SSL_CERT_FILE", ssl_cert)

    sb = E2BSandbox.connect(sandbox_id)
    try:
        content = sb.files.read(sandbox_path, format="bytes")
    except Exception as e:
        return {
            "success": False,
            "host_path": None,
            "size": 0,
            "error": f"读取沙箱文件失败: {str(e)}",
        }

    # 写宿主机
    try:
        os.makedirs(os.path.dirname(host_path), exist_ok=True)
        with open(host_path, "wb") as f:
            f.write(content)
    except Exception as e:
        return {
            "success": False,
            "host_path": host_path,
            "size": 0,
            "error": f"写入宿主机失败: {str(e)}",
        }

    return {
        "success": True,
        "host_path": host_path,
        "size": len(content),
        "error": None,
    }
    # ☝️ 注意：不调 sb.kill()，沙箱由 Agent 管理


@mcp.tool()
async def decrypt_file_to_base64(file_path: str) -> dict:
    """
    解密文件并返回base64字符串（不写入磁盘）

      :param file_path: 要解密的文件路径（加密状态）
      :return: {"success": bool, "data": bytes, "error": str, "size": int}
    """

    result = await _decrypt_file_internal(file_path)

    if result["success"] and result["data"]:
        result["data"] = base64.b64encode(result["data"]).decode("utf-8")

    return result


# ════════════════════════════════════════════════════════════════
# workflow 工具：tech-spec-pdf-diff 确定性执行（Phase 2）
# 设计：把"模型读 SKILL.md 后自觉照做"改为"结构化上只能走固定流程"。
#   stage1（解密→提取→结构对齐→差异页定位）与 stage3（报告生成）为确定性
#   黑盒；diff.json 语义判定（唯一需要 LLM 判断的步骤）保留给 agent。
#   2026-08-19 落地；依据 Anthropic workflows-vs-agents 二分。
# ════════════════════════════════════════════════════════════════

PDF_DIFF_SKILL_NAME = "tech-spec-pdf-diff"
PDF_DIFF_SCRIPTS = (
    "extract_pdf.py",
    "diff_structures.py",
    "diff_pages.py",
    "generate_report.py",
)
_REPORT_NAME_RE = re.compile(r"^[\w\u4e00-\u9fa5\-_\.]+$")


def _resolve_host_path(path: str) -> str:
    """虚拟路径（/uploads/、/reports/）→ MCP server 物理路径；否则原样返回。"""
    if path.startswith("/uploads/"):
        root = os.environ.get("UPLOAD_ROOT", "")
        if root:
            return f"{root}/{path[len('/uploads/'):]}"
    if path.startswith("/reports/"):
        root = os.environ.get("REPORT_ROOT", "")
        if root:
            return f"{root}/{path[len('/reports/'):]}"
    return path


def _find_skill_scripts_dir(skill_name: str) -> str | None:
    """全局搜索 skill 的 scripts 目录（__system__ → __agent__ → 所有 __user_*__）。"""
    agent_workspace = os.environ.get("AGENT_WORKSPACE", "")
    if not agent_workspace:
        return None
    skills_base = f"{agent_workspace}/skills"
    if not os.path.isdir(skills_base):
        return None
    source_dirs = ["__system__", "__agent__"]
    for entry in sorted(os.listdir(skills_base)):
        if entry.startswith("__user_") and os.path.isdir(f"{skills_base}/{entry}"):
            source_dirs.append(entry)
    for source in source_dirs:
        scripts_dir = f"{skills_base}/{source}/{skill_name}/scripts"
        if os.path.isdir(scripts_dir):
            return scripts_dir
    return None


def _e2b_env() -> None:
    """E2B API 连接配置（与现有工具一致）。"""
    os.environ.setdefault("E2B_API_URL", os.environ.get("E2B_API_URL", ""))
    os.environ.setdefault("E2B_API_KEY", os.environ.get("E2B_API_KEY", ""))
    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        os.environ.setdefault("SSL_CERT_FILE", ssl_cert)


def _dlp_encrypted(file_path: str) -> bool:
    """按文件头判断是否 DLP 加密（不看扩展名，txt/py 也可能被加密）。"""
    try:
        with open(file_path, "rb") as f:
            head = f.read(16)
        return any(head.startswith(sig) for sig in DLP_HEADER_SIGNATURES)
    except Exception:
        return False


async def _sandbox_write(sandbox_id: str, path: str, content: bytes) -> dict:
    """写文件到沙箱。"""
    _e2b_env()
    try:
        sb = E2BSandbox.connect(sandbox_id)
        sb.files.write(path, content)
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": f"写沙箱文件失败: {str(e)}"}


async def _sandbox_run(sandbox_id: str, cmd: str, timeout: int = 300) -> dict:
    """沙箱内执行命令（commands.run 返回结构 getattr 容错，兼容不同 SDK 版本）。"""
    _e2b_env()
    try:
        sb = E2BSandbox.connect(sandbox_id)
        res = sb.commands.run(cmd, timeout=timeout)
        exit_code = getattr(res, "exit_code", None)
        stdout = getattr(res, "stdout", None)
        stderr = getattr(res, "stderr", None)
        if exit_code is None:
            # 部分 SDK 版本无 exit_code，用 error 字段
            error = getattr(res, "error", None)
            success = not error
        else:
            success = exit_code == 0
        return {
            "success": success,
            "stdout": str(stdout or ""),
            "stderr": str(stderr or ""),
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}


async def _sandbox_read(sandbox_id: str, path: str) -> dict:
    """读沙箱文件（bytes）。"""
    _e2b_env()
    try:
        sb = E2BSandbox.connect(sandbox_id)
        content = sb.files.read(path, format="bytes")
        return {"success": True, "data": content, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": f"读沙箱文件失败: {str(e)}"}


def _fail(step: str, error: str) -> dict:
    return {"success": False, "step": step, "error": error}


@mcp.tool()
async def run_pdf_diff_stage1(
    doc_a_path: str,
    doc_b_path: str,
    sandbox_id: str,
) -> dict:
    """
    技术协议 PDF 差异比对 - 阶段1（确定性流程，禁止自行编写替代代码）。

    内部固定执行：DLP 解密（如需要）→ 上传 skill 脚本 → extract_pdf.py 提取
    两份文档 → diff_structures.py 结构粗筛 → diff_pages.py 机械定位差异页。
    每步产物确定性校验，失败返回具体 step 与错误。

    Args:
        doc_a_path: 文档 A PDF 路径（/uploads/... 虚拟路径或 MCP server 物理路径）
        doc_b_path: 文档 B PDF 路径
        sandbox_id: 目标沙箱 ID

    Returns:
        {"success": bool, "step": str|None, "error": str|None,
         "doc_a_json": "/home/user/docA.json", "doc_b_json": "/home/user/docB.json",
         "diff_pages_json": "/home/user/diff_pages.json",
         "diff_summary": str, "structure_summary": str}
        后续请基于 diff_pages 候选差异页 + extract 全文做语义判定，整理 diff.json。
    """
    a = _resolve_host_path(doc_a_path)
    b = _resolve_host_path(doc_b_path)
    for p in (a, b):
        if not os.path.isfile(p):
            return _fail("validate", f"PDF 文件不存在: {p}")
    if not _is_valid_sandbox_id(sandbox_id):
        return _fail("validate", f"sandbox_id 无效: {sandbox_id}")

    # ① 上传 skill 脚本
    scripts_dir = _find_skill_scripts_dir(PDF_DIFF_SKILL_NAME)
    if not scripts_dir:
        return _fail("scripts", f"未找到 skill {PDF_DIFF_SKILL_NAME} 的 scripts 目录（AGENT_WORKSPACE 未设置？）")
    for name in PDF_DIFF_SCRIPTS:
        src = f"{scripts_dir}/{name}"
        if not os.path.isfile(src):
            return _fail("scripts", f"脚本缺失: {name}")
        with open(src, "rb") as f:
            content = f.read()
        r = await _sandbox_write(sandbox_id, f"/home/user/scripts/{name}", content)
        if not r["success"]:
            return _fail("scripts", r["error"])

    # ② 准备 PDF（DLP 加密 → 解密写明文；否则直接上传）
    for src, dst in ((a, "docA.pdf"), (b, "docB.pdf")):
        if _dlp_encrypted(src):
            dec = await _decrypt_file_internal(src)
            if not dec["success"]:
                return _fail("decrypt", f"{src}: {dec['error']}")
            content = dec["data"]
        else:
            with open(src, "rb") as f:
                content = f.read()
        r = await _sandbox_write(sandbox_id, f"/home/user/{dst}", content)
        if not r["success"]:
            return _fail("upload", r["error"])

    # ③ 提取（pdfplumber 依赖保障：先检查，缺则装）
    r = await _sandbox_run(sandbox_id, 'cd /home/user && python -c "import pdfplumber"')
    if not r["success"]:
        r2 = await _sandbox_run(sandbox_id, "cd /home/user && pip install -q pdfplumber")
        if not r2["success"]:
            return _fail("extract", f"pdfplumber 安装失败: {r2['stderr'][:300]}")
    for dst in ("docA", "docB"):
        r = await _sandbox_run(
            sandbox_id,
            f"cd /home/user && python scripts/extract_pdf.py {dst}.pdf --chapters --out {dst}.json",
        )
        if not r["success"]:
            return _fail("extract", f"extract_pdf.py {dst}: {r['stderr'][:300]}")
        chk = await _sandbox_read(sandbox_id, f"/home/user/{dst}.json")
        if not chk["success"] or not chk["data"]:
            return _fail("extract", f"{dst}.json 未生成")

    # ④ 结构粗筛（stdout 作为摘要，不阻塞后续）
    r = await _sandbox_run(
        sandbox_id,
        "cd /home/user && python scripts/diff_structures.py docA.json docB.json --print",
    )
    struct_summary = r["stdout"] if r["success"] else f"(结构对齐失败: {r['stderr'][:200]})"

    # ⑤ 差异页定位
    r = await _sandbox_run(
        sandbox_id,
        "cd /home/user && python scripts/diff_pages.py docA.json docB.json --print --out diff_pages.json",
    )
    if not r["success"]:
        return _fail("diff_pages", r["stderr"][:300])
    chk = await _sandbox_read(sandbox_id, "/home/user/diff_pages.json")
    if not chk["success"] or not chk["data"]:
        return _fail("diff_pages", "diff_pages.json 未生成")

    return {
        "success": True,
        "step": None,
        "error": None,
        "doc_a_json": "/home/user/docA.json",
        "doc_b_json": "/home/user/docB.json",
        "diff_pages_json": "/home/user/diff_pages.json",
        "diff_summary": (r["stdout"] or "")[:2000] or "（无输出）",
        "structure_summary": struct_summary[:1000],
    }


@mcp.tool()
async def run_pdf_diff_stage3(
    diff_json_path: str,
    out_prefix: str,
    sandbox_id: str,
) -> dict:
    """
    技术协议 PDF 差异比对 - 阶段3（确定性流程，禁止自行编写替代代码）。

    基于你整理好的 diff.json 生成 Markdown/HTML 差异报告到沙箱，并返回
    沙箱内路径。下载回 reports 目录用 download_from_sandbox。

    Args:
        diff_json_path: diff.json 路径（/reports/... 虚拟路径或 MCP server 物理路径）
        out_prefix: 报告文件名前缀（如 "技术协议差异对比报告_阴极vs阳极_20260819_1"），
                    仅允许中文/字母/数字/中划线/下划线/点
        sandbox_id: 目标沙箱 ID

    Returns:
        {"success": bool, "step": str|None, "error": str|None,
         "report_md": "/home/user/技术协议差异对比报告_xxx.md",
         "report_html": "/home/user/技术协议差异对比报告_xxx.html"}
    """
    host = _resolve_host_path(diff_json_path)
    if not os.path.isfile(host):
        return _fail("validate", f"diff.json 不存在: {host}")
    if not _is_valid_sandbox_id(sandbox_id):
        return _fail("validate", f"sandbox_id 无效: {sandbox_id}")
    if not _REPORT_NAME_RE.match(out_prefix or ""):
        return _fail("validate", f"out_prefix 含非法字符: {out_prefix!r}")

    # 上传 generate_report.py + diff.json 到沙箱
    scripts_dir = _find_skill_scripts_dir(PDF_DIFF_SKILL_NAME)
    if not scripts_dir:
        return _fail("scripts", f"未找到 skill {PDF_DIFF_SKILL_NAME} 的 scripts 目录")
    with open(f"{scripts_dir}/generate_report.py", "rb") as f:
        r = await _sandbox_write(sandbox_id, "/home/user/scripts/generate_report.py", f.read())
    if not r["success"]:
        return _fail("scripts", r["error"])
    with open(host, "rb") as f:
        r = await _sandbox_write(sandbox_id, "/home/user/diff.json", f.read())
    if not r["success"]:
        return _fail("upload", r["error"])

    md_name = f"技术协议差异对比报告_{out_prefix}.md"
    html_name = f"技术协议差异对比报告_{out_prefix}.html"
    for fmt, out_name in (("md", md_name), ("html", html_name)):
        r = await _sandbox_run(
            sandbox_id,
            f"cd /home/user && python scripts/generate_report.py diff.json --format {fmt} --out \"{out_name}\"",
        )
        if not r["success"]:
            return _fail("report", f"generate_report.py {fmt}: {r['stderr'][:300]}")
        chk = await _sandbox_read(sandbox_id, f"/home/user/{out_name}")
        if not chk["success"] or not chk["data"] or len(chk["data"]) < 100:
            return _fail("report", f"{out_name} 未生成或过小")

    return {
        "success": True,
        "step": None,
        "error": None,
        "report_md": f"/home/user/{md_name}",
        "report_html": f"/home/user/{html_name}",
    }


# @mcp.tool()
# async def read_excel(file_path: str, sheet_name: Optional[str] = None) -> dict:
#     """
#     解密并读取 Excel 文件，返回结构化数据

#     :param file_path: Excel 文件路径（加密状态）
#     :param sheet_name: 工作表名称，默认第一个
#     :return: {"success": bool, "data": list[dict], "columns": list, "error": str}
#     """

#     import pandas as pd

#     try:
#         # 1.解密到内存
#         decrypt_result = await _decrypt_file_internal(file_path)
#         if not decrypt_result["success"]:
#             return {
#                 "success": False,
#                 "data": None,
#                 "columns": None,
#                 "row_count": 0,
#                 "error": decrypt_result["error"],
#             }

#         # 2. 内存中读取 Excel
#         excel_buff = io.BytesIO(decrypt_result["data"])

#         if sheet_name:
#             df = pd.read_excel(excel_buff, sheet_name=sheet_name)
#         else:
#             df = pd.read_excel(excel_buff)

#         # 3. 转换为结构化数据
#         data = df.fillna("").to_dict(orient="records")
#         columns = df.columns.to_list()

#         return {
#             "success": True,
#             "data": data,
#             "columns": columns,
#             "row_count": len(data),
#             "error": None,
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "data": None,
#             "columns": None,
#             "row_count": 0,
#             "error": str(e),
#         }


# @mcp.tool()
# async def read_pdf_text(file_path: str) -> dict:
#     """
#     解密并提取 PDF 文本内容

#     :param file_path: PDF 文件路径（加密状态）
#     :return: {"success": bool, "text": str, "page_count": int, "error": str}
#     """

#     from PyPDF2 import PdfReader

#     try:
#         decrypt_result = await _decrypt_file_internal(file_path)
#         if not decrypt_result["success"]:
#             return {
#                 "success": False,
#                 "text": None,
#                 "page_count": 0,
#                 "error": decrypt_result["error"],
#             }

#         pdf_buffer = io.BytesIO(decrypt_result["data"])
#         reader = PdfReader(pdf_buffer)

#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() + "\n"

#         page_count = len(reader.pages)
#         pdf_buffer.close()

#         return {
#             "success": True,
#             "text": text.strip(),
#             "page_count": page_count,
#             "error": None,
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "text": None,
#             "page_count": 0,
#             "error": str(e),
#         }


# @mcp.tool()
# async def filter_excel(file_path: str, column: str, value: str) -> dict:

#     import pandas as pd

#     try:
#         decrypt_result = await _decrypt_file_internal(file_path)
#         if not decrypt_result["success"]:
#             return {
#                 "success": False,
#                 "data": None,
#                 "row_count": 0,
#                 "error": decrypt_result["error"],
#             }

#         excel_buff = io.BytesIO(decrypt_result["data"])
#         df = pd.read_excel(excel_buff)
#         excel_buff.close()

#         # 筛选
#         filtered = df[df[column].astype(str) == value]

#         return {
#             "success": True,
#             "data": filtered.fillna("").to_dict(orient="records"),
#             "row_count": len(filtered),
#             "error": None,
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "data": None,
#             "row_count": 0,
#             "error": str(e),
#         }


# @mcp.tool()
# async def list_excel_sheets(file_path: str) -> dict:

#     import pandas as pd

#     try:
#         decrypt_result = await _decrypt_file_internal(file_path)
#         if not decrypt_result["success"]:
#             return {"success": False, "sheets": None, "error": decrypt_result["error"]}

#         # 读取所有sheet名称
#         excel_buffer = io.BytesIO(decrypt_result["data"])
#         xl = pd.ExcelFile(excel_buffer)
#         sheet_names = xl.sheet_names
#         excel_buffer.close()

#         return {
#             "success": True,
#             "sheets": sheet_names,
#             "error": None,
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "sheets": None,
#             "error": str(e),
#         }


if __name__ == "__main__":
    print("启动 企业文件解密 MCP 服务 (Python)")
    print("服务地址: http://127.0.0.1:8000/mcp")
    print("\n可用工具:")
    print("  - decrypt_file: 解密文件返回字节流")
    print("  - read_excel: 解密并读取 Excel")
    print("  - read_pdf_text: 解密并提取 PDF 文本")
    print("  - filter_excel: 按条件筛选 Excel 数据")
    print("  - list_excel_sheets: 列出 Excel 所有工作表")
    print("\n按 Ctrl+C 停止服务")

    mcp.run(
        host="127.0.0.1",
        port=8000,
        transport="streamable-http",
    )
