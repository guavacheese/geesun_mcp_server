import base64
import os
import io
import json
import re
import shlex
import sys
import asyncio
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Optional
import httpx
import logging
from pathlib import Path
from e2b_code_interpreter import Sandbox as E2BSandbox

load_dotenv()

logger = logging.getLogger(__name__)
# ★ 2026-10-20：MCP 侧此前一条 `[OCR]` 日志都看不到（add0229b 排查多花 1 小时）。
#   `getLogger(__name__)` 没有任何 handler，root 默认 level=WARNING 且无 handler
#   ⇒ 所有 logger.info 被 lastResort（WARNING）丢弃，只有 `[DLP]`（warning）可见。
#   这里给本模块挂一个 stderr handler（**绝不能是 stdout**：stdio 传输会把
#   stdout 当协议流）。level=INFO 只影响本模块，不动 global。
_log_handler = logging.StreamHandler(sys.stderr)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(_log_handler)
logger.setLevel(logging.INFO)

decrypt_api_url = os.getenv("DECRYPT_API_URL")
mcp = FastMCP("decrypt-file")

# v3.1 护栏：公司 DLP 加密文件头魔数。
# 判断"文件是否加密"不靠扩展名（txt/py 也可能被加密），靠文件头特征。
DLP_HEADER_SIGNATURES: tuple[bytes, ...] = (b"%TSD-Header",)

# ── 上传护栏（Stage 1，2026-10-10）──────────────────────────────────
# 背景（2026-10-09 事故 会话 cecfdd84）：256.4MiB 的 DLP 加密 PDF 走 upload_to_sandbox，
# 原实现先 `f.read()` 全量读 256.4MiB 判头，命中后再转 decrypt_and_upload_to_sandbox，
# 而后者又 read 一遍、此时调用方的 content 引用尚未释放 → 峰值 2N ≈ 512.8MiB 常驻，
# 越过当时 512MiB 的容器限额 → cgroup OOM-kill（内核 memcg 记录 10-08 6 次 / 10-09 6 次，
# swarm task 连续 exit 137）。MCP 单副本共享，一炸会把全体用户的 in-flight 工具调用一起打断，
# agent 侧表现为工具永不返回、被 LangGraph 记成 "was cancelled - another message came in"，
# 模型进而误诊为"虚拟文件系统元数据不一致"，引导用户重传（方向全错）。
# 修复分两阶段：
#   Stage 1（本项）：① 先探 16B 判头，消掉一次全量读（省 1N）；
#                    ② getsize 预检 + 结构化中文错误，让失败可读、不再以"炸掉共享容器"的方式失败；
#                    ③ 解密超时可配（原 30s 对 256MiB 级文件明显不够，会被超时误伤）。
#   Stage 2（待 spike 证实后）：全程流式（解密请求/响应 + 沙箱写入都不再全量驻留），
#                    并把阻塞的 E2B 同步调用挪出事件循环线程。
MAX_UPLOAD_MB = int(os.getenv("MCP_MAX_UPLOAD_MB", "2048"))
DECRYPT_TIMEOUT_S = int(os.getenv("DECRYPT_TIMEOUT_S", "600"))


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

    # ── 同步段全部下沉 asyncio.to_thread ──
    # 2026-10-19 事故 653d8515：`async def` 工具里直接跑同步 e2b 调用
    # （connect/files.write/commands.run）会把 MCP 单事件循环堵满整个命令时长
    # ——OCR 预算封顶 3600s，实际堵了 55 分钟，期间所有请求（含重连）在队列里
    # 干等，工具响应永远送不回 agent。e2b SDK 无 async 客户端，唯一解法是把
    # 同步段移出循环。
    # 参照 deer-flow `sandbox/tools.py` 的 `_run_sync_tool_after_async_sandbox_init`。
    def _sync() -> dict:
        with open(found_path, "rb") as f:
            content = f.read()

        # connect 必须在 try 内：E2B 控制面不可达/沙箱已销毁时 connect 抛异常
        # （e2b 解析 404 响应可能 KeyError: 'code'），不能让它冒泡成工具级 ToolException
        try:
            sb = E2BSandbox.connect(sandbox_id)
            sb.files.write(sandbox_path, content)
        except Exception as e:
            return {
                "success": False,
                "sandbox_path": sandbox_path,
                "size": 0,
                "error": str(e),
            }

        return {"success": True, "sandbox_path": sandbox_path, "size": len(content)}

    return await asyncio.to_thread(_sync)


@mcp.tool()
async def upload_to_sandbox(
    file_path: str,
    remote_path: str,
    sandbox_id: str,
) -> dict:
    """
    上传宿主机的文件到沙箱（自动处理 DLP 加密），不经过 LLM 上下文。

    适用于 XML、TXT 等不需要解密的文件；若文件被 DLP 加密（文件头 %TSD-Header），
    会**自动**转解密链路，调用方无需自己判断、也无需换工具。

    大小上限由环境变量 MCP_MAX_UPLOAD_MB 控制（当前见返回体 max_upload_mb）；
    超限不会静默截断，会返回 error_code=file_too_large 的结构化中文错误。

    Args:
        file_path: 虚拟路径（/uploads/{user_id}/{session_id}/{filename}、
                   /reports/{user_id}/{session_id}/{filename}）或 MCP server
                   物理路径；自动转换
        remote_path: 沙箱内的目标路径（如 /home/user/data.xml）
        sandbox_id: 目标沙箱 ID

    Returns:
        {"success": bool, "error_code": str | None, "sandbox_path": str | None,
         "size": int, "error": str | None}
        失败时另带 virtual_path / physical_path / size_human / max_upload_mb / hint，
        error_code ∈ {file_not_found, file_too_large, read_failed,
                      decrypt_failed, decrypt_timeout, decrypt_api_error,
                      sandbox_write_failed}
    """
    # 路径转换：/uploads/、/reports/ 虚拟路径 → 物理路径（复用 _resolve_host_path）
    # original_path 保留调用方原样传入的（通常是 /uploads/... 虚拟路径），
    # 用于结构化错误体里同时给出【虚拟路径】与【物理路径】——治"只报虚拟路径、
    # 让人看不出映射缺失"的老毛病（2026-10-09 模型正是据此误诊为虚拟 FS 元数据不一致）。
    original_path = file_path
    file_path = _resolve_host_path(file_path)

    # ── Stage 1 预检：路径存在性 + 文件大小上限 ──
    # 失败直接返回结构化中文错误：不抛异常、不进入读文件路径。
    # 超限文件在预检就被挡下，不再以"把共享容器内存撑爆"的方式失败。
    bad = _upload_precheck(file_path, original_path)
    if bad:
        return bad

    # ── v3.1 护栏升级：检测到 DLP 加密 → 自动切换为解密上传 ──
    # 不再返回错误让 LLM 自行换工具：实测长上下文下模型会忽略该提示、反复
    # 调 upload_to_sandbox 失败（server.log 已实锤 tool_calls 连续两次 upload_to_sandbox 均 FAIL）。
    # 兜底复用 decrypt_and_upload_to_sandbox 的合规解密链路（内存解密→直传沙箱，明文不落盘），
    # 安全姿态不变，仅移除对「调用方正确选工具」的依赖。解密失败如实透传，不假装成功。
    # ── Stage 1 改动：判头改为"只读 16 字节"（_dlp_encrypted）──
    # 原实现是"先 f.read() 全量读 N 字节判头 → 命中后再转 decrypt_and_upload_to_sandbox
    # 让它 read 一遍"，此时调用方的 content 引用尚未释放 → 峰值 2N 常驻
    # （256.4MiB 文件即 512.8MiB），正是 2026-10-09 OOM-kill 的直接原因。现改为先探头。
    if _dlp_encrypted(file_path):
        logger.warning(
            "[DLP] 检测到加密文件 %s（%s），upload_to_sandbox 自动切换为解密上传"
            "（等价于 decrypt_and_upload_to_sandbox，明文不落盘）",
            file_path,
            _human_size(os.path.getsize(file_path)),
        )
        return await decrypt_and_upload_to_sandbox(file_path, remote_path, sandbox_id)

    # ── 同步段下沉 asyncio.to_thread（2026-10-19 653d8515 事故，详见 copy_script_to_sandbox 注释）──
    def _read_content() -> bytes:
        with open(file_path, "rb") as f:
            return f.read()

    try:
        content = await asyncio.to_thread(_read_content)
    except Exception as e:
        return {
            "success": False,
            "error_code": "read_failed",
            "error": f"读取文件失败: {e}",
            "virtual_path": original_path,
            "physical_path": file_path,
            "sandbox_path": None,
            "size": 0,
        }

    # connect 必须在 try 内：E2B 控制面不可达/沙箱已销毁时 connect 抛异常
    # （e2b 解析 404 响应可能 KeyError: 'code'），不能让它冒泡成工具级 ToolException
    def _sync_upload() -> dict:
        try:
            sb = E2BSandbox.connect(sandbox_id)
            sb.files.write(remote_path, content)
            return {
                "success": True,
                "error_code": None,
                "sandbox_path": remote_path,
                "size": len(content),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error_code": "sandbox_write_failed",
                "error": f"连接/写入沙箱 {sandbox_id} 失败: {e}",
                "virtual_path": original_path,
                "physical_path": file_path,
                "sandbox_path": None,
                "size": 0,
            }

    return await asyncio.to_thread(_sync_upload)


@mcp.tool()
async def decrypt_and_upload_to_sandbox(
    file_path: str,
    remote_path: str,
    sandbox_id: str,
) -> dict:
    """
    读取被公司加密的文件：解密并直接上传到 CubeSandbox 沙箱内（与 upload_to_sandbox 等价，后者已内置 DLP 兜底自动解密）。

    公司 DLP 会加密 PDF / Excel / Word，以及 txt、py 等任意文本类文件
    （判断是否加密看文件头 %TSD-Header，不看扩展名）。
    本工具与 upload_to_sandbox 等价：upload_to_sandbox 现也已内置 DLP 兜底（检测到 %TSD-Header 自动解密上传），两者均可解密上传，明文不落盘。

    解密后的明文会写入沙箱文件系统，不经过 LLM 上下文，不落地宿主机磁盘。

    Args:
        file_path: 虚拟路径（/uploads/{user_id}/{session_id}/{filename}、
                   /reports/...）或 MCP server 物理路径；自动转换
        remote_path: 沙箱内的目标路径（如 /home/user/data.xlsx）
        sandbox_id: 目标沙箱 ID，从聊天上下文的"沙箱 ID"字段获取

    Returns:
        {"success": bool, "error_code": str | None, "sandbox_path": str | None,
         "size": int, "error": str | None}
        失败时另带 virtual_path / physical_path / size_human / max_upload_mb / hint。
        大小上限由 MCP_MAX_UPLOAD_MB 控制（默认 2048 MiB）；解密超时由
        DECRYPT_TIMEOUT_S 控制（默认 600s）。
    """
    # 路径转换：虚拟路径 → 物理路径（复用 _resolve_host_path，与 upload_to_sandbox 一致，
    # 同时支持 /uploads/ 与 /reports/，消除原实现只认 /uploads/ 的盲区）
    original_path = file_path
    file_path = _resolve_host_path(file_path)

    # ── Stage 1 预检：与 upload_to_sandbox 共用同一套 ──
    # 本工具会被模型直接调用（2026-10-09 会话 cecfdd84 就是绕道它+decrypt_file_to_base64），
    # 所以预检不能只挂在 upload_to_sandbox 上。
    bad = _upload_precheck(file_path, original_path)
    if bad:
        return bad

    # 1. 解密到内存
    result = await _decrypt_file_internal(file_path)
    if not result["success"]:
        return {
            "success": False,
            "error_code": result.get("error_code") or "decrypt_failed",
            "error": result["error"],
            "virtual_path": original_path,
            "physical_path": file_path,
            "sandbox_path": None,
            "size": 0,
        }

    # ── 同步段下沉 asyncio.to_thread（2026-10-19 653d8515 事故，详见 copy_script_to_sandbox 注释）──
    # connect 必须在 try 内：E2B 控制面不可达/沙箱已销毁时 connect 抛异常
    # （e2b 解析 404 响应可能 KeyError: 'code'），不能让它冒泡成工具级 ToolException
    def _sync_upload() -> dict:
        try:
            sb = E2BSandbox.connect(sandbox_id)
            sb.files.write(remote_path, result["data"])
            return {
                "success": True,
                "error_code": None,
                "sandbox_path": remote_path,
                "size": len(result["data"]),
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error_code": "sandbox_write_failed",
                "error": f"连接/写入沙箱 {sandbox_id} 失败: {e}",
                "virtual_path": original_path,
                "physical_path": file_path,
                "sandbox_path": None,
                "size": 0,
            }

    return await asyncio.to_thread(_sync_upload)
    # ☝️ 注意：不调 sb.kill()，沙箱由 Agent 管理
    #    connect 创建的对象会在函数返回后被 Python GC 回收
    #    对 Agent 的沙箱无影响，文件已成功写入


async def _decrypt_file_internal(file_path: str) -> dict:
    """内部解密函数（供其他函数调用）

    Stage 1 改动：
      - 超时由硬编码 30s 改为 DECRYPT_TIMEOUT_S（默认 600s）——256MiB 级文件的
        上传+解密远超 30s，原值会把"内存够但慢"的文件误伤成超时失败。
      - 失败带回 error_code（file_not_found / decrypt_timeout / decrypt_api_error /
        decrypt_failed），并保留原始异常文本，供上层拼结构化错误。
      - 非 200 响应附带响应体片段（截断），便于分辨是网关参数问题还是 5xx。

    注：此处 `f.read()` 全量读是 2N 峰值中"第二份"的来源（解密请求体），
    Stage 1 保留（httpx 的 multipart files= 需要已知长度的字节串），
    改流式需连同沙箱写入一起改，已列入 Stage 2 spike。
    """
    try:
        # 同步段下沉 asyncio.to_thread（2026-10-19 653d8515 事故：256MiB 文件的
        # 同步 read 会把事件循环堵死数秒，长任务场景下与 connect/commands.run 同罪）
        def _read_all() -> bytes:
            with open(file_path, "rb") as f:
                return f.read()

        file_content = await asyncio.to_thread(_read_all)
    except OSError as e:
        return {
            "success": False,
            "error_code": "file_not_found",
            "data": None,
            "error": f"读取待解密文件失败: {e}（物理路径 {file_path}）",
            "size": 0,
        }

    # 调用解密API
    files = {
        "file": (
            os.path.basename(file_path),
            file_content,
        )
    }

    try:
        async with httpx.AsyncClient(timeout=DECRYPT_TIMEOUT_S) as client:
            response = await client.post(
                decrypt_api_url,
                files=files,
            )
    except httpx.TimeoutException as e:
        return {
            "success": False,
            "error_code": "decrypt_timeout",
            "data": None,
            "error": (
                f"解密API超时（当前上限 {DECRYPT_TIMEOUT_S}s，文件 {_human_size(len(file_content))}）: {e}。"
                f"若确认是正常大文件，可调高环境变量 DECRYPT_TIMEOUT_S 后重试。"
            ),
            "size": 0,
        }
    except Exception as e:
        # 解密网关不可达 / TLS 校验失败等——原实现只回 str(e)，
        # 不说是哪一步、连的哪个地址，排查时只能猜，这里补全上下文。
        return {
            "success": False,
            "error_code": "decrypt_failed",
            "data": None,
            "error": f"调用解密API失败（{decrypt_api_url}）: {type(e).__name__}: {e}",
            "size": 0,
        }

    if response.status_code != 200:
        return {
            "success": False,
            "error_code": "decrypt_api_error",
            "data": None,
            "error": (
                f"解密API返回错误 {response.status_code}"
                f"（{decrypt_api_url}，文件 {_human_size(len(file_content))}）: "
                f"{response.content[:300].decode('utf-8', 'replace')}"
            ),
            "size": 0,
        }
    return {
        "success": True,
        "error_code": None,
        "data": response.content,
        "error": None,
        "size": len(response.content),
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

    # ── 同步段下沉 asyncio.to_thread（2026-10-19 653d8515 事故，详见 copy_script_to_sandbox 注释）──
    # connect 必须在 try 内：E2B 控制面不可达/沙箱已销毁时 connect 抛异常
    # （e2b 解析 404 响应可能 KeyError: 'code'），不能让它冒泡成工具级 ToolException
    def _sync_read() -> dict:
        try:
            sb = E2BSandbox.connect(sandbox_id)
            content = sb.files.read(sandbox_path, format="bytes")
        except Exception as e:
            return {
                "success": False,
                "host_path": None,
                "size": 0,
                "error": f"读取沙箱文件失败: {str(e)}",
            }
        return {"success": True, "content": content}

    r = await asyncio.to_thread(_sync_read)
    if not r["success"]:
        return {"success": False, "host_path": None, "size": 0, "error": r["error"]}
    content = r["content"]

    # 写宿主机同样下沉（大文件写盘 + makedirs 也是同步 IO）
    def _write_host() -> None:
        os.makedirs(os.path.dirname(host_path), exist_ok=True)
        with open(host_path, "wb") as f:
            f.write(content)

    try:
        await asyncio.to_thread(_write_host)
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

      :param file_path: 要解密的文件路径（加密状态）。支持虚拟路径
                        （/uploads/{user_id}/{session_id}/{filename}、
                        /reports/{user_id}/{session_id}/{filename}）与 MCP server
                        物理路径，自动转换。
      :return: {"success": bool, "data": bytes, "error": str, "size": int}

    ⚠️ 本工具把整个明文以 base64 返回给调用方（体积约为原文 1.33 倍，且会进入
    LLM 上下文）。仅用于小文件的"可读性确认"；需要把大文件送进沙箱请用
    upload_to_sandbox / decrypt_and_upload_to_sandbox。

    Stage 1 旁支修复：原实现漏调 _resolve_host_path，传 /uploads/... 虚拟路径时
    会直接 open() 该虚拟路径 → [Errno 2] No such file or directory，而报错里
    只有虚拟路径、看不出是映射缺失（2026-10-09 会话 cecfdd84 实测 2 次，
    并被模型误读为虚拟 FS 元数据不一致）。现与另两个上传工具对齐。
    """
    # 路径转换：虚拟路径 → 物理路径（与 upload_to_sandbox /
    # decrypt_and_upload_to_sandbox 一致；这是三者中唯一漏掉的，已补齐）
    original_path = file_path
    file_path = _resolve_host_path(file_path)

    result = await _decrypt_file_internal(file_path)

    if result["success"] and result["data"]:
        result["data"] = base64.b64encode(result["data"]).decode("utf-8")
        result["virtual_path"] = original_path
        result["physical_path"] = file_path
    else:
        # 失败时补上解析后的物理路径——只给虚拟路径会让调用方看不出是映射问题
        result["virtual_path"] = original_path
        result["physical_path"] = file_path
        result["error"] = f"{result.get('error')}（虚拟路径 {original_path} → 物理路径 {file_path}）"

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
    "align_blocks.py",
    "generate_report.py",
)
_REPORT_NAME_RE = re.compile(r"^[\w\u4e00-\u9fa5\-_\.]+$")
# 扫描件/图片型 PDF 判定：平均每页字符数低于此阈值视为无文本层（需 OCR，无法文本比对）
SCAN_MIN_AVG_CHARS = 50


def _resolve_host_path(path: str) -> str:
    """虚拟路径（/uploads/、/reports/）→ MCP server 物理路径；否则原样返回。

    幂等（Stage 1 补）：若传入的已经是解析后的物理路径（以对应 root 开头），原样返回。
    必要性：upload_to_sandbox 会先把虚拟路径解析成物理路径，命中 DLP 后再把该物理路径
    转交给 decrypt_and_upload_to_sandbox，而后者也会调本函数——若 UPLOAD_ROOT 恰好是
    "/uploads" 这类与虚拟前缀重合的值，非幂等实现会二次前缀成 /uploads/uploads/...。
    同时 _upload_precheck 需要在两个工具入口都拿到"稳定"的物理路径。
    """
    if path.startswith("/uploads/"):
        root = os.environ.get("UPLOAD_ROOT", "")
        if root and not path.startswith(root.rstrip("/") + "/"):
            return f"{root}/{path[len('/uploads/'):]}"
    if path.startswith("/reports/"):
        root = os.environ.get("REPORT_ROOT", "")
        if root and not path.startswith(root.rstrip("/") + "/"):
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


def _human_size(n: int) -> str:
    """字节数 → 人类可读（结构化错误体与日志共用，避免各处重复拼字符串）。"""
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} GiB"


def _upload_precheck(physical_path: str, virtual_path: str) -> dict | None:
    """上传前预检：物理路径存在性 + 文件大小上限。

    返回 None 表示通过；返回 dict 表示结构化错误（直接作为工具返回值，不抛异常）。

    设计要点（Stage 1）：
      - 错误体同时给【虚拟路径】与【解析后的物理路径】+ 当前 UPLOAD_ROOT/REPORT_ROOT。
        2026-10-09 事故里 MCP 只回虚拟路径，模型看不出是路径映射缺失，把
        "No such file or directory" 误读成"虚拟文件系统元数据与数据不一致"，
        进而引导用户重传文件（方向全错、浪费一轮）。
      - 超限只在预检拦下，不进读文件路径——超限文件的失败方式从
        "把共享容器内存撑爆"变成"一条可读的中文错误"。
      - 沿用既有返回键（success / sandbox_path / size / error）保持向后兼容，
        新增键为附加信息，老调用方不受影响。
    """
    if not os.path.isfile(physical_path):
        return {
            "success": False,
            "error_code": "file_not_found",
            "error": (
                f"文件不存在：{physical_path}。调用传入的路径为 {virtual_path}；"
                f"当前 UPLOAD_ROOT={os.environ.get('UPLOAD_ROOT', '(未设置)')}，"
                f"REPORT_ROOT={os.environ.get('REPORT_ROOT', '(未设置)')}"
            ),
            "virtual_path": virtual_path,
            "physical_path": physical_path,
            "sandbox_path": None,
            "size": 0,
            "size_bytes": 0,
            "max_upload_mb": MAX_UPLOAD_MB,
            "hint": (
                "请核对文件名与目录层级（虚拟路径形如 /uploads/{user_id}/{session_id}/{filename}）。"
                "注意：本工具只认物理路径，不认 agent 侧虚拟文件系统；若文件确实存在，"
                "多半是虚拟路径拼错或用的是沙箱内路径。"
            ),
        }

    size = os.path.getsize(physical_path)
    limit_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if size > limit_bytes:
        return {
            "success": False,
            "error_code": "file_too_large",
            "error": (
                f"文件超过上传上限：{physical_path} 实际 {_human_size(size)}（{size} 字节），"
                f"上限 {MAX_UPLOAD_MB} MiB（{limit_bytes} 字节）。"
                f"请在沙箱内用脚本处理，或拆分为多个较小文件后分别上传。"
            ),
            "virtual_path": virtual_path,
            "physical_path": physical_path,
            "sandbox_path": None,
            "size": size,
            "size_bytes": size,
            "size_human": _human_size(size),
            "max_upload_mb": MAX_UPLOAD_MB,
            "hint": (
                f"上限由环境变量 MCP_MAX_UPLOAD_MB 控制（当前 {MAX_UPLOAD_MB}）；"
                "如需放宽请运维调整后重建 MCP 服务。"
            ),
        }
    return None


async def _sandbox_write(sandbox_id: str, path: str, content: bytes) -> dict:
    """写文件到沙箱。

    同步段（connect + files.write）下沉 asyncio.to_thread —— 2026-10-19 653d8515
    事故的直接根因：async def 里直跑同步 e2b 会把 MCP 单事件循环堵满整个命令时长。
    """
    _e2b_env()

    def _sync() -> dict:
        try:
            sb = E2BSandbox.connect(sandbox_id)
            sb.files.write(path, content)
            return {"success": True, "error": None}
        except Exception as e:
            return {"success": False, "error": f"写沙箱文件失败: {str(e)}"}

    return await asyncio.to_thread(_sync)


async def _sandbox_run(sandbox_id: str, cmd: str, timeout: int = 300) -> dict:
    """沙箱内执行命令（commands.run 返回结构 getattr 容错，兼容不同 SDK 版本）。

    同步段（connect + commands.run）下沉 asyncio.to_thread —— 2026-10-19 653d8515
    事故的直接根因，详见 _sandbox_write 注释。
    """
    _e2b_env()

    def _sync() -> dict:
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

    return await asyncio.to_thread(_sync)


async def _sandbox_read(sandbox_id: str, path: str) -> dict:
    """读沙箱文件（bytes）。同步段下沉 asyncio.to_thread，同 _sandbox_write。"""
    _e2b_env()

    def _sync() -> dict:
        try:
            sb = E2BSandbox.connect(sandbox_id)
            content = sb.files.read(path, format="bytes")
            return {"success": True, "data": content, "error": None}
        except Exception as e:
            return {"success": False, "data": None, "error": f"读沙箱文件失败: {str(e)}"}

    return await asyncio.to_thread(_sync)


def _fail(step: str, error: str) -> dict:
    return {"success": False, "step": step, "error": error}


async def _trace_check(sandbox_id: str, diff_json_data: dict) -> list[str]:
    """
    宽松追溯校验（④）：基于沙箱 diff_pages.json 的 alignment，验证 diff.json 引用的章节号
    有机械证据（存在于对齐结果的章节集合）。凭空引用的章节号 → warning（不阻断，因 diff.json 为自由文本）。
    """
    r = await _sandbox_read(sandbox_id, "/home/user/diff_pages.json")
    if not r["success"] or not r["data"]:
        return ["[trace] 沙箱内无 diff_pages.json（未跑 stage1？），跳过追溯校验"]
    try:
        dp = json.loads(bytes(r["data"]).decode("utf-8"))
    except Exception as e:
        return [f"[trace] diff_pages.json 解析失败: {e}"]
    alignment = dp.get("alignment")
    if not alignment:
        return ["[trace] diff_pages.json 无 alignment 数据（非 aligned 产物），跳过追溯校验"]
    agg_nums = {x.get("chapter") for x in alignment.get("chapter_agg", []) if x.get("chapter")}
    text = json.dumps(diff_json_data, ensure_ascii=False)
    ref_nums = set(re.findall(r"([一二三四五六七八九十]+)、", text))
    missing = ref_nums - agg_nums
    if not missing:
        return []
    return [f"[trace] diff.json 引用了对齐结果中不存在的章节号 {sorted(missing)}（疑似凭空生成，请核对；"
            f"对齐章节号: {sorted(agg_nums)}）"]


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
        # 扫描件/图片型 PDF 检测：无文本层（平均每页字符数过低）时明确报错，不静默出空报告
        try:
            doc = json.loads(bytes(chk["data"]).decode("utf-8"))
            avg = doc.get("avg_chars_per_page", 0)
            if avg < SCAN_MIN_AVG_CHARS:
                return _fail(
                    "extract",
                    f"{dst}.pdf 疑似扫描件/图片型 PDF（平均每页仅 {avg} 字符，无文本层），无法做文本差异比对，需先 OCR。",
                )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _fail("extract", f"{dst}.json 解析失败: {e}")

    # ④ 结构粗筛（stdout 作为摘要，不阻塞后续；--aligned 内容感知对齐，消除页码偏移 cascade）
    r = await _sandbox_run(
        sandbox_id,
        "cd /home/user && python scripts/diff_structures.py docA.json docB.json --aligned --print",
    )
    struct_summary = r["stdout"] if r["success"] else f"(结构对齐失败: {r['stderr'][:200]})"

    # ⑤ 差异页定位（--aligned：条款级 LCS 反推差异页，diff_pages.json 内含 alignment 供 stage3 追溯校验）
    r = await _sandbox_run(
        sandbox_id,
        "cd /home/user && python scripts/diff_pages.py docA.json docB.json --aligned --print --out diff_pages.json",
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
        diff_json_path: diff.json 来源，二选一——
            (a) 宿主路径：/reports/... 虚拟路径 或 MCP server 物理路径
                （用 write_file 写入 reports 后传这个）；
            (b) 沙箱内路径：/home/user/... （diff.json 已在沙箱时直接传沙箱路径，
                如 /home/user/diff.json）。
            工具会自动识别前缀，两种都接受。
        out_prefix: 报告文件名前缀（如 "技术协议差异对比报告_阴极vs阳极_20260819_1"），
                    仅允许中文/字母/数字/中划线/下划线/点
        sandbox_id: 目标沙箱 ID

    Returns:
        {"success": bool, "step": str|None, "error": str|None,
         "report_md": "/home/user/技术协议差异对比报告_xxx.md",
         "report_html": "/home/user/技术协议差异对比报告_xxx.html"}
    """
    if not _is_valid_sandbox_id(sandbox_id):
        return _fail("validate", f"sandbox_id 无效: {sandbox_id}")
    if not _REPORT_NAME_RE.match(out_prefix or ""):
        return _fail("validate", f"out_prefix 含非法字符: {out_prefix!r}")

    # ── diff.json 来源识别：沙箱内路径 vs 宿主路径 ──
    in_sandbox = diff_json_path.startswith("/home/user/") or diff_json_path.startswith("/tmp/")
    diff_data = None
    if in_sandbox:
        if re.search(r"[;&|$`]", diff_json_path):
            return _fail("validate", f"diff_json_path 含非法字符: {diff_json_path!r}")
        chk = await _sandbox_read(sandbox_id, diff_json_path)
        if not chk["success"] or not chk["data"]:
            return _fail("validate", f"沙箱内 diff.json 不存在或不可读: {diff_json_path}")
        sandbox_diff_path = diff_json_path
        # JSON 合法性预校验（读回沙箱内容解析）
        try:
            diff_data = json.loads(bytes(chk["data"]).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _fail(
                "diff_json_validate",
                f"diff.json JSON 语法错误: {e}。请修复后重试（注意：JSON 不允许尾随逗号，数组/对象最后一项后不能有逗号）。",
            )
    else:
        host = _resolve_host_path(diff_json_path)
        if not os.path.isfile(host):
            return _fail(
                "validate",
                f"diff.json 不存在: {host}（若 diff.json 已在沙箱，请直接传沙箱路径如 /home/user/diff.json）",
            )
        # ── diff.json JSON 合法性预校验（黑盒入口守关口）──
        # diff.json 是唯一由模型手工产出的输入（语义判定步骤），模型手写 JSON
        # 易出尾随逗号等语法错误（2026-08-19 实测：line 60 `],` trailing comma
        # → generate_report.py json.load 抛 JSONDecodeError → exit 1，且 Traceback
        # 被 DIAG 截断，agent 看不到行/列只能盲目排查绕圈）。这里上传前先解析，
        # 失败返回精确行/列错误，agent 可直接修复后重调。
        try:
            with open(host, "r", encoding="utf-8") as _f:
                diff_data = json.load(_f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return _fail(
                "diff_json_validate",
                f"diff.json JSON 语法错误: {e}。请修复后重试（注意：JSON 不允许尾随逗号，数组/对象最后一项后不能有逗号）。",
            )
        except OSError as e:
            return _fail("validate", f"读取 diff.json 失败: {e}")
        sandbox_diff_path = "/home/user/diff.json"

    # ── ④ 追溯校验（宽松）：diff.json 引用的章节号需有 alignment 证据，凭空引用告警 ──
    trace_warnings = await _trace_check(sandbox_id, diff_data)

    # 上传 generate_report.py（diff.json 仅在宿主来源时需要上传）
    scripts_dir = _find_skill_scripts_dir(PDF_DIFF_SKILL_NAME)
    if not scripts_dir:
        return _fail("scripts", f"未找到 skill {PDF_DIFF_SKILL_NAME} 的 scripts 目录")
    with open(f"{scripts_dir}/generate_report.py", "rb") as f:
        r = await _sandbox_write(sandbox_id, "/home/user/scripts/generate_report.py", f.read())
    if not r["success"]:
        return _fail("scripts", r["error"])
    if not in_sandbox:
        with open(host, "rb") as f:
            r = await _sandbox_write(sandbox_id, sandbox_diff_path, f.read())
        if not r["success"]:
            return _fail("upload", r["error"])

    md_name = f"技术协议差异对比报告_{out_prefix}.md"
    html_name = f"技术协议差异对比报告_{out_prefix}.html"
    for fmt, out_name in (("md", md_name), ("html", html_name)):
        r = await _sandbox_run(
            sandbox_id,
            f"cd /home/user && python scripts/generate_report.py \"{sandbox_diff_path}\" --format {fmt} --out \"{out_name}\"",
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
        "trace_warnings": trace_warnings,
    }


# ══════════════════════════════════════════════════════════════════════
# OCR 提取（2026-10-17 新增）
# ══════════════════════════════════════════════════════════════════════
# 为什么做成 MCP tool，而不是再加一份 skill 文档 / prompt 提示：
#   四次实锤（aaf95335 / 60697463 / 80af41e4 / 83345f0c）——模型从不翻 /opt/ocr/，
#   每次都自己现写 OCR 脚本，且失败后一律归因为"环境/库不兼容"：
#     · 60697463 / 80af41e4：裸 PaddleOCR 撞 paddle 3.3.1 的 oneDNN C++ 崩溃
#       （ConvertPirAttribute2RuntimeAttribute … onednn_instruction.cc:116），
#       对外宣称"PaddleOCR 与当前环境 C++ 级不兼容，无法使用"；
#     · 83345f0c：改用 PaddleOCR 2.x 的参数名（use_gpu / show_log）调 3.x，
#       抛 ValueError: Unknown argument，又归因为"API compatibility issues"，
#       转 pytesseract → 中文满屏乱码（RERLEDES, BRE.），最终烧满
#       recursion_limit=200 且零交付。
#   共同点：**模型不会去找入口**。加指路牌（文档/提示）无效，必须把入口放进
#   tool list —— schema 常驻、参数被强制，模型不需要知道 PaddleOCR 的版本差异。
#   沙箱内复用 /opt/ocr/ocr_run.py（v1.1.0：enable_mkldnn=False + det 边长 1500 +
#   dpi 保护 + 逐页容错）。本工具不改它，只负责搬运、分页与收口。

_OCR_WORKDIR = "/home/user/_ocr"
#: 沙箱侧 OCR 入口。**由 MCP 启动任务前下发**（内容来自仓库
#: sandbox/ocr_run.py，随镜像 COPY 进 /app/sandbox/），不再依赖镜像里烘焙的
#: /opt/ocr/ocr_run.py：
#:   · 进度行、det 限幅这类能力随 MCP 版本走，改它不必重建 OCR 镜像
#:     （重建镜像要建模板，且踩过 artifact 按 digest 缓存的坑，成本极高）；
#:   · 单一来源，避免"镜像里 1.1.0 / MCP 期望 1.2.0"的静默漂移
#:     （漂移的后果正是"跑了但看不到进度"）。
#: 镜像内的 /opt/ocr/ocr_run.py 仍保留，供交互式排障用，不是生产路径。
_OCR_ENTRY = "%s/ocr_run.py" % _OCR_WORKDIR
_OCR_RUNNER_LOCAL = Path(__file__).with_name("sandbox") / "ocr_run.py"
#: 镜像内烘焙的封装。**只用作"这是不是 OCR 模板"的环境锚点**（S2 预检），
#: 不是执行入口——执行入口是上面那个由 MCP 下发的副本。
_OCR_IMAGE_ENTRY = "/opt/ocr/ocr_run.py"
_OCR_PREVIEW_CHARS = 2000
_OCR_DPI_MIN, _OCR_DPI_MAX = 100, 200
#: 轮询间隔（秒）。事件循环已自由（同步段全在 to_thread），5s 粒度对
#: SSE 心跳与响应性都足够；deer-flow McpTaskService 同为循环外轮询形态。
_OCR_POLL_INTERVAL = 5
#: 单次调用最长等待（秒）。硬边界来自 agent 侧 MCP 客户端：
#: mcp SDK 的 streamablehttp_client 用同一个 httpx.AsyncClient 发 POST，
#: 其 timeout=httpx.Timeout(t, read=sse_read_timeout) 默认 read=300s
#: （streamable_http.py:104/711）——工具等更久也不会有人收，POST 到
#: 300s 就被客户端 ReadTimeout 掐断。故 300 是"等到被掐"之前的极限。
_OCR_WAIT_MAX = 300
#: 后台任务启动可见性探针（2026-10-20 add0229b）。
#: ★ 探针**无条件**执行，不再由 BG_STARTED 短路：BG_STARTED 只证明 nohup
#:   这一层起来了，`bash -c` 命令串被截断时 echo 照常输出而 python3 从未
#:   执行（add0229b 第二轮实锤：11 次调用全是 running/pages_ocred=0）。
#: 判据 = err.log 有内容（ocr_run.py 第一句就打 start 行，<1s）或 done.flag
#: 出现；两者皆无 ⇒ 进程根本没起来，立刻显式失败。15s 是给沙箱冷启动留
#: 的余量（首个 stderr 行不等 PaddleOCR import，见 sandbox/ocr_run.py 的
#: start 行位置）。
_OCR_LAUNCH_PROBE_TRIES = 15
_OCR_LAUNCH_PROBE_INTERVAL = 1
#: 连续"零 stderr"续查计数：键 = sandbox_id|指纹。正常不会走到这里（启动探针
#: 已把"没起来"判死），万一进程静默退出，靠它把"第 N 次全无输出"透给模型，
#: 而不是让模型自己第 5 次才发现。best-effort，进程内计数，超 64 键清空。
_OCR_SILENT_STREAK: dict[str, int] = {}

# ── 自动分批（2026-10-20 方案 B）─────────────────────────────────────────
# 要解决的问题不是"页数太多"，而是"单批全量跑"这个形态：
#   · 262 页 × 30s/页 ≈ 110 分钟的单进程，中途死一次（沙箱 TTL / OOM /
#     网络）⇒ **一页都拿不到**，前功尽弃；
#   · 一次渲染 262 页，内存峰值高；
#   · 110 分钟里没有中间产物，进度只能靠 err.log 逐页行推断。
# 分批后：每批独立进程，跑完即落盘到自己的目录，失败最多丢当前批；
# 内存峰值按批封顶；进度天然可见（第 3/7 批）。
#
# ★ 分批 ≠ 截断：总页数**不设上限**，1000 页也是 25 批跑完，一页不少。
#   任何"最多 N 页"的做法都是静默截断，是最差的一种降级（用户不知道被截）。
#: 默认批大小（页）。40 页 × 30s ≈ 20 分钟/批，比单次 300s 的 MCP 读超时
#: 长得多（必然跨调用），但短到"死一批只丢 20 分钟"。
_OCR_BATCH_PAGES = 40
#: batch_pages 参数上限。防止模型传 9999 把"分批"退化成"一次性全量"——
#: 那正是本改动要消灭的形态。
_OCR_BATCH_PAGES_MAX = 200
#: 各批产物目录：batches/b001/{result.json,err.log,done.flag,result.txt,meta}
_OCR_BATCH_DIR = "%s/batches" % _OCR_WORKDIR
#: 分批状态（指纹 + 每批状态），沙箱内 JSON，跨调用持久化。
#: ★ 必须落沙箱：MCP 进程是无状态的，每次工具调用都是新任务，
#:   "上一批跑到哪"只能存在被操作的那一侧。
_OCR_BATCH_STATE = "%s/state.json" % _OCR_BATCH_DIR


def _bump_silent_streak(key: str) -> int:
    """记录并返回某指纹连续"零 stderr"的续查次数。"""
    if len(_OCR_SILENT_STREAK) > 64:
        _OCR_SILENT_STREAK.clear()
    n = _OCR_SILENT_STREAK.get(key, 0) + 1
    _OCR_SILENT_STREAK[key] = n
    return n
#: 进度标记：ocr_run.py 每页往 stderr 打一行 `[ocr_run] page i/n ...`。
#: pages_ocred 从这里解析——result.json 只在整批跑完才落地，轮询期间
#: 拿它统计只会恒 0（add0229b：模型看 22 分钟"0 页"判定卡死，转去解析
#: 文本层，最终被硬停）。进度必须来自增量信号。
_OCR_PROGRESS_MARK = "[ocr_run] page "
#: 判活窗口（秒）：err.log 的 mtime 距今小于此值 ⇒ 认为任务还在推进。
#: ★ 只能靠文件 mtime 判活：e2b 每次 execute 是独立 PID namespace，跨会话
#:   `kill -0` 对已死进程恒报 ALIVE（2026-10-19 bgtest4 实锤），镜像里也没有
#:   ps/pgrep。ocr_run.py 每页写一行进度、渲染每 10 页写一行 ⇒ 正常推进时
#:   mtime 永远新鲜；超过窗口没动静就允许重启（否则 SIGKILL 留下的 err.log
#:   会让某个任务永远显示"在跑"，再也重启不了）。
_OCR_LIVENESS_SEC = 180

# 沙箱内的小工具：报页数 / 按页下标拆子 PDF。单独成文件，避免把路径拼进
# `python3 -c` 的引号里（路径含空格/中文/引号时会碎）。
_OCR_HELPER_PY = '''# -*- coding: utf-8 -*-
import sys
import pymupdf

src = sys.argv[1]
doc = pymupdf.open(src)
if len(sys.argv) > 2:
    idx = [int(x) for x in sys.argv[2:]]
    out = pymupdf.open()
    for i in idx:
        out.insert_pdf(doc, from_page=i, to_page=i)
    out.save(src + ".sub.pdf")
print(doc.page_count)
'''


def _parse_ocr_progress(err_text: str) -> tuple[int, int | None]:
    """从 err.log 解析沙箱侧 OCR 进度：(已完成页数, 本次任务总页数|None)。

    ocr_run.py 每页往 stderr 打一行
        `[ocr_run] page <i>/<n> ok texts=<k> elapsed=<s>s`
    取最后一行的 i/n。没有进度行时返回 (0, None)——此时要么还在渲染
    （PDF→PNG 是整批做的，渲染完才开始逐页），要么进程没起来。调用方
    配合 err.log 是否有内容来区分这两种情况。

    ★ 为什么不用 result.json 统计：它只在整批跑完才一次性落地，轮询期间
      永远是旧文件或不存在 ⇒ pages_ocred 恒 0（add0229b 模型据此判定
      "22 分钟零进展"，转而重复上传 + 解析文本层，最终被硬停）。
    """
    done = 0
    total: int | None = None
    for line in (err_text or "").splitlines():
        if _OCR_PROGRESS_MARK not in line:
            continue
        seg = line.split(_OCR_PROGRESS_MARK, 1)[1].strip()
        head, _, rest = seg.partition("/")
        try:
            done = int(head.strip())
        except ValueError:
            continue
        if rest:
            try:
                total = int(rest.split()[0].strip())
            except (ValueError, IndexError):
                pass
    return done, total


async def _ensure_ocr_runner(sandbox_id: str) -> dict:
    """把仓库内的 sandbox/ocr_run.py 下发到沙箱工作目录（幂等覆盖）。

    返回 ``{"success": bool, "version": str, "error": str|None}``。

    为什么下发而不是用镜像里那份：
      · OCR 镜像重建要过 ctl 建模板 + artifact 缓存，慢且易踩 digest 坑；
        而"加一行进度输出"这种改动不值得动镜像。
      · 版本随之固定：镜像那份是 1.1.0（无进度行），MCP 期望 1.2.0（有），
        两处独立演进必然漂移 —— 漂移时的表现就是"OCR 在跑但 pages_ocred
        恒 0"，排查成本极高。

    版本号从脚本 `__version__` 行解析，写进日志与返回值，便于事后核实
    沙箱里实际执行的是哪一版。
    """
    try:
        payload = _OCR_RUNNER_LOCAL.read_bytes()
    except OSError as e:
        return {
            "success": False,
            "version": "",
            "error": "读取本地 %s 失败: %s" % (_OCR_RUNNER_LOCAL.name, e),
        }
    version = ""
    for line in payload.decode("utf-8", errors="replace").splitlines():
        if line.startswith("__version__"):
            version = line.split("=", 1)[1].strip().strip("\"'")
            break
    wr = await _sandbox_write(sandbox_id, _OCR_ENTRY, payload)
    if not wr.get("success"):
        return {"success": False, "version": version, "error": wr.get("error")}
    logger.info("[OCR] 下发 ocr_run.py v%s → %s (%d bytes)", version, _OCR_ENTRY, len(payload))
    return {"success": True, "version": version, "error": None}


def _build_ocr_launch_cmd(
    entry: str, ocr_input: str, dpi: int, out_json: str, out_err: str, out_done: str
) -> str:
    """拼 OCR 后台启动命令。**整串 shlex.quote，绝不手工拼单引号。**

    ★ 2026-10-20 add0229b 第二轮真根因（67 上 A/B 实测）：
      旧写法是 `nohup bash -c 'python3 %s %s ...' ...`，外层手写单引号，而
      里面的路径经过 shlex.quote()——**含空格的路径**会被包成单引号形式，
      两者嵌套 ⇒ bash 在第一个内层单引号处截断命令串，后半截（重定向 +
      `echo $? > done.flag`）全变成 `bash -c` 的位置参数：

        无空格 long.pdf                        → argv=[ocr_run.py long.pdf --json --dpi 150]，产物齐全
        含空格 "康视达 CST2026手册.pdf.sub.pdf" → argv=[ocr_run.py "/home/user/_ocr/康视达"]
                                                → result.json/err.log/done.flag 一个都不产生

      而 `echo BG_STARTED` 在前台照常输出 ⇒ MCP 判定"仍在跑"、pages_ocred 恒 0，
      模型空等 22 分钟 → 转去自己解析文本层 → 被循环检测硬停。
      修法：先拼 inner，再对整串 shlex.quote（单引号嵌套由 shlex 的 `'"'"'`
      形式保证安全）。单测 `test_launch_cmd_roundtrip_with_special_chars`
      用 shlex.split 往返锁定，等价的真实 bash 行为在 67 上另行验证。
    """
    inner = "python3 %s %s --json --dpi %d > %s 2> %s; echo $? > %s" % (
        shlex.quote(entry),
        shlex.quote(ocr_input),
        int(dpi),
        shlex.quote(out_json),
        shlex.quote(out_err),
        shlex.quote(out_done),
    )
    return "nohup bash -c %s </dev/null >/dev/null 2>&1 & echo BG_STARTED" % shlex.quote(inner)


async def _ocr_task_age_seconds(sandbox_id: str, err_path: str) -> int:
    """沙箱内 OCR 任务的「最近活动距今秒数」；-1 = err.log 不存在或无 mtime。

    判活只能用文件系统信号（PID namespace 隔离，见 _OCR_LIVENESS_SEC 注释）。
    正常推进时 err.log 每页都在增长，mtime 恒新鲜；SIGKILL（OOM 等）后
    mtime 冻结，超过 _OCR_LIVENESS_SEC 即认为任务已死、允许重启。
    """
    import shlex

    cmd = (
        "e=%s; now=$(date +%%s); if [ -f \"$e\" ]; then "
        "m=$(stat -c %%Y \"$e\" 2>/dev/null || echo 0); "
        "if [ \"$m\" = \"0\" ]; then echo AGE=-1; else echo AGE=$((now-m)); fi; "
        "else echo AGE=-1; fi" % shlex.quote(err_path)
    )
    st = await _sandbox_run(sandbox_id, cmd, timeout=30)
    text = (st.get("stdout") or "").strip()
    if not text.startswith("AGE="):
        return -1
    try:
        return int(text.split("=", 1)[1])
    except (ValueError, IndexError):
        return -1


async def _batch_state_load(
    sandbox_id: str,
    state_path: str,
    fingerprint: str,
    groups: list[list[int]],
    batch_pages: int,
    now: float,
) -> tuple[dict, bool]:
    """读分批状态；指纹不符或读不到 ⇒ 新建（返回 (state, 是否新建)）。

    指纹含「源 PDF + dpi + 目标页集合 + 批大小」——任何一项变了都必须重开，
    否则会把上一组参数的批次状态当成自己的进度（页范围错位 = 收错数据）。
    """
    raw = ""
    rd = await _sandbox_read(sandbox_id, state_path)
    if rd.get("success") and rd.get("data"):
        raw = bytes(rd["data"]).decode("utf-8", errors="replace")
    state = None
    if raw.strip():
        try:
            cand = json.loads(raw)
        except ValueError:
            cand = None
        if isinstance(cand, dict) and cand.get("fingerprint") == fingerprint:
            batches = cand.get("batches")
            if isinstance(batches, list) and len(batches) == len(groups):
                state = cand
    if state is None:
        state = _batch_state_new(fingerprint, groups, batch_pages, now)
        await _batch_state_save(sandbox_id, state_path, state)
        logger.info(
            "[OCR] 新建分批状态：%d 批 × 最多 %d 页（共 %d 页），指纹=%s",
            state["batch_count"],
            batch_pages,
            sum(len(g) for g in groups),
            fingerprint[:80],
        )
        return state, True
    return state, False


async def _batch_state_save(sandbox_id: str, state_path: str, state: dict) -> bool:
    """写回分批状态。失败只告警不中断——状态丢失的后果是"进度重算"，
    而中断的后果是整轮任务失败，前者可接受。"""
    try:
        payload = json.dumps(state, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as e:  # 理论上不会发生（全是基本类型）
        logger.warning("[OCR] 分批状态序列化失败：%s", e)
        return False
    wr = await _sandbox_write(sandbox_id, state_path, payload)
    if not wr.get("success"):
        logger.warning("[OCR] 分批状态写回失败：%s", wr.get("error"))
        return False
    return True


async def _batch_concat_texts(sandbox_id: str, out_txt: str, sources: list[str]) -> str | None:
    """把各已完成批的 result.txt 按批序拼成一份全文（覆盖写）。

    为什么不 append：续查/重跑会让同一批被收两次，append 会重复；按批序
    重拼是幂等的（同一 state 重拼结果相同）。批次数 ≤ 页数/批大小，命令
    长度安全（路径全是固定 ASCII）。
    """
    import shlex

    if not sources:
        return None
    cmd = "cat %s > %s" % (
        " ".join(shlex.quote(p) for p in sources),
        shlex.quote(out_txt),
    )
    st = await _sandbox_run(sandbox_id, cmd, timeout=120)
    if not st.get("success"):
        logger.warning("[OCR] 合并批次文本失败：%s", (st.get("stderr") or "")[:200])
        return None
    return out_txt


async def _ocr_batch_finish(
    sandbox_id: str,
    state: dict,
    out_txt: str,
    pdf_total_pages: int,
    t0: float,
    extra_warnings: list[str] | None = None,
) -> dict:
    """所有批次都跑完（含失败）时的收口：合并全文 + 汇总 + 显式列出失败批。

    ★ 失败批次**不静默**：用户要的是"整本 262 页"，少一批就是少一段内容，
      必须让模型（和它转述给用户的话）知道哪几段没跑出来、可以用
      pages="41-80" 单独补跑。这比"返回全部成功"的假绿有用得多。
    """
    import time as _time

    warnings = list(extra_warnings or [])
    summary = _batch_summary(state)
    batches = state.get("batches") or []
    sources = [
        "%s/b%03d/result.txt" % (_OCR_BATCH_DIR, int(b.get("idx") or 0))
        for b in sorted(batches, key=lambda x: int(x.get("idx") or 0))
        if str(b.get("status")) == "done"
    ]
    await _batch_concat_texts(sandbox_id, out_txt, sources)
    warnings.append(
        "分批结果：逐批结构化 JSON 在 %s/bNNN/result.json，全部文本已按批序合并到 %s"
        % (_OCR_BATCH_DIR, out_txt)
    )
    for item in summary["batches_failed"]:
        warnings.append(
            "第 %s 批（第 %s 页）OCR 失败，这部分内容缺失：%s"
            "——可用 pages=\"%s\" 单独重跑补齐"
            % (
                item["batch"],
                item["pages"],
                (item.get("error") or "未知原因")[:200],
                item["pages"],
            )
        )
    if summary["pages_ocred_doc"] < len(
        [p for b in batches for p in (b.get("pages") or [])]
    ):
        # 只提示不阻断：已跑出来的部分仍然可用（这正是分批的价值）
        logger.warning(
            "[OCR] 分批收口但页数不完整：已 OCR %d 页 / 计划 %d 页",
            summary["pages_ocred_doc"],
            len([p for b in batches for p in (b.get("pages") or [])]),
        )
    preview_st = await _sandbox_run(
        sandbox_id, "head -c %d %s 2>/dev/null" % (_OCR_PREVIEW_CHARS, shlex.quote(out_txt)),
        timeout=60,
    )
    preview = (preview_st.get("stdout") or "")[:_OCR_PREVIEW_CHARS]
    total = summary["pages_ocred_doc"]
    return {
        "success": total > 0,
        "step": None if total > 0 else "batch",
        "error": None if total > 0 else "所有批次均未产出文本（见 warnings）",
        "status": "completed",
        "batch": "%d/%d" % (summary["batches_done"], state.get("batch_count") or 0),
        "batch_count": state.get("batch_count") or 0,
        "batches_done": summary["batches_done"],
        "batches_failed": summary["batches_failed"],
        # 分批模式下 pages_* 仍是"本批/文档"两套口径，收口时两者相等
        "pages_total": len([p for b in batches for p in (b.get("pages") or [])]),
        "pdf_total_pages": pdf_total_pages,
        "pages_ocred": total,
        "pages_ocred_total": len([p for b in batches for p in (b.get("pages") or [])]),
        "pages_ocred_doc": total,
        "pages_failed": sum(int(b.get("failed") or 0) for b in batches),
        "total_texts": summary["total_texts_doc"],
        "elapsed_sec": round(_time.time() - t0, 1),
        "payload_json": None,  # 分批：每批一份，见 warnings 里的目录说明
        "payload_txt": out_txt,
        "sandbox_path": "",
        "sandbox_dir": _OCR_WORKDIR,
        "preview": preview,
        "warnings": warnings,
    }


async def _ocr_batch_settle(
    sandbox_id: str,
    state: dict,
    state_path: str,
    batch_idx: int,
    res: dict,
    pdf_total_pages: int,
    t0: float,
    out_txt: str,
) -> dict:
    """收完当前批次后结算：标状态 → 保存 → 决定"返回 running 推进下一批"还是"全部收口"。

    两种结果都从同一处出，避免"完成一批就返回 completed"把后续批次丢掉
    （那会变成静默截断——本改动要消灭的正是这个）。
    """
    import time as _time

    batches = state.get("batches") or []
    cur = batches[batch_idx]
    ok = res.get("status") == "completed"
    if ok:
        cur["status"] = "done"
        cur["ocred"] = int(res.get("pages_ocred") or 0)
        cur["failed"] = int(res.get("pages_failed") or 0)
        cur["texts"] = int(res.get("total_texts") or 0)
        cur["error"] = None
        logger.info(
            "[OCR] 第 %d/%d 批完成（%s，%d 页 / %d 条文本）",
            cur["idx"],
            state.get("batch_count") or 0,
            _page_span_text(cur.get("pages") or []),
            cur["ocred"],
            cur["texts"],
        )
    else:
        cur["status"] = "failed"
        cur["error"] = str(res.get("error") or res.get("step") or "未知错误")[:300]
        logger.warning(
            "[OCR] 第 %d/%d 批失败（%s）：%s",
            cur["idx"],
            state.get("batch_count") or 0,
            _page_span_text(cur.get("pages") or []),
            cur["error"],
        )
    state["updated_at"] = _time.time()
    await _batch_state_save(sandbox_id, state_path, state)

    total = state.get("batch_count") or 0
    nxt = _batch_current_index(state)
    if nxt is None:
        return await _ocr_batch_finish(
            sandbox_id, state, out_txt, pdf_total_pages, t0
        )
    nxt_batch = batches[nxt]
    nxt_span = _page_span_text(nxt_batch.get("pages") or [])
    summary = _batch_summary(state)
    warnings = list(res.get("warnings") or [])
    if not ok:
        warnings.append(
            "第 %s 批（第 %s 页）失败，已跳过继续后续批次：%s"
            % (cur["idx"], _page_span_text(cur.get("pages") or []), cur["error"])
        )
    return {
        "success": True,
        "step": None,
        "error": None,
        "status": "running",
        # 分批进度：batch 指"当前要处理的那一批"
        "batch": "%d/%d" % (nxt_batch.get("idx") or 0, total),
        "batch_index": nxt_batch.get("idx") or 0,
        "batch_count": total,
        "batch_pages": nxt_span,
        "next_batch": nxt_span,
        "batches_done": summary["batches_done"],
        "batches_failed": summary["batches_failed"],
        "pages_total": len(nxt_batch.get("pages") or []),
        "pdf_total_pages": pdf_total_pages,
        "pages_ocred": 0,
        "pages_ocred_total": len(nxt_batch.get("pages") or []),
        "pages_ocred_doc": summary["pages_ocred_doc"],
        "pages_failed": 0,
        "total_texts": summary["total_texts_doc"],
        "elapsed_sec": round(_time.time() - t0, 1),
        "payload_json": None,
        "payload_txt": None,
        "sandbox_path": "",
        "sandbox_dir": _OCR_WORKDIR,
        "preview": "",
        "warnings": warnings,
        "progress_hint": (
            "分批 OCR：已完成 %d/%d 批（累计 %d/%d 页）。"
            "第 %s 批（第 %s 页）现在**尚未启动**——请用完全相同的参数"
            "（含 pages/batch_pages/dpi）再调一次本工具，会自动启动并推进它。"
            "**不要重新上传、不要改用 execute 自己写脚本、不要换 pages**。"
            "每批完成的结果已落盘，中途失败只影响当前批。"
            % (
                summary["batches_done"],
                total,
                summary["pages_ocred_doc"],
                pdf_total_pages,
                nxt_batch.get("idx") or 0,
                nxt_span,
            )
        ),
    }


def _parse_page_spec(spec: str, page_count: int) -> tuple[list[int] | None, str | None]:
    """把 "all" / "1-5" / "3,7,9" / "12" 解析为 0-based 页下标（去重升序）。

    返回 (pages, error)。pages=None 表示"全量，无需拆分"。
    纯函数、不碰 IO —— 单测直接喂参数（见 tests/test_ocr_page_spec.py）。
    """
    if spec is None:
        return None, None
    raw = str(spec).strip().lower()
    if raw in ("", "all", "*"):
        return None, None
    picked: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                start, end = int(a), int(b)
            except ValueError:
                return None, f"页范围无法解析: {chunk!r}（期望形如 1-5）"
            if start > end:
                return None, f"页范围起点大于终点: {chunk!r}"
            rng = list(range(start, end + 1))
        else:
            try:
                rng = [int(chunk)]
            except ValueError:
                return None, f"页号无法解析: {chunk!r}"
        for p in rng:
            if p < 1:
                return None, f"页号从 1 开始，收到 {p}"
            if page_count and p > page_count:
                return None, f"页号 {p} 超出总页数 {page_count}"
            picked.add(p - 1)
    if not picked:
        return None, "页选择为空"
    return sorted(picked), None


def _page_span_text(pages_1based: list[int], limit: int = 6) -> str:
    """把 1-based 页号列表压成人类可读文本："81-120" / "1-3,7,9"。

    连续段折叠成区间（分批后的每批都是连续段 ⇒ 恒为 "a-b"）；非连续段
    超过 ``limit`` 个时折叠成 "a-b,c-d 等 N 页"，避免把 262 个页号糊进提示里。
    纯函数，不碰 IO。
    """
    nums = sorted({int(p) for p in pages_1based})
    if not nums:
        return "空"
    segs: list[tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        segs.append((start, prev))
        start = prev = n
    segs.append((start, prev))
    parts = [("%d" % a) if a == b else ("%d-%d" % (a, b)) for a, b in segs]
    if len(parts) > limit:
        return "%s 等 %d 页" % (",".join(parts[:limit]), len(nums))
    return ",".join(parts)


def _plan_ocr_batches(target_pages: list[int], batch_pages: int) -> list[list[int]]:
    """把目标页下标（0-based，升序）切成连续批次，每批至多 batch_pages 页。

    ★ 这里**不设总页数上限**：切多少批由页数决定（1000 页 = 25 批），
      一页都不会被丢掉。分批的目的是"失败只丢当前批"和"内存按批封顶"，
      不是"少干点活"。
    batch_pages<=0 ⇒ 不分批（返回单批）。
    """
    if batch_pages <= 0 or not target_pages:
        return [list(target_pages)]
    return [target_pages[i:i + batch_pages] for i in range(0, len(target_pages), batch_pages)]


def _batch_state_new(
    fingerprint: str, groups: list[list[int]], batch_pages: int, created_at: float
) -> dict:
    """新建分批状态。每批记 pages（1-based，便于人读/回显）与状态机字段。

    状态取值：pending（待跑）| running（已启动）| done（成功收产物）
              | failed（进程非 0 退出 / 产物不可读，不自动重试）
    """
    return {
        "fingerprint": fingerprint,
        "batch_pages": int(batch_pages),
        "batch_count": len(groups),
        "created_at": created_at,
        "updated_at": created_at,
        "batches": [
            {
                "idx": i + 1,
                "pages": [int(p) + 1 for p in grp],
                "status": "pending",
                "ocred": 0,
                "failed": 0,
                "texts": 0,
                "error": None,
            }
            for i, grp in enumerate(groups)
        ],
    }


def _batch_current_index(state: dict) -> int | None:
    """下一个要处理的批次下标（0-based）；全部结束返回 None。

    优先 pending，也接管 running（上次调用启动后 MCP 侧没有机会更新——
    进程可能仍在跑，由下游的判活闸门决定"续等"还是"重启"，不在这里猜）。
    失败的批次**不自动重试**：坏 PDF 会让它无限重试，而模型可以用
    pages="41-80" 显式重跑那一段。
    """
    for i, b in enumerate(state.get("batches") or []):
        if str(b.get("status")) in ("pending", "running"):
            return i
    return None


def _batch_summary(state: dict) -> dict:
    """已跑批次汇总：完成/失败批次、文档级已完成页数、累计文本条数。"""
    batches = state.get("batches") or []
    done = [b for b in batches if str(b.get("status")) == "done"]
    failed = [b for b in batches if str(b.get("status")) == "failed"]
    return {
        "batches_done": len(done),
        "batches_failed": [
            {
                "batch": b.get("idx"),
                "pages": _page_span_text(b.get("pages") or []),
                "error": b.get("error"),
            }
            for b in failed
        ],
        "pages_ocred_doc": sum(int(b.get("ocred") or 0) for b in done),
        "total_texts_doc": sum(int(b.get("texts") or 0) for b in done),
    }


def _first_json_value(raw: str):
    """从混杂 stdout 里抠出第一个"像结果集"的 JSON 值。

    为什么需要：`ocr_run.py --json` 的输出**并不纯净** —— paddleocr 3.x 初始化时
    会往 stdout 打模型源探测结果（2026-10-17 实测：result.json 36904 字节、
    正文完全正常，但前两行是 `<Response [404]>`），直接 json.loads 会报
    `Expecting value: line 1 column 1 (char 0)`，看起来像"OCR 失败了"，
    实际只是被第三方库的调试输出污染了前两行。

    ★ 不能只取"第一个能解析的 JSON"：污染行 `<Response [404]>` 里的 `[404]`
    本身就是合法 JSON 数组，会被先抓住，返回 `[404]` 而不是真正的页结果。
    故必须按**结构**筛选：接受 dict，或"元素为 dict 的 list"；
    `[404]` 这类纯标量数组跳过，继续往后找。

    纯函数、不碰 IO —— 单测见 tests/test_ocr_pdf_to_text.py。
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    dec = json.JSONDecoder()
    s = raw.lstrip()
    for i, ch in enumerate(s):
        if ch not in "[{":
            continue
        try:
            val = dec.raw_decode(s[i:])[0]
        except ValueError:
            continue
        if isinstance(val, dict):
            return val
        if isinstance(val, list) and (not val or isinstance(val[0], dict)):
            return val
    return None


@mcp.tool()
async def ocr_pdf_to_text(
    sandbox_id: str,
    file_path: str | None = None,
    pages: str = "20",
    dpi: int = 150,
    sandbox_pdf_path: str | None = None,
    wait_seconds: int = 90,
    batch_pages: int = 0,
) -> dict:
    """对扫描件/图片型 PDF 做 OCR 并返回文本。**这是本环境唯一的 OCR 入口。**

    已内置全部正确参数，不要自己写 OCR 脚本、不要 pip install、不要直接调
    PaddleOCR(...) 或 pytesseract —— 那些写法在本沙箱里会失败，且报错会误导你：
      · 裸 PaddleOCR 会撞 paddle 的 oneDNN C++ 崩溃（onednn_instruction.cc:116），
        看起来像"库与环境不兼容"，其实只是少了 enable_mkldnn=False；
      · PaddleOCR 3.x 已删除 use_gpu / show_log 参数，照 2.x 写法传会抛
        ValueError: Unknown argument；
      · pytesseract 没装中文包，出来的是乱码。
    本工具把这些都处理好了，你只需要给路径。

    **执行模型（2026-10-19 改，submit-and-poll）**：OCR 在沙箱内后台运行，
    本工具最多等待 wait_seconds 秒。未完成时返回 status="running"（带进度提示），
    **用完全相同的参数（file_path/sandbox_id/pages/dpi/sandbox_pdf_path/batch_pages）
    再调一次即可续查或取回结果，不会重复上传、不会重复 OCR**（沙箱内按参数指纹复用）。
    上传复用判据：沙箱内同名文件存在且字节数与宿主一致即跳过上传（续查省几十秒 +
    一次 DLP 全量解密）；宿主文件被替换（大小变化）时仍会重新上传。

    **自动分批（2026-10-20 起）**：目标页数超过 batch_pages（默认 40）时，本工具
    自动切成多批，**每批独立进程、跑完即落盘**，一次调用推进一批。返回里
    `batch: "3/7"` 表示当前第 3 批 / 共 7 批，`pages_ocred_doc` 是整本文档累计
    已 OCR 页数。看到 `status="running"` 且 `batch` 未到末批时，**同样参数再调一次**
    即可推进下一批——不要重新上传、不要换 pages、不要自己写脚本重跑。
    - 总页数**不设上限**：分批不是截断，262 页 = 7 批全部跑完，一页不少；
    - 某一批失败不影响其它批（失败批会在 `batches_failed` 里列出，可用
      `pages="41-80"` 单独补跑那一段）；
    - pages="all" 与 pages="1-262" 走同一套分批路径。

    Args:
        file_path: 虚拟路径（/uploads/{user_id}/{session_id}/{filename}）或
            宿主物理路径。自动识别 DLP 加密并解密后上传（明文不落盘）。
            **与 sandbox_pdf_path 二选一即可**——PDF 已在沙箱里时只给
            sandbox_pdf_path，不给本参数。
        sandbox_id: 目标沙箱 ID（用当前会话的沙箱）。
        pages: 页选择。"20"（默认，前 20 页）| "all"（全量，自动分批）| "1-5" |
            "3,7,9" | "12"。**实测速度约 30s/页（150dpi、4 vCPU）**。
            只关心部分页时可以显式指定，会比 all 少跑很多批。
        dpi: PDF 渲染 DPI，100~200，默认 150。超过 200 在 8GB 沙箱上有 OOM 风险。
        sandbox_pdf_path: 可选。若 PDF 已经在沙箱里（之前上传过），直接给沙箱路径，
            省掉一次上传（大文件可省几十秒）。
        wait_seconds: 单次调用最长等待秒数，0~300，默认 90。超时未完成则返回
            status="running"，按上面说明用相同参数再调。
        batch_pages: 每批页数，0 = 自动（默认 40，上限 200）。一般不用传；
            页数特别多的文档想减小单批内存占用时可设小一点（如 20）。

    Returns:
        {"success": bool, "step": str|None, "error": str|None,
         "status": "completed"|"running",
         "pages_total": int,        # 本批要 OCR 的页数
         "pdf_total_pages": int,    # 文档总页数
         "pages_ocred": int, "pages_failed": int,
         "total_texts": int, "elapsed_sec": float,
         "payload_json": str,   # 沙箱内逐页结构化结果（page/texts/scores）
         "payload_txt": str,    # 沙箱内纯文本，带 "===== page N =====" 分隔
         "preview": str,        # 前 2000 字符，够判断内容对不对
         "warnings": [str],
         # 分批模式额外字段：
         "batch": "3/7", "batch_count": int, "batch_pages": "81-120",
         "batches_done": int, "batches_failed": [{batch,pages,error}],
         "pages_ocred_doc": int}  # 整本文档累计已 OCR 页数
        status="running" 时另带 progress_hint（str）说明如何续查。
        全文不会放进返回值（262 页会直接撑爆上下文）：需要细节时用 read_file
        读 payload_txt，或按页再调一次本工具（pages="10-20"）。
    """
    import shlex
    import time as _time

    t0 = _time.time()

    if not _is_valid_sandbox_id(sandbox_id):
        return _fail("validate", f"非法 sandbox_id: {sandbox_id!r}")
    if not sandbox_pdf_path and not file_path:
        # either/or：文档明说"PDF 已在沙箱里就直接给沙箱路径"，file_path
        # 若仍必填，模型照文档办事反而吃 pydantic missing_argument。
        return _fail(
            "validate",
            "必须提供 file_path 或 sandbox_pdf_path 之一：PDF 不在沙箱时给 "
            "file_path（自动解密上传），已在沙箱则给 sandbox_pdf_path",
        )
    if not (_OCR_DPI_MIN <= int(dpi) <= _OCR_DPI_MAX):
        return _fail(
            "validate",
            f"dpi 必须在 {_OCR_DPI_MIN}~{_OCR_DPI_MAX} 之间，收到 {dpi}"
            "（超过 200 在 8GB 沙箱上有 OOM 风险；低于 100 会严重漏检）",
        )
    # wait_seconds 钳制而非报错：模型给 0 表示"只想提交/查一次"，给 9999
    # 时按上限截断（300s 是 MCP 客户端 httpx read 超时，等更久没人收）。
    wait_seconds = max(0, min(_OCR_WAIT_MAX, int(wait_seconds)))

    # ── S1：定位沙箱内的 PDF ──
    import asyncio as _asyncio
    import shlex

    if sandbox_pdf_path:
        src_pdf = sandbox_pdf_path
        chk = await _sandbox_run(sandbox_id, "test -f %s && echo OK" % shlex.quote(src_pdf), timeout=60)
        if "OK" not in (chk.get("stdout") or ""):
            return _fail("locate", f"sandbox_pdf_path 不存在: {src_pdf}")
    else:
        phys = _resolve_host_path(file_path)
        bad = _upload_precheck(phys, file_path)
        if bad:
            return bad
        src_pdf = "%s/%s" % (_OCR_WORKDIR, os.path.basename(phys))
        # ★ 续查跳过重传（2026-10-20 add0229b，P3②）
        # 生产实测：262 页任务 5 次续查各重传 **256.4 MiB**（MCP 日志 5 条 [DLP]），
        # 而本函数的 running 文案一直承诺"指纹一致不会重复上传"——文案与实现不符。
        # 模型每次续查都带同一个 file_path，S1 却无条件走一遍解密+上传。
        # 判据：沙箱内同名文件存在、且**字节数与宿主一致** ⇒ 就是上一轮传上去的
        # 那一份，直接用（大文件每轮省几十秒，DLP 全量解密也一并省掉）。
        # 为什么不用 md5：256 MiB 每轮算一次要几百 ms~数秒，而续查场景下同一路径
        # 的 PDF 不会变；大小不一致即视为不同文件、照常重传（保守，不会用错文件）。
        _local_size = -1
        try:
            _local_size = os.path.getsize(phys)
        except OSError:
            _local_size = -1
        _reused_upload = False
        if _local_size > 0:
            _probe = await _sandbox_run(
                sandbox_id,
                "test -s %s && stat -c %%s %s"
                % (shlex.quote(src_pdf), shlex.quote(src_pdf)),
                timeout=60,
            )
            _probe_lines = (_probe.get("stdout") or "").strip().splitlines()
            _remote_size = (
                int(_probe_lines[0])
                if _probe_lines and _probe_lines[0].isdigit()
                else -1
            )
            if _remote_size == _local_size:
                _reused_upload = True
                # 用 warning 而非 info：MCP 的 logging 只挂了 lastResort（WARNING），
                # info 根本不会出现在 `docker logs`（文件头注释已记录该坑）。
                logger.warning(
                    "[OCR] 沙箱内已有同名同尺寸文件（%d B），跳过重传: %s",
                    _local_size, os.path.basename(phys),
                )
        # ★ 必须重试：沙箱刚 get_or_create 出来时 envd 还没就绪，e2b 的
        #   connect/files.write 会拿到空响应 → `Expecting value: line 1 column 1`。
        #   2026-10-17 实测：建箱后立刻调，9s 就失败；等模型聊两轮再调就成功。
        #   既有 upload_to_sandbox 没有重试（它历史上一直是被"已经聊了一会儿"的
        #   会话调用），本工具是第一个会紧贴建箱调用的，故在此自保。
        last_err = None
        if not _reused_upload:
            for attempt in range(4):
                prep = await _sandbox_run(
                    sandbox_id, "mkdir -p %s" % shlex.quote(_OCR_WORKDIR), timeout=60
                )
                if prep.get("success"):
                    up = await upload_to_sandbox(file_path, src_pdf, sandbox_id)
                    if up.get("success"):
                        last_err = None
                        break
                    last_err = up.get("error")
                else:
                    last_err = prep.get("stderr")
                if attempt < 3:
                    await _asyncio.sleep(3)
            if last_err:
                return _fail("upload", f"上传沙箱失败（已重试 4 次）: {str(last_err)[:400]}")

    # ── S2：沙箱是否具备 OCR 入口（旧模板会让模型重走"自己写脚本"的老路）──
    # ★ 必须 echo OK：本判据读的是 stdout 里有没有 OK，而 grep 命中的输出是
    #   `__version__ = "1.1.0"`，不含 OK。漏掉 echo 会让 precheck 恒失败
    #   （2026-10-17 实测踩到：明明文件在，却报"模板过旧"）。
    env = await _sandbox_run(
        sandbox_id,
        "test -f %s && grep -m1 '^__version__' %s && echo OK"
        % (_OCR_IMAGE_ENTRY, _OCR_IMAGE_ENTRY),
        timeout=60,
    )
    if "OK" not in (env.get("stdout") or ""):
        return _fail(
            "precheck",
            f"沙箱内缺少 {_OCR_IMAGE_ENTRY}（不是 OCR 模板）。"
            "请在 ocr-20g-v5 及以上模板重建沙箱后再试。",
        )

    # ── S3：页数 + 可选拆分 ──
    helper = "%s/_page.py" % _OCR_WORKDIR
    w = await _sandbox_write(sandbox_id, helper, _OCR_HELPER_PY.encode("utf-8"))
    if not w["success"]:
        return _fail("prepare", w["error"])

    cnt = await _sandbox_run(
        sandbox_id, "python3 %s %s" % (shlex.quote(helper), shlex.quote(src_pdf)), timeout=180
    )
    try:
        page_count = int((cnt.get("stdout") or "0").strip().splitlines()[-1])
    except Exception:
        page_count = 0
    if page_count <= 0:
        return _fail("prepare", f"读取页数失败: {(cnt.get('stderr') or '')[:300]}")

    want, perr = _parse_page_spec(pages, page_count)
    if perr:
        return _fail("validate", perr)

    # ── S3b：分批规划（方案 B，2026-10-20）──
    # 目标页集合（0-based 升序）。pages="all" ⇒ range(page_count)，
    # 于是 "all" 与 "1-262" 走**完全相同**的分批路径（不搞两套语义）。
    #
    # 分批只在"目标页数 > 批大小"时启用：小文档/小页范围仍是原来的单批路径，
    # 行为零变化（所有既有 e2e/单测都覆盖这条路，避免无谓回归）。
    target_pages = list(range(page_count)) if want is None else list(want)
    batch_pages = int(batch_pages or 0)
    if batch_pages <= 0:
        batch_pages = _OCR_BATCH_PAGES
    batch_pages = max(1, min(batch_pages, _OCR_BATCH_PAGES_MAX))
    batching = len(target_pages) > batch_pages

    batch_state: dict | None = None
    batch_idx: int | None = None
    batch_dir = ""
    # 当前（批）产物文本 vs 整本合并文本：分批时前者落到批目录，
    # 后者是所有已完成批按序拼接的结果（_ocr_batch_finish 里重拼，幂等）。
    out_txt = "%s/result.txt" % _OCR_WORKDIR
    out_txt_all = "%s/result.txt" % _OCR_WORKDIR
    if batching:
        batch_groups = _plan_ocr_batches(target_pages, batch_pages)
        # 文档级指纹：源 PDF + dpi + 批大小 + 目标页集合。任一不同都是新任务，
        # 必须重开分批状态（否则会拿上一组参数的批次进度当自己的）。
        doc_fp = "%s|dpi=%d|batch=%d|pages=%s" % (
            src_pdf,
            int(dpi),
            batch_pages,
            _page_span_text([p + 1 for p in target_pages], limit=10 ** 6),
        )
        batch_state, _fresh = await _batch_state_load(
            sandbox_id, _OCR_BATCH_STATE, doc_fp, batch_groups, batch_pages, _time.time()
        )
        batch_idx = _batch_current_index(batch_state)
        if batch_idx is None:
            # 全部批次已结束（成功或失败）⇒ 不再启动任何进程，直接收口。
            # ★ 这里绝不能返回 completed 就跑：失败的批次要显式列出来。
            return await _ocr_batch_finish(
                sandbox_id, batch_state, out_txt_all, page_count, t0
            )
        # 切换成"只处理当前批"：want 变成该批页下标，所有产物路径落到批目录。
        # 下游（指纹复用/判活/启动/轮询/收产物）**一行不用改**——这是本方案
        # 能把改造面压到最小的关键。
        want = list(batch_groups[batch_idx])
        cur_batch = batch_state["batches"][batch_idx]
        batch_dir = "%s/b%03d" % (_OCR_BATCH_DIR, batch_idx + 1)
        out_txt = "%s/result.txt" % batch_dir
        if str(cur_batch.get("status")) != "running":
            cur_batch["status"] = "running"
            batch_state["updated_at"] = _time.time()
            await _batch_state_save(sandbox_id, _OCR_BATCH_STATE, batch_state)

    target_pdf = src_pdf
    if want is not None:
        # ★ 沙箱侧 _page.py 固定把子 PDF 写在 `src + ".sub.pdf"`（helper 的既有
        #   约定，不能改——单批路径与历史产物都依赖它）。分批模式必须再「搬」
        #   到本批目录，否则每批都会指向一个不存在的 input.sub.pdf：
        #   2026-10-20 e2e 实测正是这样翻车——三批全部
        #   `FileNotFoundError: 输入不存在: batches/bNNN/input.sub.pdf`，
        #   因为文件实际叫 `分批复核 CST2026.pdf.sub.pdf` 躺在根目录。
        #   用 ; 串联而非 &&（`&` 只在末尾才是后台化，这里没有 &，安全），
        #   末尾用 test -s 做**真实**成功判据：只看 commands.run 的 rc 会把
        #   "拆出来了但没搬过去"当成功（本次 bug 就是这样躲过启动探针的）。
        split_src = src_pdf + ".sub.pdf"
        sub = ("%s/input.sub.pdf" % batch_dir) if batching else split_src
        parts = [
            "mkdir -p %s" % shlex.quote(batch_dir or _OCR_WORKDIR),
            "python3 %s %s %s"
            % (
                shlex.quote(helper),
                shlex.quote(src_pdf),
                " ".join(str(i) for i in want),
            ),
        ]
        if batching:
            parts.append("mv -f %s %s" % (shlex.quote(split_src), shlex.quote(sub)))
        parts.append("test -s %s && echo SPLIT_OK" % shlex.quote(sub))
        sp = await _sandbox_run(sandbox_id, "; ".join(parts), timeout=600)
        if "SPLIT_OK" not in (sp.get("stdout") or ""):
            return _fail(
                "split",
                "按页拆分失败（%d 页 → %s）: %s"
                % (len(want), sub, ((sp.get("stderr") or "") or (sp.get("stdout") or ""))[:300]),
            )
        target_pdf = sub

    # ── S4：后台启动 OCR + 轮询等待（submit-and-poll 轻量版）──
    # 2026-10-19 改。原实现是同步等命令跑完（budget 封顶 3600s），事故
    # 653d8515 证明这条路必死：MCP 事件循环被同步 e2b 调用堵满命令时长
    # （本次 55 分钟），工具响应送不回 agent；沙箱 TTL 3600s 无人续订，
    # OCR 跑到约 60/110 分钟被连锅端。现改为：
    #   1) nohup 后台跑（外层 bash 等 OCR 结束后写 done.flag + 退出码），
    #      commands.run 只等 shell 退出（秒级）；
    #   2) 启动前写 meta 指纹（src/pages/dpi），启动时清旧产物；
    #   3) 轮询判据 = done.flag 文件。★ 不用 kill -0/PID：实测 e2b 每次
    #      execute 是独立 PID namespace，跨会话 kill -0 恒 ALIVE——进程
    #      自然退出 5s 后仍报活（2026-10-19 bgtest4 探针实锤），ps/pgrep
    #      镜像里也没装。文件系统才是跨 namespace 的可靠信号源；
    #   4) 相同指纹的后续调用直接复用已有结果，不重跑。
    # 参照 deer-flow mcp/tasks/ 的 McpTaskDriver 状态机（简化版）。
    if batching:
        # 每批自己的产物目录：失败/重启只影响当前批，前序批次已落盘的产物
        # 不会被清场删掉（这正是"失败只丢当前批"的落地方式）。
        out_json = "%s/result.json" % batch_dir
        out_err = "%s/err.log" % batch_dir
        out_done = "%s/done.flag" % batch_dir
        out_meta = "%s/ocr.meta" % batch_dir
    else:
        out_json = "%s/result.json" % _OCR_WORKDIR
        out_err = "%s/err.log" % _OCR_WORKDIR
        out_done = "%s/done.flag" % _OCR_WORKDIR
        out_meta = "%s/ocr.meta" % _OCR_WORKDIR
    n_pages = page_count if want is None else len(want)

    # 指纹：src_pdf + pages + dpi 全同才允许复用（pages 不同必须重跑，
    # 否则会读到上一次页范围的产物——语义错误比浪费更严重）。
    # 分批模式追加批次号 + 本批页范围：同一份文档的不同批次之间也必须互相
    # 隔离，否则第 3 批的调用会命中第 2 批留下的 done.flag 直接收错页范围。
    fingerprint = "%s|%s|%s" % (src_pdf, str(pages), int(dpi))
    if batching:
        fingerprint += "|b%03d|p%s" % (
            (batch_idx or 0) + 1,
            _page_span_text([p + 1 for p in (want or [])], limit=10 ** 6),
        )

    async def _poll_done() -> str | None:
        """done.flag 的内容（退出码字符串）；不存在返回 None（仍在跑）。"""
        st = await _sandbox_run(
            sandbox_id,
            "cat %s 2>/dev/null" % shlex.quote(out_done),
            timeout=30,
        )
        text = (st.get("stdout") or "").strip()
        return text or None

    async def _errlog_state() -> tuple[bool, int]:
        """err.log 是否存在、有多少字节。

        必须区分"文件不存在"与"文件存在但为空"：前者 = 启动命令没跑到底
        （引号截断/路径错误），后者 = shell 跑了但 python3 没执行。两种故障
        的修法完全不同，返回给模型的文案要分开（add0229b 二轮就是分不清）。
        """
        st = await _sandbox_run(
            sandbox_id,
            "if [ -e {0} ]; then wc -c < {0}; else echo MISSING; fi".format(
                shlex.quote(out_err)
            ),
            timeout=30,
        )
        raw = (st.get("stdout") or "").strip()
        if not raw or "MISSING" in raw:
            return (False, 0)
        try:
            return (True, int(raw.split()[0]))
        except (ValueError, IndexError):
            return (False, 0)

    async def _read_fingerprint() -> str | None:
        st = await _sandbox_run(
            sandbox_id, "cat %s 2>/dev/null" % shlex.quote(out_meta), timeout=30
        )
        text = (st.get("stdout") or "").strip()
        return text or None

    async def _settle(warnings_list: list[str]) -> dict:
        """收当前（批）产物；分批模式还要结算批次状态并决定下一步。

        ★ 为什么三条收产物路径都走这里：不抽出来的话"指纹复用"分支会直接
          返回 completed，把后面的批次全丢掉——那正是静默截断，本改动要
          消灭的东西。收产物必须在**一处**收口，才不会漏掉批次推进。
        """
        res = await _ocr_collect(
            sandbox_id, out_json, out_err, page_count, want, pages, dpi,
            t0, warnings_list, n_pages, target_pdf, out_txt=out_txt,
        )
        if batch_state is None or batch_idx is None:
            return res
        return await _ocr_batch_settle(
            sandbox_id, batch_state, _OCR_BATCH_STATE, batch_idx, res,
            page_count, t0, out_txt_all,
        )

    # ── 4a. 指纹一致性判定 + 幂等续查 / 启动 ──
    # ★ 指纹判定必须先于 done.flag 分支：完成信号是上一次（不同参数组合）
    #   任务留下时，其 result.json 对本次无效——直接收会读到别的页范围的
    #   产物（页数更少时 _ocr_collect 的 mapping 还会 IndexError，
    #   2026-10-19 e2e T4：pages="1" 收 pages="1-2" 的 2 条记录实锤）。
    prev_fp = await _read_fingerprint()
    done_code = await _poll_done()
    if done_code is not None and prev_fp != fingerprint:
        done_code = None  # 作废旧完成信号，按新指纹重跑（prep 会清场）
    should_launch = False
    if done_code is None:
        # ── 已有同指纹任务是否仍在推进？ ──
        # ★ 不能无脑重启：prep 的 `rm -f result.json` 会把正在跑的进程写到一半
        #   的文件 unlink 掉（进程仍往已删除的 inode 写），结果全丢。add0229b
        #   的续查形态（模型每 ~300s 用完全相同的参数再调）正好踩这条路。
        #   判活 = err.log 的 mtime 新鲜（PID namespace 隔离下 kill -0 不可用，
        #   见 _OCR_LIVENESS_SEC）。
        if prev_fp == fingerprint:
            age = await _ocr_task_age_seconds(sandbox_id, out_err)
            if 0 <= age <= _OCR_LIVENESS_SEC:
                logger.info(
                    "[OCR] 同指纹任务仍在推进（err.log 最近活动 %ds 前），跳过启动直接续等",
                    age,
                )
                should_launch = False
            else:
                # 任务已停（SIGKILL 残留 / 上次中断）⇒ 有非空产物就复用。
                # ★ 必须 test -s（非空）：`python3 ... > result.json` 在命令启动
                #   瞬间就创建了 0 字节文件，用 test -f 会把"正在跑"误判成
                #   "有产物"，复用空文件直接 _fail("read")——2026-10-20 e2e
                #   T2/T3 实锤（T1 wait_seconds=5 提前返回后，T2 立刻续查）。
                has_result = await _sandbox_run(
                    sandbox_id, "test -s %s && echo OK" % shlex.quote(out_json), timeout=30
                )
                if "OK" in (has_result.get("stdout") or ""):
                    warnings0: list[str] = [
                        "复用沙箱内已有 OCR 结果（参数指纹一致，未重跑）"
                    ]
                    _OCR_SILENT_STREAK.pop("%s|%s" % (sandbox_id, fingerprint), None)
                    return await _settle(warnings0)
                should_launch = True
        else:
            should_launch = True

    if done_code is None and should_launch:
        # 启动新的后台任务。启动命令退出码不可全信（e2b process 服务偶发
        # 竞态：同一条命令实测两次一次 exit=1 一次 exit=0），以文件信号为准。
        meta_write = await _sandbox_write(
            sandbox_id, out_meta, fingerprint.encode("utf-8")
        )
        if not meta_write["success"]:
            return _fail("prepare", meta_write["error"])
        # ★ 清场与启动必须是两条独立命令（2026-10-20 add0229b 实锤）。
        #   合成一条 `A && B && C & echo D` 时 bash 把**整个 and-list** 后台化：
        #   `{ A && B && C; } & echo D`。外层子 shell 继承了 commands.run 的
        #   stdout 管道，且要等 C（nohup 前台执行）跑完才退出 ⇒ commands.run
        #   卡满 timeout=60 抛 TimeoutException，e2b 收会话时把 python3 连锅
        #   端；而 A/B 里的 rm -f 早已执行成功 ⇒ 旧产物被清空、新产物永不生成。
        #   外部表现 = "续查 5 次、22 分钟、pages_ocred 恒 0"，模型据此判定
        #   卡死转去解析文本层，最终被循环检测硬停。
        #   沙箱 A/B 实测（_bg_ab2_inner.py）：同一条 OCR 命令，`&&` 链 25s 卡满
        #   被杀；拆开后 0.0s 返回 BG_STARTED，python3 存活并在真跑。
        prep = await _sandbox_run(
            sandbox_id,
            "mkdir -p %s; rm -f %s %s %s"
            % (
                # 分批时清场只清**本批目录**：前面批次的产物必须留着，
                # 否则"失败只丢当前批"就变成"每批都丢前面所有批"。
                shlex.quote(batch_dir or _OCR_WORKDIR),
                shlex.quote(out_json),
                shlex.quote(out_err),
                shlex.quote(out_done),
            ),
            timeout=60,
        )
        if not prep.get("success"):
            return _fail(
                "prepare",
                "清理沙箱内旧 OCR 产物失败: %s"
                % ((prep.get("stderr") or "")[:300] or "commands.run 返回非零"),
            )
        # 下发 OCR 入口脚本（幂等覆盖，依赖上一步 mkdir -p）。必须先于启动：
        # 镜像里烘焙的是 1.1.0（无逐页进度行），而 pages_ocred 真实化与
        # `[ocr_run] page i/n` 进度行依赖 1.2.0。
        runner = await _ensure_ocr_runner(sandbox_id)
        if not runner["success"]:
            return _fail("prepare", "下发沙箱侧 ocr_run.py 失败: %s" % runner["error"])
        # ── P0-3：输入路径归一化 ──
        # 把待 OCR 文件落到固定路径 `_OCR_WORKDIR/input.pdf` 再喂给 ocr_run.py，
        # 从根上免疫文件名里的空格/中文/引号/`$`（原实现直接把 target_pdf 塞进
        # 命令行，只要中间多一层 shell 就出事，见下面的引号事故）。
        # 归一化失败**不阻断**：启动命令已整串 quote，任意文件名都安全，这里
        # 只是加固，退回原路径并留 warning 即可。
        ocr_input = target_pdf
        fixed_input = "%s/input.pdf" % _OCR_WORKDIR
        for probe_cmd, mark in (
            (
                "ln -sf %s %s; test -s %s && echo LINKED"
                % (shlex.quote(target_pdf), shlex.quote(fixed_input), shlex.quote(fixed_input)),
                "LINKED",
            ),
            (
                "cp -f %s %s && test -s %s && echo COPIED"
                % (shlex.quote(target_pdf), shlex.quote(fixed_input), shlex.quote(fixed_input)),
                "COPIED",
            ),
        ):
            norm = await _sandbox_run(sandbox_id, probe_cmd, timeout=600)
            if mark in (norm.get("stdout") or ""):
                ocr_input = fixed_input
                break
        if ocr_input != fixed_input:
            logger.warning(
                "[OCR] 输入路径归一化失败（ln/cp 均不可用），退回原路径 %s：%s",
                target_pdf,
                (norm.get("stderr") or "")[:200] or "无 stderr",
            )
        # ── 启动命令：整串 quote，绝不手工拼单引号（见 _build_ocr_launch_cmd）──
        launch_cmd = _build_ocr_launch_cmd(
            _OCR_ENTRY, ocr_input, int(dpi), out_json, out_err, out_done
        )
        launch = await _sandbox_run(sandbox_id, launch_cmd, timeout=60)
        done_code = await _poll_done()
        # done_code 非 None ⇒ bash -c 秒退（python3 不存在/参数错等），
        # 直接落到下面的退出码分流，不走轮询。
        # ★ P0-2：启动可见性必须**无条件**验证，不能被 BG_STARTED 短路。
        #   BG_STARTED 只证明 nohup 起了；引号截断那种事故里它照样输出，而
        #   python3 从未执行——正是这条短路让 add0229b 静默等了 22 分钟。
        if done_code is None:
            launch_visible = False
            errlog_exists = False
            for _ in range(_OCR_LAUNCH_PROBE_TRIES):
                await _asyncio.sleep(_OCR_LAUNCH_PROBE_INTERVAL)
                done_code = await _poll_done()
                if done_code is not None:
                    launch_visible = True  # 秒退：退出码分流能给出真实原因
                    break
                errlog_exists, errlog_size = await _errlog_state()
                if errlog_size > 0:
                    launch_visible = True  # 有 stderr ⇒ python3 真起来了
                    break
            if not launch_visible:
                window = _OCR_LAUNCH_PROBE_TRIES * _OCR_LAUNCH_PROBE_INTERVAL
                return _fail(
                    "launch",
                    "OCR 后台任务未能启动：%ds 内 err.log %s。"
                    "BG_STARTED=%s 只代表 nohup 这一层起了，不代表 python3 起了"
                    "（启动命令被截断时 echo 照常输出）。"
                    "launch_stdout=%r; launch_stderr=%r"
                    % (
                        window,
                        (
                            "始终未被创建——启动命令没跑到底（路径/引号问题）"
                            if not errlog_exists
                            else "已创建但始终为空——python3 未真正执行"
                        ),
                        "收到" if "BG_STARTED" in (launch.get("stdout") or "") else "缺失",
                        (launch.get("stdout") or "")[:200],
                        (launch.get("stderr") or "")[:300],
                    ),
                )

    # ── 4b. 轮询等待（单次调用最多 wait_seconds 秒）──
    deadline = _time.time() + wait_seconds
    while done_code is None:
        if _time.time() >= deadline:
            # 进度来自 err.log 的逐页进度行，不是 result.json（后者整批跑完
            # 才落地，轮询期间永远读不到 ⇒ 恒 0 ⇒ 模型误判卡死）。
            tail = await _sandbox_run(
                sandbox_id, "tail -n 20 %s 2>/dev/null" % shlex.quote(out_err), timeout=30
            )
            err_tail = (tail.get("stdout") or "").strip()
            ocred, _parsed_total = _parse_ocr_progress(err_tail)
            # ★ pages_* 语义统一到"本批"：此前 pages_total 返回文档总页数
            #   （262）而 pages_ocred_total 返回本批页数（10），同一份返回里两个
            #   "total" 互相矛盾，模型据此算错剩余时间（add0229b 实锤）。
            #   文档总页数单独用 pdf_total_pages 给。
            ocred_total = n_pages
            streak_key = "%s|%s" % (sandbox_id, fingerprint)
            if ocred > 0:
                _OCR_SILENT_STREAK.pop(streak_key, None)
                stage = "正在逐页 OCR（已处理 %d/%d 页）" % (ocred, ocred_total)
            elif err_tail:
                _OCR_SILENT_STREAK.pop(streak_key, None)
                stage = (
                    "正在渲染 PDF 页面（渲染是整批做的，渲染完才开始逐页 OCR，"
                    "此阶段进度为 0 属正常）"
                )
            else:
                # 第四态：进程一声不响。启动探针已经把"没起来"判死，走到这里
                # 说明进程起来后静默退出/被 SIGKILL——必须让模型能识别，
                # 而不是第 5 次续查才自己发现（原三态里这一态和"正在渲染"混在
                # 一起，模型无从区分）。
                streak = _bump_silent_streak(streak_key)
                stage = (
                    "进程尚未产生任何 stderr 输出（第 %d 次，疑似已静默退出）"
                    % streak
                )
                if streak >= 2:
                    stage += "——建议按失败处理，用 execute 检查 err.log 与进程状态"
            running_payload = {
                "success": True,
                "step": None,
                "error": None,
                "status": "running",
                # pages_total = 本批要 OCR 的页数（与 pages_ocred 同量纲）
                "pages_total": n_pages,
                # PDF 总页数（pages 参数只取子集时与 pages_total 不同）
                "pdf_total_pages": page_count,
                "pages_ocred": ocred,
                "pages_ocred_total": ocred_total,
                "pages_failed": 0,
                "total_texts": 0,
                "elapsed_sec": round(_time.time() - t0, 1),
                "payload_json": None,
                "payload_txt": None,
                "sandbox_path": target_pdf,
                "sandbox_dir": _OCR_WORKDIR,
                "preview": "",
                "warnings": [],
                "progress_hint": (
                    "OCR 仍在沙箱后台运行（本次已等待 %ds；%s）。"
                    "请用完全相同的参数再调一次本工具续查/取结果——"
                    "参数指纹一致不会重复上传、也不会重跑 OCR。"
                    "PDF 在沙箱内 %s（目录 %s），**不要重新上传**。"
                    "150dpi 实测约 30s/页，本批 %d 页（文档共 %d 页）"
                    "可据此估算剩余时间；err.log 尾部：%s"
                    % (
                        wait_seconds,
                        stage,
                        target_pdf,
                        _OCR_WORKDIR,
                        n_pages,
                        page_count,
                        err_tail.splitlines()[-1][:200] if err_tail else "(空)",
                    )
                ),
            }
            if batch_state is not None and batch_idx is not None:
                # 分批：把"第几批 / 整本累计"补上去。模型据此能算出还剩多少，
                # 而不是只看到一个恒定的本批 0/40 以为卡死。
                bsum = _batch_summary(batch_state)
                cur_b = batch_state["batches"][batch_idx]
                running_payload.update({
                    "batch": "%d/%d" % (cur_b.get("idx") or 0,
                                        batch_state.get("batch_count") or 0),
                    "batch_index": cur_b.get("idx") or 0,
                    "batch_count": batch_state.get("batch_count") or 0,
                    "batch_pages": _page_span_text(cur_b.get("pages") or []),
                    "batches_done": bsum["batches_done"],
                    "batches_failed": bsum["batches_failed"],
                    "pages_ocred_doc": bsum["pages_ocred_doc"] + ocred,
                })
                running_payload["progress_hint"] = (
                    "分批 OCR：第 %s/%s 批（第 %s 页）正在后台运行；"
                    "本批已处理 %d/%d 页，整本累计 %d/%d 页。%s。"
                    "请用**完全相同的参数**（含 pages/batch_pages/dpi）再调一次本工具"
                    "续查；本批完成后会自动结算并给出下一批。"
                    "PDF 在沙箱内 %s，**不要重新上传、不要换 pages、"
                    "不要改用 execute 自己写脚本重跑**——"
                    "已完成批次的结果已落盘，重启只会重跑当前批。"
                    "150dpi 实测约 30s/页；err.log 尾部：%s"
                    % (
                        cur_b.get("idx") or 0,
                        batch_state.get("batch_count") or 0,
                        _page_span_text(cur_b.get("pages") or []),
                        ocred,
                        n_pages,
                        bsum["pages_ocred_doc"] + ocred,
                        page_count,
                        stage,
                        target_pdf,
                        err_tail.splitlines()[-1][:200] if err_tail else "(空)",
                    )
                )
            return running_payload
        await _asyncio.sleep(_OCR_POLL_INTERVAL)
        done_code = await _poll_done()

    # ── 4c. 进程结束：按退出码分流 ──
    # 终验指纹：轮询期间若有别的参数组合的任务也写了 done.flag，此时收上来
    # 的会是它的产物——宁可显式失败让模型重试，也不返回错页范围的数据。
    _OCR_SILENT_STREAK.pop("%s|%s" % (sandbox_id, fingerprint), None)
    final_fp = await _read_fingerprint()
    if final_fp is not None and final_fp != fingerprint:
        return _fail(
            "stale",
            "沙箱内 OCR 产物与本次参数指纹不一致（可能有多参数任务并发），"
            "请稍后用相同参数重试或换 sandbox_id",
        )
    warnings: list[str] = []
    try:
        exit_code = int(done_code)
    except (TypeError, ValueError):
        exit_code = -1
    if exit_code == 2:
        warnings.append("OCR 未识别出任何文字（页面可能是纯图无文字/分辨率过低）")
    elif exit_code not in (0, 2):
        tail = await _sandbox_run(sandbox_id, "tail -n 3 %s" % shlex.quote(out_err), timeout=60)
        warnings.append(
            "OCR 执行失败（exit=%d）：%s" % (exit_code, (tail.get("stdout") or "").strip()[:300])
        )
    return await _settle(warnings)


async def _ocr_collect(
    sandbox_id: str,
    out_json: str,
    out_err: str,
    page_count: int,
    want: list[int] | None,
    pages: str,
    dpi: int,
    t0: float,
    warnings: list[str],
    n_pages: int,
    sandbox_path: str = "",
    out_txt: str = "",
) -> dict:
    """S5：读回 result.json，拼纯文本，只回摘要 + 预览。

    从 ocr_pdf_to_text 抽出——submit-and-poll 改造后有三条路径汇到此处
    （轮询等到完成 / 指纹复用 / 续查时进程已结束），收集逻辑只写一份。

    ``out_txt``：纯文本落地路径。分批模式下由调用方传"批目录/result.txt"，
    这样每批的文本各自独立（后续由 _ocr_batch_finish 按批序重拼全文），
    否则第 3 批会把第 2 批的文本覆盖掉。默认沿用根路径（单批路径不变）。
    """
    import shlex
    import time as _time

    # ── S5：读回结果，拼纯文本，只回摘要 + 预览 ──
    rd = await _sandbox_read(sandbox_id, out_json)
    if not rd["success"] or not rd["data"]:
        tail = await _sandbox_run(sandbox_id, "tail -n 5 %s" % shlex.quote(out_err), timeout=60)
        return _fail(
            "read",
            "沙箱内未生成可读的 result.json：%s | stderr=%s"
            % (rd.get("error"), (tail.get("stdout") or "")[-300:]),
        )
    raw_text = bytes(rd["data"]).decode("utf-8", errors="replace")
    recs = _first_json_value(raw_text)
    if recs is None:
        tail = await _sandbox_run(sandbox_id, "tail -n 8 %s" % shlex.quote(out_err), timeout=60)
        return _fail(
            "parse",
            "result.json 里没有可用 JSON：head=%r | stderr=%s"
            % (raw_text[:200], (tail.get("stdout") or "")[-300:]),
        )

    if not isinstance(recs, list):
        return _fail("parse", "result.json 结构异常（期望 list）")

    # 子 PDF 的 page 序号 → 原 PDF 页号。want 比 recs 短时（理论上是
    # 指纹不一致的脏数据被收了上来）不得 IndexError——退化为恒等映射，
    # 由上游指纹终验兜底；这里只保证不崩。
    mapping = {
        i + 1: (want[i] + 1 if want is not None and i < len(want) else i + 1)
        for i in range(len(recs))
    }
    chunks: list[str] = []
    total_texts = 0
    failed = 0
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        sub_no = int(rec.get("page") or 0)
        real_no = mapping.get(sub_no, sub_no)
        texts = rec.get("texts") or []
        total_texts += len(texts)
        if not rec.get("ok"):
            failed += 1
        chunks.append("===== page %d =====" % real_no)
        chunks.append("\n".join(str(t) for t in texts) if texts else "(无文本)")
        chunks.append("")
    body = "\n".join(chunks)
    out_txt = out_txt or ("%s/result.txt" % _OCR_WORKDIR)
    tw = await _sandbox_write(sandbox_id, out_txt, body.encode("utf-8"))
    if not tw["success"]:
        warnings.append("纯文本回写失败：%s" % tw.get("error"))

    if want is not None:
        warnings.append("已按 pages=%r 只 OCR %d/%d 页" % (pages, len(want), page_count))
    if dpi > 150:
        warnings.append("dpi=%d 高于默认 150，单页耗时约为 150dpi 的 2 倍以上" % int(dpi))

    return {
        "success": failed == 0 or total_texts > 0,
        "step": None,
        "error": None,
        "status": "completed",
        # 与 running 分支同口径：pages_* 一律指"本批"，文档总页数用
        # pdf_total_pages（此前 pages_total=文档总页数、pages_ocred=本批页数，
        # 同一份返回里两个 total 互相矛盾）。
        "pages_total": n_pages,
        "pdf_total_pages": page_count,
        "pages_ocred": len(recs),
        # 与 running 分支同名字段对齐（e2e 实测 completed 缺这个键，模型对
        # pages_ocred 与 pages_* 的口径会对不上）
        "pages_ocred_total": n_pages,
        "pages_failed": failed,
        "total_texts": total_texts,
        "elapsed_sec": round(_time.time() - t0, 1),
        "payload_json": out_json,
        "payload_txt": out_txt,
        # 沙箱内路径回显：模型据此知道文件在哪、不必重新上传（add0229b：
        # ls /home/user 没看到 _ocr/ 子目录 ⇒ 结论"PDF 没上传" ⇒ 重传 268MB）。
        "sandbox_path": sandbox_path,
        "sandbox_dir": _OCR_WORKDIR,
        "preview": body[:_OCR_PREVIEW_CHARS],
        "warnings": warnings,
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
    print("服务地址: http://0.0.0.0:8000/mcp")
    print("\n可用工具:")
    print("  - decrypt_file: 解密文件返回字节流")
    print("  - read_excel: 解密并读取 Excel")
    print("  - read_pdf_text: 解密并提取 PDF 文本")
    print("  - filter_excel: 按条件筛选 Excel 数据")
    print("  - list_excel_sheets: 列出 Excel 所有工作表")
    print("\n按 Ctrl+C 停止服务")

    # host=0.0.0.0：容器化部署时 agent 经 appnet 服务名 geesun-mcp 访问；
    # 若仍按 dev 同机裸跑，0.0.0.0 同样接受 127.0.0.1 的本地连接，行为不变。
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="streamable-http",
    )
