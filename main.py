import base64
import os
import io
import json
import re
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
#: 后台任务启动可见性探针（2026-10-20 add0229b）。commands.run 未返回
#: BG_STARTED 时给进程这么久落首个 stderr 行：ocr_run.py 启动即打 env 行，
#: 真起来了 1~2s 就有内容；这么久还空 ⇒ 进程根本没起来，立刻显式失败。
#: 不能让模型对着一个不存在的任务续查 5 次、等 22 分钟（事故原样）。
_OCR_LAUNCH_PROBE_TRIES = 10
_OCR_LAUNCH_PROBE_INTERVAL = 1
#: 进度标记：ocr_run.py 每页往 stderr 打一行 `[ocr_run] page i/n ...`。
#: pages_ocred 从这里解析——result.json 只在整批跑完才落地，轮询期间
#: 拿它统计只会恒 0（add0229b：模型看 22 分钟"0 页"判定卡死，转去解析
#: 文本层，最终被硬停）。进度必须来自增量信号。
_OCR_PROGRESS_MARK = "[ocr_run] page "

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
    **用完全相同的参数（file_path/sandbox_id/pages/dpi/sandbox_pdf_path）再调一次
    即可续查或取回结果，不会重复上传、不会重复 OCR**（沙箱内按参数指纹复用）。
    大文档推荐分批：pages="1-20" → "21-40" …，单批 20 页约 10 分钟（150dpi 实测
    30s/页），每批一次调用，避免单次任务跑几十分钟。

    Args:
        file_path: 虚拟路径（/uploads/{user_id}/{session_id}/{filename}）或
            宿主物理路径。自动识别 DLP 加密并解密后上传（明文不落盘）。
            **与 sandbox_pdf_path 二选一即可**——PDF 已在沙箱里时只给
            sandbox_pdf_path，不给本参数。
        sandbox_id: 目标沙箱 ID（用当前会话的沙箱）。
        pages: 页选择。"20"（默认，前 20 页）| "all"（全量，慎用）| "1-5" |
            "3,7,9" | "12"。**实测速度约 30s/页（150dpi、4 vCPU）**——262 页
            全量约 110 分钟，抽 20 页约 10 分钟。只关心部分页时务必指定。
        dpi: PDF 渲染 DPI，100~200，默认 150。超过 200 在 8GB 沙箱上有 OOM 风险。
        sandbox_pdf_path: 可选。若 PDF 已经在沙箱里（之前上传过），直接给沙箱路径，
            省掉一次上传（大文件可省几十秒）。
        wait_seconds: 单次调用最长等待秒数，0~300，默认 90。超时未完成则返回
            status="running"，按上面说明用相同参数再调。

    Returns:
        {"success": bool, "step": str|None, "error": str|None,
         "status": "completed"|"running",
         "pages_total": int, "pages_ocred": int, "pages_failed": int,
         "total_texts": int, "elapsed_sec": float,
         "payload_json": str,   # 沙箱内逐页结构化结果（page/texts/scores）
         "payload_txt": str,    # 沙箱内纯文本，带 "===== page N =====" 分隔
         "preview": str,        # 前 2000 字符，够判断内容对不对
         "warnings": [str]}
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
        # ★ 必须重试：沙箱刚 get_or_create 出来时 envd 还没就绪，e2b 的
        #   connect/files.write 会拿到空响应 → `Expecting value: line 1 column 1`。
        #   2026-10-17 实测：建箱后立刻调，9s 就失败；等模型聊两轮再调就成功。
        #   既有 upload_to_sandbox 没有重试（它历史上一直是被"已经聊了一会儿"的
        #   会话调用），本工具是第一个会紧贴建箱调用的，故在此自保。
        last_err = None
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

    target_pdf = src_pdf
    if want is not None:
        sub = src_pdf + ".sub.pdf"
        sp = await _sandbox_run(
            sandbox_id,
            "python3 %s %s %s" % (
                shlex.quote(helper), shlex.quote(src_pdf), " ".join(str(i) for i in want)
            ),
            timeout=600,
        )
        if not sp.get("success"):
            return _fail("split", f"按页拆分失败: {(sp.get('stderr') or '')[:300]}")
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
    out_json = "%s/result.json" % _OCR_WORKDIR
    out_err = "%s/err.log" % _OCR_WORKDIR
    out_done = "%s/done.flag" % _OCR_WORKDIR
    out_meta = "%s/ocr.meta" % _OCR_WORKDIR
    n_pages = page_count if want is None else len(want)

    # 指纹：src_pdf + pages + dpi 全同才允许复用（pages 不同必须重跑，
    # 否则会读到上一次页范围的产物——语义错误比浪费更严重）。
    fingerprint = "%s|%s|%s" % (src_pdf, str(pages), int(dpi))

    async def _poll_done() -> str | None:
        """done.flag 的内容（退出码字符串）；不存在返回 None（仍在跑）。"""
        st = await _sandbox_run(
            sandbox_id,
            "cat %s 2>/dev/null" % shlex.quote(out_done),
            timeout=30,
        )
        text = (st.get("stdout") or "").strip()
        return text or None

    async def _read_fingerprint() -> str | None:
        st = await _sandbox_run(
            sandbox_id, "cat %s 2>/dev/null" % shlex.quote(out_meta), timeout=30
        )
        text = (st.get("stdout") or "").strip()
        return text or None

    # ── 4a. 指纹一致性判定 + 幂等续查 / 启动 ──
    # ★ 指纹判定必须先于 done.flag 分支：完成信号是上一次（不同参数组合）
    #   任务留下时，其 result.json 对本次无效——直接收会读到别的页范围的
    #   产物（页数更少时 _ocr_collect 的 mapping 还会 IndexError，
    #   2026-10-19 e2e T4：pages="1" 收 pages="1-2" 的 2 条记录实锤）。
    prev_fp = await _read_fingerprint()
    done_code = await _poll_done()
    if done_code is not None and prev_fp != fingerprint:
        done_code = None  # 作废旧完成信号，按新指纹重跑（bg_cmd 会清场）
    if done_code is None:
        has_result = (
            await _sandbox_run(
                sandbox_id, "test -f %s && echo OK" % shlex.quote(out_json), timeout=30
            )
        )
        if prev_fp == fingerprint and "OK" in (has_result.get("stdout") or ""):
            # 兜底：done.flag 被清但结果还在且指纹一致 ⇒ 复用
            warnings0: list[str] = ["复用沙箱内已有 OCR 结果（参数指纹一致，未重跑）"]
            return await _ocr_collect(
                sandbox_id, out_json, out_err, page_count, want, pages, dpi,
                t0, warnings0, n_pages, target_pdf,
            )
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
                shlex.quote(_OCR_WORKDIR),
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
        # 启动命令单独一条：`nohup ... &` 只后台化 nohup 自身，echo 在前台，
        # shell 立刻退出、管道立刻关闭 ⇒ commands.run 秒级返回。
        launch_cmd = (
            "nohup bash -c 'python3 %s %s --json --dpi %d > %s 2> %s; echo $? > %s' "
            "</dev/null >/dev/null 2>&1 & echo BG_STARTED"
            % (
                shlex.quote(_OCR_ENTRY),
                shlex.quote(target_pdf),
                int(dpi),
                shlex.quote(out_json),
                shlex.quote(out_err),
                shlex.quote(out_done),
            )
        )
        launch = await _sandbox_run(sandbox_id, launch_cmd, timeout=60)
        done_code = await _poll_done()
        # done_code 非 None ⇒ bash -c 秒退（python3 不存在/参数错等），
        # 直接落到下面的退出码分流，不走轮询。
        # 启动可见性：没看到 BG_STARTED（或 commands.run 自己异常）时**不能
        # 静默放过**——先给进程一点时间落首个 stderr 行（ocr_run.py 启动即打
        # env 行），既没 done.flag 也没 err.log 内容 ⇒ 后台任务根本没起来，
        # 立刻显式失败。add0229b 的 22 分钟空等正是这里静默造成的。
        if done_code is None and "BG_STARTED" not in (launch.get("stdout") or ""):
            launch_visible = False
            for _ in range(_OCR_LAUNCH_PROBE_TRIES):
                await _asyncio.sleep(_OCR_LAUNCH_PROBE_INTERVAL)
                done_code = await _poll_done()
                if done_code is not None:
                    launch_visible = True  # 秒退：退出码分流能给出真实原因
                    break
                err_size = await _sandbox_run(
                    sandbox_id,
                    "wc -c < %s 2>/dev/null || echo 0" % shlex.quote(out_err),
                    timeout=30,
                )
                try:
                    first = (err_size.get("stdout") or "0").strip().split()
                    if first and int(first[0]) > 0:
                        launch_visible = True  # 已有 stderr ⇒ 进程真起来了
                        break
                except (ValueError, IndexError):
                    continue
            if not launch_visible:
                return _fail(
                    "launch",
                    "OCR 后台任务未能启动：commands.run 未返回 BG_STARTED，"
                    "且 %ds 内 err.log 无任何输出（进程从未起来）。"
                    "launch_stderr=%r; launch_stdout=%r"
                    % (
                        _OCR_LAUNCH_PROBE_TRIES * _OCR_LAUNCH_PROBE_INTERVAL,
                        (launch.get("stderr") or "")[:300],
                        (launch.get("stdout") or "")[:200],
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
            ocred, ocred_total = _parse_ocr_progress(err_tail)
            if ocred_total is None:
                ocred_total = n_pages
            if ocred > 0:
                stage = "正在逐页 OCR（已处理 %d/%d 页）" % (ocred, ocred_total)
            elif err_tail:
                stage = (
                    "正在渲染 PDF 页面（渲染是整批做的，渲染完才开始逐页 OCR，"
                    "此阶段进度为 0 属正常）"
                )
            else:
                stage = "尚未产生任何 stderr 输出"
            return {
                "success": True,
                "step": None,
                "error": None,
                "status": "running",
                "pages_total": page_count,
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
                    "150dpi 实测约 30s/页，本批 %d 页可据此估算剩余时间；"
                    "err.log 尾部：%s"
                    % (
                        wait_seconds,
                        stage,
                        target_pdf,
                        _OCR_WORKDIR,
                        n_pages,
                        err_tail.splitlines()[-1][:200] if err_tail else "(空)",
                    )
                ),
            }
        await _asyncio.sleep(_OCR_POLL_INTERVAL)
        done_code = await _poll_done()

    # ── 4c. 进程结束：按退出码分流 ──
    # 终验指纹：轮询期间若有别的参数组合的任务也写了 done.flag，此时收上来
    # 的会是它的产物——宁可显式失败让模型重试，也不返回错页范围的数据。
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
    return await _ocr_collect(
        sandbox_id, out_json, out_err, page_count, want, pages, dpi,
        t0, warnings, n_pages, target_pdf,
    )


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
) -> dict:
    """S5：读回 result.json，拼纯文本，只回摘要 + 预览。

    从 ocr_pdf_to_text 抽出——submit-and-poll 改造后有三条路径汇到此处
    （轮询等到完成 / 指纹复用 / 续查时进程已结束），收集逻辑只写一份。
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
    out_txt = "%s/result.txt" % _OCR_WORKDIR
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
        "pages_total": page_count,
        "pages_ocred": len(recs),
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
