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


@mcp.tool()
async def copy_script_to_sandbox(
    script_name: str,
    sandbox_id: str,
    sandbox_path: str | None = None,
    skill_name: str = "plc-code-auditor",
) -> dict:
    """
    将 skill 脚本直传沙箱，不走 LLM 上下文。

    自动搜索所有 skill 来源目录（__system__ → __agent__ → __user_*__），
    无需指定 skill_source。

    Args:
        script_name: 脚本文件名（如 plc_audit.py）
        sandbox_id: 沙箱 ID
        sandbox_path: 沙箱目标路径（如 /home/user/plc_audit.py），
                      不传则自动取 /home/user/{script_name}
        skill_name: 技能名称（默认 plc-code-auditor）
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

    # 依次搜索，第一个找到的命中
    found_path = None
    for source in source_dirs:
        test_path = f"{skills_base}/{source}/{skill_name}/scripts/{script_name}"
        if os.path.isfile(test_path):
            found_path = test_path
            break

    if not found_path:
        return {
            "success": False,
            "error": f"在所有 skill 目录中均未找到 {skill_name}/scripts/{script_name}，"
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
            relative_path = file_path[len("/uploads/"):]
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
    【推荐方式】解密文件并直接上传到 CubeSandbox 沙箱内。

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


@mcp.tool()
async def decrypt_to_tempfile(
    file_path: str,
    output_path: str,
) -> dict:
    """
    解密文件并写入指定路径（/tmp下），供同机其他进程直接使用。
    注意：调用方用完后负责清理临时文件。
    :return: {"success": bool, "output_path": str, "error": str}
    """

    result = await _decrypt_file_internal(file_path)
    if not result["success"]:
        return {
            "success": False,
            "output_path": None,
            "error": result["error"],
        }

    with open(output_path, "wb") as f:
        f.write(result["data"])

    return {
        "success": True,
        "output_path": output_path,
        "error": None,
    }


def _is_valid_sandbox_id(sandbox_id: str) -> bool:
    """检查 sandbox_id 是否格式有效（32位hex或36位含横线UUID，且不含中文等占位文本）"""
    sid = sandbox_id.strip()
    if not sid:
        return False
    # 标准 UUID 格式 8-4-4-4-12（含横线）
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', sid, re.I):
        return True
    # 32 位 hex（无横线）
    if re.match(r'^[0-9a-f]{32}$', sid, re.I):
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

    Returns:
        {"success": bool, "host_path": str | None, "size": int, "error": str | None}
    """
    # 路径转换：虚拟 /reports/ 路径 → 物理路径
    if host_path.startswith("/reports/"):
        report_root = os.environ.get("REPORT_ROOT", "")
        if report_root:
            relative_path = host_path[len("/reports/"):]
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


@mcp.tool()
async def read_excel(file_path: str, sheet_name: Optional[str] = None) -> dict:
    """
    解密并读取 Excel 文件，返回结构化数据

    :param file_path: Excel 文件路径（加密状态）
    :param sheet_name: 工作表名称，默认第一个
    :return: {"success": bool, "data": list[dict], "columns": list, "error": str}
    """

    import pandas as pd

    try:
        # 1.解密到内存
        decrypt_result = await _decrypt_file_internal(file_path)
        if not decrypt_result["success"]:
            return {
                "success": False,
                "data": None,
                "columns": None,
                "row_count": 0,
                "error": decrypt_result["error"],
            }

        # 2. 内存中读取 Excel
        excel_buff = io.BytesIO(decrypt_result["data"])

        if sheet_name:
            df = pd.read_excel(excel_buff, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_buff)

        # 3. 转换为结构化数据
        data = df.fillna("").to_dict(orient="records")
        columns = df.columns.to_list()

        return {
            "success": True,
            "data": data,
            "columns": columns,
            "row_count": len(data),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "columns": None,
            "row_count": 0,
            "error": str(e),
        }


@mcp.tool()
async def read_pdf_text(file_path: str) -> dict:
    """
    解密并提取 PDF 文本内容

    :param file_path: PDF 文件路径（加密状态）
    :return: {"success": bool, "text": str, "page_count": int, "error": str}
    """

    from PyPDF2 import PdfReader

    try:
        decrypt_result = await _decrypt_file_internal(file_path)
        if not decrypt_result["success"]:
            return {
                "success": False,
                "text": None,
                "page_count": 0,
                "error": decrypt_result["error"],
            }

        pdf_buffer = io.BytesIO(decrypt_result["data"])
        reader = PdfReader(pdf_buffer)

        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        page_count = len(reader.pages)
        pdf_buffer.close()

        return {
            "success": True,
            "text": text.strip(),
            "page_count": page_count,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "text": None,
            "page_count": 0,
            "error": str(e),
        }


@mcp.tool()
async def filter_excel(file_path: str, column: str, value: str) -> dict:

    import pandas as pd

    try:
        decrypt_result = await _decrypt_file_internal(file_path)
        if not decrypt_result["success"]:
            return {
                "success": False,
                "data": None,
                "row_count": 0,
                "error": decrypt_result["error"],
            }

        excel_buff = io.BytesIO(decrypt_result["data"])
        df = pd.read_excel(excel_buff)
        excel_buff.close()

        # 筛选
        filtered = df[df[column].astype(str) == value]

        return {
            "success": True,
            "data": filtered.fillna("").to_dict(orient="records"),
            "row_count": len(filtered),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "row_count": 0,
            "error": str(e),
        }


@mcp.tool()
async def list_excel_sheets(file_path: str) -> dict:

    import pandas as pd

    try:
        decrypt_result = await _decrypt_file_internal(file_path)
        if not decrypt_result["success"]:
            return {"success": False, "sheets": None, "error": decrypt_result["error"]}

        # 读取所有sheet名称
        excel_buffer = io.BytesIO(decrypt_result["data"])
        xl = pd.ExcelFile(excel_buffer)
        sheet_names = xl.sheet_names
        excel_buffer.close()

        return {
            "success": True,
            "sheets": sheet_names,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "sheets": None,
            "error": str(e),
        }


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
