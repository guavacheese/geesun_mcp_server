import base64
import os
import io
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Optional
import httpx

load_dotenv()

decrypt_api_url = os.getenv("DECRYPT_API_URL")
mcp = FastMCP("decrypt-file")


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
