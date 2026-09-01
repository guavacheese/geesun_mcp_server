# geesun_mcp_server

企业内网文件解密 / 沙箱上传 MCP 服务（FastMCP + FastAPI）。提供 `decrypt_file_to_base64` / `decrypt_and_upload_to_sandbox` / `upload_to_sandbox` 等工具，供 geesun_agent 通过 MCP 调用，实现在 DLP 加密文档环境下安全解密与沙箱隔离处理。

## 环境变量

所有配置通过环境变量注入（`main.py` 启动时 `load_dotenv()` 读取 `.env`）。

| 变量 | 说明 | 本地示例 |
|---|---|---|
| `DECRYPT_API_URL` | 企业 DLP 文档解密 API 地址 | `http://10.10.10.17:8090/common-file/anony/revolveAdminFile` |
| `E2B_API_URL` | CubeSandbox cube-api 地址（裸金属） | `http://172.16.66.13:6000` |
| `E2B_API_KEY` | 沙箱凭据（敏感） | `e2b_0000000000000000000000000000000000000000` |
| `SSL_CERT_FILE` | 沙箱侧 TLS 根证书路径（留空不注入） | `/home/dhp/projects/cube-cert/rootCA.pem` |
| `AGENT_WORKSPACE` | agent 工作区路径（沙箱/挂载视角） | `/mnt/d/workspace/geesun_agent` |
| `UPLOAD_ROOT` | 上传数据根目录 | `/mnt/d/workspace/geesun_agent/data/uploads` |
| `REPORT_ROOT` | 报告输出根目录 | `/mnt/d/workspace/geesun_agent/data/reports` |

> ⚠️ `.env` 已被 `.gitignore` 忽略，**含密钥（`E2B_API_KEY`），绝不入库、不进镜像**。仓库只提交脱敏模板 `.env.example`。

## 本地开发（uv run）

```bash
# 1. 准备环境变量（首次）
cp .env.example .env
#    按实际环境编辑 .env：解密 API、cube-api、密钥、路径

# 2. 安装依赖并启动（requires-python >= 3.13）
uv sync --frozen
uv run python main.py
```

启动后监听 `0.0.0.0:8000`（本地测试可临时改 `host` 或直接访问 `http://127.0.0.1:8000`）。agent 侧通过 MCP 配置指向该地址即可联调。

## Docker 镜像

生产镜像由 `geesun_agent/deploy/build-push.sh` 的 `build_push_mcp()` 统一构建并推 Harbor `geesun_ai/geesun-mcp-server:<MCP_TAG>`（构建上下文为本仓库根）。也可单独构建：

```bash
docker build -t 172.16.220.74:8333/geesun_ai/geesun-mcp-server:1.0.0 .
```

- Dockerfile 基镜像 `python:3.13-slim-bookworm`（与 `pyproject.toml` 的 `requires-python>=3.13` 一致），非 root UID 1001（`mcpuser`，与 agent 同 UID，便于共享挂载目录属主）。
- `.dockerignore` 排除 `.venv/`、`.env`（密钥）等，确保镜像干净。
- 部署走 Docker Swarm：`geesun_agent/deploy/start_stack.sh --with=mcp`（详见 geesun_agent 仓库 README「部署」节）。

## 工具清单（MCP 方法）

- `decrypt_file_to_base64`：解密 DLP 文件为内存 base64（不解盘）
- `decrypt_and_upload_to_sandbox`：解密后上传沙箱执行
- `upload_to_sandbox`：上传文件到沙箱（检测 DLP 加密自动兜底解密上传）
- `download_from_sandbox` / `copy_script_to_sandbox`：沙箱产物回传 / 脚本注入
- `run_pdf_diff_stage1` / `run_pdf_diff_stage3`：PDF 差异比对流水线
