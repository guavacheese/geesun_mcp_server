# geesun_mcp_server 生产镜像（企业文件解密 MCP）
# 构建上下文：geesun_mcp_server 仓库根（/d/workspace/geesun_mcp_server）
# 推送到 Harbor geesun_ai 项目：172.16.220.74:8333/geesun_ai/geesun-mcp-server:<tag>
# 由 deploy/build-push.sh 的 sync_mcp() 构建并推送。
# ⚠️ 基镜像必须与 pyproject.toml 的 requires-python(>=3.13) 及 dev 运行时一致，
#    否则会出现「dev 能跑、生产镜像装不上 fastmcp/pandas」的漂移。
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# 依赖安装（利用层缓存）
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# 源码
COPY . /app

# 非 root 运行（与 agent 一致用 UID 1001，便于共享挂载目录属主）
RUN useradd --create-home --uid 1001 mcpuser \
    && chown -R mcpuser:mcpuser /app
USER mcpuser

EXPOSE 8000

# host=0.0.0.0：agent 经 appnet 服务名 geesun-mcp:8000 访问（见 main.py）
CMD ["python", "main.py"]
