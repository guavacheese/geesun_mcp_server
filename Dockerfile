# geesun_mcp_server 生产镜像（企业文件解密 MCP）
# 构建上下文：geesun_mcp_server 仓库根（/d/workspace/geesun_mcp_server）
# 推送到 Harbor geesun_ai 项目：172.16.220.74:8333/geesun_ai/geesun-mcp-server:<tag>
# 由 deploy/build-push.sh 的 sync_mcp() 构建并推送。
# ⚠️ 基镜像必须与 pyproject.toml 的 requires-python(>=3.13) 及 dev 运行时一致，
#    否则会出现「dev 能跑、生产镜像装不上 fastmcp/pandas」的漂移。
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖安装（利用层缓存）。仍以 root 装：pip 要写 /usr/local/lib/python3.13，
# 用 mcpuser 装会 permission denied（与 agent 镜像不同，那边 venv 在 /app 内）。
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# 非 root 运行（与 agent 一致用 UID 1001，便于共享挂载目录属主）
# ★ 用户提前建好、源码一律 COPY --chown，避免「先 root 拷、最后 chown -R /app」：
#   同一写法在 agent 镜像上实测 chown 层要 1149s（BuildKit 为整棵树 copy-up
#   生成一份 354MB 重复层），纯白等。
RUN useradd --create-home --uid 1001 mcpuser

COPY --chown=mcpuser:mcpuser . /app

USER mcpuser

EXPOSE 8000

# host=0.0.0.0：agent 经 appnet 服务名 geesun-mcp:8000 访问（见 main.py）
CMD ["python", "main.py"]
