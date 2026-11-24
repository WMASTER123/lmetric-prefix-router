# lmetric-prefix-router

当前开发目标为服务器预装 vLLM 0.23 的共享 DP 入口。每次选择一个完整 TP2 副本。可独立运行的开发网关与 Manager 描述符分别验证，当前 Manager 状态为 `import_only`。

## 预装 0.23 主开发线

在一个共享地址后运行两个完整的 TP2 副本。combined 在准入后、发送前读取各副本混合缓存与在途负载，有可信优势时由网关设置 rank 头，无可信正命中或新鲜等命中时交给引擎原生调度。开发入口提供 combined、同代理原生和纯最长前缀三个模式。安装、依赖、可定向的输入范围、观测与退出见[预装版运行说明](docs/native023.md)。

## 并行保留的 r10 双端点变体

r10 计划在两个独立 TP2 服务各运行冻结的 r10 Qwen3.5-35B-A3B。相同代理、相同负载及相同在途计数贯穿两个实验臂。在可信快照完整、有正命中、且需要定向时，combined 在最长命中的副本中最小化 `(未命中的输入 token 数 + 待处理 prefill token 数) × (在途数 + 1)`。全部冷、缺少可信比较或所有副本正命中相同且采样新鲜时，交给外部均衡器。外部基线始终选本代理观察到的在途请求最少的端点，平局时轮换。实验要求所有流量经过同一代理。旧 0.23 实验在共享入口委托 vLLM 原生 DP，旧结果另行保留。

## 安装与复核

在独立 Python 环境中从仓库根目录运行

```sh
python -m pip install -e ".[test]"
python -m pytest -q
vllm-hust-ext extension inspect io.github.wmaster123.lmetric-prefix-router
```

可用以下命令启动开发用外部基线。两个地址须为隔离测试服务的回环地址或平台批准的本地端口转发。CLI 仅暴露基线；组合策略的集成测试通过进程内接口提供可信快照。

```sh
python -m lmetric_prefix_router.gateway --backends http://127.0.0.1:18410 http://127.0.0.1:18420 --port 18300
```

实验网关只代理 completions 和 chat/completions，使用独立有界连接池，在整段流式响应结束或断开后释放在途记账。它要求隔离、无鉴权的测试后端；生产接入仍需要租户身份、健康成员管理及可信的 r10 缓存读端。退出本地测试可停止网关并卸载开发包。

## 来源

- 个人 MOD 使用[组织模板 4163473](https://github.com/vLLM-HUST/extension-template/tree/416347398ba197a4db5e4360db1aaecc64355b79)创建。
- 决策函数取自 WMASTER123 的 lmetric-combined 实验。传输层基于组织 Prefix Router 的每后端连接池与请求计时，保留有界准入、发送前复验和流式释放，简化静态请求头与路由配置。[组织实现 bcc9691](https://github.com/vLLM-HUST/vllm-hust-prefix-router/tree/bcc9691)
- 网站引擎兼容性目标仍为 2026.11.21.10。环境和模型资格以正式包为准。[r10 冻结制品](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r10-v1)
