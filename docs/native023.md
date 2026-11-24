# native023 本机共享引擎接入指南

此开发包适配服务器预装的 vLLM `0.23.0+empty@0fc695fc` 和 vLLM-Ascend `0.23.0@5cb98caa`。使用单机、单租户、固定本地模型目录和一个共享 HTTP 入口。网站冻结引擎 r10 使用[独立接入条件](r10-qualification.md)。

## 副本与设备

一次决策返回一个内部 DP rank 或委托原生。TP 组内的设备共同执行同一请求。当前网关支持 DP1 和 DP2。

| 引擎拓扑 | 路由行为 |
|---|---|
| DP1 × TP1 或 TP2 | 一个可选目的地。省略 rank 头，已知时仍返回缓存预测 |
| DP2 × TP1 | 两个单卡副本，网关可以选择 rank 0 或 1 |
| DP2 × TP2 | 两个双卡副本，路由目标同上；35B 四卡的当前候选仍须单独验收 |

## 运行已绑定的本地服务

在服务器创建独立环境，安装本开发包；Manager 使用仓库 requirements-manager.txt 固定的版本。使用与引擎相同的本地分词器、块哈希配置和固定 PYTHONHASHSEED。下例在两张已分配的 Ascend 910B2 上运行一个 35B 副本。启动引擎和网关的用户需要能读取同一 Linux PID 命名空间内的引擎进程与监听套接字。保持模型和配置文件在服务存续期间只读。

```sh
python -m venv --system-site-packages .venv
.venv/bin/pip install '.[native023]' -r requirements-manager.txt
export PYTHONHASHSEED=42
vllm serve /models/Qwen3.5-35B-A3B --served-model-name lab-model \
  --host 127.0.0.1 --port 18200 --data-parallel-size 1 --tensor-parallel-size 2 \
  --dtype bfloat16 --language-model-only --max-model-len 4096 --max-num-seqs 32 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --enable-chunked-prefill \
  --enable-prompt-tokens-details --kv-events-config \
  '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:18300","replay_endpoint":"tcp://127.0.0.1:18400","buffer_steps":100000}' &
engine_pid=$!
```

等待引擎 `/health` 返回 200 后启动网关。绑定模式读取实际启动命令中的 DP 和 TP，并核对 API 监听端口、每个 rank 的独立 EngineCore 重放监听端口和进程归属。可另外传入 `--dp-size` 作为断言；不一致时启动失败。

```sh
.venv/bin/python -m lmetric_prefix_router.native023 --engine-pid "$engine_pid" \
  --backend http://127.0.0.1:18200 --model-path /models/Qwen3.5-35B-A3B \
  --served-model-name lab-model --event-port 18300 --replay-port 18400 --port 18201
```

单全注意力接入面限定 Qwen2 架构、关闭滑窗、显式 `--block-size 128`。35B 混合布局使用 2048 块、全注意力连续链与三个 Mamba 组的共同终点。输入上限为 4096 tokens。适配其他模型需要分别核验块布局、哈希、分词和实际回执。

## 冷启动、恢复与可观测性

- 默认生产候选路径使用上面的进程绑定。在固定版本的调度器中，新 EngineCore 建立发布者和空块池。网关从序号 0 连续重放到结束帧，可在无手动清缓存的情况下建立视图。每次事务前后检查原进程及套接字归属，派发时检查进程存活与视图期限。
- 后台循环在每次事务结束后等待 250 ms，静默期间仍读取增量。每次最多 1024 批、8 MiB、1 秒。可重试超时保留已提交游标，恢复前暂停预测；断序或截断撤销索引。已发布但仍未处理的事件可能造成短时预测误差。1.25 秒未完成有效同步，停止使用该视图。进程换代后重新绑定新引擎并更新 Manager 配置。历史超过单次重建预算或已被覆盖时，可能持续委托原生。
- 省略 `--engine-pid` 时保留早期开发用的实时订阅路径。此路径依赖清空基线，默认按 DP2 检查，可显式指定 DP1。Manager 的绑定就绪检查只认可有进程身份的路径。
- 单条字符串或 token IDs 的 completions 可建立指纹。绑定模式启动时用单轮及多轮对话对照引擎 `/tokenize` 的完整 token IDs，通过后才开放纯文本的 system/user/assistant 消息。多模态、工具、自定义模板、salt 或未核验参数委托原生；额外身份键的存储与移除独立记账。
- `/status` 提供拓扑、观测到的指标成员、缓存视图、负载新鲜度、各原因次数及最近 64 条请求的进程内分段计时。记录解析、指纹（含分词与哈希）、准入、发送前选点、引擎响应头、首个响应体、可见流内容及结束时刻。非流式回执在完成时解析。首个 HTTP 响应块不能直接用作客户端 TTFT。
- `X-Lmetric-Reason` 将 `single_replica`、`all_miss`、`equal_hit_native` 与 `unsupported_input`、`topology_mismatch`、`topology_unverified`、`cache_view_unavailable` 分别计数。`longest_prefix` 主动选择最长命中副本；正命中相同且负载采样过期时使用本地积压估计。独立对照模式包括 `--policy native` 和 `--policy prefix`。
- 负载使用引擎 running/waiting 以及代理在途记账。委托请求的实际副本通常无法从 HTTP 回执直接辨认。串行功能探针使用引擎各 rank 的查询与命中计数增量复核；并发性能实验需要独立的归属证据。

## Manager 与回滚

部署者核对启动命令、模型和 `/status` 中的拓扑后，将当前 `engine_generation` 的 64 位十六进制摘要、HTTP 地址及每个 rank 的 replay 端口保存为部署专用 JSON。Provider 核对正在运行的包版本、插件 ID、预期引擎代际、HTTP 与 replay 地址、combined 策略、所有 rank 就绪以及引擎 HTTP 健康。Manager 单独保存启用意图。

```json
{"router_url":"http://127.0.0.1:18201","backend_url":"http://127.0.0.1:18200","replay_endpoints":{"0":"tcp://127.0.0.1:18400"},"engine_generation":"<从本次部署核对的摘要>"}
```

```sh
id=io.github.wmaster123.lmetric-prefix-router
.venv/bin/vllm-hust-ext extension configure "$id" --file binding.json
.venv/bin/vllm-hust-ext extension check "$id"
.venv/bin/vllm-hust-ext extension enable "$id"
.venv/bin/vllm-hust-ext extension status "$id"
```

运维方负责将请求入口切到网关。回滚时将新流量切回原引擎，等待 `/status.active` 归零，停止网关并执行 `extension disable`。Manager 的配置与启用不会搬迁正在生成的请求。

DP2 配置还应列出 `"1":"tcp://127.0.0.1:18401"`。确认无需保存意图后，可执行 `extension forget "$id"`，并在网关环境运行 `python -m pip uninstall lmetric-prefix-router`。

## 功能验收与组件状态

使用全新、无其他业务流量的专用引擎执行 `experiments/probe_managed023.py`。传入共享引擎和网关地址、served model、DP、TP 和块大小。探针检查初始查询计数为零，全程不调用清缓存接口。分别验证冷、单热、双热和文本聊天，各 rank 的引擎增量给出实际落点。例如单 TP2 使用 `--dp 1 --tp 2 --block 2048`。此探针不测性能。

| 组件 | 当前实现 | 尚需单独验证 |
|---|---|---|
| 决策与记账 | 复用 combined、发送前预留和终态释放；按已验证 DP 成员选点 | 大并发积压估计误差、35B 双 TP2 |
| 请求与缓存 | 原生哈希、分模型布局、纯文本聊天模板对照、隔离带盐块 | 更多模型、模板与多租户 |
| 转发与观测 | 复用连接池、有界 JSON/SSE 回执及计时工具，记录指纹和准入的在线开销 | 分场景 TTFT、停顿和有效吞吐 |
| 恢复 | 本地进程与套接字绑定、从零连续重放、后台检查尾部、超时和进程退出退化 | 大历史重建及高压下恢复时间 |
| 生命周期 | 模板 Manifest 0.3、独占前门、外部 Provider 和运维排空 | 网站冻结 r10 兼容性、真实部署切流 |

产品仓保存运行组件、回归和可复跑的功能探针。原始请求、结果和性能数据留在实验环境。传输层、计时与回执来自组织 Prefix Router，许可证为 Apache-2.0。[组织组件](https://github.com/vLLM-HUST/vllm-hust-prefix-router/tree/bcc9691) · [组织 Provider 规范示例](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/main/docs/extension-manager.md)
