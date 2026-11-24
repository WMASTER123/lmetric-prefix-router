# native023 共享入口开发指南

本宿主为预装 vLLM 0.23.0+empty、vLLM-Ascend 0.23.0，来源修订分别为 `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` 和 `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`。面向 Qwen3.5-35B-A3B BF16、文本模式，单机四张 910B2，TP2 × 内部 DP2。运行说明适用于单租户回环实验服务。r10 的准入、驻留状态和外部均衡语义另见[r10 接入条件](r10-qualification.md)。

## 运行

在已有上列版本的服务器上创建能读取系统包的独立环境，从项目根目录安装 wheel 或源码。开发环境必须提供模型匹配的分词器、msgspec、pyzmq、aiohttp 和同一版本的 vLLM。保持推理栈由服务器管理。

```sh
python -m venv --system-site-packages .venv-native023
.venv-native023/bin/python -m pip install '.[native023]'
export PYTHONHASHSEED=42
```

在该专用部署中，先在一个终端启动回环网关，模型路径换为服务器上与后端相同的快照。网关与引擎必须使用相同的哈希种子、算法、分词器和块哈希表示。

```sh
.venv-native023/bin/python -m lmetric_prefix_router.native023 \
  --backend http://127.0.0.1:18200 --event-port 18300 --port 18201 \
  --model-path /models/Qwen3.5-35B-A3B --served-model-name lab-model --policy combined
```

在另一个终端用服务器自带 Python 启动共享入口。先根据平台分配校验物理卡与逻辑卡映射，下面命令假定四张卡均已分配给当前容器。显式固定 hash 算法。仅在隔离功能验收环境设置 `VLLM_SERVER_DEV_MODE=1` 以允许操作员运行清缓存探针。

```sh
export PYTHONHASHSEED=42
vllm serve /models/Qwen3.5-35B-A3B --served-model-name lab-model \
  --host 127.0.0.1 --port 18200 --data-parallel-size 2 --tensor-parallel-size 2 \
  --dtype bfloat16 --language-model-only --max-model-len 4096 --max-num-seqs 32 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --enable-chunked-prefill \
  --enable-prompt-tokens-details --kv-events-config \
  '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://127.0.0.1:18300","replay_endpoint":"tcp://127.0.0.1:18400","buffer_steps":100000}'
```

`GET /status` 在代理端展示 rank 视图、负载新鲜度、在途数和预测比对计数。引擎就绪由引擎端的 `/health` 核对。开发网关从不主动清缓存。受控实验在发布者连接后由操作员同步已知空起点，以下命令会重置整个测试引擎的缓存。

```sh
python experiments/probe_native023.py --engine http://127.0.0.1:18200 \
  --router http://127.0.0.1:18201 --model lab-model --reset-isolated-engine
```

## 路由、回执及可靠性

- 当前正向定向覆盖普通单条字符串和 token ID completions。其他参数组合、带 salt、批量或 chat 交给同一引擎原生分配。响应头 X-Lmetric-Route 区分定向 rank 与原生委托，原生由哪个 rank 接收须另取引擎证据。客户端传入的 rank 头不参与派发。
- 对每个 rank 维护一个全注意力组、三个递归状态组和每组引用计数。块长固定 2048。全注意力链前缀需连续，三个递归组需在同一终点同时可用；最多复用至最后一个输入 token 之前。新模型、新块长、LoRA、多租户身份需要重新验证。
- 观察到清空事件后才信任该 rank。断序立即撤销，后续明确的清空事件可以重新同步。断连或无效事件关闭该 rank 的本次订阅，网关退回原生。恢复路径是修复来源、重新启动网关、重新同步。持续无流量时无法凭事件序号检测尾部丢失；自动重放和跨代身份验证属于后续验收项。
- 在共享连接池准入后读取最新完整视图，计算决策再发送一次。按 rank 的 running/waiting 异步采样。在未知目的地的委托请求存续期间，按副本均摊其在途估计；首段可见流式输出后释放待 prefill 工作量。非流式请求直到终态释放。未识别输入只计请求数。成功响应中只有预测与实际 cached_tokens 均明确时计入一致/不一致计数，其他单列 unpaired。
- `--policy native` 全部省略 rank；`--policy prefix` 对完整可信视图按最长前缀选点、平局轮换。比较时一次运行一个代理，同一负载、相同清空和预热步骤，另列引擎直连作代理开销对照。
- 首版手动启动独立服务。回滚时把新流量切回共享引擎入口，等代理在途数归零后关闭代理。Manager 的发现、独占前门和启用资格独立审查，当前描述符保留发现模式。

事件格式和负载采样依据旧适配 b63ffe2。连接池和有界回执解析来自组织 Prefix Router，沿用 Apache-2.0；实验运行器和旧策略留在原仓。[组织实现](https://github.com/vLLM-HUST/vllm-hust-prefix-router/tree/bcc9691)

## 组件复用与交付状态

本分支提供可手动运行的外部网关，Manager 入口目前只支持发现。仓库维护者可以按下表核对组件接线和后续发布条件。

| 组件 | 当前实现与改变 | 待验证项 |
|---|---|---|
| 决策与在途记账 | 抽取 combined 纯函数。将连接池等待移到选点之前，在实际发送前同步选点和预留，在流结束、失败或取消时归还。 | 目标拓扑上测同步开销和在途估计误差 |
| 请求指纹与缓存索引 | 使用部署环境的 tokenizer 和原生块哈希。按 rank 计数，验全注意力的连续链和三个 Mamba 组共同终点。 | chat 模板、额外模型或身份维度需独立核验 |
| 负载采样、HTTP 与回执 | 从组织路由器移植 rank 采样、连接池和有界 JSON/SSE 观察器。共享入口使用一个连接池，负载异步采样，预测和实际同时可见才计入比对。 | 分段计时工具已有实现；主服务尚未输出完整的分词、准入、等待与返回耗时 |
| 缓存失信与恢复 | 断序撤销索引。断连撤销对应 rank 的信任并退出订阅，整个池回退原生。 | 发布者身份、跨代同步、自动重订阅和可靠重放尚待接入 |
| 管理器与服务生命周期 | 模板提供 manifest、唯一 bundle ID、部署级前门独占声明；操作员负责启动、排空和切流。 | Provider 需绑定运行配置摘要、运行包身份、后端健康和事件就绪，再验证 configure/check/enable/disable |

产品仓保留实现、配置说明、回归测试和可重跑的功能探针。请求级原始记录、模型文件、旧策略成绩和评测数据集由实验环境保管。正式提交需要追溯结果时，用候选 revision、wheel SHA 和证据索引关联受控存储。开发态探针位于源码分发包，运行 wheel 仅包含运行模块、manifest、许可证与标准打包元数据。

组织已有的外部服务示例将启用意图、运行状态核验和部署切流分开，可参照其 Provider 绑定与回滚次序。[Prefix Router 的 Manager 接入文档](https://github.com/vLLM-HUST/vllm-hust-prefix-router/blob/main/docs/extension-manager.md)
