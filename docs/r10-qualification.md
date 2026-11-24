# r10 接入条件

目标为 vLLM-HUST Engine 2026.11.21.10 的 Qwen3.5-35B-A3B BF16。采用官方安装器和四项发布制品，核对 Python、CANN、卡型、模型配置、tokenizer、模板和物理设备映射。性能记录由实验环境单独归档。[官方分发制品](https://github.com/vLLM-HUST/vllm-hust-website/releases/tag/engine-wheels-r10-v1)

## 部署

每个服务使用独立端口和缓存目录，TP2/DP1。对外的最小路由单元为整个模型副本。组合策略与基线必须经过同一个代理，客户端不指定卡号或 rank。当前正式启动器要求恰好两个物理设备并在服务期间占用容器级模型租约。两个端点共存需要平台支持的资源准入边界。[启动器](https://github.com/DeepLinkNext/DeepLinkEngine/tree/0ff0da7ad67e632a11f1ba6e5e08c8242bdfd6b8)

## 缓存证据

Qwen35 的 LiveStateScheduler 通过 ResidentLeases.offer 检查驻留状态、写入完成、身份、持有者和预留归属。token 向量含 MTP 前瞻 token，实际可复用量取 cursor。普通注意力页和整个混合模型可恢复的状态具有不同语义。路由读端必须处理重启、失信、过期和正在变更的快照。主机恢复另行归因。

## 验证入口

- 使用 experiments/probe_resident.py 检查冷请求、精确 token 续写、异 salt 与共享文档新问题的实际 usage。
- 提供双端点后，以不对称预热和重启检验选点、代际校验和重新建账。工作负载、到达轨迹、种子和测量窗口在各臂间保持一致。
- 跟踪首字、可见输出停顿、有效吞吐、缓存回执、失败和代理在线耗时。硬件回执、运行日志和性能评估单独归档。
- Manager 描述符处于发现阶段；部署级验证完成后再审查启用契约。
