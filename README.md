# Audit Chain

A dependency-free Python reference implementation for security, audit, transaction-graph.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现按租户分区的追加审计日志：每条记录包含租户、顺序号、事件和前一条摘要，读取方可以离线校验完整历史并定位第一个损坏的位置。追加必须拒绝错误的前序状态，校验结果要明确区分缺失、顺序错误和摘要错误；日志格式保持可迁移，不依赖网络服务或远端锚定。

租户身份按稳定的 JSON 数据身份划分（`json.dumps(..., sort_keys=True)`）：对象键顺序不影响身份，而 `1`、`1.0`、`true`、`"1"` 是四个互不相干的租户；append、verify、verify_all 共用同一套分区语义。

输入与读取都遵循标准 JSON 边界：append 的 tenant 和 event 只能由 null、布尔值、有限整数/浮点数、字符串、数组和键为字符串的对象递归组成，NaN、Infinity、-Infinity、非字符串键及无法无歧义编码的值一律抛出 `ValueError`（先于任何追加动作，不写入任何字节）。verify/verify_all 以同一规则解释历史：包含重复对象键、NaN/Infinity/-Infinity 或其他非标准 JSON 内容的物理行按 missing 类损坏处理并停在首次出现处。

## 并发一致性

同一 JSONL 文件可被多个线程或进程同时追加与读取。所有 append/verify/verify_all 通过日志文件自身文件描述符上的建议锁（`flock`，无 sidecar 文件、无网络服务）协调：append 在排他锁内完成"读取—校验—追加"全过程，seq、prev、hash 始终基于记录真正落盘前的最新完整状态计算，成功结果与文件记录一一对应；`O_APPEND` 保证已存在字节永不被覆盖或丢弃。取得写入资格后若历史不再符合链，append 抛出 `AuditChainStateError` 且文件保持原样。verify/verify_all 在共享锁下读取快照，结果只对应追加前或追加完成后的合法历史，不会把半条记录报告为 missing/sequence/digest。
