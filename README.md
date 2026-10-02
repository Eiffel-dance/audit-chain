# Audit Chain

A dependency-free Python reference implementation for security, audit, transaction-graph.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现按租户分区的追加审计日志：每条记录包含租户、顺序号、事件和前一条摘要，读取方可以离线校验完整历史并定位第一个损坏的位置。追加必须拒绝错误的前序状态，校验结果要明确区分缺失、顺序错误和摘要错误；日志格式保持可迁移，不依赖网络服务或远端锚定。

租户身份按稳定的 JSON 数据身份划分（`json.dumps(..., sort_keys=True)`）：对象键顺序不影响身份，而 `1`、`1.0`、`true`、`"1"` 是四个互不相干的租户；append、verify、verify_all 共用同一套分区语义。

输入与读取都遵循标准 JSON 边界：append 的 tenant 和 event 只能由 null、布尔值、有限整数/浮点数、字符串、数组和键为字符串的对象递归组成，NaN、Infinity、-Infinity、非字符串键及无法无歧义编码的值一律抛出 `ValueError`（先于任何追加动作，不写入任何字节）。verify 的 tenant 参数采用同一定义：非法值在读取历史或触碰路径之前抛出 `ValueError`（不泄露 TypeError/RecursionError 或 JSON 编码异常），已损坏的文件与非法 tenant 同时出现时同样是 `ValueError` 优先且字节不变；合法 tenant 的校验结果（missing/sequence/digest 及 expected_count 语义）保持不变。verify/verify_all 以同一规则解释历史：包含重复对象键、NaN/Infinity/-Infinity 或其他非标准 JSON 内容的物理行按 missing 类损坏处理并停在首次出现处。

同一路径可被多个线程或进程同时读写而仍保持单一可验证历史：append 在整个"读取最新状态—计算 seq/prev/hash—追加"区间持有排他文件锁（`flock`，无第三方依赖），因此每条记录都基于其真正落盘前的最新完整状态计算，任何调用都不会覆盖既有字节；verify/verify_all 持共享锁读取，只会看到某次追加之前或之后完整自洽的文件视图，绝不会把写了一半的记录报成损坏。历史一旦损坏，并发 append 仍以相同的 tenant/seq/line/reason（missing、sequence、digest 之一）抛出 `AuditChainStateError` 且不改动文件。旧版 JSONL 文件格式不变，可继续直接验证。

`append_batch(tenant, events)` 一次提交一组同租户事件：events 必须是列表（空列表直接返回 `[]`，不创建文件、不改动字节；其他类型一律 `ValueError`），租户与列表中每个事件沿用与 append 相同的标准 JSON 边界。成功时按输入顺序分配连续 seq——首条接在该租户当前链末（prev 为提交前最后一条记录的摘要），后续逐条引用批次内前一条摘要——并在同一次排他租约内以一个字节块整体追加，因此整批对其他追加/校验是一个不可分割的可见结果：同租户追加只能整体落在批前或批后，不同租户仍可交错且各自从 1 计数。落盘为每行一个 JSON 对象，与逐条调用 append 产生的字节和 verify/verify_all 计数、摘要、首个错误位置完全一致；扫描到与 append 相同的首个损坏点时抛出同样字段（tenant、seq、reason、line）的 `AuditChainStateError`，失败不留任何部分记录。

`export_tenant(tenant)` 面向离线迁移，按租户导出单链：返回 UTF-8 编码的 `bytes`，本身不创建、不改写、不删除任何文件，也不访问网络；调用方自行把返回字节写入新的 JSONL 路径后，新的 `AuditChain` 可直接用现有 verify/verify_all 离线验证。tenant 采用与 append/verify 完全相同的标准 JSON 边界，非法值在读取任何历史之前统一抛 `ValueError` 且不读取、不创建文件。导出在单次只读共享锁快照中完成：先运行与 `verify(tenant)` 相同的完整扫描，整份快照对该租户有效后才开始拼装结果，因此源状态错误统一抛 `AuditChainStateError`（填入目标 tenant、首个应检查的 seq、reason 与物理行号），绝不会先返回已收集的前缀；reason 仅为 missing、sequence、digest，优先级与行号沿用现有校验规则。源文件不存在或为空是合法空快照，成功返回 `b""`。导出内容只包含目标租户在源文件中按物理顺序出现的记录（其他租户的交错记录被丢弃且不导致重排），每行一个 JSON 对象，沿用 append 的序列化（`sort_keys=True`、`allow_nan=False`）与换行规则，字段、值、seq、prev、hash 与源历史语义一致；JSON 身份不同的租户（如 `1`、`1.0`、`true`、`"1"`）各自独立导出、互不混入。快照未变更时重复调用得到字节完全相同的结果；并发追加期间，导出只能观察到某次追加之前或之后的完整快照，失败时不写入任何字节。
