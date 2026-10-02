# Audit Chain

A dependency-free Python reference implementation for security, audit, transaction-graph.

Run with: python3 demo.py
Tests: python3 -m unittest discover -s tests -v

## Scope

实现按租户分区的追加审计日志：每条记录包含租户、顺序号、事件和前一条摘要，读取方可以离线校验完整历史并定位第一个损坏的位置。追加必须拒绝错误的前序状态，校验结果要明确区分缺失、顺序错误和摘要错误；日志格式保持可迁移，不依赖网络服务或远端锚定。

租户身份按稳定的 JSON 数据身份划分（`json.dumps(..., sort_keys=True)`）：对象键顺序不影响身份，而 `1`、`1.0`、`true`、`"1"` 是四个互不相干的租户；append、verify、verify_all 共用同一套分区语义。

## 标准 JSON 边界

`append(tenant, event)` 的两个参数只能是标准 JSON 值：`null`、布尔值、有限整数/浮点数、字符串、数组，以及键为字符串的对象（嵌套值同样递归受限）。`NaN`、`Infinity`、`-Infinity`、非字符串对象键以及无法无歧义编码为标准 JSON 的值（元组、字节串、集合等）一律抛 `ValueError`，且校验先于任何文件操作：文件不存在、为空或已有内容时都不会写入任何字节。

读取历史按同一规则解释每个物理行：重复对象键、`NaN`、`Infinity`、`-Infinity`（含 `1e309` 这类溢出为无穷的写法）或其他非标准 JSON 内容都按 `missing` 类损坏处理，停在首次出现的物理行；`verify` 返回 `ok/at/reason`（`at` 为物理行号），`verify_all` 返回 `ok/at/tenant/reason`，无法可靠取得租户时 `tenant` 为 `None`。字段缺失、顺序号不符、摘要不符仍分别归类为 `missing`/`sequence`/`digest`，首个错误定位规则不变。成功追加只写标准 JSONL，不重写历史；验证始终只读。
