from app import AuditChain
c=AuditChain('audit.jsonl'); c.append('tenant-a',{'action':'create','id':1}); c.append('tenant-a',{'action':'update','id':1}); print(c.verify('tenant-a'))
