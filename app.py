import hashlib, json
from pathlib import Path
class AuditChain:
    def __init__(self,path): self.path=Path(path)
    def _rows(self): return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines()] if self.path.exists() else []
    @staticmethod
    def _hash(item): return hashlib.sha256(json.dumps({k:item[k] for k in ("tenant","seq","event","prev")},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    def append(self,tenant,event):
        own=[r for r in self._rows() if r["tenant"]==tenant]; item={"tenant":tenant,"seq":len(own)+1,"event":event,"prev":own[-1]["hash"] if own else "0"*64}; item["hash"]=self._hash(item)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a",encoding="utf-8") as f: f.write(json.dumps(item,sort_keys=True)+"\n")
        return item
    def verify(self,tenant):
        expected=1; prev="0"*64
        for item in [r for r in self._rows() if r["tenant"]==tenant]:
            if item["seq"]!=expected: return {"ok":False,"at":expected,"reason":"sequence"}
            if item["prev"]!=prev or item["hash"]!=self._hash(item): return {"ok":False,"at":expected,"reason":"digest"}
            expected+=1; prev=item["hash"]
        return {"ok":True,"count":expected-1}


