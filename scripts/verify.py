"""End-to-end verification against a running instance.

    ./library && uv run python scripts/verify.py

Exercises every surface for real: ingests a document, has it read and annotated, checks
the reader, searches with and without filters, runs chat turns on both models (thinking,
follow-up rewriting, stances, citation verification), inspects the library layer, deletes
the document and confirms nothing is orphaned. Takes a few minutes because it waits on
the model. Set LIBRARY_TESTDOCS to a folder of already-shelved files for the import check."""
import asyncio, json, os, subprocess, sys, tempfile, time, uuid
import httpx

API = "http://127.0.0.1:8077"
R = []
def ok(name, cond, detail=""):
    R.append((name, bool(cond), detail)); print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}", flush=True)

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

async def sse(c, body, timeout=300):
    ev=None; out={"tokens":0,"thinking":0,"sources":[],"meta":{},"done":{},"error":None,"conv":None}
    async with c.stream("POST", "/api/chat", json=body, timeout=timeout) as r:
        async for line in r.aiter_lines():
            if line.startswith("event:"): ev=line[6:].strip()
            elif line.startswith("data:"):
                d=line[5:].strip()
                if ev=="conversation": out["conv"]=json.loads(d)["conversation_id"]
                elif ev=="meta": out["meta"]=json.loads(d)
                elif ev=="sources": out["sources"]=json.loads(d)
                elif ev=="token": out["tokens"]+=1
                elif ev=="thinking": out["thinking"]+=1
                elif ev=="done": out["done"]=json.loads(d); break
                elif ev=="error": out["error"]=d; break
    return out

async def main():
    print("== services ==")
    ok("postgres", sh("pg_isready -q && echo y")=="y")
    ok("redis", sh("redis-cli ping")=="PONG")
    ok("ollama (tunnel)", sh("curl -sf --max-time 5 localhost:11434/api/tags >/dev/null && echo y")=="y")
    gen = sh("""curl -s --max-time 40 localhost:11434/api/generate -d '{"model":"qwen3:30b-a3b","prompt":"hi","stream":false,"think":false,"options":{"num_ctx":1024}}' | python3 -c "import sys,json;print('y' if json.load(sys.stdin).get('response') else 'n')" 2>/dev/null""")
    ok("ollama generates", gen=="y")
    ok("api", sh(f"curl -sf {API}/api/health >/dev/null && echo y")=="y")
    ok("worker", "worker running" in sh("./library status"))

    async with httpx.AsyncClient(base_url=API, timeout=120) as c:
        h = (await c.get("/api/health")).json()
        ok("health ok", h["ok"] and sum(h["orphan_vectors"].values())==0, f"{h['documents']} docs, {h['chunks']} chunks, orphans={sum(h['orphan_vectors'].values())}")
        ok("ui served", (await c.get("/")).status_code==200)

        print("== ingest → read → annotate → delete ==")
        md = "# Verification note\n\n## Reranking latency\n\nA cross-encoder scores query and passage together. On Apple Silicon it costs about sixty-five milliseconds per pair, so depth is the only latency lever.\n\n## Fusion\n\nReciprocal rank fusion needs no score calibration between dense and lexical retrievers, which matters because cosine and ts_rank_cd are not comparable.\n"
        r = await c.post("/api/documents", files={"file": ("verify-note.md", md, "text/markdown")})
        j = r.json(); did = j.get("document_id")
        ok("upload", r.status_code==200 and j["status"]=="ready", f"{j.get('sections')} sections, {j.get('chunks')} chunks, {j.get('elapsed_seconds')}s")
        r2 = await c.post("/api/documents", files={"file": ("verify-note.md", md, "text/markdown")})
        ok("dedup on re-upload", r2.json().get("status")=="duplicate")
        s = (await c.get("/api/search", params={"q":"reciprocal rank fusion calibration","limit":3})).json()
        ok("searchable immediately", any(x["document_id"]==did for x in s["hits"]), f"{s['elapsed_seconds']}s")
        bad = await c.post("/api/documents", files={"file": ("x.xyz", b"junk", "application/octet-stream")})
        ok("rejects unsupported type", bad.status_code==415)

        jr = await c.post(f"/api/documents/{did}/read", params={"tier":2})
        ok("queue read+annotate", jr.status_code==200 and jr.json()["state"]=="queued")
        t0=time.time(); state=None
        while time.time()-t0 < 420:
            jobs = (await c.get("/api/jobs", params={"limit":5})).json()
            mine = [x for x in jobs if x["document_id"]==did]
            if mine and mine[0]["state"] in ("done","error"): state=mine[0]; break
            await asyncio.sleep(5)
        ok("read+annotate completes", state and state["state"]=="done", f"{time.time()-t0:.0f}s" + (f" — {state['error'][:80]}" if state and state.get("error") else ""))
        doc = (await c.get(f"/api/documents/{did}")).json()
        ok("document is annotated (tier 2)", doc["tier"]==2)
        rd = (await c.get(f"/api/documents/{did}/read")).json()
        n_sum = sum(1 for s_ in rd["sections"] if s_["summary"]); n_ref = sum(1 for s_ in rd["sections"] for p in s_["passages"] if p["reflection"])
        ok("reader has summaries + marginalia", n_sum>=1 and n_ref>=1, f"{n_sum} summaries, {n_ref} reflections")
        ok("reader passages start after heading", not any(p["text"].lower().startswith(s_["title"].lower()) for s_ in rd["sections"] if s_["title"] for p in s_["passages"][:1]))
        ok("auto-rebuild scheduled", sh("redis-cli exists library:rebuild_scheduled")=="1", f"ttl {sh('redis-cli ttl library:rebuild_scheduled')}s")
        cats = (await c.get("/api/categories")).json()
        ok("categories assigned", any(cat["name"] in doc["categories"] for cat in cats) or bool(doc["categories"]), str(doc["categories"][:3]))

        print("== retrieval ==")
        s1 = (await c.get("/api/search", params={"q":"Hierarchical NSW","limit":3})).json()
        ok("keyword search", s1["hits"] and "nearest neighbor" in s1["hits"][0]["document_title"].lower(), f"{s1['elapsed_seconds']}s")
        s2 = (await c.get("/api/search", params={"q":"why do residual connections help","limit":3,"rerank":"true"})).json()
        ok("reranked search", s2["hits"] and "residual" in s2["hits"][0]["document_title"].lower(), f"{s2['elapsed_seconds']}s")
        llm = next((x["id"] for x in cats if x["name"]=="Large Language Models"), None)
        if llm:
            s3 = (await c.get("/api/search", params={"q":"how are passages encoded","limit":5,"categories":llm})).json()
            ok("category filter restricts", all("Dense Passage" not in x["document_title"] for x in s3["hits"]), f"{len(s3['hits'])} hits")
        s4 = (await c.get("/api/search", params={"q":"anything","limit":5,"categories":str(uuid.uuid4())})).json()
        ok("nonexistent category → 0 hits", len(s4["hits"])==0)

        print("== chat ==")
        m = (await c.get("/api/chat/models")).json()
        ok("models report thinking capability", m["thinking"].get("qwen3:30b-a3b") is True and m["thinking"].get("huihui_ai/qwen3-coder-abliterated:latest") is False)
        ok("stances exposed", len(m["stances"])>=5)
        a = await sse(c, {"message":"In two sentences, what does LoRA change about fine-tuning?","conversational":True,"model":"qwen3:30b-a3b"})
        ok("general: answers with citations", not a["error"] and a["tokens"]>0 and a["done"].get("markers_emitted",0)>0, f"{a['tokens']} tokens, {a['thinking']} thinking events, {a['done'].get('markers_resolved')}/{a['done'].get('markers_emitted')} verified")
        ok("general: thinking streamed separately", a["thinking"]>0)
        ok("general: citations verified", a["done"].get("validity",0)>=0.8, f"validity {a['done'].get('validity')}")
        b = await sse(c, {"message":"How does that compare to full fine-tuning?","conversational":True,"model":"qwen3:30b-a3b","conversation_id":a["conv"]})
        ok("follow-up rewritten", bool(b["meta"].get("rewritten")) and "lora" in (b["meta"].get("rewritten") or "").lower(), repr((b["meta"].get("rewritten") or "")[:60]))
        t = await sse(c, {"message":"In one sentence, what is HNSW?","conversational":False,"model":"huihui_ai/qwen3-coder-abliterated:latest"})
        ok("technical model: no 400, no thinking", not t["error"] and t["tokens"]>0 and t["thinking"]==0, f"{t['tokens']} tokens")
        st = await sse(c, {"message":"Is reranking worth it?","conversational":False,"model":"qwen3:30b-a3b","stance":"contrarian"})
        ok("stance carried in meta + persisted", st["meta"].get("stance")=="contrarian" and not st["error"])
        badst = await c.post("/api/chat", json={"message":"x","stance":"bogus"})
        ok("unknown stance rejected", badst.status_code==422)
        ok("lease released after chat", sh("redis-cli exists library:llm:chat_active")=="0")

        print("== library layer ==")
        cl = (await c.get("/api/library/clusters")).json()
        ok("cross-document themes", len(cl)>=5, f"{len(cl)} themes")
        ok("every theme has a summary", all(x["summary"] for x in cl))
        co = (await c.get("/api/library/contradictions")).json()
        ok("contradictions have real explanations", all(len(x["explanation"] or "")>=40 and not (x["explanation"] or "").lower().startswith(("explain","state")) for x in co), f"{len(co)} found")
        g = (await c.get("/api/library/graph")).json()
        ok("citation graph", len(g["edges"])>=5 and g["hubs"], f"{len(g['edges'])} edges")

        print("== cleanup ==")
        dl = await c.delete(f"/api/documents/{did}")
        ok("delete removes vectors", dl.status_code==200 and dl.json()["vectors_removed"]>0, f"{dl.json().get('vectors_removed')} vectors")
        h2 = (await c.get("/api/health")).json()
        ok("no orphans after delete", sum(h2["orphan_vectors"].values())==0)
        ok("document count restored", h2["documents"]==h["documents"])

    print("== extraction (the reported BERT anomaly) ==")
    txt = sh("psql -d library_agent -tAc \"select string_agg(text, ' ' order by order_index) from chunk c join document d on d.id=c.document_id where d.title like 'BERT%';\"")
    ok("no figure-label junk", not any(k in txt for k in ("Elikes","E[CLS]","EA EB","Position Embeddings E0")))
    ok("footnotes formatted [n]", "[5] The" in txt and "5The" not in txt)
    ok("no stray superscript in body", "QA and NLI. 6" not in txt and "QA and NLI." in txt)

    print("== tooling ==")
    tl = sh("uv run pytest tests/ -q 2>&1 | grep -E 'passed|failed'")
    ok("tests", "passed" in tl and "failed" not in tl, tl.split(" in ")[0])
    ok("lint", sh("uv run ruff check src/ tests/ 2>&1 | tail -1")=="All checks passed!")
    ok("ui js parses", sh("python3 -c \"import pathlib;h=pathlib.Path('web/index.html').read_text();pathlib.Path('/tmp/v.js').write_text(h.split('<script>')[1].split('</script>')[0])\" && node --check /tmp/v.js && echo y")=="y")
    imp = sh(f"./library import {os.environ.get('LIBRARY_TESTDOCS', 'tests')} --quiet 2>&1 | tail -1")
    ok("bulk import runs", "shelved" in imp, imp)
    with tempfile.TemporaryDirectory() as td:
        sh(f"./ops/backup.sh {td} >/dev/null 2>&1")
        made = sh(f"ls {td}/*/ 2>/dev/null")
        ok("backup produces dump + archive", "library_agent.dump" in made and "documents.tar.gz" in made)
    ok("screenshots present", all(os.path.exists(f"docs/{n}.jpg") for n in ("ask","reader","threads")))
    ok("README references them", all(f"docs/{n}.jpg" in open("README.md").read() for n in ("ask","reader","threads")))

    passed = sum(1 for _,c_,_ in R if c_); total = len(R)
    print(f"\n{'ALL PASS' if passed==total else 'FAILURES'}: {passed}/{total}")
    for n,c_,d in R:
        if not c_: print(f"  ✗ {n}  {d}")

asyncio.run(main())
