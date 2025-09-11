---api

export async function rerankCandidates(docId: string, key: string, value: string) {
  const r = await fetch(`${API}/lasso/rerank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_id: docId, key, value }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json() as Promise<{ best?: { page:number; rect:{x0:number;y0:number;x1:number;y1:number}; score:number } }>;
}

---FLE

async function onRowClick(r: FieldRow) {
  setFocusedKey(r.key);

  // 1) If rects are provided by JSON, use them
  if (r.rects?.length) {
    const byPg: Record<number, KVRect[]> = {};
    r.rects.forEach(b => (byPg[b.page] = byPg[b.page] ? [...byPg[b.page], b] : [b]));
    const pg = Number(Object.keys(byPg).sort((a,b)=> byPg[+b].length - byPg[+a].length)[0]);
    const same = byPg[pg];
    const uni = same.reduce(
      (acc, rr) => ({ page: pg, x0: Math.min(acc.x0, rr.x0), y0: Math.min(acc.y0, rr.y0), x1: Math.max(acc.x1, rr.x1), y1: Math.max(acc.y1, rr.y1) }),
      { page: pg, x0: same[0].x0, y0: same[0].y0, x1: same[0].x1, y1: same[0].y1 }
    );
    setPage(pg); setRect(uni);
    return;
  }

  // 2) Fall back to classic local token match (fast)
  const hit = autoLocateByValue(r.value, tokens);
  if (hit) {
    setPage(hit.page);
    setRect({ page: hit.page, ...hit.rect });
  } else {
    setRect(null);
  }

  // 3) If we have a doc, ask the server to disambiguate w/ DistilBERT context
  if (docId && r.value && r.key) {
    try {
      const out = await rerankCandidates(docId, r.key, r.value);
      if (out?.best && out.best.score >= 0.25) {   // light guard
        setPage(out.best.page);
        setRect({ page: out.best.page, ...out.best.rect });
      }
    } catch { /* best-effort, ignore */ }
  }
}
