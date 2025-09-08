import { useEffect, useMemo, useState } from "react";
import {
  API,
  uploadPdf,
  getMetaByDocId,
  docUrlFromId,
  guessDocIdFromUrl,
  listPromDoctypes,
  getPromCatalog,
  setDoctype,
  listFields,
  saveFieldState,
  tokenSearch,
  type FieldState,
  type PromCatalog,
  type MetaResp,
} from "../../lib/api";

type Status = "idle" | "loading" | "ready" | "error";

export default function OcrWorkbench() {
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);

  const [docId, setDocId] = useState<string>("");
  const [docUrl, setDocUrl] = useState<string>("");
  const [meta, setMeta] = useState<MetaResp | null>(null);

  const [proms, setProms] = useState<Array<{ doctype: string; file: string }>>([]);
  const [catalog, setCatalog] = useState<PromCatalog | null>(null);
  const [doctype, setDoctypeLocal] = useState<string>("");

  const [fields, setFields] = useState<FieldState[]>([]);
  const [searchQ, setSearchQ] = useState<string>("invoice");
  const [searchResults, setSearchResults] = useState<Array<{ page: number; score: number }>>([]);

  const resolvedDocId = useMemo(() => docId || guessDocIdFromUrl(docUrl) || "", [docId, docUrl]);
  const canOperate = !!resolvedDocId;

  useEffect(() => {
    (async () => {
      try {
        const ds = await listPromDoctypes();
        setProms(ds);
      } catch {}
    })();
  }, []);

  useEffect(() => {
    if (!docUrl) return;
    const maybe = guessDocIdFromUrl(docUrl);
    if (maybe && !docId) setDocId(maybe);
  }, [docUrl]); // eslint-disable-line

  useEffect(() => {
    (async () => {
      if (!resolvedDocId) return;
      setStatus("loading"); setError(null);
      try {
        const m = await getMetaByDocId(resolvedDocId);
        setMeta(m);
        const url = docUrlFromId(resolvedDocId);
        setDocUrl(url);
        const fs = await listFields({ doc_url: url });
        setFields(fs || []);
        setStatus("ready");
      } catch (e: any) {
        setStatus("error"); setError(String(e?.message || e));
      }
    })();
  }, [resolvedDocId]); // eslint-disable-line

  async function onUpload(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f) return;
    setStatus("loading"); setError(null);
    try {
      const res = await uploadPdf(f);
      setDocId(res.doc_id);
      setDocUrl(res.annotated_tokens_url);
      const m = await getMetaByDocId(res.doc_id);
      setMeta(m);
      const fs = await listFields({ doc_url: res.annotated_tokens_url });
      setFields(fs || []);
      setStatus("ready");
    } catch (e: any) {
      setStatus("error"); setError(String(e?.message || e));
    } finally {
      (ev.target as HTMLInputElement).value = "";
    }
  }

  async function onChooseDoctype(dt: string) {
    try {
      setDoctypeLocal(dt);
      const cat = await getPromCatalog(dt);
      setCatalog(cat);
      if (resolvedDocId) await setDoctype(resolvedDocId, dt);
    } catch {}
  }

  async function onSaveField(idx: number) {
    if (!docUrl) return;
    const f = fields[idx]; if (!f) return;
    const payload: FieldState = { ...f };
    if (!payload.key && payload.name) payload.key = payload.name;
    await saveFieldState({ doc_url: docUrl, field: payload });
  }

  async function onSearch() {
    if (!resolvedDocId || !searchQ) { setSearchResults([]); return; }
    try {
      const hits = await tokenSearch(resolvedDocId, searchQ, 30);
      const agg: Record<number, number> = {};
      for (let i = 0; i < hits.length; i++) {
        const h = hits[i]; const prev = agg[h.page] ?? 0;
        agg[h.page] = Math.max(prev, h.score);
      }
      const arr = Object.keys(agg).map(k => ({ page: Number(k), score: agg[Number(k)] }));
      arr.sort((a, b) => b.score - a.score);
      setSearchResults(arr);
    } catch { setSearchResults([]); }
  }

  const pages = meta?.pages?.length ?? 0;

  return (
    <div className="workbench" style={{ padding: 12 }}>
      <div className="wb-toolbar" style={{ gap: 8, display: "flex", alignItems: "center" }}>
        <input type="file" accept="application/pdf" onChange={onUpload} />
        <input
          className="input"
          placeholder="Paste /data/{doc_id}/original.pdf or any PDF URL"
          value={docUrl}
          onChange={(e) => setDocUrl(e.target.value)}
          style={{ width: 420 }}
        />
        <span style={{ marginLeft: "auto" }}>
          API: <code>{API}</code>
        </span>
      </div>

      {status === "error" && <div style={{ color: "crimson" }}>Error: {error}</div>}

      <div className="wb-split" style={{ display: "grid", gridTemplateColumns: "1fr 420px", gap: 12, height: "calc(100% - 44px)" }}>
        {/* LEFT */}
        <div className="wb-left" style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
            <strong>Document</strong>
            {resolvedDocId && <span>(id: {resolvedDocId})</span>}
            {pages ? <span style={{ marginLeft: "auto" }}>{pages} pages</span> : null}
          </div>
          {docUrl ? (
            <iframe title="pdf" src={docUrl} style={{ width: "100%", height: 420, border: "1px solid #e5e7eb", borderRadius: 6 }} />
          ) : (
            <div style={{ color: "#6b7280", fontStyle: "italic" }}>Upload or paste a PDF URL to begin.</div>
          )}

          <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
            <input
              placeholder="Search tokens…"
              value={searchQ}
              onChange={(e) => setSearchQ(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSearch()}
            />
            <button onClick={onSearch}>Search</button>
            <a href="/bbox_workbench" style={{ marginLeft: "auto" }}>Open BBox Workbench →</a>
          </div>
          {searchResults.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Top pages:</div>
              <ul>
                {searchResults.map((r) => (
                  <li key={r.page}>Page {r.page} (score {r.score.toFixed(3)})</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* RIGHT */}
        <div className="wb-right" style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 12, overflow: "auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: 8, alignItems: "center" }}>
            <label>Doctype</label>
            <select value={doctype} onChange={(e) => onChooseDoctype(e.target.value)} disabled={!canOperate}>
              <option value="">(select)</option>
              {proms.map((p) => (
                <option key={p.doctype} value={p.doctype}>{p.doctype}</option>
              ))}
            </select>

            {catalog && (
              <>
                <label>Catalog</label>
                <div style={{ fontSize: 12, color: "#6b7280" }}>{catalog.doctype} v{catalog.version}</div>
              </>
            )}
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Fields</div>
            {fields.length === 0 ? (
              <div style={{ color: "#6b7280", fontStyle: "italic" }}>No fields yet. Choose a doctype and/or add fields from the BBox Workbench.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {fields.map((f, i) => (
                  <div key={(f.key || f.name || "") + ":" + i} style={{ display: "grid", gridTemplateColumns: "140px 1fr auto", gap: 8, alignItems: "center" }}>
                    <div title={String(f.key || f.name || "")} style={{ fontFamily: "monospace" }}>{f.key || f.name}</div>
                    <input
                      value={f.value ?? ""}
                      onChange={(e) => {
                        const v = e.target.value;
                        setFields((prev) => prev.map((x, idx) => (idx === i ? { ...x, value: v } : x)));
                      }}
                    />
                    <button onClick={() => onSaveField(i)}>Save</button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {!!fields.length && (
            <div style={{ marginTop: 12, fontSize: 12, color: "#6b7280" }}>
              Tip: for **field-level** binding, open the BBox tab and lasso/click a box to bind a bbox to the field and OCR its value.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
