//FLE

// add near other types
type BBox = { page: number; x0: number; y0: number; x1: number; y1: number };
type OverlayRect = BBox & { color: string; label?: string; alpha?: number; dashed?: boolean };

// Add state for overlays (keep your existing rect for the pink edit box)
const [overlays, setOverlays] = useState<OverlayRect[]>([]);

// enable/disable drawing TF-IDF third box
const DRAW_TFIDF = false; // flip to true when you want the orange box


//

<PdfEditCanvas
  docUrl={docUrl}
  page={page}
  serverW={serverW}
  serverH={serverH}
  tokens={tokensPage}
  rect={rect}
  showTokenBoxes={showBoxes}
  editable={true}
  onRectChange={setRect}
  onRectCommit={onRectCommitted}
  zoom={zoom}
  overlays={overlays}        {/* <-- new */}
/>

//
async function onRowClick(r: FieldRow) {
  setFocusedKey(r.key);

  // ---- call your router (running wherever your API lives) ----
  // If your server is same-origin, change the URL to `${API}/match/field`
  const MATCH_API = `${API}/match/field`;

  // Build minimal OCR tokens payload for server (TokenBox -> server token shape)
  // Your TokenBox has x0,y0,x1,y1,page,text already (canvas/user-space)
  const ocrTokens = tokens.map(t => ({
    text: t.text || "",
    bbox: [t.x0, t.y0, t.x1, t.y1],
    page: t.page
  }));

  // Key: use human label derived from your KV path (same as you do for Distil)
  const label = r.key.replace(/\./g, " ").replace(/\[\d+\]/g, " ").replace(/\s+/g, " ").trim();

  let resp: {
    field: string;
    llm_value: string;
    autolocate?: { text:string; conf:number; bbox:[number,number,number,number]; page:number } | null;
    bert?:       { text:string; conf:number; bbox:[number,number,number,number]; page:number } | null;
    tfidf?:      { text:string; conf:number; bbox:[number,number,number,number]; page:number } | null;
  } | null = null;

  try {
    const rres = await fetch(MATCH_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ocr_tokens: ocrTokens,
        key: label,
        field: r.key,
        llm_value: r.value,
        use_bert: true  // toggle BERT on
      })
    });
    if (!rres.ok) throw new Error(await rres.text());
    resp = await rres.json();
  } catch (err) {
    console.warn("match/field failed; falling back to local heuristics", err);
    resp = null;
  }

  // ---- build overlays from server response (or fallbacks) ----
  const newOverlays: OverlayRect[] = [];

  const pushBox = (
    b: { bbox:[number,number,number,number], page:number } | null | undefined,
    color: string,
    label?: string
  ) => {
    if (!b || !b.bbox) return;
    const [x0,y0,x1,y1] = b.bbox;
    newOverlays.push({ page: b.page, x0, y0, x1, y1, color, label, alpha: 0.20 });
  };

  if (resp) {
    // Blue = Autolocate
    pushBox(resp.autolocate || undefined, "#1f7ae0", "autolocate");
    // Purple = BERT
    pushBox(resp.bert || undefined, "#7a1fe0", "bert");
    // Orange = TF-IDF (optional)
    if (DRAW_TFIDF) pushBox(resp.tfidf || undefined, "#e07a1f", "tfidf");
  }

  // ---- keep your existing "refine and set pink edit box" behavior ----
  // Preference order for the editable pink rect:
  //   1) BERT (if present)  2) Autolocate  3) TF-IDF  4) Distil  5) ECM rects  6) local fallback
  let chosen: { page:number; x0:number; y0:number; x1:number; y1:number } | null = null;

  const refine = (b:{page:number; bbox:[number,number,number,number]}|null|undefined) => {
    if (!b || !b.bbox) return null;
    const [x0,y0,x1,y1] = b.bbox;
    const pg = b.page;
    const refined = refineWithTokens({ page: pg, x0, y0, x1, y1 }, tokens.filter(t => t.page === pg));
    return refined;
  };

  if (resp?.bert)       chosen = refine(resp.bert);
  if (!chosen && resp?.autolocate) chosen = refine(resp.autolocate);
  if (!chosen && DRAW_TFIDF && resp?.tfidf) chosen = refine(resp.tfidf);

  // If nothing from server, fall back to your prior flow
  if (!chosen) {
    // Distil (if present & confident)
    const d = distil.find((f) => f.key === r.key && f.page != null);
    if (d && (d.confidence ?? 0) >= DISTIL_OK && d.value_union) {
      const pg = d.value_union.page!;
      chosen = refineWithTokens(
        { page: pg, x0: d.value_union.x0, y0: d.value_union.y0, x1: d.value_union.x1, y1: d.value_union.y1 },
        tokens.filter((t) => t.page === pg)
      );
    }
  }

  if (!chosen && r.rects?.length) {
    const byPg: Record<number, KVRect[]> = {};
    r.rects.forEach((b) => (byPg[b.page] = byPg[b.page] ? [...byPg[b.page], b] : [b]));
    const pg = Number(Object.keys(byPg).sort((a, b) => byPg[+b].length - byPg[+a].length)[0]);
    const same = byPg[pg];
    const uni = same.reduce(
      (acc, rr) => ({
        page: pg,
        x0: Math.min(acc.x0, rr.x0),
        y0: Math.min(acc.y0, rr.y0),
        x1: Math.max(acc.x1, rr.x1),
        y1: Math.max(acc.y1, rr.y1),
      }),
      { page: pg, x0: same[0].x0, y0: same[0].y0, x1: same[0].x1, y1: same[0].y1 }
    );
    chosen = refineWithTokens(uni, tokens.filter((t) => t.page === pg));
  }

  if (!chosen) {
    const hit = autoLocateByValue(r.value, tokens);
    if (hit) {
      chosen = refineWithTokens({ page: hit.page, ...hit.rect }, tokens.filter((t) => t.page === hit.page));
    }
  }

  // ---- apply UI state updates ----
  if (newOverlays.length) {
    // if overlays are on a different page, switch page to show them
    const firstPg = newOverlays[0].page;
    if (firstPg && firstPg !== page) setPage(firstPg);
  }
  setOverlays(newOverlays);
  if (chosen) {
    if (chosen.page !== page) setPage(chosen.page);
    setRect(chosen);
  } else {
    setRect(null);
  }
}


//PEC
// At the top with other imports/types
export type OverlayRect = {
  page: number; x0: number; y0: number; x1: number; y1: number;
  color: string; alpha?: number; dashed?: boolean; label?: string;
};

// In the component props:
type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
  tokens: TokenBox[];
  rect: EditRect | null;
  showTokenBoxes: boolean;
  editable: boolean;
  onRectChange?: (r: EditRect | null) => void;
  onRectCommit?: (r: EditRect) => void;
  zoom?: number;
  overlays?: OverlayRect[];     // <-- NEW
};

// Somewhere near your page rendering, you already compute CSS sizes; assume:
//   canvasWidthPx = serverW * zoom
//   canvasHeightPx = serverH * zoom
// Wrap the bitmap/canvas with a positioned container and add an SVG overlay:

return (
  <div className="pec-root" style={{ position: "relative", width: serverW * (zoom ?? 1), height: serverH * (zoom ?? 1) }}>
    {/* your existing page bitmap/canvas rendering here */}

    {/* overlays: draw only those for the current page */}
    {!!overlays?.length && (
      <svg
        className="pec-overlays"
        width={serverW * (zoom ?? 1)}
        height={serverH * (zoom ?? 1)}
        viewBox={`0 0 ${serverW} ${serverH}`}
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
      >
        {overlays
          .filter(o => o.page === page)
          .map((o, i) => {
            const w = Math.max(0, o.x1 - o.x0);
            const h = Math.max(0, o.y1 - o.y0);
            const a = o.alpha ?? 0.20;
            const dash = o.dashed ? "6,6" : undefined;
            return (
              <g key={`ov-${i}`}>
                <rect
                  x={o.x0}
                  y={o.y0}
                  width={w}
                  height={h}
                  fill={o.color}
                  opacity={a}
                />
                <rect
                  x={o.x0 + 0.75}
                  y={o.y0 + 0.75}
                  width={Math.max(0, w - 1.5)}
                  height={Math.max(0, h - 1.5)}
                  fill="none"
                  stroke={o.color}
                  strokeWidth={2}
                  strokeDasharray={dash}
                />
                {o.label && (
                  <text x={o.x0 + 4} y={Math.max(o.y0 - 4, 10)} fontSize={12} fill={o.color}>
                    {o.label}
                  </text>
                )}
              </g>
            );
          })}
      </svg>
    )}

    {/* your existing interactive edit rect (pink), token boxes, etc. */}
  </div>
);