// D) Server 5-method overlays (robust to any single model failing)
(async () => {
  try {
    if (!docId) return;
    const res = await matchField(docId, r.key, r.value);

    // Server returns { hits: { autolocate, tfidf, minilm, distilbert, layoutlmv3 } }
    const hits = (res as any)?.hits || {};

    // Map server keys -> UI labels/colors
    const serverToUi = [
      ["autolocate",  "fuzzy"],
      ["tfidf",       "tfidf"],
      ["minilm",      "minilm"],
      ["distilbert",  "distilbert"],
      ["layoutlmv3",  "layoutlmv3"],
    ] as const;

    const pick = (m: any) =>
      !m ? null : ({ page: m.page, x0: m.rect.x0, y0: m.rect.y0, x1: m.rect.x1, y1: m.rect.y1 });

    const ovs = serverToUi.map(([srv, ui]) => ({
      label: ui,
      color: COLORS[ui],
      rect: pick(hits[srv]),
    }));

    // If main rect is empty, jump to the first page that has any overlay
    if (!rect) {
      const first = ovs.find(o => o.rect);
      if (first?.rect?.page) setPage(first.rect.page);
    }

    setOverlays(ovs);
  } catch (e) {
    console.warn("matchField failed", e);
    setOverlays([]); // still show pink if any
  }
})();


//
// Add near the other calls
export async function matchField(
  doc_id: string,
  key: string,
  value: string,
  max_window = 12,
  models_root?: string
): Promise<any> {
  const body: any = { doc_id, key, value, max_window };
  if (models_root) body.models_root = models_root;
  const r = await fetch(`${API}/lasso/locate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json(); // { hits: {...}, pages: [...] }
}

//
# Old in each loop:
# if cnt > 1500: break  (or 1000)

# New caps (faster)
if cnt > 600: break   # autolocate/tfidf
# and for emb models:
if cnt > 400: break   # minilm / distilbert / layout proxy


max_w = max(4, min(int(req.max_window), 10))


