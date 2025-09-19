export async function saveGroundTruth(
  doc_id: string,
  key: string,
  value_text: string,
  rect: { page:number; x0:number; y0:number; x1:number; y1:number },
  models_root?: string
) {
  const r = await fetch(`${API}/lasso/groundtruth/save`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ doc_id, key, value_text, rect, models_root }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

<button
  disabled={!docId || !focusedKey || !rect}
  onClick={async () => {
    try {
      const row = rows.find(r => r.key === focusedKey);
      if (!row || !rect) return;
      const resp = await saveGroundTruth(
        docId,
        focusedKey,
        row.value || "",
        rect
      );
      // show IoUs in a quick alert, or set state to render a table
      const e = resp.eval || {};
      alert(`IoU — fuzzy:${e.fuzzy?.iou} tfidf:${e.tfidf?.iou} minilm:${e.minilm?.iou} distil:${e.distilbert?.iou} layout:${e.layoutlmv3?.iou}`);
      // you can also refresh overlays from resp.methods for instant redraw
      // ...
    } catch (err:any) {
      alert("Save GT failed: " + String(err));
    }
  }}
>
  Save GT
</button>

