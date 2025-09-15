// add below EditRect type
export type OverlayRect = {
  page: number;
  x0: number; y0: number; x1: number; y1: number;
  color: string;           // e.g. "#1f7ae0" (autolocate), "#7a1fe0" (bert), "#e07a1f" (tfidf)
  alpha?: number;          // fill opacity (default 0.20)
  dashed?: boolean;        // dashed outline
  label?: string;          // optional small tag above the box
};

//
type Props = {
  // ...existing props...
  zoom?: number;

  /** Optional: colored method overlays (autolocate/bert/tfidf) */
  overlays?: OverlayRect[];            // <-- NEW
};
//
// --- method overlays (autolocate/bert/tfidf) ---
if (Array.isArray(overlays) && overlays.length) {
  for (const o of overlays) {
    if (o.page !== page) continue;
    const d = document.createElement("div");
    d.className = "ov";
    // background & border
    const a = o.alpha ?? 0.20;
    d.style.background = hexToRgba(o.color, a);
    d.style.border = `2px solid ${o.color}`;
    if (o.dashed) d.style.borderStyle = "dashed";
    placeCss(d, o.x0, o.y0, o.x1, o.y1);
    overlay.appendChild(d);

    if (o.label) {
      const tag = document.createElement("div");
      tag.className = "ov-label";
      tag.textContent = o.label;
      // place small label near top-left (in overlay css px)
      const R = overlay.getBoundingClientRect();
      const sx = R.width / serverW, sy = R.height / serverH;
      tag.style.left = `${Math.min(o.x0, o.x1) * sx + 4}px`;
      tag.style.top  = `${Math.min(o.y0, o.y1) * sy - 16}px`;
      tag.style.color = o.color;
      overlay.appendChild(tag);
    }
  }
}

//
function hexToRgba(hex: string, alpha = 0.2) {
  const m = hex.replace("#", "");
  const n = parseInt(m.length === 3 ? m.split("").map(c => c + c).join("") : m, 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

//
.overlay .ov {
  position: absolute;
  box-shadow: 0 0 0 1px rgba(0,0,0,0.04) inset;
  pointer-events: none;
}
.overlay .ov-label {
  position: absolute;
  font: 600 12px/1 system-ui, sans-serif;
  padding: 2px 4px;
  background: rgba(255,255,255,0.8);
  border-radius: 3px;
  pointer-events: none;
  user-select: none;
}

//

useEffect(() => {
  drawOverlay();
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [tokens, rect, showTokenBoxes, page, serverW, serverH, overlays]); // <-- overlays added

