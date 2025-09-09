import React, { useEffect, useRef } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";

// Vite-friendly worker (no hard-coded path)
GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.js", import.meta.url).toString();

export type EditRect = { page:number; x0:number; y0:number; x1:number; y1:number };
type Token = { page:number; x0:number; y0:number; x1:number; y1:number; text?:string };

type Props = {
  docUrl: string;
  page: number;

  // OCR/server pixel size for the current page (from /lasso/doc/{id}/meta)
  serverW: number;
  serverH: number;

  // word/token boxes (current page only) for snapping
  tokens: Token[];

  // current pink rect (server coords)
  rect: EditRect | null;

  // show yellow token boxes
  showTokenBoxes?: boolean;

  // allow editing
  editable?: boolean;

  // notify parent while dragging and when committing
  onRectChange?: (r: EditRect|null) => void;
  onRectCommit?: (r: EditRect) => void;
};

/* ------------------------------ utils ------------------------------ */

function clamp(n:number, lo:number, hi:number) { return Math.min(hi, Math.max(lo, n)); }

function snapToTokens(r: {x0:number;y0:number;x1:number;y1:number}, tokens: Token[], tol=4) {
  const edgesX = tokens.flatMap(t => [t.x0, t.x1]);
  const edgesY = tokens.flatMap(t => [t.y0, t.y1]);

  const snap = (v:number, arr:number[]) => {
    let best = v, dmin = tol+1;
    for (const a of arr) { const d = Math.abs(a - v); if (d < dmin) { dmin = d; best = a; } }
    return dmin <= tol ? best : v;
  };

  return { x0: snap(r.x0, edgesX), x1: snap(r.x1, edgesX), y0: snap(r.y0, edgesY), y1: snap(r.y1, edgesY) };
}

/**
 * Inverse-rotate a point from viewport CSS space (after pdf.js rotation)
 * back into unrotated OCR/server image space (width=w, height=h).
 */
function invRotatePoint(x:number, y:number, w:number, h:number, rotation:number) {
  switch ((rotation % 360 + 360) % 360) {
    case 90:  return { x: y,         y: w - x };
    case 180: return { x: w - x,     y: h - y };
    case 270: return { x: h - y,     y: x     };
    default:  return { x,            y        };
  }
}

/* ------------------------------ component ------------------------------ */

export default function PdfEditCanvas({
  docUrl, page, serverW, serverH,
  tokens, rect, showTokenBoxes = true, editable = true,
  onRectChange, onRectCommit
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement|null>(null);
  const overlayRef = useRef<SVGSVGElement|null>(null);
  const pdfRef = useRef<PDFDocumentProxy|null>(null);
  const pageRef = useRef<PDFPageProxy|null>(null);

  // viewport bookkeeping for the current rendered page
  const vpRef = useRef<{ wCss:number; hCss:number; rotation:number }>({ wCss:1, hCss:1, rotation:0 });

  // pink rect latest value (to avoid stale closure on commit)
  const rectRef = useRef<EditRect|null>(rect);
  useEffect(() => { rectRef.current = rect; }, [rect]);

  // load doc once
  useEffect(() => {
    let aborted = false;
    (async () => {
      const doc = await getDocument(docUrl).promise;
      if (aborted) return;
      pdfRef.current = doc;
      await render();
    })();
    return () => { aborted = true; };
    // eslint-disable-next-line
  }, [docUrl]);

  // repaint on changes
  useEffect(() => { render(); }, [page, tokens, rect, showTokenBoxes, serverW, serverH]);

  async function render() {
    if (!pdfRef.current || !canvasRef.current || !overlayRef.current) return;

    const pg = await pdfRef.current.getPage(page);
    pageRef.current = pg;

    // choose a scale that looks good on screen (not used in math directly)
    const viewport = pg.getViewport({ scale: 1.5 });
    vpRef.current = { wCss: viewport.width, hCss: viewport.height, rotation: viewport.rotation ?? 0 };

    // draw PDF page
    const cvs = canvasRef.current, ctx = cvs.getContext("2d")!;
    cvs.width = viewport.width; cvs.height = viewport.height;
    await pg.render({ canvasContext: ctx, viewport }).promise;

    // set up overlay SVG
    const svg = overlayRef.current;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
    svg.setAttribute("width", String(viewport.width));
    svg.setAttribute("height", String(viewport.height));

    // draw token boxes (yellow)
    if (showTokenBoxes) {
      const frag = document.createDocumentFragment();
      let idx = 0;
      for (const t of tokens) {
        const [x, y, w, h] = serverRectToCssRect(t.x0, t.y0, t.x1, t.y1);
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", String(x));
        r.setAttribute("y", String(y));
        r.setAttribute("width", String(w));
        r.setAttribute("height", String(h));
        r.setAttribute("class", "bbox-rect");
        frag.appendChild(r);

        // small index tag
        const tag = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        tag.setAttribute("x", String(Math.max(x - 1, 0)));
        tag.setAttribute("y", String(Math.max(y - 14, 0)));
        tag.setAttribute("width", "16"); tag.setAttribute("height", "14");
        tag.setAttribute("rx", "2"); tag.setAttribute("ry", "2");
        tag.setAttribute("class", "box-tag");
        frag.appendChild(tag);

        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", String(Math.max(x - 1, 0) + 8));
        txt.setAttribute("y", String(Math.max(y - 14, 0) + 8));
        txt.setAttribute("class", "box-tag-text");
        txt.textContent = String(++idx);
        frag.appendChild(txt);
      }
      svg.appendChild(frag);
    }

    // draw pink edit rect
    if (rect && rect.page === page) drawEditRect(svg, rect);
  }

  /** Convert a server/OCR rect to viewport CSS rect (x,y,w,h) using rotation-aware math */
  function serverRectToCssRect(x0:number, y0:number, x1:number, y1:number): [number, number, number, number] {
    const { wCss, hCss, rotation } = vpRef.current;
    const oW = serverW || 1, oH = serverH || 1;

    const rotated = (rotation % 180) !== 0;
    const sx = rotated ? wCss / oH : wCss / oW;
    const sy = rotated ? hCss / oW : hCss / oH;

    // forward rotate points (inverse of invRotatePoint)
    function rotatePoint(x:number, y:number) {
      switch ((rotation % 360 + 360) % 360) {
        case 90:  return { x: oW - y, y: x };
        case 180: return { x: oW - x, y: oH - y };
        case 270: return { x: y,      y: oH - x };
        default:  return { x,         y };
      }
    }

    const a = rotatePoint(x0, y0);
    const b = rotatePoint(x1, y1);
    const X0 = Math.min(a.x, b.x) * sx;
    const Y0 = Math.min(a.y, b.y) * sy;
    const X1 = Math.max(a.x, b.x) * sx;
    const Y1 = Math.max(a.y, b.y) * sy;
    return [X0, Y0, X1 - X0, Y1 - Y0];
  }

  function drawEditRect(svg: SVGSVGElement, r: EditRect) {
    const [gx, gy, gw, gh] = serverRectToCssRect(r.x0, r.y0, r.x1, r.y1);

    // semi-transparent pink rectangle
    const main = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    main.setAttribute("x", String(gx)); main.setAttribute("y", String(gy));
    main.setAttribute("width", String(gw)); main.setAttribute("height", String(gh));
    main.setAttribute("fill", "rgba(236,72,153,0.18)");
    main.setAttribute("stroke", "rgba(236,72,153,0.95)");
    main.setAttribute("stroke-width", "2");
    svg.appendChild(main);

    // 8 handles
    const handles: Array<[number, number, "nw"|"ne"|"sw"|"se"|"n"|"s"|"e"|"w"]> = [
      [gx, gy, "nw"], [gx+gw, gy, "ne"], [gx, gy+gh, "sw"], [gx+gw, gy+gh, "se"],
      [gx+gw/2, gy, "n"], [gx+gw/2, gy+gh, "s"], [gx, gy+gh/2, "w"], [gx+gw, gy+gh/2, "e"],
    ];
    for (const [cx, cy, mode] of handles) {
      const h = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      h.setAttribute("x", String(cx - 4)); h.setAttribute("y", String(cy - 4));
      h.setAttribute("width", "8"); h.setAttribute("height", "8");
      h.setAttribute("fill", "#ec4899");
      h.setAttribute("stroke", "#fff");
      h.setAttribute("stroke-width", "1.2");
      if (editable) {
        h.style.cursor = "pointer";
        h.addEventListener("mousedown", (e) => startDrag(e, mode));
      }
      svg.appendChild(h);
    }

    if (editable) {
      main.style.cursor = "move";
      main.addEventListener("mousedown", (e)=> startDrag(e, "move"));
    }
  }

  /* --------------------- Drag / Resize with correct mapping --------------------- */

  const dragState = useRef<{
    mode: "move"|"nw"|"ne"|"sw"|"se"|"n"|"s"|"e"|"w"|null,
    startCssX: number, startCssY: number,
    startRect: EditRect | null
  }>({ mode: null, startCssX: 0, startCssY: 0, startRect: null });

  function svgCssPoint(e: MouseEvent | React.MouseEvent) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = (e as MouseEvent).clientX; pt.y = (e as MouseEvent).clientY;
    const m = svg.getScreenCTM();
    const p = m ? pt.matrixTransform(m.inverse()) : ({x:0,y:0} as any);
    return { x: p.x as number, y: p.y as number };
  }

  function cssToServer(xCss:number, yCss:number) {
    const { wCss, hCss, rotation } = vpRef.current;
    const oW = serverW || 1, oH = serverH || 1;
    const rotated = (rotation % 180) !== 0;

    // scale from CSS to OCR px (inverse of serverRectToCssRect)
    const sxInv = rotated ? oH / wCss : oW / wCss;
    const syInv = rotated ? oW / hCss : oH / hCss;

    const x = xCss * sxInv;
    const y = yCss * syInv;

    // inverse rotation back into OCR space
    return invRotatePoint(x, y, oW, oH, rotation);
  }

  function startDrag(e: MouseEvent | React.MouseEvent, mode: Props["editable"] extends true ? any : any) {
    e.stopPropagation(); e.preventDefault();
    const p = svgCssPoint(e);
    dragState.current = { mode, startCssX: p.x, startCssY: p.y, startRect: rectRef.current };
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", onDragEnd, { once: true });
  }

  function onDragMove(e: MouseEvent) {
    const d = dragState.current; if (!d.mode || !d.startRect) return;
    const p = svgCssPoint(e);

    // delta in server space using css→server mapping
    const a = cssToServer(d.startCssX, d.startCssY);
    const b = cssToServer(p.x, p.y);
    const dx = b.x - a.x;
    const dy = b.y - a.y;

    let { x0, y0, x1, y1 } = d.startRect;
    const norm = () => { if (x0>x1) [x0,x1] = [x1,x0]; if (y0>y1) [y0,y1] = [y1,y0]; };

    if (d.mode === "move") {
      x0 += dx; x1 += dx; y0 += dy; y1 += dy;
    } else {
      if (d.mode.includes("n")) y0 += dy;
      if (d.mode.includes("s")) y1 += dy;
      if (d.mode.includes("w")) x0 += dx;
      if (d.mode.includes("e")) x1 += dx;
    }
    norm();

    // snap to token edges
    ({ x0, x1, y0, y1 } = snapToTokens({ x0, y0, x1, y1 }, tokens, 4));

    // clamp
    x0 = clamp(x0, 0, Math.max(1, serverW-1));
    x1 = clamp(x1, 0, Math.max(1, serverW-1));
    y0 = clamp(y0, 0, Math.max(1, serverH-1));
    y1 = clamp(y1, 0, Math.max(1, serverH-1));

    const rr: EditRect = { page, x0: Math.round(x0), y0: Math.round(y0), x1: Math.round(x1), y1: Math.round(y1) };
    rectRef.current = rr;
    onRectChange?.(rr);
  }

  function onDragEnd() {
    window.removeEventListener("mousemove", onDragMove);
    const rr = rectRef.current;
    if (rr) onRectCommit?.(rr);
    dragState.current = { mode:null, startCssX:0, startCssY:0, startRect:null };
  }

  /* --------------------- Background drag to create NEW rect --------------------- */

  function onOverlayMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!editable) return;

    // If click lands on an existing rect/handle, its own handler will run.
    // Here, we only handle dragging on the empty background to create a new rect.
    const svg = overlayRef.current!;
    const startCss = svgCssPoint(e);

    // temp draw rect in CSS space
    const tmp = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    tmp.setAttribute("x", String(startCss.x));
    tmp.setAttribute("y", String(startCss.y));
    tmp.setAttribute("width", "0"); tmp.setAttribute("height", "0");
    tmp.setAttribute("fill", "rgba(0,0,0,.06)");
    tmp.setAttribute("stroke", "rgba(0,0,0,.5)");
    tmp.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(tmp);

    const move = (ev: MouseEvent) => {
      const q = svgCssPoint(ev);
      const x0 = Math.min(startCss.x, q.x), y0 = Math.min(startCss.y, q.y);
      const x1 = Math.max(startCss.x, q.x), y1 = Math.max(startCss.y, q.y);
      tmp.setAttribute("x", String(x0)); tmp.setAttribute("y", String(y0));
      tmp.setAttribute("width", String(x1-x0)); tmp.setAttribute("height", String(y1-y0));
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      const x = parseFloat(tmp.getAttribute("x")||"0");
      const y = parseFloat(tmp.getAttribute("y")||"0");
      const w = parseFloat(tmp.getAttribute("width")||"0");
      const h = parseFloat(tmp.getAttribute("height")||"0");
      tmp.remove();

      // Convert corners from CSS → server with full inverse-rotation
      const a = cssToServer(x,       y);
      const b = cssToServer(x + w,   y + h);

      let X0 = Math.min(a.x, b.x), Y0 = Math.min(a.y, b.y);
      let X1 = Math.max(a.x, b.x), Y1 = Math.max(a.y, b.y);

      // snap and clamp
      ({ x0: X0, x1: X1, y0: Y0, y1: Y1 } = snapToTokens({ x0:X0, y0:Y0, x1:X1, y1:Y1 }, tokens, 4));
      X0 = clamp(X0, 0, Math.max(1, serverW-1));
      X1 = clamp(X1, 0, Math.max(1, serverW-1));
      Y0 = clamp(Y0, 0, Math.max(1, serverH-1));
      Y1 = clamp(Y1, 0, Math.max(1, serverH-1));

      if (X1 - X0 < 2 || Y1 - Y0 < 2) return;

      const rr: EditRect = { page, x0: Math.round(X0), y0: Math.round(Y0), x1: Math.round(X1), y1: Math.round(Y1) };
      rectRef.current = rr;
      onRectChange?.(rr);
      onRectCommit?.(rr);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up, { once: true });
  }

  /* --------------------- render --------------------- */

  return (
    <div className="pdf-stage">
      <canvas ref={canvasRef} />
      <svg ref={overlayRef} className="overlay" onMouseDown={onOverlayMouseDown} />
    </div>
  );
}