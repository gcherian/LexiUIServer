import React, { useEffect, useRef } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy } from "pdfjs-dist";
GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.js", import.meta.url).toString();

export type EditRect = { page:number; x0:number; y0:number; x1:number; y1:number };
type Token = { page:number; x0:number; y0:number; x1:number; y1:number; text?:string };

type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;

  tokens: Token[];              // token boxes for snapping (this page only)
  rect: EditRect | null;        // current pink rect (server coords)
  showTokenBoxes?: boolean;     // yellow word boxes toggle
  editable?: boolean;           // allow drag/resize

  onRectChange?: (r: EditRect|null) => void;        // while dragging
  onRectCommit?: (r: EditRect) => void;             // on mouse up
};

export default function PdfEditCanvas({
  docUrl, page, serverW, serverH,
  tokens,
  rect,
  showTokenBoxes = true,
  editable = true,
  onRectChange, onRectCommit
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement|null>(null);
  const overlayRef = useRef<SVGSVGElement|null>(null);
  const pdfRef = useRef<PDFDocumentProxy|null>(null);
  const scaleRef = useRef<{sx:number; sy:number}>({sx:1, sy:1});
  const vpH = useRef<number>(1);
  const dragging = useRef<{mode:"move"|"nw"|"ne"|"sw"|"se"|"n"|"s"|"e"|"w"|null, startX:number, startY:number, startRect:EditRect|null}>({mode:null,startX:0,startY:0,startRect:null});

  // load & render
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

  useEffect(() => { render(); /* re-render tokens/rect */ }, [page, tokens, rect, showTokenBoxes, serverW, serverH]);

  async function render() {
    if (!pdfRef.current || !canvasRef.current || !overlayRef.current) return;
    const pdf = pdfRef.current;
    const pg = await pdf.getPage(page);
    const viewport = pg.getViewport({ scale: 1.5 });
    const cvs = canvasRef.current, ctx = cvs.getContext("2d")!;
    cvs.width = viewport.width; cvs.height = viewport.height;
    vpH.current = viewport.height;
    await pg.render({ canvasContext: ctx, viewport }).promise;

    // set scales (server px -> canvas px)
    const sx = viewport.width / Math.max(1, serverW);
    const sy = viewport.height / Math.max(1, serverH);
    scaleRef.current = { sx, sy };

    const svg = overlayRef.current;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", `0 0 ${viewport.width} ${viewport.height}`);
    svg.setAttribute("width", String(viewport.width));
    svg.setAttribute("height", String(viewport.height));

    // token boxes (yellow)
    if (showTokenBoxes) {
      const frag = document.createDocumentFragment();
      let idx = 0;
      for (const t of tokens) {
        const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        r.setAttribute("x", String(t.x0 * sx));
        r.setAttribute("y", String(t.y0 * sy));
        r.setAttribute("width", String((t.x1 - t.x0) * sx));
        r.setAttribute("height", String((t.y1 - t.y0) * sy));
        r.setAttribute("class", "bbox-rect"); // existing yellow style
        frag.appendChild(r);

        // small index tag (optional)
        const tag = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        tag.setAttribute("x", String(Math.max(t.x0 * sx - 1, 0)));
        tag.setAttribute("y", String(Math.max(t.y0 * sy - 14, 0)));
        tag.setAttribute("width", "16"); tag.setAttribute("height", "14");
        tag.setAttribute("rx", "2"); tag.setAttribute("ry", "2");
        tag.setAttribute("class", "box-tag");
        frag.appendChild(tag);
        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", String(Math.max(t.x0 * sx - 1, 0) + 8));
        txt.setAttribute("y", String(Math.max(t.y0 * sy - 14, 0) + 8));
        txt.setAttribute("class", "box-tag-text"); txt.textContent = String(++idx);
        frag.appendChild(txt);
      }
      svg.appendChild(frag);
    }

    // pink edit rect
    if (rect && rect.page === page) drawEditRect(svg, rect);
  }

  function drawEditRect(svg: SVGSVGElement, r: EditRect) {
    const { sx, sy } = scaleRef.current;
    const gx = r.x0 * sx, gy = r.y0 * sy, gw = (r.x1 - r.x0)*sx, gh = (r.y1 - r.y0)*sy;

    // main shape
    const main = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    main.setAttribute("x", String(gx)); main.setAttribute("y", String(gy));
    main.setAttribute("width", String(gw)); main.setAttribute("height", String(gh));
    main.setAttribute("fill", "rgba(236,72,153,0.15)"); // pink-500 @ 0.15
    main.setAttribute("stroke", "rgba(236,72,153,0.95)");
    main.setAttribute("stroke-width", "2");
    svg.appendChild(main);

    // handles (8)
    const handles: Array<[string, number, number, "nw"|"ne"|"sw"|"se"|"n"|"s"|"e"|"w"]> = [
      ["nw", gx, gy, "nw"], ["ne", gx+gw, gy, "ne"], ["sw", gx, gy+gh, "sw"], ["se", gx+gw, gy+gh, "se"],
      ["n", gx+gw/2, gy, "n"], ["s", gx+gw/2, gy+gh, "s"], ["w", gx, gy+gh/2, "w"], ["e", gx+gw, gy+gh/2, "e"],
    ];
    for (const [id, cx, cy, mode] of handles) {
      const h = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      h.setAttribute("x", String(cx - 4)); h.setAttribute("y", String(cy - 4));
      h.setAttribute("width", "8"); h.setAttribute("height", "8");
      h.setAttribute("fill", "#ec4899");
      h.setAttribute("stroke", "#fff");
      h.setAttribute("stroke-width", "1.2");
      h.dataset["mode"] = mode;
      if (editable) {
        h.style.cursor = "pointer";
        h.addEventListener("mousedown", (e) => startDrag(e, mode));
      }
      svg.appendChild(h);
    }

    // drag surface
    if (editable) {
      main.style.cursor = "move";
      main.addEventListener("mousedown", (e)=> startDrag(e, "move"));
    }
  }

  // ---- dragging / snapping ----
  function svgPoint(e: React.MouseEvent | MouseEvent) {
    const svg = overlayRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = (e as MouseEvent).clientX; pt.y = (e as MouseEvent).clientY;
    const m = svg.getScreenCTM();
    const p = m ? pt.matrixTransform(m.inverse()) : ({x:0,y:0} as any);
    return { x: p.x as number, y: p.y as number };
  }

  function startDrag(e: MouseEvent | React.MouseEvent, mode: any) {
    e.stopPropagation(); e.preventDefault();
    const p = svgPoint(e as any);
    dragging.current = { mode, startX: p.x, startY: p.y, startRect: rect };
    window.addEventListener("mousemove", onDragMove);
    window.addEventListener("mouseup", onDragEnd, { once: true });
  }

  function onDragMove(e: MouseEvent) {
    const d = dragging.current; if (!d.mode || !d.startRect) return;
    const { sx, sy } = scaleRef.current;
    const p = svgPoint(e);
    const dx = (p.x - d.startX) / sx;
    const dy = (p.y - d.startY) / sy;

    let { x0, y0, x1, y1 } = d.startRect;
    const clamp = (n:number, lo:number, hi:number) => Math.min(hi, Math.max(lo, n));
    const norm = () => { if (x0>x1) [x0,x1] = [x1,x0]; if (y0>y1) [y0,y1] = [y1,y0]; };

    if (d.mode === "move") {
      x0 += dx; x1 += dx; y0 += dy; y1 += dy;
    } else {
      if (d.mode.includes("n")) y0 += dy;
      if (d.mode.includes("s")) y1 += dy;
      if (d.mode.includes("w")) x0 += dx;
      if (d.mode.includes("e")) x1 += dx;
    }

    // snap to nearest token edges (server space) within tolerance
    ({ x0, x1, y0, y1 } = snapToTokens({ x0, y0, x1, y1 }, tokens, 4));

    // clamp to page
    x0 = clamp(x0, 0, serverW-1); x1 = clamp(x1, 0, serverW-1);
    y0 = clamp(y0, 0, serverH-1); y1 = clamp(y1, 0, serverH-1);
    norm();

    onRectChange?.({ page, x0: Math.round(x0), y0: Math.round(y0), x1: Math.round(x1), y1: Math.round(y1) });
  }

  function onDragEnd() {
    window.removeEventListener("mousemove", onDragMove);
    const d = dragging.current; dragging.current = { mode:null, startX:0, startY:0, startRect:null };
    if (rect) onRectCommit?.(rect);
  }

  // initial rect click-to-create (optional simple lasso: click-drag empty area)
  function onOverlayMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    if (!editable) return;
    // if no rect yet, allow quick create by drag
    if (rect) return;
    const p = svgPoint(e);
    const { sx, sy } = scaleRef.current;
    const start = { x: p.x, y: p.y };
    const tmp = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    tmp.setAttribute("x", String(start.x));
    tmp.setAttribute("y", String(start.y));
    tmp.setAttribute("width", "0"); tmp.setAttribute("height", "0");
    tmp.setAttribute("class", "bbox-rect drawing");
    overlayRef.current!.appendChild(tmp);

    const move = (ev: MouseEvent) => {
      const q = svgPoint(ev);
      const x0 = Math.min(start.x, q.x), y0 = Math.min(start.y, q.y);
      const x1 = Math.max(start.x, q.x), y1 = Math.max(start.y, q.y);
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

      let X0 = Math.floor(x / sx), Y0 = Math.floor(y / sy);
      let X1 = Math.ceil((x+w) / sx), Y1 = Math.ceil((y+h) / sy);
      ({ x0: X0, x1: X1, y0: Y0, y1: Y1 } = snapToTokens({ x0:X0, y0:Y0, x1:X1, y1:Y1 }, tokens, 4));

      const rr: EditRect = { page, x0: clamp(X0,0,serverW-1), y0: clamp(Y0,0,serverH-1), x1: clamp(X1,0,serverW-1), y1: clamp(Y1,0,serverH-1) };
      onRectChange?.(rr); onRectCommit?.(rr);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up, { once: true });
  }

  return (
    <div className="pdf-stage">
      <canvas ref={canvasRef} />
      <svg ref={overlayRef} className="overlay" onMouseDown={onOverlayMouseDown} />
    </div>
  );
}

// snap each edge to nearest token edge within tolerance (server px)
function snapToTokens(r: {x0:number;y0:number;x1:number;y1:number}, tokens: Token[], tol=4) {
  const edgesX = tokens.flatMap(t => [t.x0, t.x1]);
  const edgesY = tokens.flatMap(t => [t.y0, t.y1]);

  const snap = (v:number, arr:number[]) => {
    let best = v, dmin = tol+1;
    for (const a of arr) { const d = Math.abs(a - v); if (d < dmin) { dmin = d; best = a; } }
    return dmin <= tol ? best : v;
  };

  return {
    x0: snap(r.x0, edgesX),
    x1: snap(r.x1, edgesX),
    y0: snap(r.y0, edgesY),
    y1: snap(r.y1, edgesY),
  };
}

function clamp(n:number, lo:number, hi:number) { return Math.min(hi, Math.max(lo, n)); }