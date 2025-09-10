// File: src/tsp4/components/lasso/PdfEditCanvas.tsx
import React, { useEffect, useLayoutEffect, useRef } from "react";
import { GlobalWorkerOptions, getDocument, type PDFDocumentProxy, type PDFPageProxy } from "pdfjs-dist";
import "pdfjs-dist/web/pdf_viewer.css";

GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

export type TokenBox = { page: number; x0: number; y0: number; x1: number; y1: number; text?: string };
export type EditRect = { page: number; x0: number; y0: number; x1: number; y1: number };

type Props = {
  docUrl: string;
  page: number;
  serverW: number;
  serverH: number;
  tokens: TokenBox[];
  rect: EditRect | null;
  showTokenBoxes: boolean;
  editable: boolean;
  onRectChange: (r: EditRect | null) => void;
  onRectCommit: (r: EditRect) => void;
};

export default function PdfEditCanvas(props: Props) {
  const { docUrl, page, serverW, serverH, tokens, rect, showTokenBoxes, editable, onRectChange, onRectCommit } = props;

  const cRef = useRef<HTMLCanvasElement | null>(null);
  const overlay = useRef<HTMLDivElement | null>(null);

  const mode = useRef<"idle" | "lasso" | "move" | "resize">("idle");
  const start = useRef<{ x: number; y: number } | null>(null);
  const now = useRef<{ x: number; y: number } | null>(null);
  const handle = useRef<"nw"|"n"|"ne"|"e"|"se"|"s"|"sw"|"w"|null>(null);

  // Render page exactly at OCR resolution (so mapping is 1:1, but we scale with CSS automatically)
  useEffect(() => {
    let off = false;
    (async () => {
      const pdf: PDFDocumentProxy = await getDocument(docUrl).promise;
      if (off) return;
      const pg: PDFPageProxy = await pdf.getPage(page);
      if (off) return;

      const natW = pg.view[2];
      const scale = serverW / natW;
      const vp = pg.getViewport({ scale, rotation: 0 });

      const c = cRef.current!;
      const ctx = c.getContext("2d")!;
      c.width = serverW;
      c.height = serverH;

      // presentational width to keep doc stable inside container
      c.style.maxWidth = "100%";
      c.style.height = "auto";

      await pg.render({ canvasContext: ctx, viewport: vp }).promise;
      if (overlay.current) {
        overlay.current.style.width = `${c.clientWidth}px`;
        overlay.current.style.height = `${c.clientHeight}px`;
      }
      draw();
    })();
    return () => { off = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docUrl, page, serverW, serverH]);

  useLayoutEffect(() => { draw(); /* eslint-disable-next-line */ }, [tokens, rect, showTokenBoxes]);

  function R() { return overlay.current!.getBoundingClientRect(); }
  function css2ocr(xcss: number, ycss: number) {
    const r = R(); const sx = serverW / r.width; const sy = serverH / r.height;
    return { x: Math.max(0, Math.min(Math.round(xcss * sx), serverW - 1)), y: Math.max(0, Math.min(Math.round(ycss * sy), serverH - 1)) };
  }
  function ocr2css(x: number, y: number) {
    const r = R(); const sx = r.width / serverW; const sy = r.height / serverH;
    return { x: x * sx, y: y * sy };
  }
  function ocrRect2css(rr: {x0:number;y0:number;x1:number;y1:number}) {
    const a = ocr2css(rr.x0, rr.y0), b = ocr2css(rr.x1, rr.y1);
    return { left: Math.min(a.x,b.x), top: Math.min(a.y,b.y), width: Math.abs(b.x-a.x), height: Math.abs(b.y-a.y) };
  }

  function draw() {
    const host = overlay.current; if (!host) return;
    // sync overlay size to canvas client box
    const c = cRef.current!;
    host.style.width = `${c.clientWidth}px`;
    host.style.height = `${c.clientHeight}px`;

    host.innerHTML = "";

    if (showTokenBoxes) {
      for (const t of tokens) {
        if (t.page !== page) continue;
        const d = document.createElement("div");
        d.className = "tok";
        place(d, ocrRect2css({ x0: t.x0, y0: t.y0, x1: t.x1, y1: t.y1 }));
        host.appendChild(d);
      }
    }

    if (rect && rect.page === page) {
      const d = document.createElement("div");
      d.className = "pink";
      place(d, ocrRect2css({ x0: rect.x0, y0: rect.y0, x1: rect.x1, y1: rect.y1 }));
      for (const pos of ["nw","n","ne","e","se","s","sw","w"] as const) {
        const h = document.createElement("div"); h.className = `handle ${pos}`; d.appendChild(h);
      }
      host.appendChild(d);
    }

    if (mode.current === "lasso" && start.current && now.current) {
      const r = R();
      const x0 = Math.min(start.current.x, now.current.x) - r.left;
      const y0 = Math.min(start.current.y, now.current.y) - r.top;
      const x1 = Math.max(start.current.x, now.current.x) - r.left;
      const y1 = Math.max(start.current.y, now.current.y) - r.top;
      const d = document.createElement("div");
      d.className = "pink live";
      place(d, { left: clamp(x0,0,r.width), top: clamp(y0,0,r.height), width: clamp(x1-x0,0,r.width), height: clamp(y1-y0,0,r.height) });
      host.appendChild(d);
    }
  }

  function place(node: HTMLDivElement, css: {left:number;top:number;width:number;height:number}) {
    node.style.left = `${css.left}px`;
    node.style.top = `${css.top}px`;
    node.style.width = `${css.width}px`;
    node.style.height = `${css.height}px`;
  }
  const clamp = (v:number,lo:number,hi:number)=>Math.max(lo,Math.min(v,hi));

  function hitHandle(e: React.MouseEvent) {
    if (!overlay.current || !rect || rect.page !== page) return null;
    const r = ocrRect2css(rect);
    const Rr = R();
    const x = e.clientX - Rr.left, y = e.clientY - Rr.top, pad = 8;
    const inBox = (lx:number,ty:number,w:number,h:number)=> x>=lx && x<=lx+w && y>=ty && y<=ty+h;
    const hmap = {
      nw: {x:r.left-pad, y:r.top-pad}, n:{x:r.left+r.width/2-pad, y:r.top-pad},
      ne:{x:r.left+r.width-pad, y:r.top-pad}, e:{x:r.left+r.width-pad, y:r.top+r.height/2-pad},
      se:{x:r.left+r.width-pad, y:r.top+r.height-pad}, s:{x:r.left+r.width/2-pad, y:r.top+r.height-pad},
      sw:{x:r.left-pad, y:r.top+r.height-pad}, w:{x:r.left-pad, y:r.top+r.height/2-pad}
    } as const;
    for (const k of Object.keys(hmap) as (keyof typeof hmap)[]) {
      const p = hmap[k]; if (inBox(p.x,p.y,pad*2,pad*2)) return k;
    }
    if (inBox(r.left, r.top, r.width, r.height)) return "e"; // sentinel for move
    return null;
  }

  function onMouseDown(e: React.MouseEvent) {
    if (!editable || !overlay.current) return;
    e.preventDefault();
    const h = hitHandle(e);
    if (h) {
      if (h === "e") { mode.current = "move"; start.current = { x: e.clientX, y: e.clientY }; }
      else { mode.current = "resize"; handle.current = h; start.current = { x: e.clientX, y: e.clientY }; }
    } else {
      mode.current = "lasso"; start.current = { x: e.clientX, y: e.clientY }; now.current = { x: e.clientX, y: e.clientY };
    }
    draw();
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!editable || !start.current) return;
    if (mode.current === "lasso") { now.current = { x: e.clientX, y: e.clientY }; draw(); return; }
    const Rr = R();
    if (mode.current === "move" && rect) {
      const dx = e.clientX - start.current.x, dy = e.clientY - start.current.y;
      const nx0 = css2ocr(ocrRect2css(rect).left + dx, ocrRect2css(rect).top + dy).x;
      const ny0 = css2ocr(ocrRect2css(rect).left + dx, ocrRect2css(rect).top + dy).y;
      const w = rect.x1 - rect.x0, h = rect.y1 - rect.y0;
      const next: EditRect = { page, x0: clamp(nx0,0,serverW-w-1), y0: clamp(ny0,0,serverH-h-1), x1: clamp(nx0+w,0,serverW-1), y1: clamp(ny0+h,0,serverH-1) };
      onRectChange(next); draw(); return;
    }
    if (mode.current === "resize" && rect) {
      const css = ocrRect2css(rect);
      let x0 = css.left, y0 = css.top, x1 = css.left + css.width, y1 = css.top + css.height;
      const nx = clamp(e.clientX - Rr.left, 0, Rr.width);
      const ny = clamp(e.clientY - Rr.top, 0, Rr.height);
      switch (handle.current) {
        case "nw": x0 = nx; y0 = ny; break; case "n": y0 = ny; break; case "ne": x1 = nx; y0 = ny; break;
        case "e": x1 = nx; break; case "se": x1 = nx; y1 = ny; break; case "s": y1 = ny; break;
        case "sw": x0 = nx; y1 = ny; break; case "w": x0 = nx; break;
      }
      const a = css2ocr(x0, y0), b = css2ocr(x1, y1);
      onRectChange({ page, x0: Math.min(a.x,b.x), y0: Math.min(a.y,b.y), x1: Math.max(a.x,b.x), y1: Math.max(a.y,b.y) });
      draw(); return;
    }
  }

  function onMouseUp(e: React.MouseEvent) {
    if (!editable) return;
    const st = start.current; start.current = null; now.current = null;
    if (mode.current === "lasso" && overlay.current && st) {
      const Rr = R();
      const x0 = clamp(Math.min(st.x, e.clientX) - Rr.left, 0, Rr.width);
      const y0 = clamp(Math.min(st.y, e.clientY) - Rr.top,  0, Rr.height);
      const x1 = clamp(Math.max(st.x, e.clientX) - Rr.left, 0, Rr.width);
      const y1 = clamp(Math.max(st.y, e.clientY) - Rr.top,  0, Rr.height);
      const a = css2ocr(x0, y0), b = css2ocr(x1, y1);
      const rr: EditRect = { page, x0: Math.min(a.x,b.x), y0: Math.min(a.y,b.y), x1: Math.max(a.x,b.x), y1: Math.max(a.y,b.y) };
      onRectChange(rr); onRectCommit(rr);
    } else if ((mode.current === "move" || mode.current === "resize") && rect) {
      onRectCommit(rect);
    }
    mode.current = "idle"; draw();
  }

  return (
    <div className="pdf-stage" style={{ position: "relative", overflow: "auto" }}>
      <canvas ref={cRef} />
      <div
        ref={overlay}
        className="overlay"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        style={{ position: "absolute", inset: 0, background: "transparent", cursor: editable ? "crosshair" : "default", userSelect: "none" }}
      />
    </div>
  );
}