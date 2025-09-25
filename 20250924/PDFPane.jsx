// src/components/PdfPane.jsx
import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { GlobalWorkerOptions, getDocument } from "pdfjs-dist";
import { locateValue } from "../lib/match.js";

GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url
).toString();

function isBad(bb){
  if (!bb) return true;
  const S = 1e9;
  const {x,y,width,height} = bb;
  return (
    !Number.isFinite(x) || !Number.isFinite(y) ||
    !Number.isFinite(width) || !Number.isFinite(height) ||
    Math.abs(x) > S || Math.abs(y) > S ||
    Math.abs(width) > S || Math.abs(height) > S ||
    width <= 0 || height <= 0
  );
}

function PdfPaneImpl({ pdfUrl }, ref) {
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [tokens, setTokens] = useState([]); // all pages
  const [hit, setHit] = useState(null);
  const [hover, setHover] = useState(null); // {page,x0,y0,x1,y1}

  useImperativeHandle(ref, ()=>({
    locateValue: (value) => {
      if (!value) return;
      const res = locateValue(value, tokens);
      setHit(res || null);
      if (res) {
        setPage(res.page);
        requestAnimationFrame(drawOverlay);
      }
    },
    showDocAIBbox: (el) => {
      if (!el || isBad(el.bbox)) { setHover(null); drawOverlay(); return; }
      const r = { page: +el.page || page, x0: el.bbox.x, y0: el.bbox.y,
                  x1: el.bbox.x + el.bbox.width, y1: el.bbox.y + el.bbox.height };
      setHover(r); requestAnimationFrame(drawOverlay);
    }
  }));

  useEffect(()=>{ (async()=>{
    if (!pdfUrl) return;
    const doc = await getDocument(pdfUrl).promise;
    setNumPages(doc.numPages);
    // load all pages’ tokens (fast enough for 1–3 pages)
    const all = [];
    for (let p=1; p<=doc.numPages; p++){
      const pg = await doc.getPage(p);
      const vp = pg.getViewport({ scale: Math.min(1, 1400/Math.max(pg.view[2],pg.view[3])) });
      if (p === page) await renderPage(pg, vp);
      const text = await pg.getTextContent({ normalizeWhitespace:true, disableCombineTextItems:false });
      all.push(...itemsToTokens(text.items, vp, p));
    }
    setTokens(all);
  })(); }, [pdfUrl]);

  useEffect(()=>{ (async()=>{
    if (!pdfUrl) return;
    const doc = await getDocument(pdfUrl).promise;
    const pg = await doc.getPage(page);
    const vp = pg.getViewport({ scale: Math.min(1, 1400/Math.max(pg.view[2],pg.view[3])) });
    await renderPage(pg, vp);
  })(); }, [page, pdfUrl]);

  async function renderPage(pg, vp){
    const c = canvasRef.current, ctx = c.getContext("2d");
    c.width = Math.floor(vp.width); c.height = Math.floor(vp.height);
    c.style.width = `${c.width}px`; c.style.height = `${c.height}px`;
    await pg.render({ canvasContext: ctx, viewport: vp }).promise;
    drawOverlay();
  }

  function itemsToTokens(items, vp, pg){
    // pdf.js text item -> bbox in canvas space
    const toks = [];
    for (const it of items) {
      const [a,b,c,d,e,f] = it.transform; // text matrix
      const x = e, y = f;                 // bottom-left
      const fontH = Math.abs(d || b || 10);
      const width = it.width;
      // canvas y grows downward; text matrix y is bottom baseline
      const yTop = vp.height - y;
      const y0 = yTop - fontH;
      const x0 = x, x1 = x + width, y1 = yTop;
      toks.push({ page: pg, x0, y0, x1, y1, text: it.str });
    }
    // sort reading order (line, then x)
    toks.sort((A,B)=> (A.y0===B.y0 ? A.x0-B.x0 : A.y0-B.y0));
    return toks;
  }

  function pxPlace(node, x0,y0,x1,y1){
    const c = canvasRef.current;
    const R = c.getBoundingClientRect();
    const sx = R.width / c.width, sy = R.height / c.height;
    Object.assign(node.style,{
      position:"absolute",
      left:`${Math.min(x0,x1)*sx}px`,
      top:`${Math.min(y0,y1)*sy}px`,
      width:`${Math.abs(x1-x0)*sx}px`,
      height:`${Math.abs(y1-y0)*sy}px`,
    });
  }

  function drawOverlay(){
    const o = overlayRef.current; if (!o) return;
    o.innerHTML = "";

    if (hover && hover.page === page) {
      const d = document.createElement("div");
      d.className = "hover";
      pxPlace(d, hover.x0, hover.y0, hover.x1, hover.y1);
      o.appendChild(d);
    }
    if (hit && hit.page === page) {
      const d = document.createElement("div");
      d.className = "hit";
      pxPlace(d, hit.rect.x0, hit.rect.y0, hit.rect.x1, hit.rect.y1);
      o.appendChild(d);
    }
  }

  return (
    <div className="right">
      <div className="pagebar">
        <button disabled={!pdfUrl || page<=1} onClick={()=>setPage(p=>p-1)}>Prev</button>
        <span style={{margin:"0 8px"}}>Page {page}{numPages?` / ${numPages}`:""}</span>
        <button disabled={!pdfUrl || (numPages && page>=numPages)} onClick={()=>setPage(p=>p+1)}>Next</button>
      </div>
      {!pdfUrl ? (
        <div style={{opacity:.7,padding:12}}>Upload a PDF to view.</div>
      ) : (
        <div style={{position:"relative"}}>
          <canvas ref={canvasRef}/>
          <div ref={overlayRef} className="overlay" style={{position:"absolute",left:0,top:0,right:0,bottom:0}}/>
        </div>
      )}

      <style>{`
        .overlay .hit {
          border: 2px solid #ec4899;
          background: rgba(236,72,153,0.16);
          box-shadow: 0 0 0 1px rgba(236,72,153,0.3) inset;
        }
        .overlay .hover {
          border: 2px dashed #f59e0b;
          background: rgba(245,158,11,0.10);
        }
      `}</style>
    </div>
  );
}

export default forwardRef(PdfPaneImpl);