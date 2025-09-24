// src/components/PdfPane.jsx
import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { GlobalWorkerOptions, getDocument } from "pdfjs-dist";
import Tesseract from "tesseract.js";
import { locateValue } from "../lib/match.js";

GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

function isSentinelRect(bb) {
  const S = 1000000000; // 1e9
  return (
    bb == null ||
    !isFinite(bb.x) || !isFinite(bb.y) || !isFinite(bb.width) || !isFinite(bb.height) ||
    Math.abs(bb.x) > S || Math.abs(bb.y) > S || Math.abs(bb.width) > S || Math.abs(bb.height) > S ||
    bb.width <= 0 || bb.height <= 0
  );
}

function PdfPaneImpl({ pdfUrl }, ref) {
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState(0);
  const [tokens, setTokens] = useState([]); // [{page,x0,y0,x1,y1,text}]
  const [hit, setHit] = useState(null);     // true-rect from matcher
  const [hoverRect, setHoverRect] = useState(null); // {page, x0,y0,x1,y1, suspect}

  useImperativeHandle(ref, ()=>({
    async locateValue(value) {
      if (!value || !tokens.length) return;
      let res = locateValue(value, tokens);
      if (!res && canvasRef.current) {
        const ocrToks = await ocrCurrentPage(canvasRef.current);
        const merged = tokens.concat(ocrToks.map(t => ({...t, page})));
        res = locateValue(value, merged);
      }
      if (res) {
        setPage(res.page);
        setHit(res);
        requestAnimationFrame(drawOverlay);
      }
    },
    showDocAIBbox(el) {
      if (!el) { setHoverRect(null); drawOverlay(); return; }
      const bb = el.bbox || {};
      const suspect = isSentinelRect(bb);
      if (suspect) {
        // still show something as a cue — a big dashed frame spanning canvas,
        // but we annotate "suspect"
        setHoverRect({ page: el.page, x0: 2, y0: 2, x1: canvasRef.current?.width-2 || 20, y1: 40, suspect: true });
      } else {
        setHoverRect({ page: el.page, x0: bb.x, y0: bb.y, x1: bb.x + bb.width, y1: bb.y + bb.height, suspect: false });
      }
      requestAnimationFrame(drawOverlay);
    }
  }));

  useEffect(()=>{ (async ()=>{
    if(!pdfUrl) return;
    const doc = await getDocument(pdfUrl).promise;
    setNumPages(doc.numPages);
    await renderPage(doc, page);
  })(); }, [pdfUrl, page]);

  async function renderPage(doc, pageNum){
    const pg = await doc.getPage(pageNum);
    const vp = pg.getViewport({ scale: Math.min(1, 1400/Math.max(pg.view[2],pg.view[3])) });
    const c = canvasRef.current; const ctx = c.getContext("2d");
    c.width = Math.floor(vp.width); c.height = Math.floor(vp.height);
    c.style.width = `${c.width}px`; c.style.height = `${c.height}px`;
    await pg.render({ canvasContext: ctx, viewport: vp }).promise;

    const text = await pg.getTextContent();
    const pageTokens = itemsToTokens(text.items, vp, pageNum);
    setTokens(prev => [...prev.filter(t=>t.page!==pageNum), ...pageTokens]);
    drawOverlay();
  }

  function itemsToTokens(items, vp, pg){
    const toks = [];
    for(const it of items){
      const tx = it.transform; // [a,b,c,d,e,f]
      const x = tx[4], y = tx[5];
      const h = Math.abs(tx[3] || 10);
      const w = it.width;
      const yTop = vp.height - y;
      toks.push({ page: pg, x0:x, y0:yTop-h, x1:x+w, y1:yTop, text:it.str });
    }
    return toks;
  }

  async function ocrCurrentPage(canvas){
    const { data:{ words } } = await Tesseract.recognize(canvas, "eng");
    return (words||[]).map(w=>({ text:w.text, x0:w.bbox.x0, y0:w.bbox.y0, x1:w.bbox.x1, y1:w.bbox.y1 }));
  }

  function drawOverlay(){
    const o = overlayRef.current; if(!o) return;
    o.innerHTML = "";

    // hover (DocAI bbox)
    if (hoverRect && hoverRect.page === page) {
      const d = document.createElement("div");
      d.className = hoverRect.suspect ? "hover suspect" : "hover";
      place(d, hoverRect.x0, hoverRect.y0, hoverRect.x1, hoverRect.y1);
      o.appendChild(d);
    }

    // true hit (matcher)
    if (hit && hit.page===page) {
      const d = document.createElement("div");
      d.className = "hit";
      place(d, hit.rect.x0, hit.rect.y0, hit.rect.x1, hit.rect.y1);
      o.appendChild(d);
    }
  }

  function place(el,x0,y0,x1,y1){
    const c = canvasRef.current;
    const R = c.getBoundingClientRect();
    const sx = R.width / c.width, sy = R.height / c.height;
    Object.assign(el.style, {
      position:"absolute",
      left:`${Math.min(x0,x1)*sx}px`,
      top:`${Math.min(y0,y1)*sy}px`,
      width:`${Math.abs(x1-x0)*sx}px`,
      height:`${Math.abs(y1-y0)*sy}px`
    });
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
          background: rgba(245,158,11,0.12);
        }
        .overlay .hover.suspect {
          border-style: dotted;
          background: rgba(245,158,11,0.05);
        }
      `}</style>
    </div>
  );
}

export default forwardRef(PdfPaneImpl);