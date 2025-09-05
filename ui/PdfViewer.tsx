import { useMemo } from "react";
import PdfCanvas, { type Rect, type Box } from "./PdfCanvas";

type Props = {
  url: string;                                 // absolute PDF url
  page: number;
  scale: number;
  ocrSize: { width:number; height:number };
  bindKey: string | null;

  // overlay data (optional)
  boxes?: Box[];                               // base boxes (blue)
  highlights?: Box[];                          // search hits (orange)
  selected?: number[];                         // indexes into boxes[]

  // controls
  onClose: () => void;
  onChangePage: (p:number) => void;
  onZoom: (s:number) => void;

  // lasso result → parent (doc space)
  onLasso: (page:number, rect:Rect) => void;

  // click a box (optional)
  onSelectBox?: (boxIndex:number) => void;
};

export default function PdfViewerFS({
  url, page, scale, ocrSize, bindKey,
  boxes = [], highlights = [], selected = [],
  onClose, onChangePage, onZoom, onLasso, onSelectBox
}: Props) {

  // use lasso tool when binding, otherwise selection tool
  const tool: "select" | "lasso" = bindKey ? "lasso" : "select";

  // simple modal styles with no external CSS dependencies
  const modal: React.CSSProperties = {
    position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
    display: "flex", alignItems: "center", justifyContent: "center", zIndex: 9999
  };
  const frame: React.CSSProperties = {
    width: "92vw", height: "88vh", background: "#fff", borderRadius: 14,
    boxShadow: "0 10px 40px rgba(0,0,0,.4)", display: "grid",
    gridTemplateRows: "56px 1fr", overflow: "hidden"
  };
  const header: React.CSSProperties = {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    padding: "0 14px", borderBottom: "1px solid #e5e5e5", background: "#fafafa"
  };
  const stage: React.CSSProperties = {
    position: "relative", width: "100%", height: "100%", overflow: "auto",
    display: "flex", alignItems: "flex-start", justifyContent: "center", padding: 12
  };

  return (
    <div style={modal} onClick={onClose}>
      <div style={frame} onClick={e=> e.stopPropagation()}>
        <div style={header}>
          <div style={{display:"flex", alignItems:"center", gap:8}}>
            <strong>PDF Viewer</strong>
            {bindKey && (
              <span style={{fontSize:12, padding:"4px 8px", background:"#eef", borderRadius:12}}>
                Binding: <span style={{fontWeight:600}}>{bindKey}</span> — drag to lasso
              </span>
            )}
          </div>
          <div style={{display:"flex", gap:8, alignItems:"center"}}>
            <button onClick={()=> onChangePage(Math.max(1, page-1))}>Prev</button>
            <span style={{fontFamily:"monospace"}}>p{page}</span>
            <button onClick={()=> onChangePage(page+1)}>Next</button>
            <span style={{width:8}}/>
            <button onClick={()=> onZoom(Math.max(0.5, +(scale-0.25).toFixed(2)))}>-</button>
            <span style={{fontFamily:"monospace"}}>{scale.toFixed(2)}x</span>
            <button onClick={()=> onZoom(+(scale+0.25).toFixed(2))}>+</button>
            <span style={{width:8}}/>
            <button onClick={onClose}>Close</button>
          </div>
        </div>

        <div style={stage}>
          <PdfCanvas
            url={url}
            page={page}
            scale={scale}
            ocrSize={ocrSize}
            boxes={boxes}
            highlights={highlights}
            selected={selected}
            tool={tool}
            onLasso={(rect)=> onLasso(page, rect)}
            onSelectBox={onSelectBox}
          />
        </div>
      </div>
    </div>
  );
}

