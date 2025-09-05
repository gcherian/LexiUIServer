import PdfCanvas, { type Rect, type Box } from "./PdfCanvas";

type Props = {
  url: string;
  page: number;
  scale: number;
  ocrSize: { width:number; height:number };

  // layers
  ocrBoxes?: Box[];
  highlightBoxes?: Box[];
  boundBoxes?: Box[];
  selectedBoxIds?: string[];

  // toolbar / state
  bindKey?: string | null;                 // if set, lasso mode is active
  rotationMode?: "auto" | "none";          // pass through to PdfCanvas ("auto" recommended)

  // controls
  onClose: () => void;
  onChangePage: (p:number) => void;
  onZoom: (s:number) => void;

  // interactions
  onLasso: (page:number, rect:Rect) => void;
  onSelectBox?: (boxId: string, idx: number) => void;
};

export default function PdfViewerFS({
  url, page, scale, ocrSize,
  ocrBoxes = [], highlightBoxes = [], boundBoxes = [], selectedBoxIds = [],
  bindKey = null, rotationMode = "auto",
  onClose, onChangePage, onZoom, onLasso, onSelectBox
}: Props){

  const tool: "select" | "lasso" = bindKey ? "lasso" : "select";

  // Minimal, clean modal styling (no external CSS dependencies)
  const modal: React.CSSProperties = {
    position:"fixed", inset:0, background:"rgba(0,0,0,0.55)",
    display:"flex", alignItems:"center", justifyContent:"center", zIndex: 9999
  };
  const frame: React.CSSProperties = {
    width:"92vw", height:"88vh", background:"#fff", borderRadius:14,
    boxShadow:"0 10px 40px rgba(0,0,0,.4)", display:"grid",
    gridTemplateRows:"56px 1fr", overflow:"hidden"
  };
  const header: React.CSSProperties = {
    display:"flex", alignItems:"center", justifyContent:"space-between",
    padding:"0 14px", borderBottom:"1px solid #e5e5e5", background:"#fafafa"
  };
  const stage: React.CSSProperties = {
    position:"relative", width:"100%", height:"100%", overflow:"auto",
    display:"flex", alignItems:"flex-start", justifyContent:"center", padding:12
  };

  return (
    <div style={modal} onClick={onClose}>
      <div style={frame} onClick={(e)=> e.stopPropagation()}>
        <div style={header}>
          <div style={{display:"flex", alignItems:"center", gap:8}}>
            <strong>PDF Viewer</strong>
            {bindKey && (
              <span style={{fontSize:12, padding:"4px 8px", background:"#eef", borderRadius:12}}>
                Binding: <b>{bindKey}</b> — drag to lasso
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
            rotationMode={rotationMode}
            ocrBoxes={ocrBoxes}
            highlightBoxes={highlightBoxes}
            boundBoxes={boundBoxes}
            selectedBoxIds={selectedBoxIds}
            tool={tool}
            onLasso={(rect)=> onLasso(page, rect)}
            onSelectBox={onSelectBox}
          />
        </div>
      </div>
    </div>
  );
}
