// src/components/KVPane.jsx
import React from "react";

export default function KVPane({ header = {}, elements = [], onHoverDocAI, onPickValue }) {
  const headerEntries = Object.entries(header);

  return (
    <div className="left">
      <div style={{padding:"10px 12px",borderBottom:"1px solid #1f2a4a"}}>
        <b>DocAI Header</b>
        {!headerEntries.length && <div style={{opacity:.6}}>Upload DocAI JSON</div>}
      </div>
      {headerEntries.length > 0 && (
        <table className="table">
          <thead><tr><th style={{width:"42%"}}>Key</th><th>Value</th></tr></thead>
          <tbody>
            {headerEntries.map(([k, v]) => (
              <tr key={k}><td><code>{k}</code></td><td>{String(v)}</td></tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{padding:"10px 12px",borderTop:"1px solid #1f2a4a"}}>
        <b>DocAI Elements</b>
        <div style={{opacity:.7,fontSize:12}}>Hover: show DocAI bbox • Click: find true location</div>
      </div>

      {elements?.length ? (
        <table className="table">
          <thead><tr><th>Content</th><th style={{width:72}}>Page</th></tr></thead>
          <tbody>
            {elements.map((el, i) => (
              <tr
                key={i}
                onMouseEnter={() => onHoverDocAI?.(el)}
                onMouseLeave={() => onHoverDocAI?.(null)}
                onClick={() => onPickValue?.(el.content)}
                title="Click to locate on PDF"
                style={{cursor:"pointer"}}
              >
                <td>{el.content || <span style={{opacity:.6}}>(empty)</span>}</td>
                <td>{el.page}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div style={{opacity:.6,padding:"8px 12px"}}>No DocAI page elements found.</div>
      )}
    </div>
  );
}