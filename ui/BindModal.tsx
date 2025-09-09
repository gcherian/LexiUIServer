import React from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  // kept for compatibility; not used by the streamlined inline flow
  docId?: string;
  docUrl?: string;
  page?: number;
  serverW?: number;
  serverH?: number;
  allKeys?: string[];
  initialKey?: string;
  box?: any;
  onBound?: (state: any) => void;
};

export default function BindModal({ open, onClose }: Props) {
  if (!open) return null;
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        zIndex: 40,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 480,
          background: "#fff",
          borderRadius: 10,
          boxShadow: "0 10px 40px rgba(0,0,0,0.2)",
          padding: 16,
        }}
      >
        <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 8 }}>Bind</div>
        <div style={{ color: "#666", marginBottom: 12 }}>
          Inline lasso is now supported directly in the editor. This modal is kept for compatibility.
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
