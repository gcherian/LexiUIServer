// ui/src/Routes.tsx
import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import FieldLevelEditor from "./tsp4/components/lasso/FieldLevelEditor";

export default function AppRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/lasso" replace />} />
        <Route path="/lasso" element={<FieldLevelEditor />} />
        {/* add other app routes here */}
      </Routes>
    </BrowserRouter>
  );
}
