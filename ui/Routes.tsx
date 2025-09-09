import OcrWorkbench from "../components/OcrWorkbench";

const routes: TemplateRoute[] = [
  // …your existing routes…
  {
    name: "OCR Demo",
    path: "/ocr",
    element: <OcrWorkbench/>
  }
];

import FieldLevelEditor from "../components/fieldedit/FieldLevelEditor";

// ...
{ path: "/field_editor", element: <FieldLevelEditor /> },

export default routes;
