import OcrWorkbench from "../components/OcrWorkbench";

const routes: TemplateRoute[] = [
  // …your existing routes…
  {
    name: "OCR Demo",
    path: "/ocr",
    element: <OcrWorkbench/>
  }
];

export default routes;
