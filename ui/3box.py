type ThreeBoxes = {
  auto_bbox:  [number,number,number,number] | null; auto_page?:  number;
  tfidf_bbox: [number,number,number,number] | null; tfidf_page?: number;
  bert_bbox:  [number,number,number,number] | null; bert_page?:  number;
};

// Convert PDF-space rect to canvas pixels using your pdf.js viewport
function pdfRectToCanvas(viewport: any, rect: [number,number,number,number]) {
  const [x0,y0,x1,y1] = rect;
  const [vx0,vy0,vx1,vy1] = viewport.convertToViewportRectangle([x0,y0,x1,y1]);
  const x = Math.min(vx0,vx1), y = Math.min(vy0,vy1);
  const w = Math.abs(vx1-vx0), h = Math.abs(vy1-vy0);
  return { x,y,w,h };
}

async function drawThreeOnPEC(boxes: ThreeBoxes) {
  pdfEditCanvas.clearOverlayLayer("validation");

  async function drawOne(rect: any, page: number | undefined, color: string) {
    if (!rect) return;
    const p = page ?? pdfEditCanvas.currentPage();
    const { viewport } = await pdfEditCanvas.ensurePageRendered(p);
    const { x,y,w,h } = pdfRectToCanvas(viewport, rect);
    pdfEditCanvas.drawRectOverlay("validation", { x, y, w, h, color, alpha: 0.20, line: 2 });
  }
//-----
  await drawOne(boxes.auto_bbox,  boxes.auto_page,  "#1f7ae0"); // blue
  await drawOne(boxes.tfidf_bbox, boxes.tfidf_page, "#e07a1f"); // orange
  await drawOne(boxes.bert_bbox,  boxes.bert_page,  "#7a1fe0"); // const row = resp.results[0];
await drawThreeOnPEC({
  auto_bbox:  row.auto_bbox,  auto_page:  row.auto_page,
  tfidf_bbox: row.tfidf_bbox, tfidf_page: row.tfidf_page,
  bert_bbox:  row.bert_bbox,  bert_page:  row.bert_page
});


