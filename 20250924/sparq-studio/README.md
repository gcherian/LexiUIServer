Agent Studio
# Agent Studio (Node + React)

## Run
npm i
npm run dev

## Build
npm run build && npm run preview

## What it does
- Low-code canvas to add agents (A0..A10)
- Connect them visually
- Configure nodes via Inspector
- Run (mock executor) to simulate MACP flow and see logs

## Extend
- Replace mock run with calls to your FastAPI/AKS services
- Export/import Studio JSON
- Add auth (Teams, Entra ID)
- Add “Publish” to push to a marketplace (similar to Lyzr’s Studio + Marketplace ideas). 

Layout

agent-studio/
├─ package.json
├─ index.html
├─ vite.config.ts
├─ tailwind.config.js
├─ postcss.config.js
├─ src/
│  ├─ main.tsx
│  ├─ App.tsx
│  ├─ store.ts
│  ├─ types.ts
│  ├─ components/
│  │   ├─ LeftRail.tsx
│  │   ├─ Canvas.tsx
│  │   ├─ Inspector.tsx
│  │   ├─ RunPanel.tsx
│  │   └─ TopBar.tsx
│  └─ theme.css
└─ README.md
