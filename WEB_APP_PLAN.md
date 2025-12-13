# Pylinkage Web App Implementation Plan

## Overview

Build a standalone web application for interactive linkage design with:
- **Frontend**: React + TypeScript + Konva.js (canvas interactions)
- **Backend**: FastAPI REST API wrapping pylinkage library
- **Communication**: REST for CRUD/simulation, WebSocket for real-time optimization progress

## Architecture

```
pylinkage/
├── src/pylinkage/          # Existing library (unchanged)
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── linkages.py     # CRUD endpoints
│   │   ├── simulation.py   # Simulation endpoints
│   │   └── examples.py     # Prebuilt examples
│   ├── models/
│   │   └── schemas.py      # Pydantic models
│   ├── services/
│   │   └── linkage_service.py
│   └── storage/
│       └── memory.py       # In-memory storage
├── frontend/               # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/     # AppShell, Sidebar
│   │   │   ├── canvas/     # LinkageCanvas, CanvasToolbar
│   │   │   └── sidebar/    # JointList, ExampleLoader, AnimationControls
│   │   ├── stores/         # Zustand stores
│   │   ├── hooks/
│   │   ├── api/            # API client
│   │   └── types/          # TypeScript types
│   └── package.json
└── app/                    # Existing Streamlit (kept for reference)
```

---

## Phase 1: MVP (COMPLETED)

### Backend
- [x] FastAPI application with CORS
- [x] Linkage CRUD: `POST/GET/PUT/DELETE /api/linkages`
- [x] Simulation: `POST /api/linkages/{id}/simulate`
- [x] Examples: `GET /api/examples`, `POST /api/examples/{name}/load`
- [x] Pydantic schemas matching `serialization.py`
- [x] In-memory storage

### Frontend
- [x] React + TypeScript + Vite scaffold
- [x] Konva canvas with joint/link rendering
- [x] Editor modes: select, add-joint, move-joint, delete, set-ground, set-crank
- [x] Zustand stores with undo/redo (zundo)
- [x] Example loader
- [x] Animation controls (simulate, play/pause, frame slider)
- [x] Joint list with selection
- [x] View toggles (loci, grid)

### Running Phase 1

```bash
# Terminal 1: Backend
uv sync --extra api
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm install && npm run dev
```

Open http://localhost:5173

---

## Phase 2: Enhanced Interactivity (PLANNED)

### 2.1 WebSocket for Live Animation
Stream simulation frames in real-time instead of fetching all at once.

```python
# api/routers/websocket.py
@app.websocket("/ws/simulation/{linkage_id}")
async def simulation_ws(websocket: WebSocket, linkage_id: str):
    await websocket.accept()
    linkage = get_linkage(linkage_id)
    for frame in linkage.step(iterations=...):
        await websocket.send_json({"frame": frame})
        await asyncio.sleep(1/60)
```

### 2.2 Keyboard Shortcuts
- `Escape`: Switch to select mode
- `Space`: Toggle animation
- `Delete/Backspace`: Delete selected joint
- `Ctrl+Z`: Undo
- `Ctrl+Shift+Z`: Redo
- `1-6`: Quick mode selection

### 2.3 Direct Joint Manipulation
When dragging joints, recalculate constraint parameters:
- Update `distance` for Crank/Fixed
- Update `distance0`/`distance1` for Revolute
- Debounced API sync

### 2.4 Draw Link Mode
Click two joints to create a Revolute connection between them.

### Files to Create
- `api/routers/websocket.py`
- `frontend/src/hooks/useKeyboardShortcuts.ts`
- `frontend/src/hooks/useSimulationStream.ts`

---

## Phase 3: Advanced Features (PLANNED)

### 3.1 Optimization Panel
PSO optimization with progress streaming.

```python
# api/routers/optimization.py
@router.post("/jobs")
async def start_optimization(request: OptimizationRequest):
    job_id = create_job(request)
    return {"job_id": job_id}

@router.websocket("/jobs/{job_id}/ws")
async def optimization_progress(websocket: WebSocket, job_id: str):
    # Stream progress updates
```

### 3.2 Synthesis Wizards
Step-by-step UI for mechanism synthesis:
1. Choose type (function, path, motion generation)
2. Define precision points on canvas
3. Configure options (Grashof filter, bounds)
4. Generate and preview solutions
5. Load selected solution into editor

### 3.3 Export Options
- JSON (existing)
- Python code generation
- SVG via `plot_linkage_svg()`
- Animated GIF

### Files to Create
- `api/routers/optimization.py`
- `api/routers/synthesis.py`
- `api/jobs/manager.py`
- `frontend/src/components/tools/OptimizationPanel.tsx`
- `frontend/src/components/tools/SynthesisWizard.tsx`

---

## Critical Files Reference

### Existing (Integrate)
- `src/pylinkage/linkage/serialization.py` - JSON format
- `src/pylinkage/optimization/particle_swarm.py` - PSO
- `src/pylinkage/synthesis/__init__.py` - Synthesis API
- `src/pylinkage/visualizer/plotly_viz.py` - Plotly output
- `examples/paperjs_linkage_editor.html` - Reference prototype

### Created (Phase 1)
- `api/main.py` - FastAPI application
- `api/models/schemas.py` - Pydantic models
- `api/routers/linkages.py` - CRUD endpoints
- `api/routers/simulation.py` - Simulation endpoint
- `api/routers/examples.py` - Example linkages
- `api/services/linkage_service.py` - Business logic
- `api/storage/memory.py` - In-memory storage
- `frontend/src/components/canvas/LinkageCanvas.tsx` - Konva canvas
- `frontend/src/stores/editorStore.ts` - UI state
- `frontend/src/stores/linkageStore.ts` - Linkage data
- `frontend/src/types/linkage.ts` - TypeScript types

---

## Dependencies

### Python (pyproject.toml)
```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic-settings>=2.1.0",
]
```

### JavaScript (frontend/package.json)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-konva": "^18.2.10",
    "konva": "^9.3.6",
    "zustand": "^4.5.0",
    "@tanstack/react-query": "^5.17.0",
    "zundo": "^2.1.0"
  }
}
```

---

## Testing Strategy

- **Backend**: pytest + httpx TestClient
- **WebSocket**: pytest-asyncio
- **Frontend**: Vitest + React Testing Library
- **E2E**: Playwright (Phase 2+)
