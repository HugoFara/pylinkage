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

## Phase 2: Enhanced Interactivity (COMPLETED)

### 2.1 WebSocket for Live Animation
- [x] Stream simulation frames in real-time via WebSocket
- [x] Two endpoints: `/ws/simulation/{id}` (frame-by-frame) and `/ws/simulation-fast/{id}` (batch)
- [x] Progress updates during simulation
- [x] Optional streaming mode toggle in UI

### 2.2 Keyboard Shortcuts
- [x] `Escape`: Switch to select mode
- [x] `Space`: Toggle animation playback
- [x] `Delete/Backspace`: Delete selected joint
- [x] `Ctrl+Z`: Undo
- [x] `Ctrl+Shift+Z` / `Ctrl+Y`: Redo
- [x] `1-7`: Quick mode selection (select, add-joint, draw-link, move, delete, ground, crank)

### 2.3 Direct Joint Manipulation
- [x] Drag any joint type (Static, Crank, Fixed, Revolute)
- [x] Automatic constraint recalculation:
  - Update `distance`/`angle` for Crank/Fixed based on parent position
  - Update `distance0`/`distance1` for Revolute based on both parents

### 2.4 Draw Link Mode
- [x] Click first joint to start link
- [x] Visual preview line follows mouse
- [x] Click second joint to create Revolute connection at midpoint
- [x] Escape or click empty space to cancel

### Files Created (Phase 2)
- `api/routers/websocket.py` - WebSocket endpoints for simulation streaming
- `frontend/src/hooks/useKeyboardShortcuts.ts` - Global keyboard shortcut handler
- `frontend/src/hooks/useSimulationStream.ts` - WebSocket client hook

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

### Created (Phase 2)
- `api/routers/websocket.py` - WebSocket endpoints for simulation streaming
- `frontend/src/hooks/useKeyboardShortcuts.ts` - Global keyboard shortcuts
- `frontend/src/hooks/useSimulationStream.ts` - WebSocket client for streaming

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
