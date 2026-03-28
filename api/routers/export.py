"""Export endpoints for downloading mechanism in various formats."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response

from ..models.mechanism_schemas import MechanismCreate
from ..services import export_service, mechanism_service

router = APIRouter(prefix="/export", tags=["export"])


def _build_mechanism(data: MechanismCreate):
    """Build a Mechanism from the request data, or raise 400."""
    mechanism_dict = {
        "name": data.name,
        "joints": data.joints,
        "links": data.links,
        "ground": data.ground,
    }
    mechanism, is_buildable, error = mechanism_service.validate_and_build(mechanism_dict)
    if not is_buildable or mechanism is None:
        raise HTTPException(status_code=400, detail=error or "Mechanism is not buildable")
    return mechanism


@router.post("/python")
def export_python(data: MechanismCreate) -> PlainTextResponse:
    """Generate Python code that recreates the mechanism."""
    mechanism = _build_mechanism(data)
    code = export_service.generate_python_code(mechanism)
    name = data.name or "mechanism"
    return PlainTextResponse(
        content=code,
        headers={"Content-Disposition": f'attachment; filename="{name}.py"'},
    )


@router.post("/svg")
def export_svg(data: MechanismCreate) -> Response:
    """Export mechanism as SVG."""
    mechanism = _build_mechanism(data)
    try:
        svg_content = export_service.export_svg(mechanism)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVG export failed: {e}") from None
    name = data.name or "mechanism"
    return Response(
        content=svg_content,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{name}.svg"'},
    )


@router.post("/dxf")
def export_dxf(data: MechanismCreate) -> Response:
    """Export mechanism as DXF (AutoCAD/CNC)."""
    mechanism = _build_mechanism(data)
    try:
        dxf_bytes = export_service.export_dxf(mechanism)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DXF export failed: {e}") from None
    name = data.name or "mechanism"
    return Response(
        content=dxf_bytes,
        media_type="application/dxf",
        headers={"Content-Disposition": f'attachment; filename="{name}.dxf"'},
    )


@router.post("/step")
def export_step(data: MechanismCreate) -> Response:
    """Export mechanism as STEP (3D CAD)."""
    mechanism = _build_mechanism(data)
    try:
        step_bytes = export_service.export_step(mechanism)
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STEP export failed: {e}") from None
    name = data.name or "mechanism"
    return Response(
        content=step_bytes,
        media_type="application/step",
        headers={"Content-Disposition": f'attachment; filename="{name}.step"'},
    )
