"""WebSocket endpoints for real-time simulation streaming."""

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

import pylinkage as pl
from pylinkage.exceptions import UnbuildableError

from ..services import linkage_service
from ..storage.memory import storage

router = APIRouter(tags=["websocket"])


def build_linkage_from_stored(stored: dict[str, Any]) -> tuple[pl.Linkage | None, str]:
    """Build a linkage from stored data.

    Returns:
        Tuple of (linkage, error_message). If successful, error is empty.
    """
    try:
        linkage_dict = {
            "name": stored.get("name"),
            "joints": stored.get("joints", []),
            "solve_order": stored.get("solve_order"),
        }
        linkage = pl.Linkage.from_dict(linkage_dict)
        # Validate buildability
        list(linkage.step(iterations=1))
        return linkage, ""
    except UnbuildableError as e:
        return None, f"Unbuildable: {e}"
    except Exception as e:
        return None, f"Error: {e}"


@router.websocket("/ws/simulation/{linkage_id}")
async def simulation_websocket(websocket: WebSocket, linkage_id: str) -> None:
    """Stream simulation frames in real-time via WebSocket.

    The client should send a JSON message to start streaming:
    {
        "action": "start",
        "iterations": 360,  // optional, uses rotation_period if not provided
        "fps": 30           // optional, default 30
    }

    The server will stream frames:
    {
        "type": "frame",
        "step": 0,
        "positions": [[x1, y1], [x2, y2], ...]
    }

    On completion:
    {
        "type": "complete",
        "total_frames": 360
    }

    On error:
    {
        "type": "error",
        "message": "..."
    }
    """
    await websocket.accept()

    # Get linkage from storage
    stored = storage.get(linkage_id)
    if stored is None:
        await websocket.send_json({
            "type": "error",
            "message": "Linkage not found"
        })
        await websocket.close()
        return

    # Build the linkage
    linkage, error = build_linkage_from_stored(stored)
    if linkage is None:
        await websocket.send_json({
            "type": "error",
            "message": error
        })
        await websocket.close()
        return

    joint_names = linkage_service.get_joint_names(linkage)

    # Send initial info
    await websocket.send_json({
        "type": "ready",
        "joint_names": joint_names,
        "rotation_period": linkage.get_rotation_period()
    })

    try:
        while True:
            # Wait for start command
            data = await websocket.receive_json()

            if data.get("action") == "start":
                iterations = data.get("iterations") or linkage.get_rotation_period()
                fps = data.get("fps", 30)
                frame_interval = 1.0 / fps

                # Stream frames
                step = 0
                for positions in linkage.step(iterations=iterations, dt=1.0):
                    frame = [[pos[0], pos[1]] for pos in positions]
                    await websocket.send_json({
                        "type": "frame",
                        "step": step,
                        "positions": frame
                    })
                    step += 1
                    # Small delay to control frame rate for live viewing
                    await asyncio.sleep(frame_interval)

                # Signal completion
                await websocket.send_json({
                    "type": "complete",
                    "total_frames": step
                })

            elif data.get("action") == "stop":
                # Just acknowledge and wait for next command
                await websocket.send_json({"type": "stopped"})

            elif data.get("action") == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })


@router.websocket("/ws/simulation-fast/{linkage_id}")
async def simulation_fast_websocket(websocket: WebSocket, linkage_id: str) -> None:
    """Stream all simulation frames as fast as possible (no FPS limiting).

    This is useful for preloading all frames before starting playback.

    The client should send:
    {
        "action": "start",
        "iterations": 360  // optional
    }

    The server streams frames immediately with progress updates:
    {
        "type": "progress",
        "current": 50,
        "total": 360
    }

    Then sends all frames at once for efficiency:
    {
        "type": "frames",
        "frames": [[[x,y], ...], ...]
    }
    """
    await websocket.accept()

    stored = storage.get(linkage_id)
    if stored is None:
        await websocket.send_json({
            "type": "error",
            "message": "Linkage not found"
        })
        await websocket.close()
        return

    linkage, error = build_linkage_from_stored(stored)
    if linkage is None:
        await websocket.send_json({
            "type": "error",
            "message": error
        })
        await websocket.close()
        return

    joint_names = linkage_service.get_joint_names(linkage)
    rotation_period = linkage.get_rotation_period()

    await websocket.send_json({
        "type": "ready",
        "joint_names": joint_names,
        "rotation_period": rotation_period
    })

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start":
                iterations = data.get("iterations") or rotation_period

                # Collect all frames
                frames: list[list[list[float]]] = []
                progress_interval = max(1, iterations // 10)  # Report ~10 progress updates

                for step, positions in enumerate(
                    linkage.step(iterations=iterations, dt=1.0), start=1
                ):
                    frame = [[pos[0], pos[1]] for pos in positions]
                    frames.append(frame)

                    # Send progress updates periodically
                    if step % progress_interval == 0:
                        await websocket.send_json({
                            "type": "progress",
                            "current": step,
                            "total": iterations
                        })
                        # Yield to event loop occasionally
                        await asyncio.sleep(0)

                # Send all frames
                await websocket.send_json({
                    "type": "frames",
                    "frames": frames,
                    "total_frames": len(frames)
                })

            elif data.get("action") == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
