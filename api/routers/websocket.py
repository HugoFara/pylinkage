"""WebSocket endpoints for real-time simulation streaming."""

import asyncio
import contextlib
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pylinkage.mechanism import Mechanism
from pylinkage.mechanism.serialization import mechanism_from_dict
from pylinkage.exceptions import UnbuildableError

from ..services import mechanism_service
from ..storage.memory import storage

router = APIRouter(tags=["websocket"])


def build_mechanism_from_stored(
    stored: dict[str, Any],
) -> tuple[Mechanism | None, str]:
    """Build a mechanism from stored data.

    Returns:
        Tuple of (mechanism, error_message). If successful, error is empty.
    """
    try:
        mechanism_dict = {
            "name": stored.get("name"),
            "joints": stored.get("joints", []),
            "links": stored.get("links", []),
            "ground": stored.get("ground"),
        }
        mechanism = mechanism_from_dict(mechanism_dict)
        # Validate buildability
        for _ in mechanism.step(dt=1.0):
            break
        return mechanism, ""
    except UnbuildableError as e:
        return None, f"Unbuildable: {e}"
    except Exception as e:
        return None, f"Error: {e}"


@router.websocket("/ws/simulation/{mechanism_id}")
async def simulation_websocket(websocket: WebSocket, mechanism_id: str) -> None:
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

    # Get mechanism from storage
    stored = storage.get(mechanism_id)
    if stored is None:
        await websocket.send_json({"type": "error", "message": "Mechanism not found"})
        await websocket.close()
        return

    # Build the mechanism
    mechanism, error = build_mechanism_from_stored(stored)
    if mechanism is None:
        await websocket.send_json({"type": "error", "message": error})
        await websocket.close()
        return

    joint_names = mechanism_service.get_joint_names(mechanism)

    # Send initial info
    await websocket.send_json({
        "type": "ready",
        "joint_names": joint_names,
        "rotation_period": mechanism.get_rotation_period(),
    })

    try:
        while True:
            # Wait for start command
            data = await websocket.receive_json()

            if data.get("action") == "start":
                iterations = data.get("iterations") or mechanism.get_rotation_period()
                fps = data.get("fps", 30)
                frame_interval = 1.0 / fps

                # Reset mechanism to initial state
                mechanism.reset()

                # Stream frames
                step = 0
                for positions in mechanism.step(dt=1.0):
                    frame = [
                        [
                            pos[0] if pos[0] is not None else 0.0,
                            pos[1] if pos[1] is not None else 0.0,
                        ]
                        for pos in positions
                    ]
                    await websocket.send_json({
                        "type": "frame",
                        "step": step,
                        "positions": frame,
                    })
                    step += 1
                    if step >= iterations:
                        break
                    # Small delay to control frame rate for live viewing
                    await asyncio.sleep(frame_interval)

                # Signal completion
                await websocket.send_json({"type": "complete", "total_frames": step})

            elif data.get("action") == "stop":
                # Just acknowledge and wait for next command
                await websocket.send_json({"type": "stopped"})

            elif data.get("action") == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})


@router.websocket("/ws/simulation-fast/{mechanism_id}")
async def simulation_fast_websocket(websocket: WebSocket, mechanism_id: str) -> None:
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

    stored = storage.get(mechanism_id)
    if stored is None:
        await websocket.send_json({"type": "error", "message": "Mechanism not found"})
        await websocket.close()
        return

    mechanism, error = build_mechanism_from_stored(stored)
    if mechanism is None:
        await websocket.send_json({"type": "error", "message": error})
        await websocket.close()
        return

    joint_names = mechanism_service.get_joint_names(mechanism)
    rotation_period = mechanism.get_rotation_period()

    await websocket.send_json({
        "type": "ready",
        "joint_names": joint_names,
        "rotation_period": rotation_period,
    })

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "start":
                iterations = data.get("iterations") or rotation_period

                # Reset mechanism to initial state
                mechanism.reset()

                # Collect all frames
                frames: list[list[list[float]]] = []
                progress_interval = max(1, iterations // 10)  # Report ~10 progress updates

                step = 0
                for positions in mechanism.step(dt=1.0):
                    frame = [
                        [
                            pos[0] if pos[0] is not None else 0.0,
                            pos[1] if pos[1] is not None else 0.0,
                        ]
                        for pos in positions
                    ]
                    frames.append(frame)
                    step += 1

                    # Send progress updates periodically
                    if step % progress_interval == 0:
                        await websocket.send_json({
                            "type": "progress",
                            "current": step,
                            "total": iterations,
                        })
                        # Yield to event loop occasionally
                        await asyncio.sleep(0)

                    if step >= iterations:
                        break

                # Send all frames
                await websocket.send_json({
                    "type": "frames",
                    "frames": frames,
                    "total_frames": len(frames),
                })

            elif data.get("action") == "close":
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})
