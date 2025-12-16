/**
 * WebSocket-based simulation streaming hook.
 *
 * Provides real-time simulation streaming from the backend,
 * with support for both frame-by-frame streaming and fast batch loading.
 */

import { useCallback, useRef, useState, useEffect } from 'react';
import type { SimulationFrame, Position } from '../types/mechanism';

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error';

interface StreamState {
  status: ConnectionStatus;
  progress: number;
  totalFrames: number;
  error: string | null;
  jointNames: string[];
  rotationPeriod: number | null;
}

interface UseSimulationStreamOptions {
  onFrame?: (frame: SimulationFrame) => void;
  onComplete?: (frames: SimulationFrame[], jointNames: string[]) => void;
  onError?: (error: string) => void;
}

export function useSimulationStream(options: UseSimulationStreamOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null);
  const framesRef = useRef<SimulationFrame[]>([]);
  const jointNamesRef = useRef<string[]>([]);

  // Store callbacks in refs to avoid recreating connect on every render
  const onFrameRef = useRef(options.onFrame);
  const onCompleteRef = useRef(options.onComplete);
  const onErrorRef = useRef(options.onError);

  // Update refs when callbacks change
  useEffect(() => {
    onFrameRef.current = options.onFrame;
    onCompleteRef.current = options.onComplete;
    onErrorRef.current = options.onError;
  }, [options.onFrame, options.onComplete, options.onError]);

  const [state, setState] = useState<StreamState>({
    status: 'disconnected',
    progress: 0,
    totalFrames: 0,
    error: null,
    jointNames: [],
    rotationPeriod: null,
  });

  const getWebSocketUrl = useCallback((linkageId: string, fast = false) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const endpoint = fast ? 'simulation-fast' : 'simulation';
    return `${protocol}//${host}/api/ws/${endpoint}/${linkageId}`;
  }, []);

  const connect = useCallback(
    (linkageId: string, fast = false) => {
      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      framesRef.current = [];
      setState((s) => ({
        ...s,
        status: 'connecting',
        progress: 0,
        error: null,
      }));

      const ws = new WebSocket(getWebSocketUrl(linkageId, fast));
      wsRef.current = ws;

      ws.onopen = () => {
        setState((s) => ({ ...s, status: 'connected' }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          switch (data.type) {
            case 'ready':
              jointNamesRef.current = data.joint_names || [];
              setState((s) => ({
                ...s,
                jointNames: data.joint_names,
                rotationPeriod: data.rotation_period,
              }));
              break;

            case 'frame': {
              const frame: SimulationFrame = {
                step: data.step,
                positions: data.positions.map(
                  ([x, y]: [number, number]): Position => ({ x, y })
                ),
              };
              framesRef.current.push(frame);
              setState((s) => ({
                ...s,
                progress: data.step + 1,
              }));
              onFrameRef.current?.(frame);
              break;
            }

            case 'frames': {
              // Batch frames from fast endpoint
              const frames: SimulationFrame[] = data.frames.map(
                (positions: number[][], step: number): SimulationFrame => ({
                  step,
                  positions: positions.map(
                    (pos: number[]): Position => ({ x: pos[0], y: pos[1] })
                  ),
                })
              );
              framesRef.current = frames;
              // Fast endpoint may include joint_names in the frames message
              if (data.joint_names) {
                jointNamesRef.current = data.joint_names;
              }
              setState((s) => ({
                ...s,
                progress: data.total_frames,
                totalFrames: data.total_frames,
              }));
              onCompleteRef.current?.(frames, jointNamesRef.current);
              break;
            }

            case 'progress':
              setState((s) => ({
                ...s,
                progress: data.current,
                totalFrames: data.total,
              }));
              break;

            case 'complete':
              setState((s) => ({
                ...s,
                progress: data.total_frames,
                totalFrames: data.total_frames,
              }));
              onCompleteRef.current?.(framesRef.current, jointNamesRef.current);
              break;

            case 'error':
              setState((s) => ({
                ...s,
                status: 'error',
                error: data.message,
              }));
              onErrorRef.current?.(data.message);
              break;
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onerror = () => {
        setState((s) => ({
          ...s,
          status: 'error',
          error: 'WebSocket connection error',
        }));
      };

      ws.onclose = () => {
        setState((s) => ({
          ...s,
          status: 'disconnected',
        }));
        wsRef.current = null;
      };
    },
    [getWebSocketUrl]
  );

  const startStreaming = useCallback(
    (iterations?: number, fps = 30) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            action: 'start',
            iterations,
            fps,
          })
        );
      }
    },
    []
  );

  const stopStreaming = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'close' }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    setState((s) => ({ ...s, status: 'disconnected' }));
  }, []);

  const getFrames = useCallback(() => framesRef.current, []);

  return {
    ...state,
    connect,
    startStreaming,
    stopStreaming,
    disconnect,
    getFrames,
    isConnected: state.status === 'connected',
    isStreaming: state.progress > 0 && state.progress < state.totalFrames,
  };
}
