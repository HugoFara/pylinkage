/**
 * Animation playback controls with REST and WebSocket simulation options.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useEditorStore } from '../../stores/editorStore';
import { useMechanismStore } from '../../stores/mechanismStore';
import { simulationApi } from '../../api/client';
import { useSimulationStream } from '../../hooks/useSimulationStream';
import type { SimulationFrame } from '../../types/mechanism';

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  controls: {
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
  },
  button: {
    padding: '8px 16px',
    borderRadius: '6px',
    border: 'none',
    fontSize: '13px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  playButton: {
    background: '#238636',
    color: 'white',
  },
  stopButton: {
    background: '#da3633',
    color: 'white',
  },
  simulateButton: {
    background: '#1f6feb',
    color: 'white',
  },
  streamButton: {
    background: '#8957e5',
    color: 'white',
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
  slider: {
    flex: 1,
    accentColor: '#58a6ff',
  },
  frameInfo: {
    fontSize: '12px',
    color: '#8b949e',
    fontFamily: 'monospace',
  },
  viewOptions: {
    display: 'flex',
    gap: '12px',
    flexWrap: 'wrap' as const,
  },
  checkbox: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    fontSize: '12px',
    color: '#8b949e',
    cursor: 'pointer',
  },
  progressBar: {
    height: '4px',
    background: '#21262d',
    borderRadius: '2px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    background: '#58a6ff',
    transition: 'width 0.1s',
  },
  keyboardHint: {
    fontSize: '11px',
    color: '#6e7681',
    marginTop: '4px',
  },
};

export function AnimationControls() {
  const isAnimating = useEditorStore((s) => s.isAnimating);
  const setAnimating = useEditorStore((s) => s.setAnimating);
  const animationFrame = useEditorStore((s) => s.animationFrame);
  const setAnimationFrame = useEditorStore((s) => s.setAnimationFrame);
  const showLoci = useEditorStore((s) => s.showLoci);
  const showGrid = useEditorStore((s) => s.showGrid);
  const toggleLoci = useEditorStore((s) => s.toggleLoci);
  const toggleGrid = useEditorStore((s) => s.toggleGrid);

  const mechanism = useMechanismStore((s) => s.mechanism);
  const loci = useMechanismStore((s) => s.loci);
  const setLoci = useMechanismStore((s) => s.setLoci);
  const updateBuildableStatus = useMechanismStore((s) => s.updateBuildableStatus);

  const animationRef = useRef<number>();
  const [useStreaming, setUseStreaming] = useState(false);

  // Callback for when streaming completes
  const handleStreamComplete = useCallback((frames: SimulationFrame[], jointNames: string[]) => {
    setLoci(frames, jointNames);
    setAnimationFrame(0);
  }, [setLoci, setAnimationFrame]);

  // WebSocket streaming hook
  const stream = useSimulationStream({
    onComplete: handleStreamComplete,
  });

  // REST API simulation mutation
  const simulateMutation = useMutation({
    mutationFn: async () => {
      if (!mechanism) throw new Error('No mechanism loaded');

      // Use direct simulation for local mechanisms (not saved to backend)
      if (mechanism.id.startsWith('local-')) {
        return simulationApi.simulateDirect({
          name: mechanism.name,
          joints: mechanism.joints,
          links: mechanism.links,
          ground: mechanism.ground,
        });
      }

      return simulationApi.simulate(mechanism.id);
    },
    onSuccess: (result) => {
      // Update buildable status based on simulation result
      updateBuildableStatus(result.is_complete, result.error);

      if (result.is_complete) {
        setLoci(result.frames, result.joint_names);
        setAnimationFrame(0);
      }
    },
  });

  // Handle simulation (REST or WebSocket)
  const handleSimulate = useCallback(() => {
    if (!mechanism) return;

    if (useStreaming) {
      // Use WebSocket streaming
      stream.connect(mechanism.id, true); // Fast mode
      // Wait for connection, then start
      setTimeout(() => {
        stream.startStreaming();
      }, 100);
    } else {
      // Use REST API
      simulateMutation.mutate();
    }
  }, [mechanism, useStreaming, stream, simulateMutation]);

  // Animation loop
  useEffect(() => {
    if (!isAnimating || !loci || loci.length === 0) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    let lastTime = 0;
    const FPS = 30;
    const frameInterval = 1000 / FPS;

    const animate = (time: number) => {
      if (time - lastTime >= frameInterval) {
        setAnimationFrame((animationFrame + 1) % loci.length);
        lastTime = time;
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isAnimating, loci, animationFrame, setAnimationFrame]);

  // Store disconnect in a ref to avoid dependency issues
  const disconnectRef = useRef(stream.disconnect);
  disconnectRef.current = stream.disconnect;

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      disconnectRef.current();
    };
  }, []);

  const totalFrames = loci?.length ?? 0;
  const canAnimate = mechanism && loci && loci.length > 0;
  const isLoading = simulateMutation.isPending || stream.isStreaming;
  const streamProgress = stream.totalFrames > 0
    ? (stream.progress / stream.totalFrames) * 100
    : 0;

  return (
    <div style={styles.container}>
      {/* Simulate buttons */}
      <div style={styles.controls}>
        <button
          style={{
            ...styles.button,
            ...(useStreaming ? styles.streamButton : styles.simulateButton),
            ...((!mechanism || isLoading) ? styles.buttonDisabled : {}),
          }}
          onClick={handleSimulate}
          disabled={!mechanism || isLoading}
        >
          {isLoading
            ? useStreaming
              ? 'Streaming...'
              : 'Simulating...'
            : 'Run Simulation'}
        </button>
        <label style={styles.checkbox} title="Use WebSocket streaming (experimental)">
          <input
            type="checkbox"
            checked={useStreaming}
            onChange={(e) => setUseStreaming(e.target.checked)}
          />
          Stream
        </label>
      </div>

      {/* Streaming progress bar */}
      {stream.isStreaming && (
        <div style={styles.progressBar}>
          <div
            style={{
              ...styles.progressFill,
              width: `${streamProgress}%`,
            }}
          />
        </div>
      )}

      {/* Play controls */}
      <div style={styles.controls}>
        <button
          style={{
            ...styles.button,
            ...(isAnimating ? styles.stopButton : styles.playButton),
            ...(!canAnimate ? styles.buttonDisabled : {}),
          }}
          onClick={() => setAnimating(!isAnimating)}
          disabled={!canAnimate}
        >
          {isAnimating ? 'Stop' : 'Play'}
        </button>

        {/* Frame slider */}
        {canAnimate && (
          <>
            <input
              type="range"
              style={styles.slider}
              min={0}
              max={totalFrames - 1}
              value={animationFrame}
              onChange={(e) => {
                setAnimating(false);
                setAnimationFrame(parseInt(e.target.value));
              }}
            />
            <span style={styles.frameInfo}>
              {animationFrame + 1}/{totalFrames}
            </span>
          </>
        )}
      </div>

      {/* View options */}
      <div style={styles.viewOptions}>
        <label style={styles.checkbox}>
          <input
            type="checkbox"
            checked={showLoci}
            onChange={toggleLoci}
          />
          Show Paths
        </label>
        <label style={styles.checkbox}>
          <input
            type="checkbox"
            checked={showGrid}
            onChange={toggleGrid}
          />
          Show Grid
        </label>
      </div>

      {/* Keyboard hints */}
      <div style={styles.keyboardHint}>
        Space: Play/Stop | Ctrl+Z: Undo | Del: Delete selected
      </div>

      {/* Error display */}
      {mechanism?.error && (
        <p style={{ color: '#f85149', fontSize: '12px', margin: 0 }}>
          {mechanism.error}
        </p>
      )}
      {simulateMutation.error && (
        <p style={{ color: '#f85149', fontSize: '12px', margin: 0 }}>
          Error: {(simulateMutation.error as Error).message}
        </p>
      )}
      {stream.error && (
        <p style={{ color: '#f85149', fontSize: '12px', margin: 0 }}>
          Stream error: {stream.error}
        </p>
      )}
    </div>
  );
}
