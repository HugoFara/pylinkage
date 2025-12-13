/**
 * Animation playback controls.
 */

import { useEffect, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useEditorStore } from '../../stores/editorStore';
import { useLinkageStore } from '../../stores/linkageStore';
import { simulationApi } from '../../api/client';

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

  const linkage = useLinkageStore((s) => s.linkage);
  const loci = useLinkageStore((s) => s.loci);
  const setLoci = useLinkageStore((s) => s.setLoci);

  const animationRef = useRef<number>();

  // Simulate mutation
  const simulateMutation = useMutation({
    mutationFn: async () => {
      if (!linkage) throw new Error('No linkage loaded');
      return simulationApi.simulate(linkage.id);
    },
    onSuccess: (result) => {
      if (result.is_complete) {
        setLoci(result.frames);
        setAnimationFrame(0);
      }
    },
  });

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

  const totalFrames = loci?.length ?? 0;
  const canAnimate = linkage && loci && loci.length > 0;

  return (
    <div style={styles.container}>
      {/* Simulate button */}
      <button
        style={{
          ...styles.button,
          ...styles.simulateButton,
          ...((!linkage || simulateMutation.isPending) ? styles.buttonDisabled : {}),
        }}
        onClick={() => simulateMutation.mutate()}
        disabled={!linkage || simulateMutation.isPending}
      >
        {simulateMutation.isPending ? 'Simulating...' : 'Run Simulation'}
      </button>

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

      {/* Error display */}
      {simulateMutation.error && (
        <p style={{ color: '#f85149', fontSize: '12px' }}>
          Error: {(simulateMutation.error as Error).message}
        </p>
      )}
    </div>
  );
}
