/**
 * Toolbar for selecting editor modes.
 */

import { useEditorStore } from '../../stores/editorStore';
import type { EditorMode } from '../../types/linkage';

interface ToolButton {
  mode: EditorMode;
  label: string;
  icon: string;
  description: string;
}

const tools: ToolButton[] = [
  { mode: 'select', label: '1: Select', icon: '👆', description: 'Click to select joints (1)' },
  { mode: 'add-joint', label: '2: Add Joint', icon: '➕', description: 'Click canvas to add joint (2)' },
  { mode: 'draw-link', label: '3: Draw Link', icon: '🔗', description: 'Click two joints to create link (3)' },
  { mode: 'move-joint', label: '4: Move', icon: '✋', description: 'Drag joints to move (4)' },
  { mode: 'delete', label: '5: Delete', icon: '🗑️', description: 'Click to delete (5)' },
  { mode: 'set-ground', label: '6: Ground', icon: '📍', description: 'Click joint to make ground (6)' },
  { mode: 'set-crank', label: '7: Crank', icon: '⚙️', description: 'Click joint to make crank (7)' },
];

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  button: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: '#30363d',
    borderRadius: '6px',
    background: '#21262d',
    color: '#e6edf3',
    cursor: 'pointer',
    fontSize: '13px',
    textAlign: 'left' as const,
    transition: 'all 0.15s',
  },
  buttonActive: {
    background: '#1f6feb',
    borderColor: '#1f6feb',
  },
  buttonHover: {
    background: '#30363d',
    borderColor: '#8b949e',
  },
  icon: {
    width: '20px',
    textAlign: 'center' as const,
  },
};

export function CanvasToolbar() {
  const mode = useEditorStore((s) => s.mode);
  const setMode = useEditorStore((s) => s.setMode);

  return (
    <div style={styles.container}>
      {tools.map((tool) => (
        <button
          key={tool.mode}
          style={{
            ...styles.button,
            ...(mode === tool.mode ? styles.buttonActive : {}),
          }}
          onClick={() => setMode(tool.mode)}
          title={tool.description}
        >
          <span style={styles.icon}>{tool.icon}</span>
          {tool.label}
        </button>
      ))}
    </div>
  );
}
