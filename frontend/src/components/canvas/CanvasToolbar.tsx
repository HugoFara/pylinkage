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
  { mode: 'select', label: 'Select', icon: '👆', description: 'Click to select joints' },
  { mode: 'add-joint', label: 'Add Joint', icon: '➕', description: 'Click canvas to add joint' },
  { mode: 'move-joint', label: 'Move', icon: '✋', description: 'Drag joints to move' },
  { mode: 'delete', label: 'Delete', icon: '🗑️', description: 'Click to delete' },
  { mode: 'set-ground', label: 'Set Ground', icon: '📍', description: 'Click joint to make ground' },
  { mode: 'set-crank', label: 'Set Crank', icon: '⚙️', description: 'Click joint to make crank' },
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
    border: '1px solid #30363d',
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
