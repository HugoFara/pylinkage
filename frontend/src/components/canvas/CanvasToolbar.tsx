/**
 * Toolbar for selecting editor modes.
 * Updated for link-first approach.
 */

import { useEditorStore } from '../../stores/editorStore';
import type { EditorMode } from '../../types/mechanism';

interface ToolButton {
  mode: EditorMode;
  label: string;
  icon: string;
  description: string;
}

const tools: ToolButton[] = [
  {
    mode: 'select',
    label: '1: Select',
    icon: '👆',
    description: 'Click to select links/joints (1)',
  },
  {
    mode: 'draw-link',
    label: '2: Draw Link',
    icon: '🔗',
    description: 'Click-drag to draw link (2)',
  },
  {
    mode: 'move-joint',
    label: '3: Move',
    icon: '✋',
    description: 'Drag joints to move (3)',
  },
  {
    mode: 'delete',
    label: '4: Delete',
    icon: '🗑️',
    description: 'Click to delete (4)',
  },
  {
    mode: 'set-driver',
    label: '5: Driver',
    icon: '⚙️',
    description: 'Click link to make driver (5)',
  },
  {
    mode: 'set-ground',
    label: '6: Ground',
    icon: '📍',
    description: 'Click joint to make ground (6)',
  },
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
