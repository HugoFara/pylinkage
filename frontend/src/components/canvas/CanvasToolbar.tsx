/**
 * Toolbar for selecting editor modes.
 * Updated with collapsible sections for drivers and joints.
 */

import { useState } from 'react';
import { useEditorStore } from '../../stores/editorStore';
import type { EditorMode } from '../../types/mechanism';

interface ToolButton {
  mode: EditorMode;
  label: string;
  icon: string;
  shortcut?: string;
}

interface ToolGroup {
  id: string;
  label: string;
  icon: string;
  tools: ToolButton[];
}

// Primary tools - always visible
const primaryTools: ToolButton[] = [
  {
    mode: 'draw-link',
    label: 'Link',
    icon: '🔗',
    shortcut: '1',
  },
];

// Joint tools - collapsible (ground is a link type, not joint)
const jointTools: ToolGroup = {
  id: 'joints',
  label: 'Joint',
  icon: '⚪',
  tools: [
    {
      mode: 'place-tracker-joint',
      label: 'Tracker',
      icon: '🟣',
    },
    {
      mode: 'place-revolute-joint',
      label: 'Revolute',
      icon: '🔵',
    },
    {
      mode: 'place-prismatic-joint',
      label: 'Prismatic',
      icon: '🟢',
    },
  ],
};

// Driver tools - collapsible
const driverTools: ToolGroup = {
  id: 'drivers',
  label: 'Driver',
  icon: '⚙️',
  tools: [
    {
      mode: 'place-crank',
      label: 'Crank',
      icon: '🔄',
    },
    {
      mode: 'place-arccrank',
      label: 'Arc Crank',
      icon: '↩️',
    },
    {
      mode: 'place-linear',
      label: 'Linear',
      icon: '↔️',
    },
  ],
};

// Secondary tools - always visible
const secondaryTools: ToolButton[] = [
  {
    mode: 'select',
    label: 'Select',
    icon: '👆',
    shortcut: '2',
  },
  {
    mode: 'move-joint',
    label: 'Move',
    icon: '✋',
    shortcut: '3',
  },
  {
    mode: 'delete',
    label: 'Delete',
    icon: '🗑️',
    shortcut: '4',
  },
];

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  divider: {
    height: '1px',
    background: '#30363d',
    margin: '8px 0',
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
    width: '100%',
  },
  buttonActive: {
    background: '#1f6feb',
    borderColor: '#1f6feb',
  },
  groupButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
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
    width: '100%',
  },
  groupButtonExpanded: {
    borderBottomLeftRadius: '0',
    borderBottomRightRadius: '0',
    borderBottom: 'none',
    background: '#30363d',
  },
  groupButtonActive: {
    background: '#1f6feb',
    borderColor: '#1f6feb',
  },
  subToolsContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    paddingLeft: '12px',
    paddingTop: '2px',
    paddingBottom: '4px',
    marginBottom: '4px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: '#30363d',
    borderTop: 'none',
    borderBottomLeftRadius: '6px',
    borderBottomRightRadius: '6px',
    background: '#161b22',
  },
  subButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 10px',
    border: 'none',
    borderRadius: '4px',
    background: 'transparent',
    color: '#8b949e',
    cursor: 'pointer',
    fontSize: '12px',
    textAlign: 'left' as const,
    transition: 'all 0.15s',
    width: 'calc(100% - 12px)',
  },
  subButtonActive: {
    background: '#1f6feb',
    color: '#ffffff',
  },
  subButtonHover: {
    background: '#30363d',
    color: '#e6edf3',
  },
  icon: {
    width: '20px',
    textAlign: 'center' as const,
  },
  expandIcon: {
    fontSize: '10px',
    transition: 'transform 0.15s',
  },
  labelWithShortcut: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    flex: 1,
  },
  shortcut: {
    color: '#8b949e',
    fontSize: '11px',
  },
};

function ToolButton({
  tool,
  isActive,
  onClick,
  style,
}: {
  tool: ToolButton;
  isActive: boolean;
  onClick: () => void;
  style?: React.CSSProperties;
}) {
  return (
    <button
      style={{
        ...styles.button,
        ...(isActive ? styles.buttonActive : {}),
        ...style,
      }}
      onClick={onClick}
      title={tool.shortcut ? `${tool.label} (${tool.shortcut})` : tool.label}
    >
      <span style={styles.icon}>{tool.icon}</span>
      <span style={styles.labelWithShortcut}>
        {tool.label}
        {tool.shortcut && <span style={styles.shortcut}>({tool.shortcut})</span>}
      </span>
    </button>
  );
}

function CollapsibleToolGroup({
  group,
  currentMode,
  onModeChange,
}: {
  group: ToolGroup;
  currentMode: EditorMode;
  onModeChange: (mode: EditorMode) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const isGroupActive = group.tools.some((tool) => tool.mode === currentMode);
  const activeSubTool = group.tools.find((tool) => tool.mode === currentMode);

  const handleGroupClick = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div style={styles.section}>
      <button
        style={{
          ...styles.groupButton,
          ...(isExpanded ? styles.groupButtonExpanded : {}),
          ...(isGroupActive && !isExpanded ? styles.groupButtonActive : {}),
        }}
        onClick={handleGroupClick}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={styles.icon}>{group.icon}</span>
          {activeSubTool ? activeSubTool.label : group.label}
        </span>
        <span
          style={{
            ...styles.expandIcon,
            transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
        >
          ▼
        </span>
      </button>
      {isExpanded && (
        <div style={styles.subToolsContainer}>
          {group.tools.map((tool) => (
            <button
              key={tool.mode}
              style={{
                ...styles.subButton,
                ...(currentMode === tool.mode ? styles.subButtonActive : {}),
              }}
              onClick={() => {
                onModeChange(tool.mode);
                setIsExpanded(false);
              }}
            >
              <span style={styles.icon}>{tool.icon}</span>
              {tool.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function CanvasToolbar() {
  const mode = useEditorStore((s) => s.mode);
  const setMode = useEditorStore((s) => s.setMode);

  return (
    <div style={styles.container}>
      {/* Primary: Draw Link (selected by default) */}
      {primaryTools.map((tool) => (
        <ToolButton
          key={tool.mode}
          tool={tool}
          isActive={mode === tool.mode}
          onClick={() => setMode(tool.mode)}
        />
      ))}

      {/* Collapsible: Joint types */}
      <CollapsibleToolGroup
        group={jointTools}
        currentMode={mode}
        onModeChange={setMode}
      />

      {/* Collapsible: Driver types */}
      <CollapsibleToolGroup
        group={driverTools}
        currentMode={mode}
        onModeChange={setMode}
      />

      <div style={styles.divider} />

      {/* Secondary tools */}
      {secondaryTools.map((tool) => (
        <ToolButton
          key={tool.mode}
          tool={tool}
          isActive={mode === tool.mode}
          onClick={() => setMode(tool.mode)}
        />
      ))}
    </div>
  );
}
