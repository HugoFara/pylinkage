/**
 * List of joints with selection and quick actions.
 */

import { useEditorStore } from '../../stores/editorStore';
import { useLinkageStore } from '../../stores/linkageStore';
import { JOINT_COLORS } from '../../types/linkage';

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  empty: {
    color: '#8b949e',
    fontSize: '13px',
    fontStyle: 'italic',
    padding: '8px',
  },
  item: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 12px',
    borderRadius: '6px',
    background: '#21262d',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  itemSelected: {
    background: '#30363d',
    outline: '1px solid #58a6ff',
  },
  itemHover: {
    background: '#30363d',
  },
  typeBadge: {
    padding: '2px 6px',
    borderRadius: '4px',
    fontSize: '10px',
    fontWeight: 'bold',
    textTransform: 'uppercase' as const,
  },
  name: {
    flex: 1,
    fontSize: '13px',
    color: '#e6edf3',
  },
  coords: {
    fontSize: '11px',
    color: '#8b949e',
    fontFamily: 'monospace',
  },
  deleteBtn: {
    padding: '2px 6px',
    borderRadius: '4px',
    border: 'none',
    background: 'transparent',
    color: '#8b949e',
    cursor: 'pointer',
    fontSize: '12px',
  },
};

const TYPE_ABBREV: Record<string, string> = {
  Static: 'S',
  Crank: 'C',
  Fixed: 'F',
  Revolute: 'R',
  Prismatic: 'P',
};

export function JointList() {
  const linkage = useLinkageStore((s) => s.linkage);
  const deleteJoint = useLinkageStore((s) => s.deleteJoint);
  const selectedJointName = useEditorStore((s) => s.selectedJointName);
  const selectJoint = useEditorStore((s) => s.selectJoint);

  if (!linkage || linkage.joints.length === 0) {
    return <p style={styles.empty}>No joints. Load an example or add joints.</p>;
  }

  const handleDelete = (e: React.MouseEvent, name: string) => {
    e.stopPropagation();
    deleteJoint(name);
    if (selectedJointName === name) {
      selectJoint(null);
    }
  };

  return (
    <div style={styles.container}>
      {linkage.joints.map((joint) => {
        const isSelected = selectedJointName === joint.name;
        const color = JOINT_COLORS[joint.type];

        return (
          <div
            key={joint.name}
            style={{
              ...styles.item,
              ...(isSelected ? styles.itemSelected : {}),
            }}
            onClick={() => selectJoint(joint.name)}
          >
            <span
              style={{
                ...styles.typeBadge,
                background: color,
                color: '#0d1117',
              }}
            >
              {TYPE_ABBREV[joint.type] || joint.type.charAt(0)}
            </span>
            <span style={styles.name}>{joint.name}</span>
            <span style={styles.coords}>
              ({(joint.x ?? 0).toFixed(0)}, {(joint.y ?? 0).toFixed(0)})
            </span>
            <button
              style={styles.deleteBtn}
              onClick={(e) => handleDelete(e, joint.name)}
              title="Delete joint"
            >
              ✕
            </button>
          </div>
        );
      })}
    </div>
  );
}
