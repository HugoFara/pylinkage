/**
 * Link list component for sidebar.
 * Shows all links in the mechanism with their properties.
 */

import { useEditorStore } from '../../stores/editorStore';
import { useMechanismStore, calculateDistance } from '../../stores/mechanismStore';
import { LINK_COLORS, type LinkDict } from '../../types/mechanism';

export function LinkList() {
  const selectedLinkId = useEditorStore((s) => s.selectedLinkId);
  const selectLink = useEditorStore((s) => s.selectLink);

  const mechanism = useMechanismStore((s) => s.mechanism);
  const deleteLink = useMechanismStore((s) => s.deleteLink);
  const getJoint = useMechanismStore((s) => s.getJoint);

  if (!mechanism || mechanism.links.length === 0) {
    return (
      <div style={{ color: '#8b949e', fontSize: '14px', padding: '8px 0' }}>
        No links. Use Draw Link mode to create links.
      </div>
    );
  }

  // Calculate link length from joint positions
  const getLinkLength = (link: LinkDict): number => {
    if (link.joints.length < 2) return 0;

    const joint1 = getJoint(link.joints[0]);
    const joint2 = getJoint(link.joints[1]);

    if (!joint1 || !joint2) return 0;

    const x1 = joint1.position[0] ?? 0;
    const y1 = joint1.position[1] ?? 0;
    const x2 = joint2.position[0] ?? 0;
    const y2 = joint2.position[1] ?? 0;

    return calculateDistance(x1, y1, x2, y2);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {mechanism.links.map((link) => {
        const isSelected = selectedLinkId === link.id;
        const color = LINK_COLORS[link.type];
        const length = getLinkLength(link);

        return (
          <div
            key={link.id}
            onClick={() => selectLink(link.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px',
              backgroundColor: isSelected ? '#21262d' : 'transparent',
              borderRadius: '4px',
              cursor: 'pointer',
              border: isSelected ? '1px solid #30363d' : '1px solid transparent',
            }}
          >
            {/* Type badge */}
            <div
              style={{
                width: '24px',
                height: '4px',
                backgroundColor: color,
                borderRadius: '2px',
              }}
            />

            {/* Link info */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: '14px',
                  color: '#c9d1d9',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
              >
                {link.name || link.id}
              </div>
              <div style={{ fontSize: '12px', color: '#8b949e' }}>
                {link.type} · {length.toFixed(1)} px
              </div>
            </div>

            {/* Connected joints */}
            <div
              style={{
                display: 'flex',
                gap: '4px',
                flexWrap: 'wrap',
              }}
            >
              {link.joints.map((jointId) => (
                <span
                  key={jointId}
                  style={{
                    fontSize: '10px',
                    padding: '2px 4px',
                    backgroundColor: '#30363d',
                    borderRadius: '2px',
                    color: '#8b949e',
                  }}
                >
                  {jointId}
                </span>
              ))}
            </div>

            {/* Delete button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                deleteLink(link.id);
                if (isSelected) {
                  selectLink(null);
                }
              }}
              style={{
                padding: '4px',
                backgroundColor: 'transparent',
                border: 'none',
                color: '#8b949e',
                cursor: 'pointer',
                fontSize: '14px',
                lineHeight: 1,
              }}
              title="Delete link"
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}
