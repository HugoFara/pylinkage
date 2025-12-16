/**
 * Dialog for configuring link properties after drawing.
 */

import { useState, useEffect } from 'react';
import { useEditorStore } from '../../stores/editorStore';
import {
  useMechanismStore,
  generateJointId,
  generateLinkId,
  calculateDistance,
} from '../../stores/mechanismStore';
import type { JointDict, LinkDict, LinkType } from '../../types/mechanism';

export function LinkPropertiesDialog() {
  const linkDialog = useEditorStore((s) => s.linkDialog);
  const closeLinkDialog = useEditorStore((s) => s.closeLinkDialog);

  const mechanism = useMechanismStore((s) => s.mechanism);
  const addLink = useMechanismStore((s) => s.addLink);

  // Form state
  const [linkType, setLinkType] = useState<LinkType>('link');
  const [linkName, setLinkName] = useState('');
  const [linkLength, setLinkLength] = useState(0);
  const [angularVelocity, setAngularVelocity] = useState(0.1);

  // Initialize form when dialog opens
  useEffect(() => {
    if (linkDialog.isOpen && linkDialog.tempLink) {
      const { startPoint, endPoint } = linkDialog.tempLink;
      const length = calculateDistance(
        startPoint.x,
        startPoint.y,
        endPoint.x,
        endPoint.y
      );
      setLinkLength(Math.round(length * 10) / 10);
      setLinkName(generateLinkId('link'));
      setLinkType('link');
      setAngularVelocity(0.1);
    }
  }, [linkDialog.isOpen, linkDialog.tempLink]);

  if (!linkDialog.isOpen || !linkDialog.tempLink) {
    return null;
  }

  const { startPoint, endPoint, startJointId, endJointId } = linkDialog.tempLink;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Create new joints if needed
    const newJoints: JointDict[] = [];
    let startJoint = startJointId;
    let endJoint = endJointId;

    // If no existing joint at start, create one
    if (!startJoint) {
      const newJoint: JointDict = {
        id: generateJointId('revolute'),
        type: 'revolute',
        position: [startPoint.x, startPoint.y],
      };
      newJoints.push(newJoint);
      startJoint = newJoint.id;
    }

    // If no existing joint at end, create one
    if (!endJoint) {
      const newJoint: JointDict = {
        id: generateJointId('revolute'),
        type: 'revolute',
        position: [endPoint.x, endPoint.y],
      };
      newJoints.push(newJoint);
      endJoint = newJoint.id;
    }

    // Create the link
    const newLink: LinkDict = {
      id: linkName,
      type: linkType,
      joints: [startJoint, endJoint],
    };

    // Add driver-specific properties
    if (linkType === 'driver') {
      newLink.angular_velocity = angularVelocity;
      newLink.initial_angle = 0;
      // Use start joint as motor if it's a ground joint
      const startIsGround = mechanism?.joints.find(
        (j) => j.id === startJoint && j.type === 'ground'
      );
      if (startIsGround) {
        newLink.motor_joint = startJoint;
      }
    }

    addLink(newLink, newJoints);
    closeLinkDialog();
  };

  const handleCancel = () => {
    closeLinkDialog();
  };

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) handleCancel();
      }}
    >
      <div
        style={{
          backgroundColor: '#161b22',
          borderRadius: '8px',
          padding: '24px',
          minWidth: '320px',
          border: '1px solid #30363d',
        }}
      >
        <h3 style={{ margin: '0 0 16px 0', color: '#c9d1d9' }}>
          New Link
        </h3>

        <form onSubmit={handleSubmit}>
          {/* Link Name */}
          <div style={{ marginBottom: '16px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '4px',
                color: '#8b949e',
                fontSize: '14px',
              }}
            >
              Name
            </label>
            <input
              type="text"
              value={linkName}
              onChange={(e) => setLinkName(e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '4px',
                color: '#c9d1d9',
                fontSize: '14px',
                boxSizing: 'border-box',
              }}
            />
          </div>

          {/* Link Length */}
          <div style={{ marginBottom: '16px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '4px',
                color: '#8b949e',
                fontSize: '14px',
              }}
            >
              Length
            </label>
            <input
              type="number"
              value={linkLength}
              onChange={(e) => setLinkLength(parseFloat(e.target.value) || 0)}
              step="0.1"
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '4px',
                color: '#c9d1d9',
                fontSize: '14px',
                boxSizing: 'border-box',
              }}
            />
          </div>

          {/* Link Type */}
          <div style={{ marginBottom: '16px' }}>
            <label
              style={{
                display: 'block',
                marginBottom: '4px',
                color: '#8b949e',
                fontSize: '14px',
              }}
            >
              Type
            </label>
            <select
              value={linkType}
              onChange={(e) => setLinkType(e.target.value as LinkType)}
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#0d1117',
                border: '1px solid #30363d',
                borderRadius: '4px',
                color: '#c9d1d9',
                fontSize: '14px',
                boxSizing: 'border-box',
              }}
            >
              <option value="link">Regular Link</option>
              <option value="driver">Driver (Motor)</option>
              <option value="ground">Ground Link</option>
            </select>
          </div>

          {/* Driver-specific: Angular Velocity */}
          {linkType === 'driver' && (
            <div style={{ marginBottom: '16px' }}>
              <label
                style={{
                  display: 'block',
                  marginBottom: '4px',
                  color: '#8b949e',
                  fontSize: '14px',
                }}
              >
                Angular Velocity (rad/step)
              </label>
              <input
                type="number"
                value={angularVelocity}
                onChange={(e) =>
                  setAngularVelocity(parseFloat(e.target.value) || 0.1)
                }
                step="0.01"
                style={{
                  width: '100%',
                  padding: '8px',
                  backgroundColor: '#0d1117',
                  border: '1px solid #30363d',
                  borderRadius: '4px',
                  color: '#c9d1d9',
                  fontSize: '14px',
                  boxSizing: 'border-box',
                }}
              />
            </div>
          )}

          {/* Snapped joints info */}
          <div
            style={{
              marginBottom: '16px',
              padding: '8px',
              backgroundColor: '#0d1117',
              borderRadius: '4px',
              fontSize: '12px',
              color: '#8b949e',
            }}
          >
            <div>
              Start:{' '}
              {startJointId ? (
                <span style={{ color: '#58a6ff' }}>{startJointId}</span>
              ) : (
                <span style={{ color: '#3fb950' }}>New joint</span>
              )}
            </div>
            <div>
              End:{' '}
              {endJointId ? (
                <span style={{ color: '#58a6ff' }}>{endJointId}</span>
              ) : (
                <span style={{ color: '#3fb950' }}>New joint</span>
              )}
            </div>
          </div>

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
            <button
              type="button"
              onClick={handleCancel}
              style={{
                padding: '8px 16px',
                backgroundColor: 'transparent',
                border: '1px solid #30363d',
                borderRadius: '4px',
                color: '#c9d1d9',
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              style={{
                padding: '8px 16px',
                backgroundColor: '#238636',
                border: 'none',
                borderRadius: '4px',
                color: '#ffffff',
                cursor: 'pointer',
              }}
            >
              Create Link
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
