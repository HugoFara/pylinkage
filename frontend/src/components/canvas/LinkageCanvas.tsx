/**
 * Main Konva canvas for linkage visualization and interaction.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Stage, Layer, Line, Circle, Text, Group } from 'react-konva';
import type Konva from 'konva';
import { useEditorStore } from '../../stores/editorStore';
import { useLinkageStore, generateJointName } from '../../stores/linkageStore';
import { JOINT_COLORS, type JointDict, type Position } from '../../types/linkage';

// Canvas configuration
const JOINT_RADIUS = 8;
const LINK_WIDTH = 4;
const GRID_SIZE = 50;

export function LinkageCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Store state
  const mode = useEditorStore((s) => s.mode);
  const selectedJointName = useEditorStore((s) => s.selectedJointName);
  const selectJoint = useEditorStore((s) => s.selectJoint);
  const hoveredJointName = useEditorStore((s) => s.hoveredJointName);
  const setHoveredJoint = useEditorStore((s) => s.setHoveredJoint);
  const showGrid = useEditorStore((s) => s.showGrid);
  const showLoci = useEditorStore((s) => s.showLoci);
  const animationFrame = useEditorStore((s) => s.animationFrame);

  const linkage = useLinkageStore((s) => s.linkage);
  const loci = useLinkageStore((s) => s.loci);
  const addJoint = useLinkageStore((s) => s.addJoint);
  const updateJoint = useLinkageStore((s) => s.updateJoint);
  const deleteJoint = useLinkageStore((s) => s.deleteJoint);

  // Handle window resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Get joint positions (from loci if animating, else from joint data)
  const getJointPosition = useCallback(
    (joint: JointDict, index: number): Position => {
      if (loci && loci.length > 0 && animationFrame < loci.length) {
        return loci[animationFrame].positions[index];
      }
      return { x: joint.x ?? 0, y: joint.y ?? 0 };
    },
    [loci, animationFrame]
  );

  // Handle stage click
  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;

    const pos = stage.getPointerPosition();
    if (!pos) return;

    // Convert to canvas coordinates (centered)
    const canvasX = pos.x - dimensions.width / 2;
    const canvasY = dimensions.height / 2 - pos.y; // Flip Y

    // Check if clicked on empty space
    if (e.target === stage) {
      if (mode === 'add-joint') {
        // Add new Static joint
        const name = generateJointName('Static');
        addJoint({
          type: 'Static',
          name,
          x: canvasX,
          y: canvasY,
        });
        selectJoint(name);
      } else if (mode === 'select') {
        selectJoint(null);
      }
    }
  };

  // Handle joint click
  const handleJointClick = (joint: JointDict) => {
    if (mode === 'delete') {
      deleteJoint(joint.name);
      if (selectedJointName === joint.name) {
        selectJoint(null);
      }
    } else if (mode === 'set-ground') {
      updateJoint(joint.name, { type: 'Static' });
    } else if (mode === 'set-crank') {
      // Find a ground joint to reference
      const groundJoint = linkage?.joints.find((j) => j.type === 'Static');
      if (groundJoint && groundJoint.name !== joint.name) {
        const dx = (joint.x ?? 0) - (groundJoint.x ?? 0);
        const dy = (joint.y ?? 0) - (groundJoint.y ?? 0);
        const distance = Math.sqrt(dx * dx + dy * dy);
        const angle = Math.atan2(dy, dx);

        updateJoint(joint.name, {
          type: 'Crank',
          joint0: { ref: groundJoint.name },
          distance,
          angle,
        });
      }
    } else {
      selectJoint(joint.name);
    }
  };

  // Handle joint drag
  const handleJointDrag = (joint: JointDict, newPos: Position) => {
    if (joint.type === 'Static') {
      updateJoint(joint.name, { x: newPos.x, y: newPos.y });
    }
    // For other joint types, we might need to update constraints
  };

  // Render grid
  const renderGrid = () => {
    if (!showGrid) return null;

    const lines: JSX.Element[] = [];
    const { width, height } = dimensions;

    // Vertical lines
    for (let x = 0; x <= width; x += GRID_SIZE) {
      lines.push(
        <Line
          key={`v-${x}`}
          points={[x, 0, x, height]}
          stroke="#21262d"
          strokeWidth={1}
        />
      );
    }

    // Horizontal lines
    for (let y = 0; y <= height; y += GRID_SIZE) {
      lines.push(
        <Line
          key={`h-${y}`}
          points={[0, y, width, y]}
          stroke="#21262d"
          strokeWidth={1}
        />
      );
    }

    // Center axes
    lines.push(
      <Line
        key="x-axis"
        points={[0, height / 2, width, height / 2]}
        stroke="#30363d"
        strokeWidth={2}
      />
    );
    lines.push(
      <Line
        key="y-axis"
        points={[width / 2, 0, width / 2, height]}
        stroke="#30363d"
        strokeWidth={2}
      />
    );

    return <Group>{lines}</Group>;
  };

  // Render loci (trajectory paths)
  const renderLoci = () => {
    if (!showLoci || !loci || loci.length < 2 || !linkage) return null;

    return linkage.joints.map((joint, jointIndex) => {
      const points: number[] = [];
      for (let frame = 0; frame < loci.length; frame++) {
        const pos = loci[frame].positions[jointIndex];
        const screenX = pos.x + dimensions.width / 2;
        const screenY = dimensions.height / 2 - pos.y;
        points.push(screenX, screenY);
      }

      return (
        <Line
          key={`loci-${joint.name}`}
          points={points}
          stroke={JOINT_COLORS[joint.type]}
          strokeWidth={1}
          opacity={0.5}
          dash={[4, 4]}
          closed
        />
      );
    });
  };

  // Render links (connections between joints)
  const renderLinks = () => {
    if (!linkage) return null;

    const links: JSX.Element[] = [];

    linkage.joints.forEach((joint, index) => {
      // Get parent references
      const parents: string[] = [];
      if (joint.joint0 && 'ref' in joint.joint0) parents.push(joint.joint0.ref);
      if (joint.joint1 && 'ref' in joint.joint1) parents.push(joint.joint1.ref);

      const jointPos = getJointPosition(joint, index);
      const screenX = jointPos.x + dimensions.width / 2;
      const screenY = dimensions.height / 2 - jointPos.y;

      parents.forEach((parentName) => {
        const parentIndex = linkage.joints.findIndex((j) => j.name === parentName);
        if (parentIndex === -1) return;

        const parentJoint = linkage.joints[parentIndex];
        const parentPos = getJointPosition(parentJoint, parentIndex);
        const parentScreenX = parentPos.x + dimensions.width / 2;
        const parentScreenY = dimensions.height / 2 - parentPos.y;

        links.push(
          <Line
            key={`link-${joint.name}-${parentName}`}
            points={[parentScreenX, parentScreenY, screenX, screenY]}
            stroke="#8b949e"
            strokeWidth={LINK_WIDTH}
            lineCap="round"
          />
        );
      });
    });

    return <Group>{links}</Group>;
  };

  // Render joints
  const renderJoints = () => {
    if (!linkage) return null;

    return linkage.joints.map((joint, index) => {
      const pos = getJointPosition(joint, index);
      const screenX = pos.x + dimensions.width / 2;
      const screenY = dimensions.height / 2 - pos.y;

      const isSelected = selectedJointName === joint.name;
      const isHovered = hoveredJointName === joint.name;
      const color = JOINT_COLORS[joint.type];

      return (
        <Group key={joint.name}>
          {/* Joint circle */}
          <Circle
            x={screenX}
            y={screenY}
            radius={JOINT_RADIUS}
            fill={color}
            stroke={isSelected ? '#ffffff' : isHovered ? '#c9d1d9' : '#0d1117'}
            strokeWidth={isSelected ? 3 : 2}
            onClick={() => handleJointClick(joint)}
            onTap={() => handleJointClick(joint)}
            onMouseEnter={() => setHoveredJoint(joint.name)}
            onMouseLeave={() => setHoveredJoint(null)}
            draggable={mode === 'move-joint' && joint.type === 'Static'}
            onDragEnd={(e) => {
              const newX = e.target.x() - dimensions.width / 2;
              const newY = dimensions.height / 2 - e.target.y();
              handleJointDrag(joint, { x: newX, y: newY });
            }}
            style={{ cursor: mode === 'delete' ? 'not-allowed' : 'pointer' }}
          />
          {/* Joint label */}
          <Text
            x={screenX + JOINT_RADIUS + 4}
            y={screenY - 6}
            text={joint.name}
            fontSize={12}
            fill="#8b949e"
          />
        </Group>
      );
    });
  };

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', overflow: 'hidden' }}
    >
      <Stage
        width={dimensions.width}
        height={dimensions.height}
        onClick={handleStageClick}
        onTap={handleStageClick}
      >
        <Layer>
          {renderGrid()}
          {renderLoci()}
          {renderLinks()}
          {renderJoints()}
        </Layer>
      </Stage>
    </div>
  );
}
