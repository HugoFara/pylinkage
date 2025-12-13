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
  const [mousePos, setMousePos] = useState<Position | null>(null);

  // Store state
  const mode = useEditorStore((s) => s.mode);
  const selectedJointName = useEditorStore((s) => s.selectedJointName);
  const selectJoint = useEditorStore((s) => s.selectJoint);
  const hoveredJointName = useEditorStore((s) => s.hoveredJointName);
  const setHoveredJoint = useEditorStore((s) => s.setHoveredJoint);
  const showGrid = useEditorStore((s) => s.showGrid);
  const showLoci = useEditorStore((s) => s.showLoci);
  const animationFrame = useEditorStore((s) => s.animationFrame);
  const linkStartJoint = useEditorStore((s) => s.linkStartJoint);
  const setLinkStartJoint = useEditorStore((s) => s.setLinkStartJoint);

  const linkage = useLinkageStore((s) => s.linkage);
  const loci = useLinkageStore((s) => s.loci);
  const addJoint = useLinkageStore((s) => s.addJoint);
  const updateJoint = useLinkageStore((s) => s.updateJoint);
  const deleteJoint = useLinkageStore((s) => s.deleteJoint);
  const getJoint = useLinkageStore((s) => s.getJoint);

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

  // Convert screen to canvas coordinates
  const screenToCanvas = useCallback(
    (screenX: number, screenY: number): Position => ({
      x: screenX - dimensions.width / 2,
      y: dimensions.height / 2 - screenY,
    }),
    [dimensions]
  );

  // Convert canvas to screen coordinates
  const canvasToScreen = useCallback(
    (canvasX: number, canvasY: number): Position => ({
      x: canvasX + dimensions.width / 2,
      y: dimensions.height / 2 - canvasY,
    }),
    [dimensions]
  );

  // Calculate distance between two points
  const calcDistance = (p1: Position, p2: Position): number => {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Calculate angle from p1 to p2
  const calcAngle = (p1: Position, p2: Position): number => {
    return Math.atan2(p2.y - p1.y, p2.x - p1.x);
  };

  // Handle mouse move on stage
  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;

    const pos = stage.getPointerPosition();
    if (!pos) return;

    setMousePos(screenToCanvas(pos.x, pos.y));
  };

  // Handle stage click
  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;

    const pos = stage.getPointerPosition();
    if (!pos) return;

    const canvasPos = screenToCanvas(pos.x, pos.y);

    // Check if clicked on empty space
    if (e.target === stage) {
      if (mode === 'add-joint') {
        // Add new Static joint
        const name = generateJointName('Static');
        addJoint({
          type: 'Static',
          name,
          x: canvasPos.x,
          y: canvasPos.y,
        });
        selectJoint(name);
      } else if (mode === 'select') {
        selectJoint(null);
      } else if (mode === 'draw-link') {
        // Cancel link drawing if clicked on empty space
        setLinkStartJoint(null);
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
      // Convert to Static, preserve position
      updateJoint(joint.name, {
        type: 'Static',
        joint0: null,
        joint1: null,
        distance: undefined,
        angle: undefined,
        distance0: undefined,
        distance1: undefined,
      });
    } else if (mode === 'set-crank') {
      // Find a ground joint to reference
      const groundJoint = linkage?.joints.find(
        (j) => j.type === 'Static' && j.name !== joint.name
      );
      if (groundJoint) {
        const jointPos = { x: joint.x ?? 0, y: joint.y ?? 0 };
        const groundPos = { x: groundJoint.x ?? 0, y: groundJoint.y ?? 0 };
        const distance = calcDistance(groundPos, jointPos);
        const angle = calcAngle(groundPos, jointPos);

        updateJoint(joint.name, {
          type: 'Crank',
          joint0: { ref: groundJoint.name },
          joint1: null,
          distance,
          angle,
          distance0: undefined,
          distance1: undefined,
        });
      }
    } else if (mode === 'draw-link') {
      if (!linkStartJoint) {
        // First joint selected
        setLinkStartJoint(joint.name);
      } else if (linkStartJoint !== joint.name) {
        // Second joint selected - create Revolute joint
        const startJoint = getJoint(linkStartJoint);
        if (startJoint) {
          const startPos = { x: startJoint.x ?? 0, y: startJoint.y ?? 0 };
          const endPos = { x: joint.x ?? 0, y: joint.y ?? 0 };

          // Create a new Revolute joint at midpoint
          const midX = (startPos.x + endPos.x) / 2;
          const midY = (startPos.y + endPos.y) / 2;
          const distance0 = calcDistance(startPos, { x: midX, y: midY });
          const distance1 = calcDistance(endPos, { x: midX, y: midY });

          const name = generateJointName('Revolute');
          addJoint({
            type: 'Revolute',
            name,
            x: midX,
            y: midY,
            joint0: { ref: linkStartJoint },
            joint1: { ref: joint.name },
            distance0,
            distance1,
          });
          selectJoint(name);
        }
        setLinkStartJoint(null);
      }
    } else {
      selectJoint(joint.name);
    }
  };

  // Handle joint drag for direct manipulation
  const handleJointDrag = (joint: JointDict, newPos: Position) => {
    if (joint.type === 'Static') {
      // Static joints just update position
      updateJoint(joint.name, { x: newPos.x, y: newPos.y });
    } else if (joint.type === 'Crank' || joint.type === 'Fixed') {
      // For Crank/Fixed, update distance and angle based on parent
      if (joint.joint0 && 'ref' in joint.joint0) {
        const parentJoint = getJoint(joint.joint0.ref);
        if (parentJoint) {
          const parentPos = { x: parentJoint.x ?? 0, y: parentJoint.y ?? 0 };
          const distance = calcDistance(parentPos, newPos);
          const angle = calcAngle(parentPos, newPos);
          updateJoint(joint.name, {
            x: newPos.x,
            y: newPos.y,
            distance,
            angle,
          });
        }
      }
    } else if (joint.type === 'Revolute') {
      // For Revolute, update both distances
      if (
        joint.joint0 &&
        'ref' in joint.joint0 &&
        joint.joint1 &&
        'ref' in joint.joint1
      ) {
        const parent0 = getJoint(joint.joint0.ref);
        const parent1 = getJoint(joint.joint1.ref);
        if (parent0 && parent1) {
          const pos0 = { x: parent0.x ?? 0, y: parent0.y ?? 0 };
          const pos1 = { x: parent1.x ?? 0, y: parent1.y ?? 0 };
          const distance0 = calcDistance(pos0, newPos);
          const distance1 = calcDistance(pos1, newPos);
          updateJoint(joint.name, {
            x: newPos.x,
            y: newPos.y,
            distance0,
            distance1,
          });
        }
      }
    }
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
        const screenPos = canvasToScreen(pos.x, pos.y);
        points.push(screenPos.x, screenPos.y);
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

  // Render link preview line for draw-link mode
  const renderLinkPreview = () => {
    if (mode !== 'draw-link' || !linkStartJoint || !mousePos || !linkage)
      return null;

    const startJoint = linkage.joints.find((j) => j.name === linkStartJoint);
    if (!startJoint) return null;

    const startPos = canvasToScreen(startJoint.x ?? 0, startJoint.y ?? 0);
    const endPos = canvasToScreen(mousePos.x, mousePos.y);

    return (
      <Line
        points={[startPos.x, startPos.y, endPos.x, endPos.y]}
        stroke="#58a6ff"
        strokeWidth={2}
        dash={[8, 4]}
        opacity={0.7}
      />
    );
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
      const screenPos = canvasToScreen(jointPos.x, jointPos.y);

      parents.forEach((parentName) => {
        const parentIndex = linkage.joints.findIndex(
          (j) => j.name === parentName
        );
        if (parentIndex === -1) return;

        const parentJoint = linkage.joints[parentIndex];
        const parentPos = getJointPosition(parentJoint, parentIndex);
        const parentScreenPos = canvasToScreen(parentPos.x, parentPos.y);

        links.push(
          <Line
            key={`link-${joint.name}-${parentName}`}
            points={[
              parentScreenPos.x,
              parentScreenPos.y,
              screenPos.x,
              screenPos.y,
            ]}
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
      const screenPos = canvasToScreen(pos.x, pos.y);

      const isSelected = selectedJointName === joint.name;
      const isHovered = hoveredJointName === joint.name;
      const isLinkStart = linkStartJoint === joint.name;
      const color = JOINT_COLORS[joint.type];

      // Determine if joint is draggable based on mode and type
      const isDraggable =
        mode === 'move-joint' &&
        (joint.type === 'Static' ||
          joint.type === 'Crank' ||
          joint.type === 'Fixed' ||
          joint.type === 'Revolute');

      return (
        <Group key={joint.name}>
          {/* Joint circle */}
          <Circle
            x={screenPos.x}
            y={screenPos.y}
            radius={JOINT_RADIUS}
            fill={color}
            stroke={
              isLinkStart
                ? '#58a6ff'
                : isSelected
                  ? '#ffffff'
                  : isHovered
                    ? '#c9d1d9'
                    : '#0d1117'
            }
            strokeWidth={isSelected || isLinkStart ? 3 : 2}
            onClick={() => handleJointClick(joint)}
            onTap={() => handleJointClick(joint)}
            onMouseEnter={() => setHoveredJoint(joint.name)}
            onMouseLeave={() => setHoveredJoint(null)}
            draggable={isDraggable}
            onDragEnd={(e) => {
              const newPos = screenToCanvas(e.target.x(), e.target.y());
              handleJointDrag(joint, newPos);
            }}
            style={{ cursor: mode === 'delete' ? 'not-allowed' : 'pointer' }}
          />
          {/* Joint label */}
          <Text
            x={screenPos.x + JOINT_RADIUS + 4}
            y={screenPos.y - 6}
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
        onMouseMove={handleMouseMove}
      >
        <Layer>
          {renderGrid()}
          {renderLoci()}
          {renderLinks()}
          {renderLinkPreview()}
          {renderJoints()}
        </Layer>
      </Stage>
    </div>
  );
}
