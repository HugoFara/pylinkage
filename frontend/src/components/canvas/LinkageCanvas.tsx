/**
 * Main Konva canvas for mechanism visualization and interaction.
 * Updated for link-first approach with thick solid links.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { Stage, Layer, Line, Circle, Text, Group } from 'react-konva';
import type Konva from 'konva';
import { useEditorStore } from '../../stores/editorStore';
import {
  useMechanismStore,
  calculateDistance,
} from '../../stores/mechanismStore';
import {
  LINK_COLORS,
  JOINT_COLORS,
  LINK_STYLES,
  JOINT_STYLES,
  MIN_LINK_LENGTH,
  type JointDict,
  type LinkDict,
  type Position,
} from '../../types/mechanism';

// Canvas configuration
const GRID_SIZE = 50;

export function LinkageCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Store state
  const mode = useEditorStore((s) => s.mode);
  const selectedLinkId = useEditorStore((s) => s.selectedLinkId);
  const selectLink = useEditorStore((s) => s.selectLink);
  const selectedJointId = useEditorStore((s) => s.selectedJointId);
  const selectJoint = useEditorStore((s) => s.selectJoint);
  const hoveredLinkId = useEditorStore((s) => s.hoveredLinkId);
  const hoveredJointId = useEditorStore((s) => s.hoveredJointId);
  const setHoveredLink = useEditorStore((s) => s.setHoveredLink);
  const setHoveredJoint = useEditorStore((s) => s.setHoveredJoint);
  const showGrid = useEditorStore((s) => s.showGrid);
  const showLoci = useEditorStore((s) => s.showLoci);
  const animationFrame = useEditorStore((s) => s.animationFrame);
  const drawState = useEditorStore((s) => s.drawState);
  const setDrawState = useEditorStore((s) => s.setDrawState);
  const resetDrawState = useEditorStore((s) => s.resetDrawState);
  const openLinkDialog = useEditorStore((s) => s.openLinkDialog);

  const mechanism = useMechanismStore((s) => s.mechanism);
  const loci = useMechanismStore((s) => s.loci);
  const lociJointNames = useMechanismStore((s) => s.lociJointNames);
  const deleteLink = useMechanismStore((s) => s.deleteLink);
  const updateJointPosition = useMechanismStore((s) => s.updateJointPosition);
  const deleteJoint = useMechanismStore((s) => s.deleteJoint);
  const findJointAtPosition = useMechanismStore((s) => s.findJointAtPosition);
  const updateLink = useMechanismStore((s) => s.updateLink);

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

  // Get joint position at current animation frame
  const getJointPosition = useCallback(
    (joint: JointDict): Position => {
      if (loci && loci.length > 0 && animationFrame < loci.length && lociJointNames) {
        // Find the index of this joint in the simulation's joint order
        const lociIndex = lociJointNames.indexOf(joint.id);
        if (lociIndex !== -1) {
          return loci[animationFrame].positions[lociIndex];
        }
      }
      return { x: joint.position[0] ?? 0, y: joint.position[1] ?? 0 };
    },
    [loci, lociJointNames, animationFrame]
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

  // Handle mouse down for draw-link mode
  const handleMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    if (mode !== 'draw-link') return;

    const stage = e.target.getStage();
    if (!stage) return;

    const pos = stage.getPointerPosition();
    if (!pos) return;

    const canvasPos = screenToCanvas(pos.x, pos.y);

    // Check for snap to existing joint
    const snappedJoint = findJointAtPosition(canvasPos.x, canvasPos.y);

    setDrawState({
      isDrawing: true,
      startPoint: snappedJoint
        ? {
            x: snappedJoint.position[0] ?? canvasPos.x,
            y: snappedJoint.position[1] ?? canvasPos.y,
          }
        : canvasPos,
      endPoint: canvasPos,
      snappedToJoint: snappedJoint?.id ?? null,
      snappedEndJoint: null,
    });
  };

  // Handle mouse move
  const handleMouseMove = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const stage = e.target.getStage();
    if (!stage) return;

    const pos = stage.getPointerPosition();
    if (!pos) return;

    const canvasPos = screenToCanvas(pos.x, pos.y);

    if (mode === 'draw-link' && drawState.isDrawing) {
      // Check for snap to existing joint at end point
      const snappedJoint = findJointAtPosition(canvasPos.x, canvasPos.y);

      setDrawState({
        endPoint: snappedJoint
          ? {
              x: snappedJoint.position[0] ?? canvasPos.x,
              y: snappedJoint.position[1] ?? canvasPos.y,
            }
          : canvasPos,
        snappedEndJoint: snappedJoint?.id ?? null,
      });
    }
  };

  // Handle mouse up for draw-link mode
  const handleMouseUp = (_e: Konva.KonvaEventObject<MouseEvent>) => {
    if (mode !== 'draw-link' || !drawState.isDrawing) return;

    const { startPoint, endPoint, snappedToJoint, snappedEndJoint } = drawState;

    if (!startPoint || !endPoint) {
      resetDrawState();
      return;
    }

    // Check minimum length
    const length = calculateDistance(startPoint.x, startPoint.y, endPoint.x, endPoint.y);
    if (length < MIN_LINK_LENGTH) {
      resetDrawState();
      return;
    }

    // Open dialog to configure link properties
    openLinkDialog({
      startPoint,
      endPoint,
      startJointId: snappedToJoint,
      endJointId: snappedEndJoint,
    });
  };

  // Handle stage click
  const handleStageClick = (e: Konva.KonvaEventObject<MouseEvent>) => {
    // Only handle clicks on the stage itself (empty space)
    if (e.target !== e.target.getStage()) return;

    if (mode === 'select') {
      selectLink(null);
      selectJoint(null);
    }
  };

  // Handle link click
  const handleLinkClick = (link: LinkDict) => {
    if (mode === 'delete') {
      deleteLink(link.id);
      if (selectedLinkId === link.id) {
        selectLink(null);
      }
    } else if (mode === 'set-driver' && link.type !== 'ground') {
      // Convert link to driver
      const groundJoints = mechanism?.joints.filter((j) => j.type === 'ground') ?? [];
      const motorJoint = groundJoints.find((gj) => link.joints.includes(gj.id));

      if (motorJoint) {
        updateLink(link.id, {
          type: 'driver',
          motor_joint: motorJoint.id,
          angular_velocity: 0.1,
          initial_angle: 0,
        });
      }
    } else {
      selectLink(link.id);
    }
  };

  // Handle joint click
  const handleJointClick = (joint: JointDict) => {
    if (mode === 'delete') {
      deleteJoint(joint.id);
      if (selectedJointId === joint.id) {
        selectJoint(null);
      }
    } else if (mode === 'set-ground' && joint.type !== 'ground') {
      // Convert joint to ground
      updateJointPosition(joint.id, joint.position[0] ?? 0, joint.position[1] ?? 0);
      // Note: Would need to update joint type, which requires mechanism store change
    } else {
      selectJoint(joint.id);
    }
  };

  // Handle joint drag for move-joint mode
  const handleJointDrag = (joint: JointDict, newPos: Position) => {
    updateJointPosition(joint.id, newPos.x, newPos.y);
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
    if (!showLoci || !loci || loci.length < 2 || !mechanism || !lociJointNames) return null;

    return mechanism.joints.map((joint) => {
      // Find the index of this joint in the simulation's joint order
      const lociIndex = lociJointNames.indexOf(joint.id);
      if (lociIndex === -1) return null;

      const points: number[] = [];
      for (let frame = 0; frame < loci.length; frame++) {
        const pos = loci[frame].positions[lociIndex];
        if (!pos) continue;
        const screenPos = canvasToScreen(pos.x, pos.y);
        points.push(screenPos.x, screenPos.y);
      }

      if (points.length < 4) return null; // Need at least 2 points

      return (
        <Line
          key={`loci-${joint.id}`}
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

  // Render draw preview line
  const renderDrawPreview = () => {
    if (mode !== 'draw-link' || !drawState.isDrawing) return null;

    const { startPoint, endPoint, snappedToJoint, snappedEndJoint } = drawState;
    if (!startPoint || !endPoint) return null;

    const startScreen = canvasToScreen(startPoint.x, startPoint.y);
    const endScreen = canvasToScreen(endPoint.x, endPoint.y);

    return (
      <Group>
        {/* Preview line */}
        <Line
          points={[startScreen.x, startScreen.y, endScreen.x, endScreen.y]}
          stroke="#58a6ff"
          strokeWidth={LINK_STYLES.strokeWidth.link}
          opacity={0.7}
          dash={[8, 4]}
        />
        {/* Start snap indicator */}
        {snappedToJoint && (
          <Circle
            x={startScreen.x}
            y={startScreen.y}
            radius={JOINT_STYLES.hoverRadius}
            stroke="#58a6ff"
            strokeWidth={2}
            fill="transparent"
          />
        )}
        {/* End snap indicator */}
        {snappedEndJoint && (
          <Circle
            x={endScreen.x}
            y={endScreen.y}
            radius={JOINT_STYLES.hoverRadius}
            stroke="#58a6ff"
            strokeWidth={2}
            fill="transparent"
          />
        )}
      </Group>
    );
  };

  // Render links as thick solid lines
  const renderLinks = () => {
    if (!mechanism) return null;

    return mechanism.links.map((link) => {
      // Get positions of joints in this link
      const jointPositions: Position[] = link.joints
        .map((jointId) => {
          const joint = mechanism.joints.find((j) => j.id === jointId);
          if (!joint) return null;
          return getJointPosition(joint);
        })
        .filter((p): p is Position => p !== null);

      if (jointPositions.length < 2) return null;

      const isSelected = selectedLinkId === link.id;
      const isHovered = hoveredLinkId === link.id;
      const color = LINK_COLORS[link.type];
      const baseWidth = LINK_STYLES.strokeWidth[link.type];
      const strokeWidth = isSelected
        ? LINK_STYLES.selectedStrokeWidth
        : isHovered
          ? LINK_STYLES.hoverStrokeWidth
          : baseWidth;

      // For binary links, draw a single line
      if (link.joints.length === 2) {
        const start = canvasToScreen(jointPositions[0].x, jointPositions[0].y);
        const end = canvasToScreen(jointPositions[1].x, jointPositions[1].y);

        return (
          <Line
            key={link.id}
            points={[start.x, start.y, end.x, end.y]}
            stroke={color}
            strokeWidth={strokeWidth}
            lineCap="round"
            onClick={() => handleLinkClick(link)}
            onTap={() => handleLinkClick(link)}
            onMouseEnter={() => setHoveredLink(link.id)}
            onMouseLeave={() => setHoveredLink(null)}
            style={{ cursor: mode === 'delete' ? 'not-allowed' : 'pointer' }}
          />
        );
      }

      // For ternary+ links, draw lines between all pairs
      const lines: JSX.Element[] = [];
      for (let i = 0; i < jointPositions.length; i++) {
        for (let j = i + 1; j < jointPositions.length; j++) {
          const start = canvasToScreen(jointPositions[i].x, jointPositions[i].y);
          const end = canvasToScreen(jointPositions[j].x, jointPositions[j].y);

          lines.push(
            <Line
              key={`${link.id}-${i}-${j}`}
              points={[start.x, start.y, end.x, end.y]}
              stroke={color}
              strokeWidth={strokeWidth}
              lineCap="round"
              onClick={() => handleLinkClick(link)}
              onTap={() => handleLinkClick(link)}
              onMouseEnter={() => setHoveredLink(link.id)}
              onMouseLeave={() => setHoveredLink(null)}
            />
          );
        }
      }

      return <Group key={link.id}>{lines}</Group>;
    });
  };

  // Render joints as small circles at endpoints
  const renderJoints = () => {
    if (!mechanism) return null;

    return mechanism.joints.map((joint) => {
      const pos = getJointPosition(joint);
      const screenPos = canvasToScreen(pos.x, pos.y);

      const isSelected = selectedJointId === joint.id;
      const isHovered = hoveredJointId === joint.id;
      const color = JOINT_COLORS[joint.type];
      const radius = isSelected
        ? JOINT_STYLES.selectedRadius
        : isHovered
          ? JOINT_STYLES.hoverRadius
          : JOINT_STYLES.radius;

      // Determine if joint is draggable based on mode
      const isDraggable = mode === 'move-joint';

      return (
        <Group key={joint.id}>
          {/* Joint circle */}
          <Circle
            x={screenPos.x}
            y={screenPos.y}
            radius={radius}
            fill={color}
            stroke={isSelected ? '#ffffff' : isHovered ? '#c9d1d9' : '#0d1117'}
            strokeWidth={JOINT_STYLES.strokeWidth}
            onClick={() => handleJointClick(joint)}
            onTap={() => handleJointClick(joint)}
            onMouseEnter={() => setHoveredJoint(joint.id)}
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
            x={screenPos.x + radius + 4}
            y={screenPos.y - 6}
            text={joint.name || joint.id}
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
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
      >
        <Layer>
          {renderGrid()}
          {renderLoci()}
          {renderLinks()}
          {renderDrawPreview()}
          {renderJoints()}
        </Layer>
      </Stage>
    </div>
  );
}
