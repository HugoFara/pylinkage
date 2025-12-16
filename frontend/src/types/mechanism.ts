/**
 * TypeScript types matching mechanism module serialization format.
 * These match the JSON format from src/pylinkage/mechanism/serialization.py
 */

// Joint types
export type JointType = 'ground' | 'revolute' | 'prismatic' | 'tracker';

export interface JointDict {
  id: string;
  type: JointType;
  position: [number | null, number | null];
  name?: string;
  // Prismatic-specific
  axis?: [number, number];
  slide_distance?: number;
  // Tracker-specific
  ref_joint1_id?: string;
  ref_joint2_id?: string;
  distance?: number;
  angle?: number;
}

// Link types
export type LinkType = 'ground' | 'driver' | 'arc_driver' | 'link';

export interface LinkDict {
  id: string;
  type: LinkType;
  joints: string[]; // References to joint IDs
  name?: string;
  // Driver-specific
  angular_velocity?: number;
  initial_angle?: number;
  motor_joint?: string; // Reference to ground joint ID
  // Arc-driver-specific
  arc_start?: number;
  arc_end?: number;
}

// Mechanism container (link-first format)
export interface MechanismDict {
  name: string;
  joints: JointDict[];
  links: LinkDict[];
  ground?: string; // Reference to ground link ID
}

// Extended response from API
export interface MechanismResponse extends MechanismDict {
  id: string;
  created_at: string;
  updated_at: string;
  is_buildable: boolean;
  rotation_period: number | null;
  error: string | null;
}

export interface MechanismListItem {
  id: string;
  name: string;
  joint_count: number;
  link_count: number;
  created_at: string;
  is_buildable: boolean;
}

// Simulation types
export interface Position {
  x: number;
  y: number;
}

export interface SimulationFrame {
  step: number;
  positions: Position[];
}

export interface SimulationResponse {
  mechanism_id: string;
  iterations: number;
  frames: SimulationFrame[];
  joint_names: string[];
  is_complete: boolean;
  error: string | null;
}

export interface TrajectoryResponse {
  mechanism_id: string;
  iterations: number;
  positions: number[][][]; // [frame][joint][x,y]
  joint_names: string[];
}

// Example types
export interface ExampleInfo {
  name: string;
  description: string;
  joint_count: number;
  link_count: number;
}

// Editor types (updated for link-first)
export type EditorMode =
  | 'select'
  | 'draw-link'
  | 'move-joint'
  | 'delete'
  | 'set-driver'
  | 'set-ground';

// Draw state for link drawing interaction
export interface DrawState {
  isDrawing: boolean;
  startPoint: { x: number; y: number } | null;
  endPoint: { x: number; y: number } | null;
  snappedToJoint: string | null; // Joint ID if snapped to existing joint
  snappedEndJoint: string | null; // Joint ID if end snapped to existing joint
}

// Dialog state for specifying link properties after drawing
export interface LinkPropertiesDialog {
  isOpen: boolean;
  tempLink: {
    startPoint: { x: number; y: number };
    endPoint: { x: number; y: number };
    startJointId: string | null;
    endJointId: string | null;
  } | null;
}

// Color scheme for link types
export const LINK_COLORS: Record<LinkType, string> = {
  ground: '#f85149', // Red for ground link
  driver: '#d29922', // Orange for driver link
  arc_driver: '#d29922', // Orange for arc driver link (same as driver)
  link: '#58a6ff', // Blue for regular links
};

// Color scheme for joint types
export const JOINT_COLORS: Record<JointType, string> = {
  ground: '#f85149', // Red for ground
  revolute: '#58a6ff', // Blue for revolute
  prismatic: '#3fb950', // Green for prismatic
  tracker: '#a371f7', // Purple for tracker (tracer points)
};

// Link style constants
export const LINK_STYLES = {
  strokeWidth: {
    ground: 8,
    driver: 6,
    arc_driver: 6,
    link: 6,
  },
  hoverStrokeWidth: 8,
  selectedStrokeWidth: 10,
};

// Joint style constants
export const JOINT_STYLES = {
  radius: 6,
  hoverRadius: 8,
  selectedRadius: 10,
  strokeWidth: 2,
};

// Snap threshold in pixels
export const SNAP_THRESHOLD = 15;

// Minimum link length in pixels
export const MIN_LINK_LENGTH = 20;
