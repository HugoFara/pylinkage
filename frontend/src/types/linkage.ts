/**
 * TypeScript types matching pylinkage serialization format.
 * These match the JSON format from src/pylinkage/linkage/serialization.py
 */

// Joint reference types
export interface JointRef {
  ref: string;
}

export interface InlineStatic {
  inline: true;
  type: 'Static';
  x: number;
  y: number;
  name?: string;
}

export type JointReference = JointRef | InlineStatic | null;

// Joint types
export type JointType = 'Static' | 'Crank' | 'Fixed' | 'Revolute' | 'Prismatic';

export interface JointDict {
  type: JointType;
  name: string;
  x: number | null;
  y: number | null;

  // Parent references
  joint0?: JointReference;
  joint1?: JointReference;
  joint2?: JointReference; // For Prismatic

  // Type-specific attributes
  distance?: number; // For Crank, Fixed
  angle?: number; // For Crank, Fixed
  distance0?: number; // For Revolute
  distance1?: number; // For Revolute
  revolute_radius?: number; // For Prismatic
}

// Linkage types
export interface LinkageDict {
  name: string;
  joints: JointDict[];
  solve_order?: string[] | null;
}

export interface LinkageResponse extends LinkageDict {
  id: string;
  created_at: string;
  updated_at: string;
  is_buildable: boolean;
  rotation_period: number | null;
  error: string | null;
}

export interface LinkageListItem {
  id: string;
  name: string;
  joint_count: number;
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
  linkage_id: string;
  iterations: number;
  frames: SimulationFrame[];
  joint_names: string[];
  is_complete: boolean;
  error: string | null;
}

export interface TrajectoryResponse {
  linkage_id: string;
  iterations: number;
  positions: number[][][]; // [frame][joint][x,y]
  joint_names: string[];
}

// Example types
export interface ExampleInfo {
  name: string;
  description: string;
  joint_count: number;
}

// Editor types
export type EditorMode =
  | 'select'
  | 'add-joint'
  | 'draw-link'
  | 'move-joint'
  | 'delete'
  | 'set-ground'
  | 'set-crank';

// Color scheme for joint types
export const JOINT_COLORS: Record<JointType, string> = {
  Static: '#f85149', // Red for ground
  Crank: '#d29922', // Orange for crank
  Fixed: '#a371f7', // Purple for fixed
  Revolute: '#58a6ff', // Blue for revolute
  Prismatic: '#3fb950', // Green for prismatic
};
