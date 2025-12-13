/**
 * Editor state store using Zustand.
 * Manages UI state like current mode, animation, and selection.
 */

import { create } from 'zustand';
import type { EditorMode } from '../types/linkage';

interface EditorState {
  // Current editor mode
  mode: EditorMode;
  setMode: (mode: EditorMode) => void;

  // Animation state
  isAnimating: boolean;
  animationFrame: number;
  setAnimating: (isAnimating: boolean) => void;
  setAnimationFrame: (frame: number) => void;

  // Selection state
  selectedJointName: string | null;
  selectJoint: (name: string | null) => void;

  // Hover state
  hoveredJointName: string | null;
  setHoveredJoint: (name: string | null) => void;

  // View settings
  showLoci: boolean;
  showDimensions: boolean;
  showGrid: boolean;
  toggleLoci: () => void;
  toggleDimensions: () => void;
  toggleGrid: () => void;

  // Drawing state (for draw-link mode)
  linkStartJoint: string | null;
  setLinkStartJoint: (name: string | null) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  // Mode
  mode: 'select',
  setMode: (mode) => set({ mode, linkStartJoint: null }),

  // Animation
  isAnimating: false,
  animationFrame: 0,
  setAnimating: (isAnimating) => set({ isAnimating }),
  setAnimationFrame: (animationFrame) => set({ animationFrame }),

  // Selection
  selectedJointName: null,
  selectJoint: (name) => set({ selectedJointName: name }),

  // Hover
  hoveredJointName: null,
  setHoveredJoint: (name) => set({ hoveredJointName: name }),

  // View
  showLoci: true,
  showDimensions: false,
  showGrid: true,
  toggleLoci: () => set((s) => ({ showLoci: !s.showLoci })),
  toggleDimensions: () => set((s) => ({ showDimensions: !s.showDimensions })),
  toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),

  // Drawing
  linkStartJoint: null,
  setLinkStartJoint: (name) => set({ linkStartJoint: name }),
}));
