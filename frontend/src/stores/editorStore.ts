/**
 * Editor state store using Zustand.
 * Manages UI state like current mode, animation, selection, and draw state.
 * Updated for link-first approach.
 */

import { create } from 'zustand';
import type { DrawState, EditorMode } from '../types/mechanism';

interface EditorState {
  // Current editor mode
  mode: EditorMode;
  setMode: (mode: EditorMode) => void;

  // Animation state
  isAnimating: boolean;
  animationFrame: number;
  setAnimating: (isAnimating: boolean) => void;
  setAnimationFrame: (frame: number) => void;

  // Selection state (link-centric)
  selectedLinkId: string | null;
  selectLink: (id: string | null) => void;

  // Secondary selection (for moving joints)
  selectedJointId: string | null;
  selectJoint: (id: string | null) => void;

  // Hover state
  hoveredLinkId: string | null;
  hoveredJointId: string | null;
  setHoveredLink: (id: string | null) => void;
  setHoveredJoint: (id: string | null) => void;

  // Drawing state (for draw-link mode)
  drawState: DrawState;
  setDrawState: (state: Partial<DrawState>) => void;
  resetDrawState: () => void;

  // View settings
  showLoci: boolean;
  showDimensions: boolean;
  showGrid: boolean;
  toggleLoci: () => void;
  toggleDimensions: () => void;
  toggleGrid: () => void;
}

const initialDrawState: DrawState = {
  isDrawing: false,
  startPoint: null,
  endPoint: null,
  snappedToJoint: null,
  snappedEndJoint: null,
};

export const useEditorStore = create<EditorState>((set) => ({
  // Mode - default to draw-link for immediate drawing
  mode: 'draw-link',
  setMode: (mode) =>
    set({
      mode,
      drawState: initialDrawState,
      selectedLinkId: null,
      selectedJointId: null,
    }),

  // Animation
  isAnimating: false,
  animationFrame: 0,
  setAnimating: (isAnimating) => set({ isAnimating }),
  setAnimationFrame: (animationFrame) => set({ animationFrame }),

  // Selection (link-centric)
  selectedLinkId: null,
  selectLink: (id) => set({ selectedLinkId: id }),

  // Secondary selection
  selectedJointId: null,
  selectJoint: (id) => set({ selectedJointId: id }),

  // Hover
  hoveredLinkId: null,
  hoveredJointId: null,
  setHoveredLink: (id) => set({ hoveredLinkId: id }),
  setHoveredJoint: (id) => set({ hoveredJointId: id }),

  // Drawing state
  drawState: initialDrawState,
  setDrawState: (state) =>
    set((s) => ({
      drawState: { ...s.drawState, ...state },
    })),
  resetDrawState: () => set({ drawState: initialDrawState }),

  // View
  showLoci: true,
  showDimensions: false,
  showGrid: true,
  toggleLoci: () => set((s) => ({ showLoci: !s.showLoci })),
  toggleDimensions: () => set((s) => ({ showDimensions: !s.showDimensions })),
  toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),
}));
