/**
 * Keyboard shortcuts hook for the linkage editor.
 *
 * Shortcuts:
 * - Escape: Switch to select mode
 * - Space: Toggle animation playback
 * - Delete/Backspace: Delete selected joint
 * - Ctrl+Z: Undo
 * - Ctrl+Shift+Z / Ctrl+Y: Redo
 * - 1: Select mode
 * - 2: Add joint mode
 * - 3: Draw link mode
 * - 4: Move joint mode
 * - 5: Delete mode
 * - 6: Set ground mode
 * - 7: Set crank mode
 */

import { useEffect, useCallback } from 'react';
import { useEditorStore } from '../stores/editorStore';
import { useLinkageStore } from '../stores/linkageStore';
import type { EditorMode } from '../types/linkage';

const MODE_SHORTCUTS: Record<string, EditorMode> = {
  '1': 'select',
  '2': 'add-joint',
  '3': 'draw-link',
  '4': 'move-joint',
  '5': 'delete',
  '6': 'set-ground',
  '7': 'set-crank',
};

export function useKeyboardShortcuts() {
  const setMode = useEditorStore((s) => s.setMode);
  const isAnimating = useEditorStore((s) => s.isAnimating);
  const setAnimating = useEditorStore((s) => s.setAnimating);
  const selectedJointName = useEditorStore((s) => s.selectedJointName);
  const selectJoint = useEditorStore((s) => s.selectJoint);

  const loci = useLinkageStore((s) => s.loci);
  const deleteJoint = useLinkageStore((s) => s.deleteJoint);

  // Access temporal store for undo/redo via the store's temporal property
  const temporalStore = useLinkageStore.temporal;
  const undo = temporalStore.getState().undo;
  const redo = temporalStore.getState().redo;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Don't handle shortcuts when typing in inputs
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const key = event.key;
      const ctrlOrCmd = event.ctrlKey || event.metaKey;
      const shift = event.shiftKey;

      // Escape: Switch to select mode
      if (key === 'Escape') {
        event.preventDefault();
        setMode('select');
        return;
      }

      // Space: Toggle animation
      if (key === ' ') {
        event.preventDefault();
        if (loci && loci.length > 0) {
          setAnimating(!isAnimating);
        }
        return;
      }

      // Delete/Backspace: Delete selected joint
      if ((key === 'Delete' || key === 'Backspace') && selectedJointName) {
        event.preventDefault();
        deleteJoint(selectedJointName);
        selectJoint(null);
        return;
      }

      // Ctrl+Z: Undo
      if (ctrlOrCmd && key === 'z' && !shift) {
        event.preventDefault();
        undo();
        return;
      }

      // Ctrl+Shift+Z or Ctrl+Y: Redo
      if ((ctrlOrCmd && shift && key === 'z') || (ctrlOrCmd && key === 'y')) {
        event.preventDefault();
        redo();
        return;
      }

      // Number keys for mode shortcuts
      if (key in MODE_SHORTCUTS && !ctrlOrCmd && !shift) {
        event.preventDefault();
        setMode(MODE_SHORTCUTS[key]);
        return;
      }
    },
    [
      setMode,
      isAnimating,
      setAnimating,
      loci,
      selectedJointName,
      deleteJoint,
      selectJoint,
      undo,
      redo,
    ]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
}
