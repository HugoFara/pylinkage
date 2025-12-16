/**
 * Keyboard shortcuts hook for the linkage editor.
 * Updated for link-first approach with collapsible toolbar.
 *
 * Shortcuts:
 * - Escape: Switch to select mode
 * - Space: Toggle animation playback
 * - Delete/Backspace: Delete selected link
 * - Ctrl+Z: Undo
 * - Ctrl+Shift+Z / Ctrl+Y: Redo
 * - 1: Draw link mode (default)
 * - 2: Select mode
 * - 3: Move joint mode
 * - 4: Delete mode
 */

import { useEffect, useCallback } from 'react';
import { useEditorStore } from '../stores/editorStore';
import { useMechanismStore } from '../stores/mechanismStore';
import type { EditorMode } from '../types/mechanism';

const MODE_SHORTCUTS: Record<string, EditorMode> = {
  '1': 'draw-link',
  '2': 'select',
  '3': 'move-joint',
  '4': 'delete',
};

export function useKeyboardShortcuts() {
  const setMode = useEditorStore((s) => s.setMode);
  const isAnimating = useEditorStore((s) => s.isAnimating);
  const setAnimating = useEditorStore((s) => s.setAnimating);
  const selectedLinkId = useEditorStore((s) => s.selectedLinkId);
  const selectLink = useEditorStore((s) => s.selectLink);

  const loci = useMechanismStore((s) => s.loci);
  const deleteLink = useMechanismStore((s) => s.deleteLink);

  // Access temporal store for undo/redo via the store's temporal property
  const temporalStore = useMechanismStore.temporal;
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

      // Delete/Backspace: Delete selected link
      if ((key === 'Delete' || key === 'Backspace') && selectedLinkId) {
        event.preventDefault();
        deleteLink(selectedLinkId);
        selectLink(null);
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
      selectedLinkId,
      deleteLink,
      selectLink,
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
