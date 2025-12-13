/**
 * Linkage data store using Zustand with undo/redo support.
 */

import { create } from 'zustand';
import { temporal } from 'zundo';
import type {
  JointDict,
  JointType,
  LinkageResponse,
  Position,
  SimulationFrame,
} from '../types/linkage';

interface LinkageState {
  // Current linkage data
  linkage: LinkageResponse | null;
  setLinkage: (linkage: LinkageResponse | null) => void;

  // Simulation results cache
  loci: SimulationFrame[] | null;
  setLoci: (loci: SimulationFrame[] | null) => void;

  // Joint operations
  addJoint: (joint: JointDict) => void;
  updateJoint: (name: string, updates: Partial<JointDict>) => void;
  deleteJoint: (name: string) => void;

  // Get joint by name
  getJoint: (name: string) => JointDict | undefined;

  // Get joint position at a frame
  getJointPositionAtFrame: (name: string, frame: number) => Position | null;
}

export const useLinkageStore = create<LinkageState>()(
  temporal(
    (set, get) => ({
      linkage: null,
      setLinkage: (linkage) => set({ linkage, loci: null }),

      loci: null,
      setLoci: (loci) => set({ loci }),

      addJoint: (joint) => {
        const { linkage } = get();
        if (!linkage) return;

        set({
          linkage: {
            ...linkage,
            joints: [...linkage.joints, joint],
          },
          loci: null, // Invalidate cache
        });
      },

      updateJoint: (name, updates) => {
        const { linkage } = get();
        if (!linkage) return;

        set({
          linkage: {
            ...linkage,
            joints: linkage.joints.map((j) =>
              j.name === name ? { ...j, ...updates } : j
            ),
          },
          loci: null,
        });
      },

      deleteJoint: (name) => {
        const { linkage } = get();
        if (!linkage) return;

        // Also update any joints that reference this joint
        const updatedJoints = linkage.joints
          .filter((j) => j.name !== name)
          .map((j) => {
            const updated = { ...j };
            // Clear references to deleted joint
            if (j.joint0 && 'ref' in j.joint0 && j.joint0.ref === name) {
              updated.joint0 = null;
            }
            if (j.joint1 && 'ref' in j.joint1 && j.joint1.ref === name) {
              updated.joint1 = null;
            }
            if (j.joint2 && 'ref' in j.joint2 && j.joint2.ref === name) {
              updated.joint2 = null;
            }
            return updated;
          });

        set({
          linkage: {
            ...linkage,
            joints: updatedJoints,
          },
          loci: null,
        });
      },

      getJoint: (name) => {
        const { linkage } = get();
        return linkage?.joints.find((j) => j.name === name);
      },

      getJointPositionAtFrame: (name, frame) => {
        const { linkage, loci } = get();
        if (!linkage || !loci || frame >= loci.length) return null;

        const jointIndex = linkage.joints.findIndex((j) => j.name === name);
        if (jointIndex === -1) return null;

        return loci[frame].positions[jointIndex];
      },
    }),
    {
      limit: 50, // Keep 50 history states
      equality: (a, b) => JSON.stringify(a.linkage) === JSON.stringify(b.linkage),
    }
  )
);

// Helper to generate unique joint names
let jointCounter = 0;
export function generateJointName(type: JointType): string {
  jointCounter++;
  const prefix = type.charAt(0);
  return `${prefix}${jointCounter}`;
}

export function resetJointCounter(): void {
  jointCounter = 0;
}
