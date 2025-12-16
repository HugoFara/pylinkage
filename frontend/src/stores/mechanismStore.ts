/**
 * Mechanism data store using Zustand with undo/redo support.
 * Link-first approach: links are primary, joints are derived from connections.
 */

import { create } from 'zustand';
import { temporal } from 'zundo';
import type {
  JointDict,
  JointType,
  LinkDict,
  LinkType,
  MechanismResponse,
  Position,
  SimulationFrame,
} from '../types/mechanism';
import { SNAP_THRESHOLD } from '../types/mechanism';

interface MechanismState {
  // Current mechanism data
  mechanism: MechanismResponse | null;
  setMechanism: (mechanism: MechanismResponse | null) => void;
  updateBuildableStatus: (is_buildable: boolean, error?: string | null) => void;

  // Simulation results cache
  loci: SimulationFrame[] | null;
  lociJointNames: string[] | null; // Joint names order for loci positions
  setLoci: (loci: SimulationFrame[] | null, jointNames?: string[] | null) => void;

  // Link operations (primary)
  addLink: (link: LinkDict, newJoints?: JointDict[]) => void;
  updateLink: (id: string, updates: Partial<LinkDict>) => void;
  deleteLink: (id: string) => void;

  // Joint operations (secondary)
  addJoint: (joint: JointDict) => void;
  updateJoint: (id: string, updates: Partial<JointDict>) => void;
  updateJointPosition: (id: string, x: number, y: number) => void;
  deleteJoint: (id: string) => void;

  // Lookups
  getLink: (id: string) => LinkDict | undefined;
  getJoint: (id: string) => JointDict | undefined;
  getLinksForJoint: (jointId: string) => LinkDict[];

  // Utility: Find joint at position
  findJointAtPosition: (
    x: number,
    y: number,
    threshold?: number
  ) => JointDict | null;

  // Get joint position at a frame
  getJointPositionAtFrame: (id: string, frame: number) => Position | null;
}

export const useMechanismStore = create<MechanismState>()(
  temporal(
    (set, get) => ({
      mechanism: null,
      setMechanism: (mechanism) => set({ mechanism, loci: null }),
      updateBuildableStatus: (is_buildable, error = null) => {
        const { mechanism } = get();
        if (!mechanism) return;
        set({
          mechanism: {
            ...mechanism,
            is_buildable,
            error,
          },
        });
      },

      loci: null,
      lociJointNames: null,
      setLoci: (loci, jointNames = null) => set({ loci, lociJointNames: jointNames }),

      addLink: (link, newJoints) => {
        const { mechanism } = get();

        // Create empty mechanism if none exists
        const baseMechanism = mechanism ?? createEmptyMechanism();

        const updatedJoints = newJoints
          ? [...baseMechanism.joints, ...newJoints]
          : baseMechanism.joints;

        set({
          mechanism: {
            ...baseMechanism,
            joints: updatedJoints,
            links: [...baseMechanism.links, link],
          },
          loci: null, // Invalidate cache
        });
      },

      updateLink: (id, updates) => {
        const { mechanism } = get();
        if (!mechanism) return;

        set({
          mechanism: {
            ...mechanism,
            links: mechanism.links.map((l) =>
              l.id === id ? { ...l, ...updates } : l
            ),
          },
          loci: null,
        });
      },

      deleteLink: (id) => {
        const { mechanism } = get();
        if (!mechanism) return;

        const linkToDelete = mechanism.links.find((l) => l.id === id);
        if (!linkToDelete) return;

        // Remove the link
        const updatedLinks = mechanism.links.filter((l) => l.id !== id);

        // Find orphaned joints (joints not referenced by any remaining link)
        const referencedJointIds = new Set<string>();
        for (const link of updatedLinks) {
          for (const jointId of link.joints) {
            referencedJointIds.add(jointId);
          }
        }

        const updatedJoints = mechanism.joints.filter((j) =>
          referencedJointIds.has(j.id)
        );

        set({
          mechanism: {
            ...mechanism,
            links: updatedLinks,
            joints: updatedJoints,
          },
          loci: null,
        });
      },

      addJoint: (joint) => {
        const { mechanism } = get();
        if (!mechanism) return;

        set({
          mechanism: {
            ...mechanism,
            joints: [...mechanism.joints, joint],
          },
          loci: null,
        });
      },

      updateJoint: (id, updates) => {
        const { mechanism } = get();
        if (!mechanism) return;

        set({
          mechanism: {
            ...mechanism,
            joints: mechanism.joints.map((j) =>
              j.id === id ? { ...j, ...updates } : j
            ),
          },
          loci: null,
        });
      },

      updateJointPosition: (id, x, y) => {
        const { mechanism } = get();
        if (!mechanism) return;

        set({
          mechanism: {
            ...mechanism,
            joints: mechanism.joints.map((j) =>
              j.id === id ? { ...j, position: [x, y] as [number, number] } : j
            ),
          },
          loci: null,
        });
      },

      deleteJoint: (id) => {
        const { mechanism } = get();
        if (!mechanism) return;

        // Delete all links that reference this joint
        const updatedLinks = mechanism.links.filter(
          (l) => !l.joints.includes(id)
        );

        // Find all referenced joints in remaining links
        const referencedJointIds = new Set<string>();
        for (const link of updatedLinks) {
          for (const jointId of link.joints) {
            referencedJointIds.add(jointId);
          }
        }

        // Keep only referenced joints
        const updatedJoints = mechanism.joints.filter((j) =>
          referencedJointIds.has(j.id)
        );

        set({
          mechanism: {
            ...mechanism,
            links: updatedLinks,
            joints: updatedJoints,
          },
          loci: null,
        });
      },

      getLink: (id) => {
        const { mechanism } = get();
        return mechanism?.links.find((l) => l.id === id);
      },

      getJoint: (id) => {
        const { mechanism } = get();
        return mechanism?.joints.find((j) => j.id === id);
      },

      getLinksForJoint: (jointId) => {
        const { mechanism } = get();
        if (!mechanism) return [];
        return mechanism.links.filter((l) => l.joints.includes(jointId));
      },

      findJointAtPosition: (x, y, threshold = SNAP_THRESHOLD) => {
        const { mechanism } = get();
        if (!mechanism) return null;

        for (const joint of mechanism.joints) {
          const [jx, jy] = joint.position;
          if (jx === null || jy === null) continue;

          const distance = Math.sqrt((x - jx) ** 2 + (y - jy) ** 2);
          if (distance <= threshold) {
            return joint;
          }
        }
        return null;
      },

      getJointPositionAtFrame: (id, frame) => {
        const { mechanism, loci } = get();
        if (!mechanism || !loci || frame >= loci.length) return null;

        const jointIndex = mechanism.joints.findIndex((j) => j.id === id);
        if (jointIndex === -1) return null;

        return loci[frame].positions[jointIndex];
      },
    }),
    {
      limit: 50, // Keep 50 history states
      equality: (a, b) =>
        JSON.stringify(a.mechanism) === JSON.stringify(b.mechanism),
    }
  )
);

// Counter for generating unique IDs
let linkCounter = 0;
let jointCounter = 0;

export function generateLinkId(type: LinkType): string {
  linkCounter++;
  const prefix = type === 'ground' ? 'G' : type === 'driver' ? 'D' : 'L';
  return `${prefix}${linkCounter}`;
}

export function generateJointId(type: JointType): string {
  jointCounter++;
  const prefixMap: Record<JointType, string> = {
    prismatic: 'P',
    revolute: 'J',
    tracker: 'T',
  };
  const prefix = prefixMap[type] ?? 'J';
  return `${prefix}${jointCounter}`;
}

export function resetCounters(): void {
  linkCounter = 0;
  jointCounter = 0;
}

// Helper to create an empty mechanism
export function createEmptyMechanism(): MechanismResponse {
  return {
    id: `local-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    name: 'New Mechanism',
    joints: [],
    links: [],
    ground: undefined,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    is_buildable: false,
    rotation_period: null,
    error: null,
  };
}

// Helper to calculate distance between two points
export function calculateDistance(
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}
