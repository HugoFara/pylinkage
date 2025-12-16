/**
 * API client for pylinkage backend.
 * Updated for mechanism module (link-first approach).
 */

import type {
  ExampleInfo,
  MechanismDict,
  MechanismListItem,
  MechanismResponse,
  SimulationResponse,
  TrajectoryResponse,
} from '../types/mechanism';

const API_BASE = '/api';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Mechanism CRUD
export const mechanismApi = {
  list: () => fetchJson<MechanismListItem[]>(`${API_BASE}/mechanisms`),

  get: (id: string) => fetchJson<MechanismResponse>(`${API_BASE}/mechanisms/${id}`),

  create: (data: MechanismDict) =>
    fetchJson<MechanismResponse>(`${API_BASE}/mechanisms/`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<MechanismDict>) =>
    fetchJson<MechanismResponse>(`${API_BASE}/mechanisms/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: async (id: string) => {
    const response = await fetch(`${API_BASE}/mechanisms/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`Failed to delete mechanism: ${response.status}`);
    }
  },
};

// Simulation
export const simulationApi = {
  simulate: (id: string, iterations?: number, dt = 1.0) =>
    fetchJson<SimulationResponse>(`${API_BASE}/mechanisms/${id}/simulate`, {
      method: 'POST',
      body: JSON.stringify({ iterations, dt }),
    }),

  trajectory: (id: string, iterations?: number, dt = 1.0) =>
    fetchJson<TrajectoryResponse>(`${API_BASE}/mechanisms/${id}/trajectory`, {
      method: 'POST',
      body: JSON.stringify({ iterations, dt }),
    }),

  rotationPeriod: (id: string) =>
    fetchJson<{ rotation_period: number | null; error: string | null }>(
      `${API_BASE}/mechanisms/${id}/rotation-period`
    ),
};

// Examples
export const examplesApi = {
  list: () => fetchJson<ExampleInfo[]>(`${API_BASE}/examples`),

  get: (name: string) => fetchJson<MechanismDict>(`${API_BASE}/examples/${name}`),

  load: (name: string) =>
    fetchJson<MechanismResponse>(`${API_BASE}/examples/${name}/load`, {
      method: 'POST',
    }),
};
