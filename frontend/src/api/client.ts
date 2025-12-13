/**
 * API client for pylinkage backend.
 */

import type {
  ExampleInfo,
  LinkageDict,
  LinkageListItem,
  LinkageResponse,
  SimulationResponse,
  TrajectoryResponse,
} from '../types/linkage';

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

// Linkage CRUD
export const linkageApi = {
  list: () => fetchJson<LinkageListItem[]>(`${API_BASE}/linkages`),

  get: (id: string) => fetchJson<LinkageResponse>(`${API_BASE}/linkages/${id}`),

  create: (data: LinkageDict) =>
    fetchJson<LinkageResponse>(`${API_BASE}/linkages/`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: Partial<LinkageDict>) =>
    fetchJson<LinkageResponse>(`${API_BASE}/linkages/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  delete: async (id: string) => {
    const response = await fetch(`${API_BASE}/linkages/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`Failed to delete linkage: ${response.status}`);
    }
  },
};

// Simulation
export const simulationApi = {
  simulate: (id: string, iterations?: number, dt = 1.0) =>
    fetchJson<SimulationResponse>(`${API_BASE}/linkages/${id}/simulate`, {
      method: 'POST',
      body: JSON.stringify({ iterations, dt }),
    }),

  trajectory: (id: string, iterations?: number, dt = 1.0) =>
    fetchJson<TrajectoryResponse>(`${API_BASE}/linkages/${id}/trajectory`, {
      method: 'POST',
      body: JSON.stringify({ iterations, dt }),
    }),

  rotationPeriod: (id: string) =>
    fetchJson<{ rotation_period: number | null; error: string | null }>(
      `${API_BASE}/linkages/${id}/rotation-period`
    ),
};

// Examples
export const examplesApi = {
  list: () => fetchJson<ExampleInfo[]>(`${API_BASE}/examples`),

  get: (name: string) => fetchJson<LinkageDict>(`${API_BASE}/examples/${name}`),

  load: (name: string) =>
    fetchJson<LinkageResponse>(`${API_BASE}/examples/${name}/load`, {
      method: 'POST',
    }),
};
