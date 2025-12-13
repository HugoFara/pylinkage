/**
 * Dropdown to load prebuilt example linkages.
 */

import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { examplesApi, simulationApi } from '../../api/client';
import { useLinkageStore, resetJointCounter } from '../../stores/linkageStore';

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  select: {
    padding: '8px 12px',
    borderRadius: '6px',
    border: '1px solid #30363d',
    background: '#21262d',
    color: '#e6edf3',
    fontSize: '13px',
    cursor: 'pointer',
    width: '100%',
  },
  button: {
    padding: '8px 16px',
    borderRadius: '6px',
    border: 'none',
    background: '#238636',
    color: 'white',
    fontSize: '13px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  buttonDisabled: {
    background: '#21262d',
    cursor: 'not-allowed',
    opacity: 0.6,
  },
  error: {
    color: '#f85149',
    fontSize: '12px',
  },
  info: {
    color: '#8b949e',
    fontSize: '12px',
  },
};

export function ExampleLoader() {
  const [selectedExample, setSelectedExample] = useState<string>('');
  const setLinkage = useLinkageStore((s) => s.setLinkage);
  const setLoci = useLinkageStore((s) => s.setLoci);

  // Fetch available examples
  const {
    data: examples,
    isLoading: loadingExamples,
    error: examplesError,
  } = useQuery({
    queryKey: ['examples'],
    queryFn: examplesApi.list,
  });

  // Load example mutation
  const loadMutation = useMutation({
    mutationFn: examplesApi.load,
    onSuccess: async (linkage) => {
      resetJointCounter();
      setLinkage(linkage);

      // Also fetch simulation data
      try {
        const simResult = await simulationApi.simulate(linkage.id);
        if (simResult.is_complete) {
          setLoci(simResult.frames);
        }
      } catch (e) {
        console.error('Failed to simulate:', e);
      }
    },
  });

  const handleLoad = () => {
    if (selectedExample) {
      loadMutation.mutate(selectedExample);
    }
  };

  const selectedInfo = examples?.find((e) => e.name === selectedExample);

  return (
    <div style={styles.container}>
      <select
        style={styles.select}
        value={selectedExample}
        onChange={(e) => setSelectedExample(e.target.value)}
        disabled={loadingExamples}
      >
        <option value="">Select an example...</option>
        {examples?.map((ex) => (
          <option key={ex.name} value={ex.name}>
            {ex.name} ({ex.joint_count} joints)
          </option>
        ))}
      </select>

      {selectedInfo && (
        <p style={styles.info}>{selectedInfo.description}</p>
      )}

      <button
        style={{
          ...styles.button,
          ...((!selectedExample || loadMutation.isPending) ? styles.buttonDisabled : {}),
        }}
        onClick={handleLoad}
        disabled={!selectedExample || loadMutation.isPending}
      >
        {loadMutation.isPending ? 'Loading...' : 'Load Example'}
      </button>

      {examplesError && (
        <p style={styles.error}>Failed to load examples</p>
      )}
      {loadMutation.error && (
        <p style={styles.error}>
          Failed to load: {(loadMutation.error as Error).message}
        </p>
      )}
    </div>
  );
}
