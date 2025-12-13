/**
 * Sidebar with controls for linkage editing.
 */

import { useLinkageStore } from '../../stores/linkageStore';
import { ExampleLoader } from '../sidebar/ExampleLoader';
import { JointList } from '../sidebar/JointList';
import { CanvasToolbar } from '../canvas/CanvasToolbar';
import { AnimationControls } from '../sidebar/AnimationControls';

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  title: {
    margin: 0,
    fontSize: '18px',
    fontWeight: 600,
    color: '#58a6ff',
  },
  subtitle: {
    fontSize: '12px',
    color: '#8b949e',
    marginBottom: '8px',
  },
  section: {
    borderBottom: '1px solid #30363d',
    paddingBottom: '16px',
  },
  sectionTitle: {
    fontSize: '11px',
    textTransform: 'uppercase' as const,
    color: '#8b949e',
    marginBottom: '8px',
    letterSpacing: '0.5px',
  },
  stats: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '8px',
  },
  stat: {
    padding: '8px',
    background: '#21262d',
    borderRadius: '4px',
    textAlign: 'center' as const,
  },
  statValue: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#58a6ff',
  },
  statLabel: {
    fontSize: '11px',
    color: '#8b949e',
  },
};

export function Sidebar() {
  const linkage = useLinkageStore((s) => s.linkage);

  const jointCount = linkage?.joints.length ?? 0;
  const groundCount = linkage?.joints.filter((j) => j.type === 'Static').length ?? 0;
  const crankCount = linkage?.joints.filter((j) => j.type === 'Crank').length ?? 0;

  return (
    <div style={styles.container}>
      {/* Header */}
      <div>
        <h1 style={styles.title}>Pylinkage Editor</h1>
        <p style={styles.subtitle}>Interactive linkage design tool</p>
      </div>

      {/* Example Loader */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Examples</div>
        <ExampleLoader />
      </div>

      {/* Tools */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Tools</div>
        <CanvasToolbar />
      </div>

      {/* Animation */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Animation</div>
        <AnimationControls />
      </div>

      {/* Stats */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Statistics</div>
        <div style={styles.stats}>
          <div style={styles.stat}>
            <div style={styles.statValue}>{jointCount}</div>
            <div style={styles.statLabel}>Joints</div>
          </div>
          <div style={styles.stat}>
            <div style={styles.statValue}>{groundCount}</div>
            <div style={styles.statLabel}>Ground</div>
          </div>
          <div style={styles.stat}>
            <div style={styles.statValue}>{crankCount}</div>
            <div style={styles.statLabel}>Cranks</div>
          </div>
          <div style={styles.stat}>
            <div style={{ ...styles.statValue, color: linkage?.is_buildable ? '#3fb950' : '#f85149' }}>
              {linkage?.is_buildable ? 'OK' : 'ERR'}
            </div>
            <div style={styles.statLabel}>Status</div>
          </div>
        </div>
      </div>

      {/* Joint List */}
      <div>
        <div style={styles.sectionTitle}>Joints</div>
        <JointList />
      </div>
    </div>
  );
}
