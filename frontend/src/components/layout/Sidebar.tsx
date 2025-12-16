/**
 * Sidebar with controls for mechanism editing.
 * Updated for link-first approach.
 */

import { useMechanismStore } from '../../stores/mechanismStore';
import { ExampleLoader } from '../sidebar/ExampleLoader';
import { LinkList } from '../sidebar/LinkList';
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
  const mechanism = useMechanismStore((s) => s.mechanism);

  const linkCount = mechanism?.links.length ?? 0;
  const jointCount = mechanism?.joints.length ?? 0;
  const driverCount =
    mechanism?.links.filter((l) => l.type === 'driver').length ?? 0;

  return (
    <div style={styles.container}>
      {/* Header */}
      <div>
        <h1 style={styles.title}>Pylinkage Editor</h1>
        <p style={styles.subtitle}>Link-first mechanism design</p>
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
            <div style={styles.statValue}>{linkCount}</div>
            <div style={styles.statLabel}>Links</div>
          </div>
          <div style={styles.stat}>
            <div style={styles.statValue}>{jointCount}</div>
            <div style={styles.statLabel}>Joints</div>
          </div>
          <div style={styles.stat}>
            <div style={styles.statValue}>{driverCount}</div>
            <div style={styles.statLabel}>Drivers</div>
          </div>
          <div style={styles.stat}>
            <div
              style={{
                ...styles.statValue,
                color: mechanism?.is_buildable ? '#3fb950' : '#f85149',
              }}
            >
              {mechanism?.is_buildable ? 'OK' : 'ERR'}
            </div>
            <div style={styles.statLabel}>Status</div>
          </div>
        </div>
      </div>

      {/* Link List */}
      <div>
        <div style={styles.sectionTitle}>Links</div>
        <LinkList />
      </div>
    </div>
  );
}
