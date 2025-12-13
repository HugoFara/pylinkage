/**
 * Main application shell with sidebar and canvas layout.
 */

import { Sidebar } from './Sidebar';
import { LinkageCanvas } from '../canvas/LinkageCanvas';

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
  },
  sidebar: {
    width: '300px',
    minWidth: '300px',
    height: '100%',
    borderRight: '1px solid #30363d',
    background: '#161b22',
    overflow: 'auto',
  },
  main: {
    flex: 1,
    height: '100%',
    position: 'relative',
    background: '#0d1117',
  },
};

export function AppShell() {
  return (
    <div style={styles.container}>
      <aside style={styles.sidebar}>
        <Sidebar />
      </aside>
      <main style={styles.main}>
        <LinkageCanvas />
      </main>
    </div>
  );
}
