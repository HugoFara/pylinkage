import { AppShell } from './components/layout/AppShell';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

function App() {
  // Enable keyboard shortcuts globally
  useKeyboardShortcuts();

  return <AppShell />;
}

export default App;
