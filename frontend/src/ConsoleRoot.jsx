import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';

import AppShell from './app/AppShell';
import AuthGate from './app/AuthGate';
import MemoryExplorer from './features/memory/MemoryExplorer';

export default function ConsoleRoot() {
  return (
    <AuthGate>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<Navigate to="/memory" replace />} />
            <Route path="/memory" element={<MemoryExplorer />} />
            <Route path="*" element={<Navigate to="/memory" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthGate>
  );
}
