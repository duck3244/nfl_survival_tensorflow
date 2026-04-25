import { Route, Routes } from 'react-router-dom';

import { Header } from './components/Header';
import { ErrorBoundary } from './components/ErrorBoundary';
import { PredictPage } from './pages/PredictPage';
import { FamousPage } from './pages/FamousPage';

export default function App() {
  return (
    <div className="min-h-full flex flex-col">
      <Header />
      <main className="flex-1 mx-auto w-full max-w-6xl px-4 py-6">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/famous" element={<FamousPage />} />
            <Route path="*" element={<div className="text-slate-500">Not found.</div>} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}
