import { useCallback, useState } from 'react';
import { AlertCircle, KeyRound, Loader2 } from 'lucide-react';

import { getDomains } from '../lib/memoryApi';
import Button from '../shared/components/Button';

export default function TokenLogin({ onAuthenticated }) {
  const [token, setToken] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = useCallback(async (event) => {
    event.preventDefault();
    const trimmed = token.trim();

    setLoading(true);
    setError('');
    if (trimmed) {
      localStorage.setItem('api_token', trimmed);
    } else {
      localStorage.removeItem('api_token');
    }

    try {
      await getDomains();
      onAuthenticated();
    } catch (err) {
      localStorage.removeItem('api_token');
      if (err.response?.status === 401) {
        setError('Token required or invalid.');
      } else {
        setError('Memory service is unavailable.');
      }
    } finally {
      setLoading(false);
    }
  }, [onAuthenticated, token]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-white px-4 text-slate-950">
      <div className="w-full max-w-sm rounded-lg border border-slate-200 bg-white p-6 shadow-2xl">
        <div className="mb-6">
          <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-md border border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
            <KeyRound size={18} />
          </div>
          <h1 className="text-lg font-semibold">SpatialClaw Memory</h1>
          <p className="mt-1 text-sm text-slate-500">Use an API token, or leave blank for local loopback access.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <label className="block">
            <span className="mb-2 block text-xs font-medium uppercase tracking-wide text-slate-500">Token</span>
            <input
              id="api-token"
              type="password"
              value={token}
              disabled={loading}
              onChange={(event) => {
                setToken(event.target.value);
                setError('');
              }}
              placeholder="Optional for local access"
              className="h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-950 outline-none transition placeholder:text-slate-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 disabled:opacity-60"
            />
          </label>

          {error ? (
            <div className="flex items-center gap-2 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">
              <AlertCircle size={15} />
              {error}
            </div>
          ) : null}

          <Button
            type="submit"
            variant="primary"
            className="w-full"
            disabled={loading}
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            Connect
          </Button>
        </form>
      </div>
    </div>
  );
}
