import { useCallback, useEffect, useState } from 'react';

import TokenLogin from './TokenLogin';
import { AUTH_ERROR_EVENT, getDomains } from '../lib/memoryApi';
import LoadingState from '../shared/components/LoadingState';

export default function AuthGate({ children }) {
  const [authenticated, setAuthenticated] = useState(() => {
    return Boolean(localStorage.getItem('api_token'));
  });
  const [checking, setChecking] = useState(true);

  const handleAuthError = useCallback(() => {
    setAuthenticated(false);
  }, []);

  useEffect(() => {
    let mounted = true;

    async function checkAuth() {
      try {
        await getDomains();
        if (mounted) setAuthenticated(true);
      } catch {
        if (mounted) setAuthenticated(false);
      } finally {
        if (mounted) setChecking(false);
      }
    }

    checkAuth();
    return () => {
      mounted = false;
    };
  }, [authenticated]);

  useEffect(() => {
    window.addEventListener(AUTH_ERROR_EVENT, handleAuthError);
    return () => window.removeEventListener(AUTH_ERROR_EVENT, handleAuthError);
  }, [handleAuthError]);

  if (checking) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 text-slate-600">
        <LoadingState label="Connecting to memory service" />
      </div>
    );
  }

  if (!authenticated) {
    return <TokenLogin onAuthenticated={() => setAuthenticated(true)} />;
  }

  return children;
}
