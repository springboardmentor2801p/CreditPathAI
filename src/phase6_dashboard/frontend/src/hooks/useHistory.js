import { useState, useCallback } from 'react';

const STORAGE_KEY = 'creditpathai_history';
const MAX_ITEMS   = 50;

function load() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
  catch { return []; }
}

export function useHistory() {
  const [history, setHistory] = useState(load);

  const addEntry = useCallback((entry) => {
    setHistory((prev) => {
      const updated = [
        { ...entry, id: Date.now(), timestamp: new Date().toISOString() },
        ...prev,
      ].slice(0, MAX_ITEMS);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const clearHistory = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setHistory([]);
  }, []);

  return { history, addEntry, clearHistory };
}
