/**
 * Thin wrapper around fetch that:
 * 1. Automatically attaches x-user-id from localStorage
 * 2. Sets Content-Type: application/json for body requests
 *
 * Use this for all API calls so user identity is always carried.
 */
export const API = "http://127.0.0.1:8000";

export async function apiFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const userId = localStorage.getItem("user_id");

  const headers: Record<string, string> = {
    ...(options.body ? { "Content-Type": "application/json" } : {}),
    ...(userId ? { "x-user-id": userId } : {}),
    ...(options.headers as Record<string, string> | undefined ?? {}),
  };

  return fetch(`${API}${path}`, { ...options, headers });
}
