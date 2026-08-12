export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
export const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000'
export const PREFECT_URL = import.meta.env.VITE_PREFECT_URL || 'http://localhost:4200'

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options)
  if (response.status === 204) return null
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map((item) => item.msg).join(', ')
      : body.detail || 'La solicitud no pudo completarse'
    throw new Error(detail)
  }
  return body
}

export function jsonOptions(method, body) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  }
}
