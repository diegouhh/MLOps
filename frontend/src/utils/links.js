import { MLFLOW_URL, PREFECT_URL } from '../services/api'

function join(base, path) {
  return path ? `${base.replace(/\/$/, '')}${path}` : base
}

export function mlflowTrackingUrl(tracking) {
  return join(MLFLOW_URL, tracking?.path)
}

export function prefectTrackingUrl(tracking) {
  return join(PREFECT_URL, tracking?.path)
}

export function mlflowModelUrl(name) {
  return `${MLFLOW_URL.replace(/\/$/, '')}/#/models/${encodeURIComponent(name)}`
}
