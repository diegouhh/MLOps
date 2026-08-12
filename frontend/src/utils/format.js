export function formatDate(value) {
  if (!value) return '—'
  return new Intl.DateTimeFormat('es-CO', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

export function formatMetric(value) {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric.toFixed(4) : '—'
}

export function formatBytes(value = 0) {
  if (value < 1024) return `${value} B`
  if (value < 1048576) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1048576).toFixed(1)} MB`
}

export function titleCase(value = '') {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}
