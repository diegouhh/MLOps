import { Badge, Tag } from 'antd'

const config = {
  completed: ['success', 'Completado'],
  healthy: ['success', 'Activo'],
  ready: ['success', 'Listo'],
  incomplete: ['warning', 'Incompleto'],
  embedded: ['processing', 'Integrado'],
  running: ['processing', 'En curso'],
  installing: ['processing', 'Instalando'],
  queued: ['warning', 'En cola'],
  failed: ['error', 'Fallido'],
  cancelled: ['default', 'Cancelado'],
  unavailable: ['error', 'No disponible'],
  unreachable: ['error', 'Sin conexión'],
  unhealthy: ['error', 'Con errores'],
  champion: ['success', 'Champion'],
  challenger: ['warning', 'Challenger'],
}

export default function StatusTag({ value, badge = false }) {
  const [status, label] = config[value] || ['default', value || 'Sin alias']
  if (badge) return <Badge status={status} text={label} />
  const color = status === 'default' ? undefined : status
  return <Tag color={color}>{label}</Tag>
}
