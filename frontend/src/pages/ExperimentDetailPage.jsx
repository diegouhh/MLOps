import { ArrowLeftOutlined, ExperimentOutlined, RedoOutlined, SettingOutlined, StopOutlined, TrophyOutlined } from '@ant-design/icons'
import { Alert, App, Button, Card, Col, Descriptions, Form, Input, Modal, Progress, Row, Select, Space, Tag, Tooltip, Typography } from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { apiRequest, jsonOptions } from '../services/api'
import { formatDate, formatMetric, titleCase } from '../utils/format'
import { mlflowTrackingUrl, prefectTrackingUrl } from '../utils/links'

const { Paragraph, Text, Title } = Typography

const stageProgress = {
  queued: 5,
  validate_dataset: 15,
  load_data: 28,
  preprocess: 42,
  extract_features: 55,
  build_training_data: 65,
  train_model: 76,
  evaluate: 86,
  save_artifacts: 94,
  completed: 100,
  failed: 100,
}

function defaultModelName(experiment, run) {
  const base = experiment?.registered_model_name || `${experiment?.pipeline_id || 'neuroops'}-${run?.model_id || 'model'}`
  return base.toLowerCase().replace(/[^a-z0-9._ -]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 200)
}

export default function ExperimentDetailPage() {
  const { experimentId } = useParams()
  const navigate = useNavigate()
  const { message } = App.useApp()
  const experiment = useResource(`/experiments/${experimentId}`, 2500)
  const tracking = useResource(`/experiments/${experimentId}/tracking`, 5000)
  const [selectedRun, setSelectedRun] = useState(null)
  const [modelName, setModelName] = useState('')
  const [alias, setAlias] = useState('champion')
  const [saving, setSaving] = useState(false)
  const [actionError, setActionError] = useState('')
  const item = experiment.data

  const descriptions = useMemo(() => item ? [
    { key: 'dataset', label: 'Dataset', children: <Space><Text copyable>{item.dataset_id}</Text><Tag>v{item.dataset_version}</Tag></Space> },
    { key: 'pipeline', label: 'Pipeline', children: item.pipeline_id },
    { key: 'validation', label: 'Validación', children: titleCase(item.validation_strategy) },
    { key: 'metric', label: 'Métrica principal', children: titleCase(item.primary_metric) },
    { key: 'seed', label: 'Semilla', children: item.random_seed },
    { key: 'created', label: 'Creado', children: formatDate(item.created_at) },
  ] : [], [item])

  async function runAction(action) {
    setActionError('')
    try {
      const result = await apiRequest(`/experiments/${experimentId}/${action}`, jsonOptions('POST'))
      if (action === 'rerun') navigate(`/experiments/${result.id}`)
      else experiment.reload()
      message.success(action === 'rerun' ? 'Reejecución creada' : 'Cancelación solicitada')
    } catch (error) {
      setActionError(error.message)
    }
  }

  function openRegistration(run) {
    setSelectedRun(run)
    setAlias('champion')
    setModelName(defaultModelName(item, run))
  }

  async function register() {
    setSaving(true)
    setActionError('')
    try {
      await apiRequest(`/experiments/${experimentId}/winner`, jsonOptions('POST', { training_run_id: selectedRun.id, alias, model_name: modelName }))
      message.success(`${modelName} registrado como ${alias}`)
      setSelectedRun(null)
      experiment.reload()
    } catch (error) {
      setActionError(error.message)
    } finally {
      setSaving(false)
    }
  }

  if (!item && experiment.loading) return <Card loading />
  if (!item) return <Alert type="error" showIcon message={experiment.error || 'Experimento no encontrado'} />

  const mlflowLink = tracking.data?.mlflow?.path ? mlflowTrackingUrl(tracking.data.mlflow) : null
  const prefectLink = tracking.data?.prefect?.path ? prefectTrackingUrl(tracking.data.prefect) : null

  return (
    <>
      <PageHeading
        title={item.name}
        actions={<><Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/experiments')}>Volver</Button>{mlflowLink && <Button href={mlflowLink} target="_blank" icon={<ExperimentOutlined />}>View in MLflow ↗</Button>}{prefectLink && <Button href={prefectLink} target="_blank" icon={<SettingOutlined />}>View in Prefect ↗</Button>}</>}
      />
      {(experiment.error || tracking.error || actionError || item.error_message) && <Alert type="error" showIcon message={actionError || item.error_message || experiment.error || tracking.error} style={{ marginBottom: 16 }} />}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[18, 18]} align="middle">
          <Col flex="auto"><Space direction="vertical"><StatusTag value={item.status} /><Title level={4} style={{ margin: 0 }}>{item.pipeline_id}</Title><Text type="secondary">{item.runs.length} candidatos · {titleCase(item.primary_metric)}</Text></Space></Col>
          <Col><Space wrap>{['queued', 'running'].includes(item.status) && <Button danger icon={<StopOutlined />} onClick={() => runAction('cancel')}>Cancelar</Button>}<Button type="primary" icon={<RedoOutlined />} onClick={() => runAction('rerun')}>Reejecutar</Button></Space></Col>
        </Row>
      </Card>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        {item.runs.map((run) => {
          const runTracking = tracking.data?.runs?.[run.id]
          const runLink = runTracking?.path ? mlflowTrackingUrl(runTracking) : null
          const winner = run.candidate_id && run.candidate_id === item.winner_candidate_id
          return (
            <Col xs={24} lg={12} xl={8} key={run.id}>
              <Card className={`candidate-card ${winner ? 'winner' : ''}`} title={<Space>{winner && <TrophyOutlined style={{ color: '#52c41a' }} />}<span>{run.model_id}</span></Space>} extra={<StatusTag value={run.status} />}>
                <Text type="secondary">Etapa: {titleCase(run.stage)}</Text>
                <Progress percent={stageProgress[run.stage] || (run.status === 'running' ? 70 : 10)} status={run.status === 'failed' ? 'exception' : run.status === 'completed' ? 'success' : 'active'} size="small" />
                {run.error_message && <Alert type="error" showIcon message={run.error_message} style={{ marginTop: 10 }} />}
                <div className="metrics-grid">{Object.entries(run.metrics || {}).map(([metric, value]) => <div className="metric-tile" key={metric}><span>{titleCase(metric)}</span><strong>{formatMetric(value)}</strong></div>)}</div>
                <Space wrap style={{ marginTop: 16 }}>
                  {runLink ? <Button href={runLink} target="_blank" icon={<ExperimentOutlined />}>View in MLflow</Button> : <Tooltip title="El run aparecerá cuando MLflow lo cree"><Button disabled>View in MLflow</Button></Tooltip>}
                  {run.status === 'completed' && run.candidate_id && <Button type="primary" onClick={() => openRegistration(run)}>Registrar / promover</Button>}
                </Space>
                {run.mlflow_run_id && <Paragraph copyable={{ text: run.mlflow_run_id }} className="mono" ellipsis style={{ marginTop: 12, marginBottom: 0 }}>{run.mlflow_run_id}</Paragraph>}
              </Card>
            </Col>
          )
        })}
      </Row>
      <Card title="Configuración reproducible">
        <Descriptions bordered column={{ xs: 1, md: 2 }} items={descriptions} />
        <Space wrap style={{ marginTop: 16 }}>{Object.entries(item.pipeline_config || {}).map(([key, value]) => <Tag key={key}>{titleCase(key)}: {Array.isArray(value) ? value.join(', ') || '—' : String(value ?? '—')}</Tag>)}</Space>
      </Card>
      <Modal title={`Registrar ${selectedRun?.model_id || 'candidato'}`} open={Boolean(selectedRun)} onCancel={() => setSelectedRun(null)} onOk={register} okText="Registrar versión" confirmLoading={saving}>
        <Form layout="vertical">
          <Form.Item label="Nombre en Model Registry" required extra="Puedes reutilizar este nombre en futuros experimentos para crear nuevas versiones."><Input value={modelName} maxLength={200} onChange={(event) => setModelName(event.target.value)} /></Form.Item>
          <Form.Item label="Alias"><Select value={alias} onChange={setAlias} options={[{ value: 'champion', label: 'Champion (producción)' }, { value: 'challenger', label: 'Challenger (candidato)' }]} /></Form.Item>
        </Form>
        {actionError && <Alert type="error" showIcon message={actionError} />}
      </Modal>
    </>
  )
}
