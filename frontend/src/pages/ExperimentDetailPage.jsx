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

const metricLabels = {
  accuracy: 'Accuracy',
  balanced_accuracy: 'Balanced accuracy',
  precision_macro: 'Precision macro',
  recall_macro: 'Recall macro',
  f1_macro: 'F1 macro',
  f1_weighted: 'F1 weighted',
  roc_auc: 'ROC AUC',
}

function MetricTiles({ entries, cv = false }) {
  if (!entries.length) return null
  return (
    <div className="metrics-grid">
      {entries.map(([metric, value, deviation]) => (
        <div className="metric-tile" key={metric}>
          <span>{metricLabels[metric] || titleCase(metric)}</span>
          <strong>
            {formatMetric(value)}{cv && deviation !== undefined ? ` ± ${formatMetric(deviation)}` : ''}
          </strong>
        </div>
      ))}
    </div>
  )
}

function RunMetrics({ metrics = {} }) {
  const evaluationUnit = metrics._evaluation_unit || 'record'
  const primary = Object.entries(metrics)
    .filter(([key, value]) => (
      !key.startsWith('_')
      && !key.startsWith('cv_')
      && !key.startsWith('epoch_')
      && !key.startsWith('sample_')
      && Number.isFinite(Number(value))
    ))
    .map(([key, value]) => [key, value])
  const cv = Object.entries(metrics)
    .filter(([key]) => key.startsWith('cv_mean_'))
    .map(([key, value]) => {
      const metric = key.replace('cv_mean_', '')
      return [metric, value, metrics[`cv_std_${metric}`]]
    })
  const secondaryPrefix = evaluationUnit === 'subject' ? 'epoch_' : 'sample_'
  const secondary = Object.entries(metrics)
    .filter(([key, value]) => key.startsWith(secondaryPrefix) && Number.isFinite(Number(value)))
    .map(([key, value]) => [key.replace(secondaryPrefix, ''), value])

  return (
    <>
      {primary.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <Text strong>
            Evaluación principal{evaluationUnit === 'subject' ? ' · Por sujeto' : ''}
          </Text>
          <MetricTiles entries={primary} />
        </div>
      )}
      {cv.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <Text strong>Estabilidad entre folds</Text>
          <MetricTiles entries={cv} cv />
        </div>
      )}
      {secondary.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <Text strong>Evaluación secundaria · Por época</Text>
          <MetricTiles entries={secondary} />
        </div>
      )}
    </>
  )
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
        actions={<><Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/experiments')}>Volver</Button>{mlflowLink && <Button href={mlflowLink} target="_blank" icon={<ExperimentOutlined />}>Ver en MLflow</Button>}{prefectLink && <Button href={prefectLink} target="_blank" icon={<SettingOutlined />}>Ver en Prefect</Button>}</>}
      />
      {(experiment.error || tracking.error || actionError || item.error_message) && <Alert type="error" showIcon message={actionError || item.error_message || experiment.error || tracking.error} style={{ marginBottom: 16 }} />}
      <Card style={{ marginBottom: 16 }}>
        <Row gutter={[18, 18]} align="middle">
          <Col flex="auto"><Space direction="vertical"><StatusTag value={item.status} /><Title level={4} style={{ margin: 0 }}>{item.pipeline_id}</Title><Text type="secondary">{item.runs.length} candidatos - {titleCase(item.primary_metric)}</Text></Space></Col>
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
                <RunMetrics metrics={run.metrics || {}} />
                <Space wrap style={{ marginTop: 16 }}>
                  {runLink ? <Button href={runLink} target="_blank" icon={<ExperimentOutlined />}>Ver en MLflow</Button> : <Tooltip title="El run aparecerá cuando MLflow lo cree"><Button disabled>Ver en MLflow</Button></Tooltip>}
                  {run.status === 'completed' && run.candidate_id && <Button type="primary" onClick={() => openRegistration(run)}>Registrar / promover</Button>}
                </Space>
                {run.mlflow_run_id && <Paragraph copyable={{ text: run.mlflow_run_id }} className="mono" ellipsis style={{ marginTop: 12, marginBottom: 0 }}>{run.mlflow_run_id}</Paragraph>}
              </Card>
            </Col>
          )
        })}
      </Row>
      <Card title="Trazabilidad" style={{ marginBottom: 16 }}>
        <Descriptions bordered column={{ xs: 1, md: 2 }} items={[
          { key: 'neuroops', label: 'NeuroOps ID', children: <Text copyable className="mono">{item.id}</Text> },
          { key: 'prefect', label: 'Prefect Flow ID', children: item.prefect_flow_run_id ? <Text copyable className="mono">{item.prefect_flow_run_id}</Text> : '-' },
          { key: 'mlflow', label: 'MLflow Parent Run ID', children: item.mlflow_parent_run_id ? <Text copyable className="mono">{item.mlflow_parent_run_id}</Text> : '-' },
          { key: 'registry', label: 'Model Registry', children: item.registered_model_name || '-' },
        ]} />
      </Card>
      <Card title="Configuración reproducible">
        <Descriptions bordered column={{ xs: 1, md: 2 }} items={descriptions} />
        <Space wrap style={{ marginTop: 16 }}>{Object.entries(item.pipeline_config || {}).map(([key, value]) => <Tag key={key}>{titleCase(key)}: {Array.isArray(value) ? value.join(', ') || '-' : String(value ?? '-')}</Tag>)}</Space>
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
