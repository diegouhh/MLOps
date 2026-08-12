import { ExperimentOutlined, EyeOutlined, RocketOutlined, TrophyOutlined } from '@ant-design/icons'
import { Alert, App, Button, Card, Col, Descriptions, Empty, Form, Input, Modal, Row, Select, Space, Statistic, Table, Tag, Typography } from 'antd'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { apiRequest, jsonOptions } from '../services/api'
import { formatDate, formatMetric, titleCase } from '../utils/format'
import { mlflowModelUrl, mlflowTrackingUrl, prefectTrackingUrl } from '../utils/links'

const { Text, Title } = Typography

export default function ModelManagementPage() {
  const { message } = App.useApp()
  const navigate = useNavigate()
  const registry = useResource('/registry/models', 5000)
  const experiments = useResource('/experiments', 5000)
  const tracking = useResource('/experiments/tracking-summary', 5000)
  const [selectedName, setSelectedName] = useState('')
  const [registration, setRegistration] = useState(null)
  const [modelName, setModelName] = useState('')
  const [alias, setAlias] = useState('champion')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!selectedName && registry.data?.length) setSelectedName(registry.data[0].name)
  }, [registry.data, selectedName])

  const model = (registry.data || []).find((item) => item.name === selectedName)
  const champion = model?.versions.find((version) => version.alias === 'champion') || null

  async function setVersionAlias(version, nextAlias) {
    setError('')
    try {
      await apiRequest(`/registry/models/${encodeURIComponent(selectedName)}/aliases`, jsonOptions('POST', { version, alias: nextAlias }))
      message.success(`Versión ${version} promovida a ${nextAlias}`)
      registry.reload()
    } catch (requestError) {
      setError(requestError.message)
    }
  }

  function openRegistration(experiment, run) {
    setRegistration({ experiment, run })
    setModelName(selectedName || experiment.registered_model_name || `neuroops-${experiment.pipeline_id}`)
    setAlias(selectedName ? 'challenger' : 'champion')
  }

  async function registerCandidate() {
    setSaving(true)
    setError('')
    try {
      await apiRequest(`/experiments/${registration.experiment.id}/winner`, jsonOptions('POST', { training_run_id: registration.run.id, alias, model_name: modelName }))
      message.success(`${modelName} registrado como ${alias}`)
      setSelectedName(modelName)
      setRegistration(null)
      registry.reload()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSaving(false)
    }
  }

  const versionColumns = [
    { title: 'Versión', dataIndex: 'version', render: (value) => <strong>v{value}</strong> },
    { title: 'Alias', dataIndex: 'alias', render: (value) => <StatusTag value={value} /> },
    { title: 'Modelo base', dataIndex: 'model_id', render: (value) => <Tag>{value}</Tag> },
    { title: 'Dataset', dataIndex: 'dataset_name', render: (value, row) => value || `${row.dataset_id.slice(0, 8)}…` },
    { title: 'Métrica principal', render: (_, row) => `${titleCase(row.primary_metric)}: ${formatMetric(row.metrics?.[row.primary_metric])}` },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
    { title: 'Acciones', fixed: 'right', width: 300, render: (_, row) => { const runLink = tracking.data?.[row.experiment_id]?.runs?.[row.training_run_id]; return <div className="table-actions">{runLink?.path && <Button size="small" href={mlflowTrackingUrl(runLink)} target="_blank" icon={<EyeOutlined />}>View in MLflow</Button>}<Button size="small" type={row.alias === 'champion' ? 'primary' : 'default'} onClick={() => setVersionAlias(row.version, 'champion')}>Champion</Button><Button size="small" onClick={() => setVersionAlias(row.version, 'challenger')}>Challenger</Button></div> } },
  ]

  const experimentColumns = [
    { title: 'Experimento', dataIndex: 'name' },
    { title: 'Pipeline', dataIndex: 'pipeline_id' },
    { title: 'Candidatos', dataIndex: 'runs', render: (runs) => runs.length },
    { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
    { title: 'Seguimiento', width: 330, render: (_, row) => <div className="table-actions">{tracking.data?.[row.id]?.mlflow?.path && <Button size="small" href={mlflowTrackingUrl(tracking.data[row.id].mlflow)} target="_blank">View in MLflow</Button>}{tracking.data?.[row.id]?.prefect?.path && <Button size="small" href={prefectTrackingUrl(tracking.data[row.id].prefect)} target="_blank">View in Prefect</Button>}<Button size="small" type="link" onClick={() => navigate(`/experiments/${row.id}`)}>Detalle</Button></div> },
  ]

  const expandable = {
    expandedRowRender: (experiment) => <Table size="small" pagination={false} rowKey="id" dataSource={experiment.runs} columns={[
      { title: 'Modelo', dataIndex: 'model_id' },
      { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
      { title: 'Resultado', render: (_, run) => `${titleCase(experiment.primary_metric)}: ${formatMetric(run.metrics?.[experiment.primary_metric])}` },
      { title: '', render: (_, run) => run.status === 'completed' && run.candidate_id ? <Button size="small" type="primary" onClick={() => openRegistration(experiment, run)}>Registrar versión</Button> : null },
    ]} />,
    rowExpandable: (experiment) => experiment.runs.length > 0,
  }

  return (
    <>
      <PageHeading
        title="Model management"
        actions={selectedName && <Button href={mlflowModelUrl(selectedName)} target="_blank" icon={<ExperimentOutlined />}>View model in MLflow ↗</Button>}
      />
      {(registry.error || experiments.error || tracking.error || error) && <Alert type="error" showIcon message={error || registry.error || experiments.error || tracking.error} style={{ marginBottom: 16 }} />}
      <Card style={{ marginBottom: 16 }}>
        <Form layout="vertical"><Form.Item label="Modelo registrado" style={{ marginBottom: 0 }}><Select showSearch allowClear placeholder="Selecciona un modelo" value={selectedName || undefined} onChange={setSelectedName} options={(registry.data || []).map((item) => ({ value: item.name, label: `${item.name} · ${item.versions.length} versiones` }))} /></Form.Item></Form>
      </Card>
      {!model ? <Card><Empty description="Registra un candidato desde un experimento completado" /></Card> : <>
        <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
          <Col xs={24} lg={10}>
            <Card className="current-model-card" title={<Space><TrophyOutlined /><span>Current Production Model</span></Space>}>
              {champion ? <>
                <Title level={4} className="current-model-title">{selectedName}</Title>
                <Space wrap><Tag color="cyan">Version: {champion.version}</Tag><Tag color="blue">{champion.model_id}</Tag><StatusTag value="champion" /></Space>
                <Row gutter={[12, 18]} style={{ marginTop: 18 }}>
                  <Col span={12}><Statistic title="Métrica principal" value={champion.metrics?.[champion.primary_metric]} precision={4} /></Col>
                  <Col span={12}><Statistic title="Versiones" value={model.versions.length} /></Col>
                </Row>
                <Descriptions size="small" column={1} style={{ marginTop: 18 }} items={[{ key: 'dataset', label: 'Dataset', children: champion.dataset_name }, { key: 'pipeline', label: 'Pipeline', children: champion.pipeline_id }, { key: 'date', label: 'Actualizado', children: formatDate(champion.created_at) }]} />
              </> : <Empty description="Este modelo no tiene alias champion" />}
            </Card>
          </Col>
          <Col xs={24} lg={14}>
            <Card title="Métricas del champion" className="current-model-card">
              {champion ? <Row gutter={[12, 12]}>{Object.entries(champion.metrics || {}).map(([metric, value]) => <Col xs={12} md={8} key={metric}><Statistic title={titleCase(metric)} value={Number(value)} precision={4} /></Col>)}</Row> : <Empty />}
            </Card>
          </Col>
        </Row>
        <Card title="Model versions" style={{ marginBottom: 16 }} extra={<Tag color="blue">MLflow Registry</Tag>}>
          <Table columns={versionColumns} dataSource={model.versions} rowKey={(row) => `${row.version}-${row.candidate_id}`} scroll={{ x: 1100 }} />
        </Card>
      </>}
      <Card title={<Space><RocketOutlined /><span>Training & experiment history</span></Space>}>
        <Table columns={experimentColumns} dataSource={experiments.data || []} rowKey="id" loading={experiments.loading} expandable={expandable} scroll={{ x: 800 }} />
      </Card>
      <Modal title="Registrar candidato" open={Boolean(registration)} onCancel={() => setRegistration(null)} onOk={registerCandidate} okText="Registrar versión" confirmLoading={saving}>
        <Form layout="vertical">
          <Form.Item label="Nombre en Model Registry" required><Input value={modelName} onChange={(event) => setModelName(event.target.value)} maxLength={200} /></Form.Item>
          <Form.Item label="Alias"><Select value={alias} onChange={setAlias} options={[{ value: 'champion', label: 'Champion (producción)' }, { value: 'challenger', label: 'Challenger (candidato)' }]} /></Form.Item>
        </Form>
        {registration && <Text type="secondary">{registration.experiment.name} · {registration.run.model_id}</Text>}
        {error && <Alert type="error" showIcon message={error} style={{ marginTop: 12 }} />}
      </Modal>
    </>
  )
}
