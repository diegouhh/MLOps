import { CopyOutlined, DeleteOutlined, ExperimentOutlined, PlusOutlined, SendOutlined, SettingOutlined } from '@ant-design/icons'
import { Alert, App, Button, Card, Col, Descriptions, Empty, Form, Input, InputNumber, Row, Select, Space, Switch, Table, Tabs, Tag, Typography } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { apiRequest, jsonOptions } from '../services/api'
import { formatDate } from '../utils/format'
import { mlflowTrackingUrl, prefectTrackingUrl } from '../utils/links'

const { Text, Title } = Typography

function RecordField({ field, value, onChange }) {
  if (field.type === 'boolean') return <Switch checked={Boolean(value)} onChange={onChange} />
  if (field.type === 'integer' || field.type === 'number') return <InputNumber className="full-width" step={field.type === 'integer' ? 1 : 0.01} value={value} onChange={onChange} placeholder="Escribe un valor" />
  return <Input value={value ?? ''} onChange={(event) => onChange(event.target.value)} placeholder="Escribe un valor" />
}

function hasValue(value) {
  return value !== undefined && value !== null && value !== ''
}

function previewValue(value) {
  if (!hasValue(value)) return <Text type="secondary">Sin valor</Text>
  if (typeof value === 'boolean') return value ? 'Sí' : 'No'
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

export default function PredictionsPage() {
  const { message } = App.useApp()
  const registry = useResource('/registry/models', 5000)
  const history = useResource('/predictions', 3000)
  const tracking = useResource('/predictions/tracking-summary', 5000)
  const [modelName, setModelName] = useState('')
  const [selector, setSelector] = useState('champion')
  const schema = useResource(modelName && selector ? `/registry/models/${encodeURIComponent(modelName)}/versions/${encodeURIComponent(selector)}/input-schema` : null)
  const [records, setRecords] = useState([])
  const [jsonValue, setJsonValue] = useState('[]')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!modelName && registry.data?.length) setModelName(registry.data[0].name)
  }, [modelName, registry.data])

  const selectedModel = (registry.data || []).find((item) => item.name === modelName)
  const selectorOptions = useMemo(() => {
    if (!selectedModel) return []
    const aliases = selectedModel.versions.filter((item) => item.alias).map((item) => ({ value: item.alias, label: `${item.alias} · v${item.version}` }))
    const versions = selectedModel.versions.map((item) => ({ value: item.version, label: `Versión ${item.version}${item.alias ? ` · ${item.alias}` : ''}` }))
    return [...aliases, ...versions]
  }, [selectedModel])

  useEffect(() => {
    if (!selectedModel) return
    const hasChampion = selectedModel.versions.some((item) => item.alias === 'champion')
    const valid = selectorOptions.some((item) => item.value === selector)
    if (!valid) setSelector(hasChampion ? 'champion' : selectedModel.versions[0]?.version || '')
  }, [selectedModel, selector, selectorOptions])

  useEffect(() => {
    if (!schema.data) return
    const initial = [{ ...schema.data.example }]
    setRecords(initial)
  }, [schema.data])

  useEffect(() => {
    setJsonValue(JSON.stringify(records, null, 2))
  }, [records])

  const fields = schema.data?.fields || []
  const requiredFields = fields.filter((field) => field.required)
  const requiredTotal = records.length * requiredFields.length
  const requiredComplete = records.reduce((total, record) => total + requiredFields.filter((field) => hasValue(record[field.name])).length, 0)
  const readyToSubmit = records.length > 0 && requiredComplete === requiredTotal

  const jsonSummary = useMemo(() => {
    try {
      const parsed = JSON.parse(jsonValue)
      if (!Array.isArray(parsed)) return { count: 0, error: 'El JSON debe ser una lista de registros.' }
      return { count: parsed.length, error: '' }
    } catch {
      return { count: 0, error: 'El JSON todavía no es válido.' }
    }
  }, [jsonValue])

  function updateRecord(index, field, value) {
    setRecords((current) => current.map((record, recordIndex) => recordIndex === index ? { ...record, [field]: value } : record))
  }

  function duplicateRecord(index) {
    setRecords((current) => {
      const next = [...current]
      next.splice(index + 1, 0, { ...current[index] })
      return next
    })
  }

  function applyJson() {
    try {
      const parsed = JSON.parse(jsonValue)
      if (!Array.isArray(parsed) || parsed.length === 0) throw new Error('La entrada necesita al menos un registro')
      setRecords(parsed)
      message.success('JSON aplicado al formulario')
    } catch (parseError) {
      message.error(parseError.message)
    }
  }

  async function submit() {
    if (!records.length) return message.warning('Agrega al menos un registro')
    const fields = schema.data?.fields || []
    const missing = records.flatMap((record, index) => fields.filter((field) => field.required && (record[field.name] === undefined || record[field.name] === null || record[field.name] === '')).map((field) => `Registro ${index + 1}: ${field.name}`))
    if (missing.length) return message.warning(`Completa ${missing.join(', ')}`)
    setSaving(true)
    setError('')
    try {
      const result = await apiRequest('/predictions', jsonOptions('POST', { model_name: modelName, version_or_alias: selector, records }))
      message.success(`Predicción ${result.id.slice(0, 8)} enviada`)
      history.reload()
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSaving(false)
    }
  }

  const historyColumns = [
    { title: 'ID', dataIndex: 'id', render: (value) => <Text copyable={{ text: value }} className="mono">{value.slice(0, 8)}…</Text> },
    { title: 'Modelo', dataIndex: 'registered_model_name' },
    { title: 'Alias / versión', dataIndex: 'version_or_alias', render: (value) => <Tag color="blue">{value}</Tag> },
    { title: 'Registros', dataIndex: 'input_payload', align: 'center', render: (value) => value.length },
    { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
    { title: '', render: (_, job) => {
      const jobTracking = tracking.data?.[job.id]
      return <Space wrap>
        {jobTracking?.mlflow?.path && <Button size="small" href={mlflowTrackingUrl(jobTracking.mlflow)} target="_blank" icon={<ExperimentOutlined />}>View in MLflow</Button>}
        {jobTracking?.prefect?.path && <Button size="small" href={prefectTrackingUrl(jobTracking.prefect)} target="_blank" icon={<SettingOutlined />}>View in Prefect</Button>}
      </Space>
    } },
  ]

  const tabItems = [
    {
      key: 'form',
      label: `Formulario (${records.length})`,
      children: <>
        {records.map((record, index) => <Card size="small" className="prediction-record" key={index}>
          <div className="prediction-record-header">
            <strong>Registro {index + 1}</strong>
            <div className="prediction-record-actions">
              <Button type="text" size="small" icon={<CopyOutlined />} onClick={() => duplicateRecord(index)}>Duplicar</Button>
              <Button type="text" size="small" danger icon={<DeleteOutlined />} disabled={records.length === 1} onClick={() => setRecords((current) => current.filter((_, itemIndex) => itemIndex !== index))}>Eliminar</Button>
            </div>
          </div>
          <div className="prediction-fields-grid">{fields.map((field) => <Form.Item className="prediction-field" label={<span title={field.name}>{field.name}</span>} required={field.required} extra={field.dtype} key={field.name}><RecordField field={field} value={record[field.name]} onChange={(value) => updateRecord(index, field.name, value)} /></Form.Item>)}</div>
        </Card>)}
        <Button className="add-record-button" icon={<PlusOutlined />} onClick={() => setRecords((current) => [...current, { ...schema.data.example }])}>Agregar registro</Button>
      </>,
    },
    {
      key: 'preview',
      label: `Vista previa (${records.length})`,
      children: <Table
        className="prediction-preview-table"
        size="small"
        rowKey={(_, index) => index}
        dataSource={records}
        columns={[
          { title: '#', width: 54, fixed: 'left', render: (_, __, index) => index + 1 },
          ...fields.map((field) => ({ title: field.name, dataIndex: field.name, width: 150, ellipsis: true, render: previewValue })),
        ]}
        pagination={records.length > 5 ? { pageSize: 5, size: 'small', showSizeChanger: false } : false}
        scroll={{ x: Math.max(560, fields.length * 150 + 54) }}
        locale={{ emptyText: <Empty description="Agrega un registro para previsualizarlo" /> }}
      />,
    },
    {
      key: 'json',
      label: 'JSON avanzado',
      children: <>
        <div className="json-editor-summary">
          <Text type={jsonSummary.error ? 'danger' : 'secondary'}>{jsonSummary.error || `${jsonSummary.count} ${jsonSummary.count === 1 ? 'registro detectado' : 'registros detectados'}`}</Text>
          <Button disabled={Boolean(jsonSummary.error) || jsonSummary.count === 0} onClick={applyJson}>Aplicar al formulario</Button>
        </div>
        <Input.TextArea className="mono" rows={14} value={jsonValue} onChange={(event) => setJsonValue(event.target.value)} />
      </>,
    },
  ]

  return (
    <>
      <PageHeading title="Predicciones" />
      {(registry.error || history.error || tracking.error || schema.error || error) && <Alert type="error" showIcon message={error || schema.error || registry.error || history.error || tracking.error} style={{ marginBottom: 16 }} />}
      <Row gutter={[18, 18]}>
        <Col xs={24} xxl={10}>
          <Card className="prediction-composer-card" title="Nueva predicción" extra={schema.data?.fields?.length ? <Tag color="blue">{records.length} {records.length === 1 ? 'registro' : 'registros'}</Tag> : null}>
            <Form layout="vertical">
              <Form.Item label="Modelo registrado" required><Select showSearch value={modelName || undefined} onChange={setModelName} placeholder="Seleccionar…" options={(registry.data || []).map((item) => ({ value: item.name, label: item.name }))} /></Form.Item>
              <Form.Item label="Alias o versión" required><Select value={selector || undefined} onChange={setSelector} options={selectorOptions} /></Form.Item>
            </Form>
            {schema.loading && <Card loading />}
            {schema.data && schema.data.fields.length > 0 && <>
              <Alert type="info" showIcon message={`${schema.data.dataset_name} · v${schema.data.dataset_version}`} description={`No incluyas la columna objetivo “${schema.data.target_column}”.`} style={{ marginBottom: 16 }} />
              <div className="prediction-summary" role="status">
                <div><Text type="secondary">Registros</Text><strong>{records.length}</strong></div>
                <div><Text type="secondary">Variables por registro</Text><strong>{fields.length}</strong></div>
                <div><Text type="secondary">Campos obligatorios</Text><strong>{requiredComplete}/{requiredTotal}</strong></div>
                <StatusTag value={readyToSubmit ? 'ready' : 'incomplete'} />
              </div>
              <Tabs items={tabItems} />
              <Button type="primary" size="large" icon={<SendOutlined />} loading={saving} disabled={!readyToSubmit} block style={{ marginTop: 18 }} onClick={submit}>Ejecutar predicción ({records.length})</Button>
            </>}
            {schema.data && schema.data.fields.length === 0 && <Alert type="warning" showIcon message="Este modelo no expone todavía un esquema de entrada tabular." />}
            {!modelName && <Empty description="Selecciona un modelo registrado" />}
          </Card>
        </Col>
        <Col xs={24} xxl={14}>
          <Card title={`Historial (${(history.data || []).length})`}>
            <Table
              columns={historyColumns}
              dataSource={history.data || []}
              rowKey="id"
              loading={history.loading}
              scroll={{ x: 900 }}
              expandable={{
                expandedRowRender: (job) => <Row gutter={[16, 16]}><Col xs={24} lg={12}><Title level={5}>Entradas</Title><pre className="json-preview">{JSON.stringify(job.input_payload, null, 2)}</pre></Col><Col xs={24} lg={12}><Title level={5}>Resultado</Title>{job.result_payload ? <Descriptions bordered size="small" column={1} items={Object.entries(job.result_payload).map(([key, value]) => ({ key, label: key, children: <Text className="mono">{JSON.stringify(value)}</Text> }))} /> : job.error_message ? <Alert type="error" showIcon message={job.error_message} /> : <Empty description="Procesando…" />}</Col></Row>,
              }}
              locale={{ emptyText: <Empty description="Aún no hay predicciones" /> }}
            />
          </Card>
        </Col>
      </Row>
    </>
  )
}
