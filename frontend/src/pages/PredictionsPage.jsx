import { CopyOutlined, DeleteOutlined, ExperimentOutlined, PlusOutlined, SendOutlined, SettingOutlined } from '@ant-design/icons'
import { Alert, App, Button, Card, Col, Descriptions, Empty, Form, Input, InputNumber, Row, Select, Space, Switch, Table, Tabs, Tag, Typography } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { apiRequest, jsonOptions } from '../services/api'
import { formatDate } from '../utils/format'
import { mlflowTrackingUrl, prefectTrackingUrl } from '../utils/links'

const { Paragraph, Text, Title } = Typography

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

function resultSummary(job) {
  if (!job.result_payload) return null
  if (job.result_payload.mode === 'eeg_recording') {
    return <Descriptions bordered size="small" column={1} items={[
      { key: 'prediction', label: 'Predicción', children: <strong>{String(job.result_payload.prediction ?? 'Sin resultado')}</strong> },
      { key: 'epochs', label: 'Épocas analizadas', children: job.result_payload.epochs_analyzed ?? '-' },
      { key: 'distribution', label: 'Distribución', children: <Text className="mono">{JSON.stringify(job.result_payload.class_distribution || {})}</Text> },
      ...(job.result_payload.mean_probabilities ? [{ key: 'probabilities', label: 'Probabilidad media', children: <Text className="mono">{JSON.stringify(job.result_payload.mean_probabilities)}</Text> }] : []),
    ]} />
  }
  return <Descriptions bordered size="small" column={1} items={Object.entries(job.result_payload).map(([key, value]) => ({ key, label: key, children: <Text className="mono">{JSON.stringify(value)}</Text> }))} />
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
  const [selectedExample, setSelectedExample] = useState(0)
  const [selectedRecording, setSelectedRecording] = useState('')
  const [jsonValue, setJsonValue] = useState('[]')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!modelName && registry.data?.length) setModelName(registry.data[0].name)
  }, [modelName, registry.data])

  const selectedModel = (registry.data || []).find((item) => item.name === modelName)
  const selectorOptions = useMemo(() => {
    if (!selectedModel) return []
    const aliases = selectedModel.versions.filter((item) => item.alias).map((item) => ({ value: item.alias, label: `${item.alias} - v${item.version}` }))
    const versions = selectedModel.versions.map((item) => ({ value: item.version, label: `Versión ${item.version}${item.alias ? ` - ${item.alias}` : ''}` }))
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
    setError('')
    if (schema.data.input_mode === 'eeg_recording') {
      const first = schema.data.recordings?.[0]?.value || ''
      setSelectedRecording(first)
      setRecords(first ? [{ recording: first }] : [])
      setJsonValue('[]')
      return
    }
    const initial = schema.data.examples?.length ? { ...schema.data.examples[0] } : { ...schema.data.example }
    setSelectedExample(0)
    setRecords([initial])
  }, [schema.data])

  useEffect(() => {
    if (schema.data?.input_mode === 'tabular') setJsonValue(JSON.stringify(records, null, 2))
  }, [records, schema.data?.input_mode])

  const fields = schema.data?.fields || []
  const requiredFields = fields.filter((field) => field.required)
  const requiredTotal = records.length * requiredFields.length
  const requiredComplete = records.reduce((total, record) => total + requiredFields.filter((field) => hasValue(record[field.name])).length, 0)
  const isEeg = schema.data?.input_mode === 'eeg_recording'
  const readyToSubmit = isEeg ? Boolean(selectedRecording) : records.length > 0 && requiredComplete === requiredTotal

  const jsonSummary = useMemo(() => {
    try {
      const parsed = JSON.parse(jsonValue)
      if (!Array.isArray(parsed)) return { count: 0, error: 'El JSON debe ser una lista de registros.' }
      return { count: parsed.length, error: '' }
    } catch {
      return { count: 0, error: 'El JSON todavía no es válido.' }
    }
  }, [jsonValue])

  function chooseExample(index) {
    const example = schema.data?.examples?.[index]
    if (!example) return
    setSelectedExample(index)
    setRecords([{ ...example }])
  }

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
      message.success('JSON aplicado')
    } catch (parseError) {
      message.error(parseError.message)
    }
  }

  function selectRecording(value) {
    setSelectedRecording(value)
    setRecords(value ? [{ recording: value }] : [])
  }

  async function submit() {
    if (!records.length) return message.warning(isEeg ? 'Selecciona un registro EEG' : 'Agrega al menos un registro')
    if (!isEeg) {
      const missing = records.flatMap((record, index) => fields.filter((field) => field.required && !hasValue(record[field.name])).map((field) => `Registro ${index + 1}: ${field.name}`))
      if (missing.length) return message.warning(`Completa ${missing.join(', ')}`)
    }
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
    { title: 'ID', dataIndex: 'id', render: (value) => <Text copyable={{ text: value }} className="mono">{value.slice(0, 8)}...</Text> },
    { title: 'Modelo', dataIndex: 'registered_model_name' },
    { title: 'Versión solicitada', dataIndex: 'version_or_alias', render: (value) => <Tag>{value}</Tag> },
    { title: 'Versión usada', dataIndex: 'resolved_model_version', render: (value) => value ? <Tag color="blue">v{value}</Tag> : '-' },
    { title: 'Entradas', dataIndex: 'input_payload', align: 'center', render: (value) => value.length },
    { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
    { title: '', render: (_, job) => {
      const jobTracking = tracking.data?.[job.id]
      return <Space wrap>
        {jobTracking?.mlflow?.path && <Button size="small" href={mlflowTrackingUrl(jobTracking.mlflow)} target="_blank" icon={<ExperimentOutlined />}>Ver en MLflow</Button>}
        {jobTracking?.prefect?.path && <Button size="small" href={prefectTrackingUrl(jobTracking.prefect)} target="_blank" icon={<SettingOutlined />}>Ver en Prefect</Button>}
      </Space>
    } },
  ]

  const manualTabItems = [
    {
      key: 'manual',
      label: 'Editar datos',
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
        <Button className="add-record-button" icon={<PlusOutlined />} onClick={() => setRecords((current) => [...current, { ...schema.data.example }])}>Agregar otro registro</Button>
      </>,
    },
    {
      key: 'json',
      label: 'JSON avanzado',
      children: <>
        <div className="json-editor-summary">
          <Text type={jsonSummary.error ? 'danger' : 'secondary'}>{jsonSummary.error || `${jsonSummary.count} ${jsonSummary.count === 1 ? 'registro detectado' : 'registros detectados'}`}</Text>
          <Button disabled={Boolean(jsonSummary.error) || jsonSummary.count === 0} onClick={applyJson}>Aplicar JSON</Button>
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
          <Card className="prediction-composer-card" title="Nueva predicción">
            <Form layout="vertical">
              <Form.Item label="1. Modelo" required><Select showSearch value={modelName || undefined} onChange={setModelName} placeholder="Selecciona un modelo" options={(registry.data || []).map((item) => ({ value: item.name, label: item.name }))} /></Form.Item>
              <Form.Item label="2. Versión" required><Select value={selector || undefined} onChange={setSelector} options={selectorOptions} /></Form.Item>
            </Form>

            {schema.loading && <Card loading />}

            {schema.data && <Alert type="info" showIcon message={`${schema.data.dataset_name} - v${schema.data.dataset_version}`} description={schema.data.help} style={{ marginBottom: 16 }} />}

            {schema.data && isEeg && <>
              {schema.data.recordings?.length ? <>
                <Form layout="vertical">
                  <Form.Item label="3. Registro EEG" required extra="No necesitas escribir edad, sexo, participant_id ni otras variables manualmente.">
                    <Select
                      showSearch
                      optionFilterProp="label"
                      value={selectedRecording || undefined}
                      onChange={selectRecording}
                      placeholder="Selecciona el registro que quieres analizar"
                      options={(schema.data.recordings || []).map((item) => ({ value: item.value, label: item.label }))}
                    />
                  </Form.Item>
                </Form>
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Space direction="vertical" size={4}>
                    <Text type="secondary">NeuroOps hará automáticamente</Text>
                    <Text>Filtrado, referencia, segmentación, extracción PSD y predicción con la versión fijada del modelo.</Text>
                  </Space>
                </Card>
                <Button type="primary" size="large" icon={<SendOutlined />} loading={saving} disabled={!readyToSubmit} block onClick={submit}>Ejecutar predicción EEG</Button>
              </> : <Alert type="warning" showIcon message="No hay registros EEG compatibles disponibles en la versión del dataset usada para entrenar este modelo." />}
            </>}

            {schema.data && !isEeg && fields.length > 0 && <>
              {schema.data.examples?.length > 0 && <Card size="small" title="3. Datos de prueba" style={{ marginBottom: 16 }}>
                <Paragraph type="secondary">Ya cargamos automáticamente una fila real del dataset. Puedes ejecutar directamente o escoger otro ejemplo.</Paragraph>
                <Form layout="vertical">
                  <Form.Item label="Ejemplo">
                    <Select value={selectedExample} onChange={chooseExample} options={schema.data.examples.map((_, index) => ({ value: index, label: `Ejemplo ${index + 1}` }))} />
                  </Form.Item>
                </Form>
                <Table
                  size="small"
                  rowKey={() => 'selected-example'}
                  dataSource={records.slice(0, 1)}
                  columns={fields.slice(0, 5).map((field) => ({ title: field.name, dataIndex: field.name, ellipsis: true, render: previewValue }))}
                  pagination={false}
                  scroll={{ x: true }}
                />
                {fields.length > 5 && <Text type="secondary">Se muestran 5 de {fields.length} variables. El resto también está cargado.</Text>}
              </Card>}

              {!schema.data.examples?.length && <Alert type="warning" showIcon message="No encontramos una fila completa para precargar. Puedes introducir los datos manualmente." style={{ marginBottom: 16 }} />}

              <Button type="primary" size="large" icon={<SendOutlined />} loading={saving} disabled={!readyToSubmit} block onClick={submit}>Ejecutar predicción</Button>

              <Tabs style={{ marginTop: 18 }} items={manualTabItems} />
            </>}

            {schema.data && !isEeg && fields.length === 0 && <Alert type="warning" showIcon message="Este modelo no expone un esquema de entrada tabular." />}
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
              scroll={{ x: 980 }}
              expandable={{
                expandedRowRender: (job) => <Row gutter={[16, 16]}><Col xs={24} lg={12}><Title level={5}>Entrada utilizada</Title><pre className="json-preview">{JSON.stringify(job.input_payload, null, 2)}</pre></Col><Col xs={24} lg={12}><Title level={5}>Resultado</Title>{job.result_payload ? resultSummary(job) : job.error_message ? <Alert type="error" showIcon message={job.error_message} /> : <Empty description="Procesando" />}</Col></Row>,
              }}
              locale={{ emptyText: <Empty description="Aún no hay predicciones" /> }}
            />
          </Card>
        </Col>
      </Row>
    </>
  )
}
