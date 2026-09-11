import { ArrowLeftOutlined, ArrowRightOutlined, CheckOutlined, ExperimentOutlined } from '@ant-design/icons'
import { Alert, App, Button, Card, Checkbox, Col, Descriptions, Form, Input, InputNumber, Radio, Row, Select, Space, Steps, Tag, Typography } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import SchemaFields from '../components/SchemaFields'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { apiRequest, jsonOptions } from '../services/api'
import { titleCase } from '../utils/format'
import { schemaDefaults } from '../utils/schema'

const { Paragraph, Text, Title } = Typography

const stepItems = [
  { title: 'Dataset' },
  { title: 'Pipeline' },
  { title: 'Configuración' },
  { title: 'Modelos' },
  { title: 'Validación' },
  { title: 'Revisar' },
]

export default function NewExperimentPage() {
  const { message } = App.useApp()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const datasets = useResource('/datasets')
  const pipelines = useResource('/pipelines')
  const models = useResource('/models/catalog')
  const [step, setStep] = useState(0)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [name, setName] = useState('')
  const [datasetId, setDatasetId] = useState(searchParams.get('dataset') || '')
  const [pipelineId, setPipelineId] = useState(searchParams.get('pipeline') || '')
  const [pipelineConfig, setPipelineConfig] = useState({})
  const [registeredModelName, setRegisteredModelName] = useState('')
  const [selectedModels, setSelectedModels] = useState([])
  const [modelParameters, setModelParameters] = useState({})
  const [validation, setValidation] = useState({ strategy: 'train_test_split', test_size: 0.2, n_splits: 5, metric: 'f1_macro', seed: 42 })

  const dataset = (datasets.data || []).find((item) => item.id === datasetId)
  const pipeline = (pipelines.data || []).find((item) => item.id === pipelineId)
  const columns = dataset?.versions.at(-1)?.schema_summary?.columns || []
  const compatiblePipelines = (pipelines.data || []).filter((item) => item.available && (!dataset || item.data_type === dataset.data_type))
  const compatibleModels = (models.data || []).filter(
    (item) => item.available && (!pipeline || pipeline.supported_models.includes(item.id)),
  )
  const isEegPipeline = pipeline?.data_type === 'eeg_bids'

  useEffect(() => {
    if (!models.data || selectedModels.length) return
    const requestedMany = (searchParams.get('models') || '')
      .split(',')
      .map((value) => value.trim())
      .filter(Boolean)
    const requestedOne = searchParams.get('model')
    const requested = requestedMany.length
      ? requestedMany
      : [requestedOne].filter(Boolean)
    const selected = requested
      .map((modelId) => models.data.find((item) => item.id === modelId))
      .filter(Boolean)
    if (!selected.length) return
    const unique = [...new Map(selected.map((item) => [item.id, item])).values()]
    setSelectedModels(unique.map((item) => item.id))
    setModelParameters(Object.fromEntries(
      unique.map((item) => [item.id, { ...item.default_parameters }]),
    ))
  }, [models.data, searchParams, selectedModels.length])

  useEffect(() => {
    if (!pipeline) return
    setSelectedModels((current) => current.filter(
      (modelId) => pipeline.supported_models.includes(modelId),
    ))
    setPipelineConfig((current) => Object.keys(current).length ? current : schemaDefaults(pipeline.config_schema))
    if (pipeline.data_type === 'eeg_bids') {
      setValidation((current) => ['group_kfold', 'stratified_group_kfold'].includes(current.strategy)
        ? current
        : { ...current, strategy: 'stratified_group_kfold' })
    }
  }, [pipeline])

  function choosePipeline(id) {
    const next = (pipelines.data || []).find((item) => item.id === id)
    setPipelineId(id)
    setPipelineConfig(schemaDefaults(next?.config_schema))
    setRegisteredModelName(`neuroops-${id}`)
    setSelectedModels((current) => current.filter(
      (modelId) => !next || next.supported_models.includes(modelId),
    ))
  }

  function toggleModel(model, checked) {
    setSelectedModels((current) => checked ? [...current, model.id] : current.filter((id) => id !== model.id))
    setModelParameters((current) => ({ ...current, [model.id]: current[model.id] || { ...model.default_parameters } }))
  }

  function validateStep() {
    if (step === 0 && !datasetId) return 'Selecciona un dataset'
    if (step === 1 && !pipelineId) return 'Selecciona un pipeline disponible'
    if (step === 2) {
      const missing = (pipeline?.config_schema.required || []).filter((field) => pipelineConfig[field] === undefined || pipelineConfig[field] === null || pipelineConfig[field] === '')
      if (missing.length) return `Completa: ${missing.map(titleCase).join(', ')}`
    }
    if (step === 3 && !selectedModels.length) return 'Selecciona al menos un modelo'
    if (step === 4 && !isEegPipeline && ['group_kfold', 'stratified_group_kfold'].includes(validation.strategy) && !pipelineConfig.group_column) return 'Selecciona una columna de grupo antes de usar validación por grupos'
    return ''
  }

  function next() {
    const validationError = validateStep()
    setError(validationError)
    if (!validationError) setStep((current) => Math.min(current + 1, stepItems.length - 1))
  }

  async function execute() {
    setSaving(true)
    setError('')
    try {
      const validationConfig = validation.strategy === 'train_test_split'
        ? { test_size: Number(validation.test_size) }
        : { n_splits: Number(validation.n_splits) }
      const result = await apiRequest('/experiments', jsonOptions('POST', {
        name: name.trim() || `Experimento ${new Date().toLocaleString('es-CO')}`,
        dataset_id: datasetId,
        dataset_version: dataset.current_version,
        pipeline_id: pipelineId,
        registered_model_name: registeredModelName || `neuroops-${pipelineId}`,
        pipeline_config: pipelineConfig,
        models: selectedModels.map((modelId) => ({ model_id: modelId, parameters: modelParameters[modelId] || {} })),
        validation_strategy: validation.strategy,
        validation_config: validationConfig,
        primary_metric: validation.metric,
        random_seed: Number(validation.seed),
      }))
      message.success('Experimento enviado a Prefect')
      navigate(`/experiments/${result.id}`)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSaving(false)
    }
  }

  const reviewItems = useMemo(() => [
    { key: 'dataset', label: 'Dataset', children: dataset?.name },
    { key: 'pipeline', label: 'Pipeline', children: pipeline?.display_name },
    { key: 'registry', label: 'Modelo registrado', children: registeredModelName || `neuroops-${pipelineId}` },
    { key: 'models', label: 'Modelos', children: selectedModels.join(', ') },
    { key: 'validation', label: 'Validación', children: titleCase(validation.strategy) },
    { key: 'metric', label: 'Métrica principal', children: titleCase(validation.metric) },
    { key: 'seed', label: 'Semilla', children: validation.seed },
  ], [dataset, pipeline, pipelineId, registeredModelName, selectedModels, validation])

  return (
    <>
      <PageHeading title="Nuevo experimento" />
      <Steps current={step} items={stepItems} responsive style={{ marginBottom: 28 }} />
      <Card>
        {step === 0 && <>
          <Title level={3}>Selecciona el dataset</Title>
          <Row gutter={[14, 14]}>{(datasets.data || []).map((item) => <Col xs={24} md={12} xl={8} key={item.id}><Card className={`choice-card ${datasetId === item.id ? 'selected' : ''}`} onClick={() => { setDatasetId(item.id); setPipelineId(''); setPipelineConfig({}) }}><Space direction="vertical"><Space><strong>{item.name}</strong>{datasetId === item.id && <CheckOutlined style={{ color: '#1677ff' }} />}</Space><Text type="secondary">{item.data_type} - v{item.current_version}</Text><StatusTag value={item.status} /></Space></Card></Col>)}</Row>
        </>}
        {step === 1 && <>
          <Title level={3}>Selecciona el pipeline</Title>
          <Row gutter={[14, 14]}>{compatiblePipelines.map((item) => <Col xs={24} md={12} key={item.id}><Card className={`choice-card ${pipelineId === item.id ? 'selected' : ''}`} onClick={() => choosePipeline(item.id)}><Space direction="vertical"><Space><strong>{item.display_name}</strong>{pipelineId === item.id && <CheckOutlined style={{ color: '#1677ff' }} />}</Space><Text type="secondary">{item.description}</Text><Space wrap>{item.steps.map((itemStep) => <Tag key={itemStep}>{itemStep}</Tag>)}</Space></Space></Card></Col>)}</Row>
        </>}
        {step === 2 && <>
          <Title level={3}>Configura el pipeline</Title>
          <Form layout="vertical"><SchemaFields schema={pipeline?.config_schema} value={pipelineConfig} onChange={setPipelineConfig} columnOptions={columns} /></Form>
        </>}
        {step === 3 && <>
          <Title level={3}>Selecciona y configura modelos</Title>
          <Space direction="vertical" className="full-width" size="middle">
            {compatibleModels.map((model) => {
              const selected = selectedModels.includes(model.id)
              return <Card className="model-param-card" key={model.id} title={<Checkbox checked={selected} onChange={(event) => toggleModel(model, event.target.checked)}>{model.display_name}</Checkbox>} extra={model.supports_probability && <Tag color="blue">Probabilidades</Tag>}><Paragraph type="secondary">{model.description}</Paragraph>{selected && <Form layout="vertical"><SchemaFields schema={model.parameter_schema} value={modelParameters[model.id]} onChange={(next) => setModelParameters((current) => ({ ...current, [model.id]: next }))} /></Form>}</Card>
            })}
          </Space>
        </>}
        {step === 4 && <>
          <Title level={3}>Validación científica</Title>
          <Form layout="vertical">
            <Form.Item label="Estrategia"><Radio.Group value={validation.strategy} onChange={(event) => setValidation({ ...validation, strategy: event.target.value })}><Space direction="vertical"><Radio value="train_test_split" disabled={isEegPipeline}>Train/test estratificado</Radio><Radio value="stratified_kfold" disabled={isEegPipeline}>Stratified K-Fold</Radio><Radio value="group_kfold">Group K-Fold</Radio><Radio value="stratified_group_kfold">Stratified Group K-Fold</Radio></Space></Radio.Group></Form.Item>
            <Row gutter={16}>
              <Col xs={24} md={8}>{validation.strategy === 'train_test_split' ? <Form.Item label="Proporción de prueba"><InputNumber min={0.05} max={0.5} step={0.05} value={validation.test_size} onChange={(value) => setValidation({ ...validation, test_size: value })} className="full-width" /></Form.Item> : <Form.Item label="Número de folds"><InputNumber min={2} max={20} value={validation.n_splits} onChange={(value) => setValidation({ ...validation, n_splits: value })} className="full-width" /></Form.Item>}</Col>
              <Col xs={24} md={8}><Form.Item label="Métrica principal"><Select value={validation.metric} onChange={(value) => setValidation({ ...validation, metric: value })} options={['f1_macro', 'balanced_accuracy', 'accuracy', 'precision_macro', 'recall_macro', 'f1_weighted', 'roc_auc'].map((value) => ({ value, label: titleCase(value) }))} /></Form.Item></Col>
              <Col xs={24} md={8}><Form.Item label="Semilla"><InputNumber min={0} value={validation.seed} onChange={(value) => setValidation({ ...validation, seed: value })} className="full-width" /></Form.Item></Col>
            </Row>
            {isEegPipeline && <Alert type="info" showIcon message="Los experimentos EEG se validan por sujeto para evitar que épocas del mismo sujeto aparezcan en entrenamiento y validación." />}
            {!isEegPipeline && ['group_kfold', 'stratified_group_kfold'].includes(validation.strategy) && !pipelineConfig.group_column && <Alert type="warning" showIcon message="Esta estrategia necesita una columna de grupo en la configuración del pipeline." />}
          </Form>
        </>}
        {step === 5 && <>
          <Title level={3}>Revisa y ejecuta</Title>
          <Form layout="vertical">
            <Form.Item label="Nombre del experimento"><Input value={name} onChange={(event) => setName(event.target.value)} placeholder="Ej. Iris - comparación inicial" /></Form.Item>
            <Form.Item label="Nombre en Model Registry" required extra="Los siguientes experimentos con este nombre crearán nuevas versiones del mismo modelo."><Input value={registeredModelName} maxLength={200} onChange={(event) => setRegisteredModelName(event.target.value)} /></Form.Item>
          </Form>
          <Descriptions bordered column={{ xs: 1, md: 2 }} items={reviewItems} />
          <Title level={5} style={{ marginTop: 22 }}>Configuración del pipeline</Title><Space wrap>{Object.entries(pipelineConfig).map(([key, value]) => <Tag key={key}>{titleCase(key)}: {Array.isArray(value) ? value.join(', ') || '-' : String(value ?? '-')}</Tag>)}</Space>
        </>}
        {(error || datasets.error || pipelines.error || models.error) && <Alert type="error" showIcon message={error || datasets.error || pipelines.error || models.error} style={{ marginTop: 18 }} />}
        <div className="wizard-footer">
          <Button icon={<ArrowLeftOutlined />} disabled={step === 0 || saving} onClick={() => { setError(''); setStep((current) => current - 1) }}>Atrás</Button>
          {step < stepItems.length - 1 ? <Button type="primary" onClick={next}>Continuar <ArrowRightOutlined /></Button> : <Button type="primary" icon={<ExperimentOutlined />} loading={saving} onClick={execute}>Ejecutar experimento</Button>}
        </div>
      </Card>
    </>
  )
}
