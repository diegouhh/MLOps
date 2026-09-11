import {
  AppstoreOutlined,
  CheckCircleOutlined,
  ExperimentOutlined,
  SearchOutlined,
  SlidersOutlined,
  SwapOutlined,
} from '@ant-design/icons'
import {
  Alert,
  App,
  Button,
  Card,
  Checkbox,
  Col,
  Collapse,
  Descriptions,
  Drawer,
  Empty,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { titleCase } from '../utils/format'

const { Paragraph, Text, Title } = Typography

const taskLabels = {
  classification: 'Clasificación',
  regression: 'Regresión',
  preprocessing: 'Preprocesamiento',
}

const dataTypeLabels = {
  tabular: 'Tabular',
  eeg_bids: 'EEG BIDS',
}

const familyLabels = {
  linear: 'Lineal',
  ensemble: 'Ensamble',
  kernel: 'Kernel',
  neighbors: 'Vecinos',
  other: 'Otro',
}

const levelLabels = {
  high: 'Alta',
  medium: 'Media',
  low: 'Baja',
}

const costLabels = {
  high: 'Alto',
  medium: 'Medio',
  low: 'Bajo',
}

function labels(values, dictionary) {
  return (values || []).map((value) => dictionary[value] || titleCase(value))
}

function uniqueOptions(items, key, dictionary) {
  const values = [...new Set((items || []).map((item) => item[key]).filter(Boolean))]
  return values.map((value) => ({ value, label: dictionary[value] || titleCase(value) }))
}

function uniqueTaskOptions(items) {
  const values = [...new Set((items || []).flatMap((item) => item.task_types || []))]
  return values.map((value) => ({ value, label: taskLabels[value] || titleCase(value) }))
}

function compactSteps(steps) {
  if (!steps?.length) return 'Sin etapas declaradas'
  const visible = steps.slice(0, 4).map(titleCase)
  const remaining = steps.length - visible.length
  return `${visible.join(' → ')}${remaining > 0 ? ` +${remaining}` : ''}`
}

function listItems(values) {
  if (!values?.length) return <Text type="secondary">Sin información adicional</Text>
  return (
    <ul className="catalog-detail-list">
      {values.map((value) => <li key={value}>{value}</li>)}
    </ul>
  )
}

export default function CatalogPage({ type }) {
  const { message } = App.useApp()
  const isPipeline = type === 'pipelines'
  const navigate = useNavigate()
  const pipelines = useResource('/pipelines')
  const models = useResource('/models/catalog')
  const datasets = useResource(isPipeline ? '/datasets' : null)
  const resource = isPipeline ? pipelines : models

  const [search, setSearch] = useState('')
  const [taskFilter, setTaskFilter] = useState('')
  const [stateFilter, setStateFilter] = useState('')
  const [dataTypeFilter, setDataTypeFilter] = useState('')
  const [familyFilter, setFamilyFilter] = useState('')
  const [interpretabilityFilter, setInterpretabilityFilter] = useState('')
  const [costFilter, setCostFilter] = useState('')
  const [pipelineFilter, setPipelineFilter] = useState('')
  const [datasetFilter, setDatasetFilter] = useState('')
  const [selected, setSelected] = useState(null)
  const [comparisonOpen, setComparisonOpen] = useState(false)
  const [comparedModelIds, setComparedModelIds] = useState([])

  const pipelineById = useMemo(
    () => Object.fromEntries((pipelines.data || []).map((item) => [item.id, item])),
    [pipelines.data],
  )
  const modelById = useMemo(
    () => Object.fromEntries((models.data || []).map((item) => [item.id, item])),
    [models.data],
  )
  const selectedDataset = useMemo(
    () => (datasets.data || []).find((item) => item.id === datasetFilter),
    [datasetFilter, datasets.data],
  )
  const compatiblePipelinesByModel = useMemo(() => {
    const result = {}
    for (const model of models.data || []) {
      result[model.id] = (pipelines.data || []).filter(
        (pipeline) => pipeline.supported_models?.includes(model.id),
      )
    }
    return result
  }, [models.data, pipelines.data])

  const filtered = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase()
    return (resource.data || []).filter((item) => {
      const text = `${item.display_name} ${item.description} ${item.id}`.toLowerCase()
      if (normalizedSearch && !text.includes(normalizedSearch)) return false
      if (taskFilter && !(item.task_types || []).includes(taskFilter)) return false
      if (stateFilter === 'available' && !item.available) return false
      if (stateFilter === 'unavailable' && item.available) return false

      if (isPipeline) {
        if (dataTypeFilter && item.data_type !== dataTypeFilter) return false
        return true
      }

      if (familyFilter && item.family !== familyFilter) return false
      if (interpretabilityFilter && item.interpretability !== interpretabilityFilter) {
        return false
      }
      if (costFilter && item.compute_cost !== costFilter) return false
      if (
        pipelineFilter
        && !pipelineById[pipelineFilter]?.supported_models?.includes(item.id)
      ) {
        return false
      }
      return true
    })
  }, [
    costFilter,
    dataTypeFilter,
    familyFilter,
    interpretabilityFilter,
    isPipeline,
    pipelineById,
    pipelineFilter,
    resource.data,
    search,
    stateFilter,
    taskFilter,
  ])

  const comparedModels = comparedModelIds.map((modelId) => modelById[modelId]).filter(Boolean)
  const resourceError = resource.error || pipelines.error || models.error || datasets.error

  function resetFilters() {
    setSearch('')
    setTaskFilter('')
    setStateFilter('')
    setDataTypeFilter('')
    setFamilyFilter('')
    setInterpretabilityFilter('')
    setCostFilter('')
    setPipelineFilter('')
    setDatasetFilter('')
  }

  function pipelineCompatibility(item) {
    if (!selectedDataset) return null
    return item.data_type === selectedDataset.data_type
  }

  function openPipelineExperiment(item) {
    if (!item.available) {
      setSelected(item)
      return
    }
    if (pipelineCompatibility(item) === false) {
      setSelected(item)
      return
    }
    const params = new URLSearchParams({ pipeline: item.id })
    if (selectedDataset) params.set('dataset', selectedDataset.id)
    navigate(`/experiments/new?${params.toString()}`)
  }

  function openModelExperiment(item) {
    navigate(`/experiments/new?model=${item.id}`)
  }

  function toggleCompare(modelId, checked) {
    if (!checked) {
      setComparedModelIds((current) => current.filter((id) => id !== modelId))
      return
    }
    if (comparedModelIds.includes(modelId)) return
    if (comparedModelIds.length >= 3) {
      message.warning('Puedes comparar hasta 3 modelos a la vez')
      return
    }
    setComparedModelIds((current) => [...current, modelId])
  }

  function createExperimentFromComparison() {
    if (!comparedModelIds.length) return
    const params = new URLSearchParams({ models: comparedModelIds.join(',') })
    if (pipelineFilter) params.set('pipeline', pipelineFilter)
    navigate(`/experiments/new?${params.toString()}`)
  }

  function pipelineAction(item, compact = false) {
    const compatible = pipelineCompatibility(item)
    if (!item.available) {
      return <Button onClick={() => setSelected(item)}>Ver requisitos</Button>
    }
    if (compatible === false) {
      return <Button onClick={() => setSelected(item)}>Ver compatibilidad</Button>
    }
    return (
      <Button type="primary" onClick={() => openPipelineExperiment(item)}>
        {compact ? 'Crear experimento' : 'Crear experimento con este pipeline'}
      </Button>
    )
  }

  function renderFilters() {
    return (
      <div className="catalog-filter-bar">
        <Input
          className="catalog-search"
          prefix={<SearchOutlined />}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder={`Buscar ${isPipeline ? 'pipeline' : 'modelo'}…`}
          allowClear
        />
        <Select
          allowClear
          className="catalog-filter"
          placeholder="Tarea"
          value={taskFilter || undefined}
          onChange={(value) => setTaskFilter(value || '')}
          options={uniqueTaskOptions(resource.data)}
        />
        <Select
          allowClear
          className="catalog-filter"
          placeholder="Estado"
          value={stateFilter || undefined}
          onChange={(value) => setStateFilter(value || '')}
          options={[
            { value: 'available', label: 'Disponible' },
            { value: 'unavailable', label: 'No disponible' },
          ]}
        />
        {isPipeline ? (
          <>
            <Select
              allowClear
              className="catalog-filter"
              placeholder="Entrada"
              value={dataTypeFilter || undefined}
              onChange={(value) => setDataTypeFilter(value || '')}
              options={uniqueOptions(resource.data, 'data_type', dataTypeLabels)}
            />
            <Select
              allowClear
              className="catalog-filter catalog-filter-wide"
              placeholder="Comprobar con dataset"
              value={datasetFilter || undefined}
              onChange={(value) => setDatasetFilter(value || '')}
              options={(datasets.data || []).map((item) => ({
                value: item.id,
                label: item.name,
              }))}
            />
          </>
        ) : (
          <>
            <Select
              allowClear
              className="catalog-filter"
              placeholder="Familia"
              value={familyFilter || undefined}
              onChange={(value) => setFamilyFilter(value || '')}
              options={uniqueOptions(resource.data, 'family', familyLabels)}
            />
            <Select
              allowClear
              className="catalog-filter"
              placeholder="Interpretabilidad"
              value={interpretabilityFilter || undefined}
              onChange={(value) => setInterpretabilityFilter(value || '')}
              options={uniqueOptions(resource.data, 'interpretability', levelLabels)}
            />
            <Select
              allowClear
              className="catalog-filter"
              placeholder="Costo"
              value={costFilter || undefined}
              onChange={(value) => setCostFilter(value || '')}
              options={uniqueOptions(resource.data, 'compute_cost', costLabels)}
            />
            <Select
              allowClear
              className="catalog-filter catalog-filter-wide"
              placeholder="Pipeline"
              value={pipelineFilter || undefined}
              onChange={(value) => setPipelineFilter(value || '')}
              options={(pipelines.data || []).map((item) => ({
                value: item.id,
                label: item.display_name,
              }))}
            />
          </>
        )}
        <Button icon={<SlidersOutlined />} onClick={resetFilters}>Limpiar</Button>
        <Text type="secondary" className="catalog-result-count">
          {filtered.length} resultados
        </Text>
      </div>
    )
  }

  function renderPipelineCard(item) {
    const compatible = pipelineCompatibility(item)
    return (
      <Card
        className={`catalog-card catalog-card-compact ${!item.available ? 'catalog-card-unavailable' : ''}`}
        title={(
          <span className="catalog-card-title">
            <AppstoreOutlined />
            <span title={item.display_name}>{item.display_name}</span>
          </span>
        )}
        extra={<StatusTag value={item.available ? 'ready' : 'unavailable'} />}
      >
        <Paragraph className="catalog-description" ellipsis={{ rows: 2 }}>
          {item.description}
        </Paragraph>
        <div className="catalog-tags">
          <Tag color="blue">{dataTypeLabels[item.data_type] || titleCase(item.data_type)}</Tag>
          {(item.task_types || []).slice(0, 2).map((task) => (
            <Tag key={task}>{taskLabels[task] || titleCase(task)}</Tag>
          ))}
          <Tag>v{item.version}</Tag>
          {compatible === true && <Tag color="green">Compatible con dataset</Tag>}
          {compatible === false && <Tag color="red">No compatible con dataset</Tag>}
        </div>
        <div className="catalog-compact-line" title={(item.steps || []).map(titleCase).join(' → ')}>
          {compactSteps(item.steps)}
        </div>
        <div className="catalog-facts">
          <span><strong>{item.supported_models?.length || 0}</strong> modelos compatibles</span>
          <span>{item.category || 'Pipeline'}</span>
        </div>
        {!item.available && item.required_packages?.length > 0 && (
          <Text type="secondary" className="catalog-inline-note">
            Faltan requisitos opcionales. Revisa los detalles para ver cuáles.
          </Text>
        )}
        <div className="catalog-actions catalog-actions-compact">
          <Button onClick={() => setSelected(item)}>Ver detalles</Button>
          {pipelineAction(item, true)}
        </div>
      </Card>
    )
  }

  function renderModelCard(item) {
    const compatiblePipelines = compatiblePipelinesByModel[item.id] || []
    const checked = comparedModelIds.includes(item.id)
    return (
      <Card
        className={`catalog-card catalog-card-compact ${!item.available ? 'catalog-card-unavailable' : ''}`}
        title={(
          <span className="catalog-card-title">
            <AppstoreOutlined />
            <span title={item.display_name}>{item.display_name}</span>
          </span>
        )}
        extra={(
          <Space size={4}>
            <Checkbox
              checked={checked}
              onChange={(event) => toggleCompare(item.id, event.target.checked)}
              aria-label={`Comparar ${item.display_name}`}
            />
            <StatusTag value={item.available ? 'ready' : 'unavailable'} />
          </Space>
        )}
      >
        <Paragraph className="catalog-description" ellipsis={{ rows: 2 }}>
          {item.description}
        </Paragraph>
        <div className="catalog-tags">
          {(item.task_types || []).slice(0, 2).map((task) => (
            <Tag color="purple" key={task}>{taskLabels[task] || titleCase(task)}</Tag>
          ))}
          <Tag>{familyLabels[item.family] || titleCase(item.family || 'other')}</Tag>
          {item.supports_probability && <Tag color="blue">Probabilidades</Tag>}
        </div>
        <div className="catalog-facts catalog-facts-three">
          <span><strong>{levelLabels[item.interpretability] || '—'}</strong> interpretabilidad</span>
          <span><strong>{costLabels[item.compute_cost] || '—'}</strong> costo</span>
          <span><strong>{compatiblePipelines.length}</strong> pipelines</span>
        </div>
        <Text type="secondary" className="catalog-inline-note">
          {(item.strengths || [])[0] || 'Consulta la ficha para ver ventajas y limitaciones.'}
        </Text>
        <div className="catalog-actions catalog-actions-compact">
          <Button onClick={() => setSelected(item)}>Ver detalles</Button>
          <Button
            type="primary"
            disabled={!item.available}
            onClick={() => openModelExperiment(item)}
          >
            Crear experimento
          </Button>
        </div>
      </Card>
    )
  }

  function renderPipelineDetails(item) {
    const compatible = pipelineCompatibility(item)
    const modelNames = (item.supported_models || []).map(
      (modelId) => modelById[modelId]?.display_name || modelId,
    )
    const configFields = Object.keys(item.config_schema?.properties || {})
    return (
      <>
        {!item.available && (
          <Alert
            type="warning"
            showIcon
            message="Pipeline no disponible"
            description={item.unavailable_reason || 'No se cumplen todos los requisitos.'}
            style={{ marginBottom: 16 }}
          />
        )}
        {compatible === false && selectedDataset && (
          <Alert
            type="error"
            showIcon
            message="No es compatible con el dataset seleccionado"
            description={`${selectedDataset.name} usa ${dataTypeLabels[selectedDataset.data_type] || selectedDataset.data_type}, mientras este pipeline espera ${item.input_label || dataTypeLabels[item.data_type] || item.data_type}.`}
            style={{ marginBottom: 16 }}
          />
        )}
        <Descriptions
          size="small"
          column={1}
          items={[
            { key: 'status', label: 'Estado', children: <StatusTag value={item.available ? 'ready' : 'unavailable'} /> },
            { key: 'input', label: 'Entrada', children: item.input_label || dataTypeLabels[item.data_type] || item.data_type },
            { key: 'task', label: 'Tarea', children: labels(item.task_types, taskLabels).join(', ') || 'No declarada' },
            { key: 'version', label: 'Versión', children: item.version },
            { key: 'category', label: 'Categoría', children: item.category || 'Pipeline' },
          ]}
        />
        <div className="catalog-detail-section">
          <Title level={5}>Etapas</Title>
          <ol className="catalog-step-list">
            {(item.steps || []).map((step) => <li key={step}>{titleCase(step)}</li>)}
          </ol>
        </div>
        <div className="catalog-detail-section">
          <Title level={5}>Modelos compatibles</Title>
          <Space wrap size={[6, 6]}>
            {modelNames.length
              ? modelNames.map((name) => <Tag key={name}>{name}</Tag>)
              : <Text type="secondary">No hay modelos habilitados para este pipeline.</Text>}
          </Space>
        </div>
        {item.required_packages?.length > 0 && (
          <div className="catalog-detail-section">
            <Title level={5}>Requisitos</Title>
            <Space wrap size={[6, 6]}>
              {item.required_packages.map((dependency) => <Tag key={dependency}>{dependency}</Tag>)}
            </Space>
          </div>
        )}
        <Collapse
          className="catalog-technical-collapse"
          items={[
            {
              key: 'technical',
              label: 'Configuración técnica',
              children: configFields.length ? (
                <Space wrap size={[6, 6]}>
                  {configFields.map((field) => <Tag key={field}>{titleCase(field)}</Tag>)}
                </Space>
              ) : <Text type="secondary">Este pipeline no expone parámetros configurables.</Text>,
            },
          ]}
        />
      </>
    )
  }

  function renderModelDetails(item) {
    const compatiblePipelines = compatiblePipelinesByModel[item.id] || []
    return (
      <>
        {!item.available && (
          <Alert
            type="warning"
            showIcon
            message="Modelo no disponible"
            description={item.unavailable_reason || 'No se cumplen todos los requisitos.'}
            style={{ marginBottom: 16 }}
          />
        )}
        <Descriptions
          size="small"
          column={1}
          items={[
            { key: 'status', label: 'Estado', children: <StatusTag value={item.available ? 'ready' : 'unavailable'} /> },
            { key: 'task', label: 'Tarea', children: labels(item.task_types, taskLabels).join(', ') || 'No declarada' },
            { key: 'family', label: 'Familia', children: familyLabels[item.family] || titleCase(item.family || 'other') },
            { key: 'interpretability', label: 'Interpretabilidad', children: levelLabels[item.interpretability] || 'No declarada' },
            { key: 'cost', label: 'Costo computacional', children: costLabels[item.compute_cost] || 'No declarado' },
            { key: 'probability', label: 'Probabilidades', children: item.supports_probability ? 'Sí' : 'No' },
          ]}
        />
        <div className="catalog-detail-grid">
          <div className="catalog-detail-section">
            <Title level={5}>Ventajas</Title>
            {listItems(item.strengths)}
          </div>
          <div className="catalog-detail-section">
            <Title level={5}>Limitaciones</Title>
            {listItems(item.limitations)}
          </div>
        </div>
        <div className="catalog-detail-section">
          <Title level={5}>Pipelines compatibles</Title>
          <Space wrap size={[6, 6]}>
            {compatiblePipelines.length
              ? compatiblePipelines.map((pipeline) => (
                <Tag key={pipeline.id}>{pipeline.display_name}</Tag>
              ))
              : <Text type="secondary">No hay pipelines compatibles declarados.</Text>}
          </Space>
        </div>
        <Collapse
          className="catalog-technical-collapse"
          items={[
            {
              key: 'parameters',
              label: 'Parámetros configurables',
              children: item.editable_parameters?.length ? (
                <Space wrap size={[6, 6]}>
                  {item.editable_parameters.map((parameter) => (
                    <Tag key={parameter}>
                      {parameter}: {String(item.default_parameters?.[parameter] ?? '—')}
                    </Tag>
                  ))}
                </Space>
              ) : <Text type="secondary">No hay parámetros editables.</Text>,
            },
          ]}
        />
      </>
    )
  }

  const comparisonColumns = [
    { title: 'Característica', dataIndex: 'label', key: 'label', width: 170 },
    ...comparedModels.map((model) => ({
      title: model.display_name,
      dataIndex: model.id,
      key: model.id,
      width: 220,
    })),
  ]

  const comparisonRows = [
    {
      key: 'family',
      label: 'Familia',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        familyLabels[model.family] || titleCase(model.family || 'other'),
      ])),
    },
    {
      key: 'interpretability',
      label: 'Interpretabilidad',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        levelLabels[model.interpretability] || 'No declarada',
      ])),
    },
    {
      key: 'cost',
      label: 'Costo computacional',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        costLabels[model.compute_cost] || 'No declarado',
      ])),
    },
    {
      key: 'probability',
      label: 'Probabilidades',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        model.supports_probability ? 'Sí' : 'No',
      ])),
    },
    {
      key: 'pipelines',
      label: 'Pipelines compatibles',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        String(compatiblePipelinesByModel[model.id]?.length || 0),
      ])),
    },
    {
      key: 'strength',
      label: 'Ventaja principal',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        model.strengths?.[0] || 'No declarada',
      ])),
    },
    {
      key: 'limitation',
      label: 'Limitación principal',
      ...Object.fromEntries(comparedModels.map((model) => [
        model.id,
        model.limitations?.[0] || 'No declarada',
      ])),
    },
  ]

  return (
    <>
      <PageHeading
        title={isPipeline ? 'Catálogo de pipelines' : 'Catálogo de modelos'}
        description={isPipeline
          ? 'Explora, filtra y comprueba qué pipeline encaja con tus datos.'
          : 'Compara modelos y elige una opción adecuada antes de crear el experimento.'}
        actions={(
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            onClick={() => navigate('/experiments/new')}
          >
            Crear experimento
          </Button>
        )}
      />
      {resourceError && (
        <Alert type="error" showIcon message={resourceError} style={{ marginBottom: 16 }} />
      )}
      {renderFilters()}
      {selectedDataset && isPipeline && (
        <div className="catalog-context-strip">
          <CheckCircleOutlined />
          <span>
            Comprobando compatibilidad con <strong>{selectedDataset.name}</strong>
            {' '}({dataTypeLabels[selectedDataset.data_type] || selectedDataset.data_type}).
          </span>
        </div>
      )}
      {filtered.length === 0 && !resource.loading ? (
        <Empty description="No hay resultados con estos filtros" />
      ) : (
        <Row gutter={[14, 14]}>
          {filtered.map((item) => (
            <Col xs={24} md={12} xl={8} key={item.id}>
              {isPipeline ? renderPipelineCard(item) : renderModelCard(item)}
            </Col>
          ))}
        </Row>
      )}

      {!isPipeline && comparedModelIds.length > 0 && (
        <div className="catalog-comparison-bar">
          <div>
            <strong>{comparedModelIds.length}</strong>
            <span>{comparedModelIds.length === 1 ? ' modelo seleccionado' : ' modelos seleccionados'}</span>
          </div>
          <Space wrap>
            <Button onClick={() => setComparedModelIds([])}>Limpiar</Button>
            <Button
              icon={<SwapOutlined />}
              disabled={comparedModelIds.length < 2}
              onClick={() => setComparisonOpen(true)}
            >
              Comparar
            </Button>
            <Button
              type="primary"
              icon={<ExperimentOutlined />}
              onClick={createExperimentFromComparison}
            >
              Crear experimento
            </Button>
          </Space>
        </div>
      )}

      <Drawer
        title={selected?.display_name}
        open={Boolean(selected)}
        onClose={() => setSelected(null)}
        width={540}
        extra={selected && (
          isPipeline
            ? pipelineAction(selected)
            : (
              <Button
                type="primary"
                disabled={!selected.available}
                onClick={() => openModelExperiment(selected)}
              >
                Crear experimento
              </Button>
            )
        )}
      >
        {selected && (
          <>
            <Paragraph>{selected.description}</Paragraph>
            {isPipeline ? renderPipelineDetails(selected) : renderModelDetails(selected)}
          </>
        )}
      </Drawer>

      <Modal
        title="Comparar modelos"
        open={comparisonOpen}
        onCancel={() => setComparisonOpen(false)}
        width={980}
        footer={(
          <Space>
            <Button onClick={() => setComparisonOpen(false)}>Cerrar</Button>
            <Button
              type="primary"
              icon={<ExperimentOutlined />}
              onClick={createExperimentFromComparison}
            >
              Crear experimento con estos modelos
            </Button>
          </Space>
        )}
      >
        <Paragraph type="secondary">
          Compara características orientativas antes de evaluarlos con el mismo dataset y la
          misma estrategia de validación.
        </Paragraph>
        <Table
          className="catalog-comparison-table"
          columns={comparisonColumns}
          dataSource={comparisonRows}
          pagination={false}
          size="small"
          scroll={{ x: 650 }}
        />
      </Modal>
    </>
  )
}
