import { AppstoreOutlined, CheckCircleOutlined, SearchOutlined } from '@ant-design/icons'
import { Alert, Button, Card, Col, Empty, Input, Modal, Row, Space, Steps, Tag, Typography } from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { titleCase } from '../utils/format'

const { Paragraph, Text, Title } = Typography

export default function CatalogPage({ type }) {
  const isPipeline = type === 'pipelines'
  const navigate = useNavigate()
  const resource = useResource(isPipeline ? '/pipelines' : '/models/catalog')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState(null)
  const filtered = useMemo(
    () => (resource.data || []).filter((item) => `${item.display_name} ${item.description} ${item.id}`.toLowerCase().includes(search.toLowerCase())),
    [resource.data, search],
  )

  function pipelineAction(item, compact = false) {
    if (item.available) {
      return <Button type="primary" onClick={() => navigate(`/experiments/new?pipeline=${item.id}`)}>{compact ? 'Usar' : 'Usar en experimento'}</Button>
    }
    return <Button disabled>No disponible</Button>
  }

  return (
    <>
      <PageHeading
        title={isPipeline ? 'Catálogo de pipelines' : 'Catálogo de modelos'}
        actions={<Button type="primary" onClick={() => navigate('/experiments/new')}>Crear experimento</Button>}
      />
      {resource.error && <Alert type="error" showIcon message={resource.error} style={{ marginBottom: 16 }} />}
      <div className="catalog-toolbar">
        <Input prefix={<SearchOutlined />} value={search} onChange={(event) => setSearch(event.target.value)} placeholder={`Buscar ${isPipeline ? 'pipeline' : 'modelo'}…`} allowClear />
        <Text type="secondary">{filtered.length} resultados</Text>
      </div>
      {filtered.length === 0 && !resource.loading ? <Empty description="No hay resultados" /> : (
        <Row gutter={[16, 16]}>
          {filtered.map((item) => (
            <Col xs={24} lg={12} xxl={8} key={item.id}>
              <Card
                className={`catalog-card ${!item.available ? 'catalog-card-unavailable' : ''}`}
                hoverable
                title={<span className="catalog-card-title"><AppstoreOutlined /><span title={item.display_name}>{item.display_name}</span></span>}
                extra={<StatusTag value={item.available ? 'ready' : 'unavailable'} />}
              >
                <Paragraph className="catalog-description" ellipsis={{ rows: 2 }}>{item.description}</Paragraph>
                <div className="catalog-tags">
                  {isPipeline ? <><Tag color="blue">{item.data_type}</Tag><Tag>v{item.version}</Tag><Tag color="cyan">Incluido</Tag></> : item.task_types.map((task) => <Tag color="purple" key={task}>{task}</Tag>)}
                </div>
                {isPipeline && !item.available && item.required_packages?.length > 0 && (
                  <div className="pipeline-dependencies">
                    <Text type="secondary">Requiere reconstruir la imagen con</Text>
                    <div className="dependency-list">{item.required_packages.map((dependency) => <Tag key={dependency}>{dependency}</Tag>)}</div>
                  </div>
                )}
                <div className="catalog-meta">
                  <Text type="secondary">{isPipeline ? `${item.steps.length} etapas · ${item.supported_models.length} modelos` : `${item.editable_parameters.length} parámetros · ${item.supports_probability ? 'con probabilidades' : 'sin probabilidades'}`}</Text>
                  <div className="catalog-actions">
                    <Button onClick={() => setSelected(item)}>Ver detalles</Button>
                    {isPipeline ? pipelineAction(item, true) : <Button type="primary" disabled={!item.available} onClick={() => navigate(`/experiments/new?model=${item.id}`)}>Usar</Button>}
                  </div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      )}
      <Modal
        title={selected?.display_name}
        open={Boolean(selected)}
        onCancel={() => setSelected(null)}
        footer={selected && (isPipeline ? pipelineAction(selected) : <Button type="primary" disabled={!selected.available} onClick={() => navigate(`/experiments/new?model=${selected.id}`)}>Usar en experimento</Button>)}
        width={720}
      >
        {selected && <>
          <Paragraph>{selected.description}</Paragraph>
          {!selected.available && <Alert type="warning" showIcon message={selected.unavailable_reason} description="Las dependencias de los pipelines se administran en pyproject.toml y se incluyen al construir la imagen; la interfaz no instala paquetes en ejecución." style={{ marginBottom: 16 }} />}
          {isPipeline ? <>
            <Title level={5}>Etapas</Title>
            <Steps direction="vertical" size="small" items={selected.steps.map((step) => ({ title: titleCase(step), icon: <CheckCircleOutlined /> }))} />
            <Title level={5}>Modelos compatibles</Title><Space wrap>{selected.supported_models.map((model) => <Tag key={model}>{model}</Tag>)}</Space>
          </> : <>
            <Title level={5}>Parámetros editables</Title>
            <Space wrap>{selected.editable_parameters.map((parameter) => <Tag key={parameter}>{parameter}: {String(selected.default_parameters[parameter])}</Tag>)}</Space>
          </>}
          {selected.required_packages?.length > 0 && <><Title level={5}>Dependencias incluidas</Title><Text code>{selected.required_packages.join(', ')}</Text></>}
        </>}
      </Modal>
    </>
  )
}
