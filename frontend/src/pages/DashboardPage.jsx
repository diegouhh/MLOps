import {
  AppstoreOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  PlusOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { Alert, Button, Card, Col, Empty, List, Row, Space, Statistic, Table, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { formatDate } from '../utils/format'

const { Text, Title } = Typography

export default function DashboardPage() {
  const navigate = useNavigate()
  const experiments = useResource('/experiments', 5000)
  const registry = useResource('/registry/models', 5000)
  const dependencies = useResource('/system/dependencies', 10000)
  const rows = experiments.data || []
  const registeredVersions = (registry.data || []).reduce((total, item) => total + item.versions.length, 0)

  const columns = [
    { title: 'Experimento', dataIndex: 'name', render: (name, row) => <Button type="link" onClick={() => navigate(`/experiments/${row.id}`)}>{name}</Button> },
    { title: 'Pipeline', dataIndex: 'pipeline_id' },
    { title: 'Modelos', dataIndex: 'model_ids', align: 'center', render: (models) => models.length },
    { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
  ]

  return (
    <>
      <PageHeading
        title="Dashboard"
        actions={<Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>Nuevo experimento</Button>}
      />
      {experiments.error && <Alert type="error" showIcon message={experiments.error} style={{ marginBottom: 16 }} />}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} xl={6}><Card className="metric-card"><Statistic title="Experimentos" value={rows.length} prefix={<ExperimentOutlined />} /></Card></Col>
        <Col xs={24} sm={12} xl={6}><Card className="metric-card"><Statistic title="Completados" value={rows.filter((item) => item.status === 'completed').length} valueStyle={{ color: '#3f8600' }} prefix={<CheckCircleOutlined />} /></Card></Col>
        <Col xs={24} sm={12} xl={6}><Card className="metric-card"><Statistic title="Fallidos" value={rows.filter((item) => item.status === 'failed').length} valueStyle={{ color: '#cf1322' }} prefix={<CloseCircleOutlined />} /></Card></Col>
        <Col xs={24} sm={12} xl={6}><Card className="metric-card"><Statistic title="Versiones registradas" value={registeredVersions} prefix={<SettingOutlined />} /></Card></Col>
      </Row>
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} lg={16}>
          <Card className="section-card" title="Experimentos recientes" extra={<Button type="link" onClick={() => navigate('/experiments')}>Ver todos</Button>}>
            <Table columns={columns} dataSource={rows.slice(0, 6)} rowKey="id" pagination={false} loading={experiments.loading} scroll={{ x: 700 }} locale={{ emptyText: <Empty description="Aún no hay experimentos" /> }} />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card className="section-card" title="Servicios conectados">
            {dependencies.error && <Alert type="error" showIcon message={dependencies.error} />}
            {Object.entries(dependencies.data?.services || {}).map(([name, service]) => (
              <div className="service-row" key={name}>
                {name === 'mlflow' ? <ExperimentOutlined /> : name === 'prefect' ? <SettingOutlined /> : <AppstoreOutlined />}
                <div className="service-copy"><strong>{name === 'api' ? 'NeuroOps API' : name === 'mlflow' ? 'MLflow' : name === 'prefect_worker' ? 'Prefect Worker' : 'Prefect'}</strong><span>{service.url || 'Servicio interno'}</span></div>
                <StatusTag value={service.status} badge />
              </div>
            ))}
          </Card>
        </Col>
      </Row>
      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card className="navigation-card" hoverable onClick={() => navigate('/model-management')}>
            <Space align="start"><SettingOutlined style={{ fontSize: 26, color: '#1677ff' }} /><div><Title level={4}>Model management</Title><Text type="secondary">Compara versiones, revisa el champion y promueve candidatos con trazabilidad en MLflow.</Text></div></Space>
          </Card>
        </Col>
        <Col xs={24} md={12}>
          <Card className="navigation-card" hoverable onClick={() => navigate('/predictions')}>
            <Space align="start"><LineChartOutlined style={{ fontSize: 26, color: '#1677ff' }} /><div><Title level={4}>Predicciones</Title><Text type="secondary">Usa modelos registrados mediante formularios generados desde sus variables de entrada.</Text></div></Space>
          </Card>
        </Col>
      </Row>
    </>
  )
}
