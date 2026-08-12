import {
  AppstoreOutlined,
  BarChartOutlined,
  CloudServerOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  RocketOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import { Breadcrumb, Button, Grid, Layout, Menu, Space, Tooltip } from 'antd'
import { useMemo, useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import brainIcon from '../assets/brain.svg'
import { API_URL, MLFLOW_URL, PREFECT_URL } from '../services/api'

const { Header, Sider, Content } = Layout

const menuItems = [
  { key: '/dashboard', icon: <DashboardOutlined />, label: 'Dashboard' },
  { key: '/datasets', icon: <DatabaseOutlined />, label: 'Datasets' },
  {
    key: 'catalog',
    icon: <AppstoreOutlined />,
    label: 'Catálogos',
    children: [
      { key: '/catalog/pipelines', icon: <RocketOutlined />, label: 'Pipelines' },
      { key: '/catalog/models', icon: <BarChartOutlined />, label: 'Modelos' },
    ],
  },
  { key: '/experiments', icon: <ExperimentOutlined />, label: 'Experimentos' },
  { key: '/model-management', icon: <SettingOutlined />, label: 'Model management' },
  { key: '/predictions', icon: <LineChartOutlined />, label: 'Predicciones' },
]

const labels = {
  dashboard: 'Dashboard',
  datasets: 'Datasets',
  catalog: 'Catálogos',
  pipelines: 'Pipelines',
  models: 'Modelos',
  experiments: 'Experimentos',
  new: 'Nuevo experimento',
  'model-management': 'Model management',
  predictions: 'Predicciones',
}

export default function AppLayout() {
  const screens = Grid.useBreakpoint()
  const compact = !screens.lg
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()

  const selectedKey = useMemo(() => {
    if (location.pathname.startsWith('/experiments')) return '/experiments'
    return menuItems
      .flatMap((item) => item.children || [item])
      .find((item) => location.pathname.startsWith(item.key))?.key || '/dashboard'
  }, [location.pathname])

  const breadcrumbItems = useMemo(() => [
    { title: 'Inicio', onClick: () => navigate('/dashboard') },
    ...location.pathname.split('/').filter(Boolean).map((segment) => ({
      title: labels[segment] || (segment.length > 18 ? `${segment.slice(0, 8)}…` : segment),
    })),
  ], [location.pathname, navigate])

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header className="app-header">
        <div className="brand">
          <Button
            className="mobile-trigger"
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed((current) => !current)}
          />
          <img src={brainIcon} alt="NeuroOps" />
          <span className="brand-name">NeuroOps</span>
          <span className="brand-version">Research MLOps · v0.3.0</span>
        </div>
        <div className="header-actions">
          <Tooltip title="Abrir la documentación de la API">
            <Button href={`${API_URL.replace('/api/v1', '')}/docs`} target="_blank" icon={<CloudServerOutlined />}>
              <span className="header-label">API</span>
            </Button>
          </Tooltip>
          <Button href={MLFLOW_URL} target="_blank" icon={<ExperimentOutlined />}>
            <span className="header-label">MLflow ↗</span>
          </Button>
          <Button href={PREFECT_URL} target="_blank" icon={<SettingOutlined />}>
            <span className="header-label">Prefect ↗</span>
          </Button>
        </div>
      </Header>
      <Layout>
        <Sider
          className="app-sider"
          width={230}
          collapsedWidth={compact ? 0 : 80}
          breakpoint="lg"
          onBreakpoint={setCollapsed}
          collapsible={!compact}
          collapsed={collapsed}
          onCollapse={setCollapsed}
          theme="light"
        >
          <Menu
            className="app-menu"
            mode="inline"
            selectedKeys={[selectedKey]}
            defaultOpenKeys={['catalog']}
            items={menuItems}
            onClick={({ key }) => key.startsWith('/') && navigate(key)}
          />
        </Sider>
        <Layout className="app-content-shell">
          <Breadcrumb className="app-breadcrumb" items={breadcrumbItems} />
          <Content className="app-content">
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </Layout>
  )
}
