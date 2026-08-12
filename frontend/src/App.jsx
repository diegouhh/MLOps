import { App as AntApp, ConfigProvider, Spin } from 'antd'
import esES from 'antd/locale/es_ES'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import AppLayout from './components/AppLayout'

const CatalogPage = lazy(() => import('./pages/CatalogPage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DatasetsPage = lazy(() => import('./pages/DatasetsPage'))
const ExperimentDetailPage = lazy(() => import('./pages/ExperimentDetailPage'))
const ExperimentsPage = lazy(() => import('./pages/ExperimentsPage'))
const ModelManagementPage = lazy(() => import('./pages/ModelManagementPage'))
const NewExperimentPage = lazy(() => import('./pages/NewExperimentPage'))
const PredictionsPage = lazy(() => import('./pages/PredictionsPage'))

const theme = {
  token: {
    colorPrimary: '#1677ff',
    colorInfo: '#1677ff',
    borderRadius: 8,
    colorBgLayout: '#f5f7fa',
    colorText: '#1f2937',
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  components: {
    Layout: { headerBg: '#ffffff', siderBg: '#ffffff' },
    Menu: { itemSelectedBg: '#e6f4ff', itemSelectedColor: '#1677ff' },
    Card: { headerBg: '#ffffff' },
  },
}

export default function App() {
  return (
    <ConfigProvider locale={esES} theme={theme}>
      <AntApp>
        <BrowserRouter>
          <Suspense fallback={<div style={{ minHeight: '100vh', display: 'grid', placeItems: 'center' }}><Spin size="large" /></div>}>
            <Routes>
              <Route path="/" element={<AppLayout />}>
                <Route index element={<Navigate to="/dashboard" replace />} />
                <Route path="dashboard" element={<DashboardPage />} />
                <Route path="datasets" element={<DatasetsPage />} />
                <Route path="catalog/pipelines" element={<CatalogPage type="pipelines" />} />
                <Route path="catalog/models" element={<CatalogPage type="models" />} />
                <Route path="experiments" element={<ExperimentsPage />} />
                <Route path="experiments/new" element={<NewExperimentPage />} />
                <Route path="experiments/:experimentId" element={<ExperimentDetailPage />} />
                <Route path="model-management" element={<ModelManagementPage />} />
                <Route path="predictions" element={<PredictionsPage />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AntApp>
    </ConfigProvider>
  )
}
