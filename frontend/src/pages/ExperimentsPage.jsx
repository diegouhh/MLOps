import { PlusOutlined, SearchOutlined } from '@ant-design/icons'
import { Alert, Button, Card, Empty, Input, Select, Space, Table, Tag } from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { formatDate, titleCase } from '../utils/format'

export default function ExperimentsPage() {
  const navigate = useNavigate()
  const resource = useResource('/experiments', 4000)
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('all')
  const rows = useMemo(() => (resource.data || []).filter((item) => {
    const matchesSearch = `${item.name} ${item.pipeline_id} ${item.model_ids.join(' ')}`.toLowerCase().includes(search.toLowerCase())
    return matchesSearch && (status === 'all' || item.status === status)
  }), [resource.data, search, status])

  const columns = [
    { title: 'Experimento', dataIndex: 'name', render: (name, row) => <Button type="link" onClick={() => navigate(`/experiments/${row.id}`)}>{name}</Button> },
    { title: 'Pipeline', dataIndex: 'pipeline_id', render: (value) => <Tag color="blue">{value}</Tag> },
    { title: 'Modelos candidatos', dataIndex: 'model_ids', render: (values) => <Space wrap>{values.map((value) => <Tag key={value}>{value}</Tag>)}</Space> },
    { title: 'Métrica', dataIndex: 'primary_metric', render: titleCase },
    { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
  ]

  return (
    <>
      <PageHeading title="Experimentos" actions={<Button type="primary" icon={<PlusOutlined />} onClick={() => navigate('/experiments/new')}>Nuevo experimento</Button>} />
      {resource.error && <Alert type="error" showIcon message={resource.error} style={{ marginBottom: 16 }} />}
      <Card>
        <div className="filter-toolbar">
          <Input className="filter-search" prefix={<SearchOutlined />} value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Buscar experimento…" allowClear />
          <Select className="filter-select" value={status} onChange={setStatus} options={[{ value: 'all', label: 'Todos los estados' }, ...['queued', 'running', 'completed', 'failed', 'cancelled'].map((value) => ({ value, label: titleCase(value) }))]} />
        </div>
        <Table columns={columns} dataSource={rows} rowKey="id" loading={resource.loading} scroll={{ x: 980 }} locale={{ emptyText: <Empty description="No hay experimentos" /> }} />
      </Card>
    </>
  )
}
