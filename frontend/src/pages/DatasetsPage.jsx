import { DeleteOutlined, FolderOpenOutlined, InboxOutlined, PlusOutlined, ReloadOutlined, UploadOutlined } from '@ant-design/icons'
import { Alert, App, Button, Card, Col, Empty, Form, Input, Popconfirm, Row, Select, Space, Table, Tabs, Tag, Typography, Upload } from 'antd'
import { useMemo, useState } from 'react'
import PageHeading from '../components/PageHeading'
import StatusTag from '../components/StatusTag'
import useResource from '../hooks/useResource'
import { apiRequest, jsonOptions } from '../services/api'
import { formatBytes, formatDate } from '../utils/format'

const { Dragger } = Upload
const { Paragraph, Text } = Typography

function latestVersion(dataset) {
  return dataset.versions.at(-1)
}

function isMounted(dataset) {
  return latestVersion(dataset)?.schema_summary?.source === 'mounted'
}

export default function DatasetsPage() {
  const { message } = App.useApp()
  const [uploadForm] = Form.useForm()
  const [mountedForm] = Form.useForm()
  const resource = useResource('/datasets')
  const mountedResource = useResource('/datasets/mounted')
  const [file, setFile] = useState(null)
  const [dataType, setDataType] = useState('tabular')
  const [saving, setSaving] = useState(false)
  const [registering, setRegistering] = useState(false)
  const [versioning, setVersioning] = useState('')

  const mountedOptions = useMemo(() => (mountedResource.data || []).map((item) => ({
    value: item.relative_path,
    disabled: item.registered || !item.valid,
    label: item.valid
      ? `${item.name} · ${item.subject_count} sujetos · ${formatBytes(item.size_bytes)}${item.registered ? ' · registrada' : ''}`
      : `${item.relative_path} · ${item.validation_error}`,
    candidate: item,
  })), [mountedResource.data])

  async function upload(values) {
    if (!file) return message.warning('Selecciona un archivo')
    setSaving(true)
    try {
      const body = new FormData()
      body.append('file', file)
      body.append('name', values.name)
      body.append('data_type', values.data_type)
      if (values.description) body.append('description', values.description)
      await apiRequest('/datasets', { method: 'POST', body })
      message.success('Dataset registrado y validado')
      uploadForm.resetFields()
      uploadForm.setFieldValue('data_type', 'tabular')
      setDataType('tabular')
      setFile(null)
      resource.reload()
    } catch (error) {
      message.error(error.message)
    } finally {
      setSaving(false)
    }
  }

  async function registerMounted(values) {
    setRegistering(true)
    try {
      await apiRequest('/datasets/mounted', jsonOptions('POST', values))
      message.success('Carpeta BIDS registrada sin copiar sus archivos')
      mountedForm.resetFields()
      resource.reload()
      mountedResource.reload()
    } catch (error) {
      message.error(error.message)
    } finally {
      setRegistering(false)
    }
  }

  async function remove(id) {
    try {
      await apiRequest(`/datasets/${id}`, { method: 'DELETE' })
      message.success('Registro de dataset eliminado')
      resource.reload()
      mountedResource.reload()
    } catch (error) {
      message.error(error.message)
    }
  }

  async function uploadVersion(dataset, nextFile) {
    setVersioning(dataset.id)
    try {
      const body = new FormData()
      body.append('file', nextFile)
      await apiRequest(`/datasets/${dataset.id}/versions`, { method: 'POST', body })
      message.success(`Versión v${dataset.current_version + 1} registrada`)
      resource.reload()
    } catch (error) {
      message.error(error.message)
    } finally {
      setVersioning('')
    }
    return false
  }

  function selectMounted(relativePath) {
    const candidate = (mountedResource.data || []).find((item) => item.relative_path === relativePath)
    if (candidate) mountedForm.setFieldValue('name', candidate.name)
  }

  const columns = [
    { title: 'Dataset', dataIndex: 'name', render: (name, row) => <Space direction="vertical" size={0}><strong>{name}</strong><Text type="secondary">{row.description || latestVersion(row)?.original_filename}</Text></Space> },
    { title: 'Tipo', dataIndex: 'data_type', render: (value) => <Tag color="blue">{value === 'tabular' ? 'CSV tabular' : 'EEG · BIDS'}</Tag> },
    { title: 'Origen', render: (_, row) => isMounted(row) ? <Tag icon={<FolderOpenOutlined />} color="cyan">Local montado</Tag> : <Tag>Subido</Tag> },
    { title: 'Versión', dataIndex: 'current_version', render: (value) => `v${value}` },
    { title: 'Dimensiones', render: (_, row) => { const version = latestVersion(row); const summary = version?.schema_summary || {}; return isMounted(row) ? `${summary.subject_count} sujetos · ${summary.eeg_file_count} EEG` : version?.row_count != null ? `${version.row_count} × ${version.column_count}` : 'BIDS' } },
    { title: 'Tamaño', render: (_, row) => formatBytes(latestVersion(row)?.size_bytes) },
    { title: 'Estado', dataIndex: 'status', render: (value) => <StatusTag value={value} /> },
    { title: 'Fecha', dataIndex: 'created_at', render: formatDate },
    {
      title: 'Acciones',
      width: 220,
      render: (_, row) => <div className="table-actions">
        {isMounted(row) ? <Tag>Solo lectura</Tag> : <Upload accept={row.data_type === 'tabular' ? '.csv' : '.zip'} showUploadList={false} beforeUpload={(nextFile) => uploadVersion(row, nextFile)}><Button size="small" icon={<UploadOutlined />} loading={versioning === row.id}>Nueva versión</Button></Upload>}
        <Popconfirm title="¿Eliminar este dataset?" description={isMounted(row) ? 'Se elimina el registro, no la carpeta local.' : 'Solo se eliminará si ningún experimento lo usa.'} onConfirm={() => remove(row.id)}><Button danger type="text" icon={<DeleteOutlined />} aria-label={`Eliminar ${row.name}`} /></Popconfirm>
      </div>,
    },
  ]

  const uploadPanel = <Form form={uploadForm} layout="vertical" initialValues={{ data_type: 'tabular' }} onFinish={upload}>
    <Form.Item name="name" label="Nombre" rules={[{ required: true, message: 'Escribe un nombre' }]}><Input placeholder="Ej. Iris demo" /></Form.Item>
    <Form.Item name="data_type" label="Tipo de datos"><Select onChange={setDataType} options={[{ value: 'tabular', label: 'Datos tabulares (CSV)' }, { value: 'eeg_bids', label: 'EEG BIDS pequeño (ZIP)' }]} /></Form.Item>
    <Form.Item name="description" label="Descripción"><Input.TextArea rows={2} placeholder="Propósito y procedencia" /></Form.Item>
    <Form.Item label="Archivo" required>
      <Dragger fileList={file ? [file] : []} accept={dataType === 'tabular' ? '.csv' : '.zip'} maxCount={1} beforeUpload={(next) => { setFile(next); return false }} onRemove={() => setFile(null)}>
        <p className="ant-upload-drag-icon"><InboxOutlined /></p>
        <p>Arrastra el archivo o haz clic</p>
        <p className="ant-upload-hint">{dataType === 'tabular' ? 'CSV con encabezados.' : 'Para colecciones grandes usa Carpeta local BIDS.'}</p>
      </Dragger>
    </Form.Item>
    <Button htmlType="submit" type="primary" loading={saving} block>Registrar y validar</Button>
  </Form>

  const mountedPanel = <>
    <Alert type="info" showIcon message="Sin carga ni copia" description={<span>Coloca el dataset en <Text code>datasets/nombre-del-dataset</Text> y actualiza la lista.</span>} style={{ marginBottom: 16 }} />
    <Form form={mountedForm} layout="vertical" onFinish={registerMounted}>
      <Form.Item name="relative_path" label="Carpeta BIDS" rules={[{ required: true, message: 'Selecciona una carpeta' }]}>
        <Select showSearch optionFilterProp="label" placeholder="Seleccionar carpeta detectada" options={mountedOptions} onChange={selectMounted} notFoundContent="No se encontraron raíces BIDS" />
      </Form.Item>
      <Form.Item name="name" label="Nombre" rules={[{ required: true, message: 'Escribe un nombre' }]}><Input /></Form.Item>
      <Form.Item name="description" label="Descripción"><Input.TextArea rows={2} placeholder="Procedencia o propósito" /></Form.Item>
      <Space direction="vertical" className="full-width" size={10}>
        <Button icon={<ReloadOutlined />} onClick={mountedResource.reload} loading={mountedResource.loading} block>Actualizar carpetas</Button>
        <Button htmlType="submit" type="primary" icon={<FolderOpenOutlined />} loading={registering} block>Registrar carpeta</Button>
      </Space>
    </Form>
    {mountedResource.error && <Alert type="error" showIcon message={mountedResource.error} style={{ marginTop: 14 }} />}
  </>

  return (
    <>
      <PageHeading title="Datasets" />
      {resource.error && <Alert type="error" showIcon message={resource.error} style={{ marginBottom: 16 }} />}
      <Row gutter={[18, 18]}>
        <Col xs={24} xl={8}>
          <Card title="Registrar dataset" extra={<PlusOutlined />} className="dataset-register-card">
            <Tabs size="small" items={[{ key: 'upload', label: 'Subir archivo', children: uploadPanel }, { key: 'mounted', label: 'Carpeta local BIDS', children: mountedPanel }]} />
          </Card>
        </Col>
        <Col xs={24} xl={16}>
          <Card title={`Datasets disponibles (${(resource.data || []).length})`}>
            <Table columns={columns} dataSource={resource.data || []} rowKey="id" loading={resource.loading} scroll={{ x: 1100 }} pagination={{ pageSize: 8, hideOnSinglePage: true }} locale={{ emptyText: <Empty description="Sube o monta el primer dataset" /> }} />
          </Card>
        </Col>
      </Row>
    </>
  )
}
