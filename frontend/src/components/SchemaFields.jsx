import { Form, Input, InputNumber, Select, Switch } from 'antd'
import { titleCase } from '../utils/format'

function normalizedType(spec = {}) {
  const type = Array.isArray(spec.type) ? spec.type.find((item) => item !== 'null') : spec.type
  return type || (spec.enum ? typeof spec.enum.find((item) => item !== null) : 'string')
}

export default function SchemaFields({ schema = {}, value = {}, onChange, columnOptions = [] }) {
  const required = new Set(schema.required || [])
  const update = (name, nextValue) => onChange({ ...value, [name]: nextValue })

  return (
    <div className="schema-grid">
      {Object.entries(schema.properties || {}).map(([name, spec]) => {
        const type = normalizedType(spec)
        const label = titleCase(name)
        let field
        if (name === 'target_column' && columnOptions.length) {
          field = <Select showSearch options={columnOptions.map((column) => ({ label: column, value: column }))} value={value[name]} onChange={(next) => update(name, next)} />
        } else if (name === 'group_column' && columnOptions.length) {
          field = <Select allowClear showSearch options={columnOptions.map((column) => ({ label: column, value: column }))} value={value[name]} onChange={(next) => update(name, next || null)} />
        } else if (spec.enum) {
          const options = spec.enum.map((option) => ({ label: option === null ? 'Ninguno' : String(option), value: option === null ? '__null__' : option }))
          const current = value[name] === null ? '__null__' : value[name]
          field = <Select options={options} value={current} onChange={(next) => update(name, next === '__null__' ? null : next)} />
        } else if (type === 'array') {
          field = <Select mode="tags" value={value[name] || []} onChange={(next) => update(name, next)} tokenSeparators={[',']} />
        } else if (type === 'boolean') {
          field = <Switch checked={Boolean(value[name])} onChange={(next) => update(name, next)} />
        } else if (type === 'integer' || type === 'number') {
          field = <InputNumber className="full-width" min={spec.minimum} max={spec.maximum} step={type === 'integer' ? 1 : 0.01} value={value[name]} onChange={(next) => update(name, next)} />
        } else {
          field = <Input value={value[name] ?? ''} onChange={(event) => update(name, event.target.value)} />
        }
        return (
          <Form.Item key={name} label={label} required={required.has(name)} extra={spec.description}>
            {field}
          </Form.Item>
        )
      })}
    </div>
  )
}
