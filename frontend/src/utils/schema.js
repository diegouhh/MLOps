export function schemaDefaults(schema = {}) {
  return Object.fromEntries(Object.entries(schema.properties || {}).flatMap(([name, spec]) => {
    if (Object.hasOwn(spec, 'default')) return [[name, spec.default]]
    return []
  }))
}
