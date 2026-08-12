import { useCallback, useEffect, useState } from 'react'
import { apiRequest } from '../services/api'

export default function useResource(path, interval = 0) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(Boolean(path))

  const reload = useCallback(async () => {
    if (!path) {
      setData(null)
      setLoading(false)
      return null
    }
    try {
      const result = await apiRequest(path)
      setData(result)
      setError('')
      return result
    } catch (requestError) {
      setError(requestError.message)
      return null
    } finally {
      setLoading(false)
    }
  }, [path])

  useEffect(() => {
    reload()
    if (!interval || !path) return undefined
    const timer = window.setInterval(reload, interval)
    return () => window.clearInterval(timer)
  }, [interval, path, reload])

  return { data, error, loading, reload, setData }
}
