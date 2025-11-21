export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export function makeApiUrl(path) {
  return new URL(path, API_BASE_URL).toString()
}

export function makeTemplateApiUrl(pathTemplate) {
  const base = API_BASE_URL.replace(/\/+$/, '')
  const path = pathTemplate.startsWith('/') ? pathTemplate : `/${pathTemplate}`
  return `${base}${path}`
}
