async function request(url, options = {}) {
  const res = await fetch(url, options)
  if (!res.ok) {
    let detail = `${res.status}`
    try {
      const data = await res.json()
      detail = data.detail || JSON.stringify(data)
    } catch {}
    throw new Error(detail)
  }
  const type = res.headers.get('content-type') || ''
  return type.includes('application/json') ? res.json() : res
}

export const api = {
  health: () => request('/api/health'),
  testCases: () => request('/api/test-cases'),
  templates: () => request('/api/templates'),
  createFromTestdata: (name) => request('/api/cases/from-testdata', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({name})
  }),
  uploadCase: ({fdi, upper, lower}) => {
    const body = new FormData()
    body.append('fdi', fdi)
    body.append('upper', upper)
    body.append('lower', lower)
    return request('/api/cases/upload', {method:'POST', body})
  },
  case: (id) => request(`/api/cases/${id}`),
  generate: (id) => request(`/api/cases/${id}/generate`, {method:'POST'}),
}
