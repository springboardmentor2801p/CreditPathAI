const BASE = 'http://localhost:8000'

export async function postRecommend(payload) {
  const res = await fetch(`${BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}

export async function getRandomBorrower() {
  const res = await fetch(`${BASE}/random-borrower`)
  if (!res.ok) throw new Error('Could not load random borrower')
  return res.json()
}

export async function getBatchCases(n = 15) {
  const res = await fetch(`${BASE}/batch-cases?n=${n}`)
  if (!res.ok) throw new Error('Could not load batch cases')
  return res.json()
}

