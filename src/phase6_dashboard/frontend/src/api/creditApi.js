import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export async function checkHealth() {
  const { data } = await api.get('/health-check');
  return data;
}

export async function predictRisk(borrower, loan) {
  const { data } = await api.post('/predict-risk', { borrower, loan });
  return data;
}

export async function getRecommendation(borrower, loan) {
  const { data } = await api.post('/get-recommendation', { borrower, loan });
  return data;
}
