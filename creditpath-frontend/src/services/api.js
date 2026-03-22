import axios from 'axios'

const API_URL = 'http://127.0.0.1:8000'

export const predictRisk = async (borrowerData) => {
  const response = await axios.post(`${API_URL}/predict`, borrowerData)
  return response.data
}