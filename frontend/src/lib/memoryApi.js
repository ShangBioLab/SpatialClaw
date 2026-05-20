import axios from 'axios';

export const AUTH_ERROR_EVENT = 'SPATIALCLAW:auth-error';

export const api = axios.create({
  baseURL: '/api',
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('api_token');
  if (token) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('api_token');
      window.dispatchEvent(new CustomEvent(AUTH_ERROR_EVENT));
    }
    return Promise.reject(error);
  },
);

const encodeId = (id) => encodeURIComponent(id);

// ============ Browse API ============

export const getDomains = () =>
  api.get('/browse/domains').then((res) => res.data);

export const getNode = (path, domain = 'session') =>
  api.get(`/browse/node?path=${encodeId(path)}&domain=${domain}`).then((res) => res.data);

export const getChildren = (nodeUuid, domain, path) => {
  const params = new URLSearchParams({ node_uuid: nodeUuid });
  if (domain) params.append('domain', domain);
  if (path) params.append('path', path);
  return api.get(`/browse/children?${params}`).then((res) => res.data);
};

export const getAllPaths = (domain) => {
  const params = domain ? `?domain=${domain}` : '';
  return api.get(`/browse/paths${params}`).then((res) => res.data);
};

export const searchMemories = (query, limit = 10, domain) => {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  if (domain) params.append('domain', domain);
  return api.get(`/browse/search?${params}`).then((res) => res.data);
};

export const getRecent = (limit = 10) =>
  api.get(`/browse/recent?limit=${limit}`).then((res) => res.data);

export const createMemory = (data) =>
  api.post('/browse/create', data).then((res) => res.data);

export const updateNode = (path, domain, data) =>
  api.put(`/browse/node?path=${encodeId(path)}&domain=${domain}`, data).then((res) => res.data);

export const addPath = (data) =>
  api.post('/browse/add-path', data).then((res) => res.data);

// ============ Glossary API ============

export const getAllGlossary = () =>
  api.get('/browse/glossary').then((res) => res.data);

export const addGlossary = (keyword, nodeUuid) =>
  api.post('/browse/glossary', { keyword, node_uuid: nodeUuid }).then((res) => res.data);

export const removeGlossary = (keyword, nodeUuid) =>
  api.delete('/browse/glossary', { data: { keyword, node_uuid: nodeUuid } }).then((res) => res.data);

// ============ Health API ============

export const getHealth = () =>
  axios.get('/health').then((res) => res.data);

export default api;
