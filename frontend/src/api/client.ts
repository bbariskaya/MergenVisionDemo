import { ApiError, type ApiErrorResponse } from './types'

const API_BASE = '/api/v1'

async function parseError(response: Response): Promise<ApiError> {
  let body: ApiErrorResponse | undefined
  try {
    body = (await response.json()) as ApiErrorResponse
  } catch {
    body = undefined
  }
  const message =
    typeof body?.detail === 'string'
      ? body.detail
      : `${response.status} ${response.statusText}`
  return new ApiError(message, response.status, body)
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init.headers || {}),
    },
  })
  if (!response.ok) {
    throw await parseError(response)
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export async function apiUpload<T>(
  path: string,
  formData: FormData,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init.headers || {}),
    },
  })
  if (!response.ok) {
    throw await parseError(response)
  }
  return (await response.json()) as T
}
