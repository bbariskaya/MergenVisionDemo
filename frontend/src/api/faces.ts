import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch, apiUpload } from './client'
import { queryKeys } from './queryKeys'
import type {
  BulkEnrollResponse,
  EnrollRequest,
  EnrollResponse,
  FaceDetail,
  FaceHistoryEntry,
  FaceListResponse,
  RecognizeResponse,
} from './types'

export interface RecognizeVariables {
  image: File
  topK: number
  threshold: number
}

export function useRecognizeMutation() {
  return useMutation<RecognizeResponse, Error, RecognizeVariables>({
    mutationFn: async ({ image, topK, threshold }) => {
      const formData = new FormData()
      formData.append('image', image)
      return apiUpload<RecognizeResponse>(
        `/faces/recognize?top_k=${topK}&threshold=${threshold}`,
        formData,
      )
    },
  })
}

export function useEnrollMutation() {
  const queryClient = useQueryClient()
  return useMutation<EnrollResponse, Error, EnrollRequest>({
    mutationFn: async ({ name, nationalId, image, metadata }) => {
      const formData = new FormData()
      formData.append('name', name)
      formData.append('nationalId', nationalId)
      formData.append('image', image)
      if (metadata) {
        formData.append('metadata', JSON.stringify(metadata))
      }
      return apiUpload<EnrollResponse>('/faces/enroll', formData)
    },
    onSuccess: (_, variables) => {
      // Enroll changes the vector store; invalidate any face-related caches.
      queryClient.invalidateQueries({ queryKey: ['face'] })
      void variables
    },
  })
}

export function useBulkEnrollMutation() {
  return useMutation<BulkEnrollResponse, Error, { images: File[]; entries: Array<{ name: string; nationalId: string; metadata?: Record<string, unknown> }> }>({
    mutationFn: async ({ images, entries }) => {
      const formData = new FormData()
      images.forEach((image) => formData.append('images', image))
      formData.append('entries', JSON.stringify(entries))
      return apiUpload<BulkEnrollResponse>('/faces/enroll/bulk', formData)
    },
  })
}

export interface FaceListParams {
  search?: string
  isActive?: boolean
  limit?: number
  offset?: number
}

export async function listFaces(params: FaceListParams = {}): Promise<FaceListResponse> {
  const query = new URLSearchParams()
  if (params.search) query.set('search', params.search)
  if (params.isActive !== undefined) query.set('is_active', String(params.isActive))
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  return apiFetch<FaceListResponse>(`/faces?${query.toString()}`)
}

export function useFaceList(params: FaceListParams = {}) {
  return useQuery({
    queryKey: ['faces', 'list', params] as const,
    queryFn: () => listFaces(params),
  })
}

export function useFace(faceId: string) {
  return useQuery({
    queryKey: queryKeys.face(faceId),
    queryFn: () => apiFetch<FaceDetail>(`/faces/${faceId}`),
    enabled: faceId.length > 0,
  })
}

export function useFaceHistory(faceId: string) {
  return useQuery({
    queryKey: queryKeys.faceHistory(faceId),
    queryFn: () => apiFetch<FaceHistoryEntry[]>(`/faces/${faceId}/history`),
    enabled: faceId.length > 0,
  })
}

export function useDeleteFaceMutation() {
  const queryClient = useQueryClient()
  return useMutation<void, Error, string>({
    mutationFn: (faceId) => apiFetch(`/faces/${faceId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['face'] })
    },
  })
}

export interface FacesQueryParams {
  search?: string
  isActive?: boolean | null
  limit?: number
  offset?: number
}

export function useFaces(params: FacesQueryParams = {}) {
  const { search = '', isActive = null, limit = 20, offset = 0 } = params
  return useQuery({
    queryKey: queryKeys.faces(params),
    queryFn: () => {
      const query = new URLSearchParams()
      if (search) query.set('search', search)
      if (isActive !== null && isActive !== undefined) query.set('is_active', String(isActive))
      query.set('limit', String(limit))
      query.set('offset', String(offset))
      return apiFetch<FaceListResponse>(`/faces?${query.toString()}`)
    },
  })
}
