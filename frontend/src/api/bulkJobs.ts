import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
import { queryKeys } from './queryKeys'
import type { VggfaceBulkJob } from './types'

const POLL_INTERVAL_MS = 2000

export function useBulkJob(jobId: string) {
  return useQuery({
    queryKey: queryKeys.bulkJob(jobId),
    queryFn: () => apiFetch<VggfaceBulkJob>(`/bulk-jobs/${jobId}`),
    enabled: jobId.length > 0,
    refetchInterval: POLL_INTERVAL_MS,
    refetchIntervalInBackground: true,
  })
}

export function useLatestBulkJob() {
  return useQuery({
    queryKey: queryKeys.latestBulkJob(),
    queryFn: () => apiFetch<VggfaceBulkJob>('/bulk-jobs/latest'),
    retry: false,
  })
}

export function useStartVggfaceBulkJob() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (maxPhotos?: number) =>
      apiFetch<VggfaceBulkJob>('/bulk-jobs/vggface', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ maxPhotos }),
      }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.bulkJob(data.jobId), data)
      queryClient.setQueryData(queryKeys.latestBulkJob(), data)
    },
  })
}

export function useCancelBulkJob() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) =>
      apiFetch<VggfaceBulkJob>(`/bulk-jobs/${jobId}/cancel`, { method: 'POST' }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.bulkJob(data.jobId), data)
    },
  })
}

export function useResumeBulkJob() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) =>
      apiFetch<VggfaceBulkJob>(`/bulk-jobs/${jobId}/resume`, { method: 'POST' }),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.bulkJob(data.jobId), data)
    },
  })
}
