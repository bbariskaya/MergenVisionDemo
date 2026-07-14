import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'
import { queryKeys } from './queryKeys'
import type { ProcessDetail } from './types'

export function useProcess(processId: string) {
  return useQuery({
    queryKey: queryKeys.process(processId),
    queryFn: () => apiFetch<ProcessDetail>(`/processes/${processId}`),
    enabled: processId.length > 0,
  })
}
