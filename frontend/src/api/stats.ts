import { useQuery } from '@tanstack/react-query'
import { apiFetch } from './client'
import { queryKeys } from './queryKeys'
import type { EnrollmentStats } from './types'

export async function getEnrollmentStats(): Promise<EnrollmentStats> {
  return apiFetch<EnrollmentStats>('/stats')
}

export function useEnrollmentStats(poll = false) {
  return useQuery({
    queryKey: queryKeys.enrollmentStats(),
    queryFn: getEnrollmentStats,
    refetchInterval: poll ? 3000 : false,
  })
}
