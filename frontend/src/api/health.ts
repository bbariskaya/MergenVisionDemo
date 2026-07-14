import { useQuery } from '@tanstack/react-query'
import { queryKeys } from './queryKeys'
import type { HealthReadyResponse } from './types'

async function fetchLive(): Promise<{ status: string }> {
  const response = await fetch('/health/live')
  if (!response.ok) throw new Error('live probe failed')
  return response.json() as Promise<{ status: string }>
}

async function fetchReady(): Promise<HealthReadyResponse> {
  const response = await fetch('/health/ready')
  const body = (await response.json()) as HealthReadyResponse
  if (!response.ok) {
    return { status: 'unavailable', components: body.components || [] }
  }
  return body
}

export function useHealthLive() {
  return useQuery({
    queryKey: queryKeys.health.live,
    queryFn: fetchLive,
    refetchInterval: 10_000,
  })
}

export function useHealthReady() {
  return useQuery({
    queryKey: queryKeys.health.ready,
    queryFn: fetchReady,
    refetchInterval: 10_000,
  })
}
