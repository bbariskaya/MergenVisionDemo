export const queryKeys = {
  health: {
    live: ['health', 'live'] as const,
    ready: ['health', 'ready'] as const,
  },
  face: (faceId: string) => ['face', faceId] as const,
  faceHistory: (faceId: string) => ['face', faceId, 'history'] as const,
  faces: (params: { search?: string; isActive?: boolean | null; limit?: number; offset?: number }) =>
    ['faces', params] as const,
  process: (processId: string) => ['process', processId] as const,
  enrollmentStats: () => ['stats', 'enrollment'] as const,
  bulkJob: (jobId: string) => ['bulk-job', jobId] as const,
  latestBulkJob: () => ['bulk-job', 'latest'] as const,
}
