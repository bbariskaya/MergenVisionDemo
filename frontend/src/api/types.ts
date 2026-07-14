export type UUID = string

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface FaceCandidate {
  faceId: UUID
  personId: UUID
  photoId: UUID
  score: number
  name: string | null
}

export interface RecognizedFace {
  faceIndex: number
  faceId: UUID | null
  personId: UUID | null
  photoId: UUID | null
  status: 'known' | 'unknown' | string
  name: string | null
  metadata: Record<string, unknown> | null
  boundingBox: BoundingBox
  landmarks: number[][]
  confidence: number | null
  candidates: FaceCandidate[]
}

export interface RecognizeResponse {
  processId: UUID
  faceCount: number
  faces: RecognizedFace[]
}

export interface EnrollRequest {
  name: string
  nationalId: string
  image: File
  metadata?: Record<string, unknown>
}

export interface EnrollResponse {
  faceId: UUID
  personId: UUID
  photoId: UUID
  status: string
  name: string
  createdAt: string
}

export interface FacePhoto {
  photoId: UUID
  status: string
  createdAt: string
}

export interface FaceDetail {
  faceId: UUID
  personId: UUID
  photoId: UUID | null
  name: string
  nationalIdMasked: string
  status: string
  boundingBox: BoundingBox | null
  landmarks: number[][] | null
  metadata: Record<string, unknown> | null
  createdAt: string
  photos: FacePhoto[]
}

export interface AddPhotoResponse {
  faceId: UUID
  personId: UUID
  photoId: UUID
  sampleId: UUID
  status: string
  name: string
  createdAt: string
}

export interface FaceHistoryEntry {
  processId: UUID
  status: string
  timestamp: string
}

export interface ProcessFace {
  faceIndex: number
  status: string
  faceId: UUID | null
  score: number | null
  boundingBox: BoundingBox
}

export interface ProcessDetail {
  processId: UUID
  status: string
  faceCount: number | null
  createdAt: string
  completedAt: string | null
  faces: ProcessFace[]
}

export interface BulkEnrollRecord {
  index: number
  faceId: UUID
  personId: UUID
  photoId: UUID
  status: string
  name: string
}

export interface BulkEnrollResponse {
  enrolled: BulkEnrollRecord[]
  errors: Array<{ index?: number; message?: string }>
}

export interface FaceListItem {
  faceId: UUID
  personId: UUID
  photoId: UUID | null
  name: string
  nationalIdMasked: string
  status: string
  createdAt: string
  photoCount: number
}

export interface FaceListResponse {
  items: FaceListItem[]
  total: number
  limit: number
  offset: number
}

export interface EnrollmentStats {
  personCount: number
  faceCount: number
  photoCount: number
  recognitionCount: number
  activePersonCount: number
}

export interface BulkJobShard {
  workerId: string
  shardIndex: number
  processId: UUID
  status: string
  progress: Record<string, unknown>
}

export interface VggfaceBulkJob {
  jobId: UUID
  status: string
  datasetType: string
  assignedWorkers: string[]
  targetTotalActivePhotos: number
  startingActivePhotos: number
  currentActivePhotos: number
  photosAddedByJob: number
  requestedPhotos: number
  totalEnrolled: number
  totalDuplicate: number
  totalNoFace: number
  totalErrors: number
  elapsedSeconds: number
  avgPhotosPerSecond: number
  probeP50Ms: number | null
  probeP95Ms: number | null
  shards: BulkJobShard[]
  createdAt: string
  completedAt: string | null
}

export interface HealthComponent {
  name: string
  status: string
  details?: Record<string, unknown>
}

export interface HealthReadyResponse {
  status: 'ready' | 'unavailable'
  components: HealthComponent[]
}

export interface ApiErrorDetail {
  loc: Array<string | number>
  msg: string
  type: string
}

export interface ApiErrorResponse {
  detail?: string | ApiErrorDetail[]
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public body?: ApiErrorResponse,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}
