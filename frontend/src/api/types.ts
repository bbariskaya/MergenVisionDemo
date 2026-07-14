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
}

export interface RecognizedFace {
  faceIndex: number
  faceId: UUID | null
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
  photoId: UUID
  name: string
  nationalIdMasked: string
  status: string
  boundingBox: BoundingBox
  landmarks: number[][]
  metadata: Record<string, unknown> | null
  createdAt: string
  photos: FacePhoto[]
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
  photoId: UUID
  name: string
  nationalIdMasked: string
  status: string
  createdAt: string
}

export interface FaceListResponse {
  items: FaceListItem[]
  total: number
  limit: number
  offset: number
}

export interface FaceListItem {
  faceId: UUID
  personId: UUID
  photoId: UUID
  name: string
  nationalIdMasked: string
  status: string
  createdAt: string
}

export interface FaceListResponse {
  items: FaceListItem[]
  total: number
  limit: number
  offset: number
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
