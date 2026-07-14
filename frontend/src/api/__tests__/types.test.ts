import { describe, expect, it } from 'vitest'
import type { EnrollResponse, FaceDetail } from '../types'

describe('API types', () => {
  it('EnrollResponse has required fields', () => {
    const r: EnrollResponse = {
      faceId: 'a',
      personId: 'b',
      photoId: 'c',
      status: 'known',
      name: 'Ali',
      createdAt: new Date().toISOString(),
    }
    expect(r.name).toBe('Ali')
  })

  it('FaceDetail exposes masked national ID only', () => {
    const r: FaceDetail = {
      faceId: 'a',
      personId: 'b',
      photoId: 'c',
      name: 'Ali',
      nationalIdMasked: '123456*****',
      status: 'active',
      boundingBox: { x1: 0, y1: 0, x2: 1, y2: 1 },
      landmarks: [],
      metadata: null,
      createdAt: new Date().toISOString(),
      photos: [],
    }
    expect(r.nationalIdMasked).toBe('123456*****')
    // @ts-expect-error raw national_id must not exist on detail type
    expect(r.nationalId).toBeUndefined()
  })
})
