import { describe, expect, it } from 'vitest'
import { clamp, cn, mapRecognizeStatus, statusColor } from '../utils'

describe('utils', () => {
  it('cn merges tailwind classes', () => {
    expect(cn('px-2 py-1', 'px-4')).toBe('py-1 px-4')
  })

  it('clamps value between min and max', () => {
    expect(clamp(5, 0, 10)).toBe(5)
    expect(clamp(-1, 0, 10)).toBe(0)
    expect(clamp(11, 0, 10)).toBe(10)
  })

  it('maps recognize statuses to Turkish', () => {
    expect(mapRecognizeStatus('known')).toBe('Bulundu')
    expect(mapRecognizeStatus('unknown')).toBe('Bulunamadı')
    expect(mapRecognizeStatus('no_face')).toBe('Yüz Algılanmadı')
  })

  it('returns color classes for known status', () => {
    expect(statusColor('known')).toContain('emerald')
  })
})
