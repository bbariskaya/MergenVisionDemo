import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

export function formatDate(iso: string | Date): string {
  const d = typeof iso === 'string' ? new Date(iso) : iso
  return new Intl.DateTimeFormat('tr-TR', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(d)
}

export function mapRecognizeStatus(status: string): string {
  switch (status) {
    case 'known':
      return 'Bulundu'
    case 'unknown':
      return 'Bulunamadı'
    case 'no_face':
      return 'Yüz Algılanmadı'
    default:
      return status
  }
}

export function mapProcessStatus(status: string): string {
  switch (status) {
    case 'completed':
      return 'Tamamlandı'
    case 'pending':
      return 'Bekliyor'
    case 'failed':
      return 'Hata'
    default:
      return status
  }
}

export function mapFaceStatus(status: string): string {
  switch (status) {
    case 'active':
      return 'Aktif'
    case 'deleted':
      return 'Silinmiş'
    default:
      return status
  }
}

export function statusColor(status: string): string {
  switch (status) {
    case 'known':
    case 'completed':
    case 'ok':
    case 'active':
      return 'bg-emerald-100 text-emerald-800 border-emerald-200'
    case 'unknown':
      return 'bg-slate-100 text-slate-700 border-slate-200'
    case 'failed':
    case 'unavailable':
    case 'no_face':
    case 'deleted':
      return 'bg-red-100 text-red-800 border-red-200'
    case 'pending':
      return 'bg-amber-100 text-amber-800 border-amber-200'
    default:
      return 'bg-slate-100 text-slate-700 border-slate-200'
  }
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
