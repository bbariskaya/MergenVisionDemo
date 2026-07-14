import { cn } from '@/lib/utils'
import { Menu } from 'lucide-react'
import { useState } from 'react'
import { useLocation } from 'react-router'
import { HealthIndicator } from './HealthIndicator'
import { Sidebar } from './Sidebar'

const pageTitles: Record<string, { title: string; subtitle?: string }> = {
  '/': { title: 'Genel Bakış', subtitle: 'Operasyonel özet' },
  '/enroll': { title: 'Yeni Kişi Kaydı', subtitle: 'Kişi bilgilerini ve yüz görselini ekleyin' },
  '/identify': { title: 'Yüz Tanıma', subtitle: 'Bir görseldeki yüzleri kayıtlı kişilerle eşleştirin' },
  '/search-face': { title: 'Kişiler', subtitle: 'Kayıtlı kimlikleri arayın ve görüntüleyin' },
  '/system': { title: 'Sistem Durumu', subtitle: 'Altyapı ve servis sağlığı' },
  '/settings': { title: 'Ayarlar' },
}

const firstSegmentLabels: Record<string, string> = {
  enroll: 'Yeni Kişi Kaydı',
  identify: 'Yüz Tanıma',
  'search-face': 'Kişiler',
  system: 'Sistem Durumu',
  settings: 'Ayarlar',
  faces: 'Kişi Detayı',
  processes: 'İşlem Detayı',
}

function PageTitle() {
  const { pathname } = useLocation()
  const meta = pageTitles[pathname]
  const segments = pathname.split('/').filter(Boolean)
  const firstSegment = segments[0]
  const title = meta?.title ?? (firstSegment ? firstSegmentLabels[firstSegment] ?? firstSegment : 'Ana Sayfa')
  const subtitle = meta?.subtitle ?? (segments.length > 1 ? segments.slice(1).join(' / ') : undefined)

  return (
    <div className="min-w-0">
      <span className="block truncate text-lg font-semibold text-navy-900">{title}</span>
      {subtitle && <span className="block truncate text-xs text-navy-500">{subtitle}</span>}
    </div>
  )
}

function DemoBadge() {
  return (
    <span className="inline-flex items-center rounded-full border border-navy-200 bg-white px-2.5 py-1 text-xs font-medium text-navy-600">
      Demo Ortamı
    </span>
  )
}

export interface LayoutProps {
  children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div className="flex flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-navy-200 bg-white/90 px-4 backdrop-blur lg:px-6">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setMobileOpen(true)}
              className="rounded p-2 text-navy-600 hover:bg-navy-100 focus-visible:ring-primary lg:hidden"
              aria-label="Menüyü aç"
            >
              <Menu className="h-6 w-6" />
            </button>
            <PageTitle />
          </div>
          <div className="flex items-center gap-3">
            <DemoBadge />
            <HealthIndicator />
          </div>
        </header>
        <main className={cn('flex-1 p-4 lg:p-6')}>
          <div className="mx-auto max-w-7xl">{children}</div>
        </main>
      </div>
    </div>
  )
}
