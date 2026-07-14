import { cn } from '@/lib/utils'
import { Menu } from 'lucide-react'
import { useState } from 'react'
import { useLocation } from 'react-router'
import { HealthIndicator } from './HealthIndicator'
import { Sidebar } from './Sidebar'

function Breadcrumb() {
  const location = useLocation()
  const segments = location.pathname.split('/').filter(Boolean)

  const labels: Record<string, string> = {
    '': 'Ana Sayfa',
    enroll: 'Kayıt Yap',
    identify: 'Yüz Tanı',
    'search-face': 'Yüz Ara',
    settings: 'Ayarlar',
    faces: 'Yüz Detayı',
    processes: 'İşlem Detayı',
  }

  return (
    <nav aria-label="Breadcrumb" className="text-sm text-slate-500">
      <ol className="flex items-center gap-2">
        <li>
          <span className="font-medium text-slate-800">
            {segments.length === 0 ? 'Ana Sayfa' : labels[segments[0]] || segments[0]}
          </span>
        </li>
        {segments.length > 1 && (
          <li>
            <span className="text-slate-400">/</span>
            <span className="ml-2 text-slate-600">{segments.slice(1).join(' / ')}</span>
          </li>
        )}
      </ol>
    </nav>
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
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-slate-200 bg-white/80 px-4 backdrop-blur lg:px-8">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setMobileOpen(true)}
              className="rounded p-2 text-slate-600 hover:bg-slate-100 lg:hidden"
              aria-label="Menüyü aç"
            >
              <Menu className="h-6 w-6" />
            </button>
            <Breadcrumb />
          </div>
          <HealthIndicator />
        </header>
        <main className={cn('flex-1 p-4 lg:p-8')}>
          <div className="mx-auto max-w-7xl">{children}</div>
        </main>
      </div>
    </div>
  )
}
