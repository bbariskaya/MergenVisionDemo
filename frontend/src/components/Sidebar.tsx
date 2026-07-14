import { cn } from '@/lib/utils'
import { Activity, Database, LayoutDashboard, ScanFace, UserPlus, Users, X } from 'lucide-react'
import { Link, useLocation } from 'react-router'

export interface SidebarProps {
  mobileOpen: boolean
  onClose: () => void
}

interface NavItem {
  to: string
  label: string
  icon: typeof LayoutDashboard
}

const mainItems: NavItem[] = [
  { to: '/', label: 'Genel Bakış', icon: LayoutDashboard },
  { to: '/enroll', label: 'Yeni Kişi Kaydı', icon: UserPlus },
  { to: '/identify', label: 'Yüz Tanıma', icon: ScanFace },
  { to: '/search-face', label: 'Kişiler', icon: Users },
]

const adminItems: NavItem[] = [
  { to: '/bulk-enrollment', label: 'Toplu Kayıt İşlemi', icon: Database },
  { to: '/system', label: 'Sistem Durumu', icon: Activity },
]

export function Sidebar({ mobileOpen, onClose }: SidebarProps) {
  const location = useLocation()

  function isActive(path: string) {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  function NavLink({ item, onClick }: { item: NavItem; onClick?: () => void }) {
    const Icon = item.icon
    const active = isActive(item.to)
    return (
      <Link
        to={item.to}
        onClick={onClick}
        className={cn(
          'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
          active
            ? 'bg-white/10 text-white'
            : 'text-navy-300 hover:bg-white/5 hover:text-white',
        )}
        aria-current={active ? 'page' : undefined}
      >
        <span
          className={cn(
            'absolute left-0 h-5 w-0.5 rounded-r-full transition-all',
            active ? 'bg-primary' : 'bg-transparent',
          )}
          aria-hidden="true"
        />
        <Icon className="h-5 w-5" aria-hidden="true" />
        {item.label}
      </Link>
    )
  }

  function NavGroup({ items, onClick, label }: { items: NavItem[]; onClick?: () => void; label: string }) {
    return (
      <nav className="flex flex-col gap-1 px-3" aria-label={label}>
        {items.map((item) => (
          <NavLink key={item.to} item={item} onClick={onClick} />
        ))}
      </nav>
    )
  }

  const brand = (
    <div className="flex items-center gap-4">
      <img
        src="/interprobe_logo.jpeg"
        alt="Interprobe"
        className="h-14 w-auto rounded-md"
      />
      <div className="flex flex-col">
        <span className="text-lg font-bold leading-tight text-white">Interprobe</span>
        <span className="text-xs text-navy-300">Yüz Tanıma Platformu</span>
      </div>
    </div>
  )

  const content = (
    <>
      <div className="flex-1 space-y-6 py-5">
        <NavGroup items={mainItems} label="Ana navigasyon" />
        <div>
          <p className="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-navy-400">
            Yönetim
          </p>
          <NavGroup items={adminItems} label="Yönetim navigasyonu" />
        </div>
      </div>
      <div className="border-t border-navy-800 p-4">
        <p className="text-[11px] leading-relaxed text-navy-400">
          Interprobe Yüz Tanıma Platformu
          <span className="block text-navy-500">v0.1.0 · Demo Ortamı</span>
        </p>
      </div>
    </>
  )

  return (
    <>
      <aside className="hidden w-64 flex-col bg-navy-900 lg:flex">
        <div className="flex h-20 items-center px-5">{brand}</div>
        {content}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden" data-testid="mobile-drawer">
          <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden="true" />
          <div className="absolute left-0 top-0 flex h-full w-64 flex-col bg-navy-900 shadow-xl">
            <div className="flex h-20 items-center justify-between px-4">
              {brand}
              <button
                onClick={onClose}
                className="rounded p-1.5 text-navy-300 hover:bg-navy-800 hover:text-white focus-visible:ring-primary"
                aria-label="Menüyü kapat"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            {content}
          </div>
        </div>
      )}
    </>
  )
}
