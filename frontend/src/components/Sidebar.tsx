import { cn } from '@/lib/utils'
import { LayoutDashboard, ScanFace, Search, Settings, UserPlus, X } from 'lucide-react'
import { Link, useLocation } from 'react-router'

export interface SidebarProps {
  mobileOpen: boolean
  onClose: () => void
}

const navItems = [
  { to: '/', label: 'Ana Sayfa', icon: LayoutDashboard },
  { to: '/enroll', label: 'Kayıt Yap', icon: UserPlus },
  { to: '/identify', label: 'Yüz Tanı', icon: ScanFace },
  { to: '/search-face', label: 'Kayıtlı Yüzler', icon: Search },
  { to: '/settings', label: 'Ayarlar', icon: Settings },
]

export function Sidebar({ mobileOpen, onClose }: SidebarProps) {
  const location = useLocation()

  function isActive(path: string) {
    if (path === '/') return location.pathname === '/'
    return location.pathname.startsWith(path)
  }

  function NavList({ onItemClick }: { onItemClick?: () => void }) {
    return (
      <nav className="flex flex-col gap-1 px-3">
        {navItems.map((item) => {
          const Icon = item.icon
          const active = isActive(item.to)
          return (
            <Link
              key={item.to}
              to={item.to}
              onClick={onItemClick}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors',
                active ? 'bg-primary text-white' : 'text-slate-300 hover:bg-navy-800 hover:text-white',
              )}
              aria-current={active ? 'page' : undefined}
            >
              <Icon className="h-5 w-5" aria-hidden="true" />
              {item.label}
            </Link>
          )
        })}
      </nav>
    )
  }

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden w-64 flex-col bg-navy-900 lg:flex">
        <div className="flex h-16 items-center gap-3 px-6">
          <img
            src="/interprobe_logo.jpeg"
            alt="Interprobe"
            className="h-8 w-auto rounded"
          />
          <span className="text-lg font-bold text-white">Interprobe</span>
        </div>
        <div className="flex-1 py-4">
          <NavList />
        </div>
        <div className="border-t border-navy-800 p-4 text-xs text-slate-400">
          Interprobe Operasyon Merkezi v0.1.0
        </div>
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden="true" />
          <div className="absolute left-0 top-0 h-full w-64 bg-navy-900 shadow-xl">
            <div className="flex h-16 items-center justify-between px-4">
              <div className="flex items-center gap-3">
                <img
                  src="/interprobe_logo.jpeg"
                  alt="Interprobe"
                  className="h-8 w-auto rounded"
                />
                <span className="text-lg font-bold text-white">Interprobe</span>
              </div>
              <button
                onClick={onClose}
                className="rounded p-1 text-slate-300 hover:bg-navy-800 hover:text-white"
                aria-label="Menüyü kapat"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <div className="py-4">
              <NavList onItemClick={onClose} />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
