import { useHealthLive, useHealthReady } from '@/api/health'
import { cn } from '@/lib/utils'
import { Activity, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router'

export function HealthIndicator() {
  const live = useHealthLive()
  const ready = useHealthReady()
  const [showDetails, setShowDetails] = useState(false)

  const isLoading = live.isLoading || ready.isLoading
  const isReady = ready.data?.status === 'ready'

  const statusClasses = isLoading
    ? 'bg-navy-50 text-navy-500 border-navy-200'
    : isReady
      ? 'bg-successSubtle text-success border-emerald-200'
      : 'bg-dangerSubtle text-danger border-red-200'

  return (
    <div className="relative">
      <button
        onClick={() => setShowDetails((v) => !v)}
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget)) setShowDetails(false)
        }}
        className={cn(
          'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
          statusClasses,
        )}
        aria-expanded={showDetails}
        aria-haspopup="true"
      >
        {isLoading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
        ) : isReady ? (
          <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
        ) : (
          <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
        )}
        <span>{isLoading ? 'Kontrol' : isReady ? 'Hazır' : 'Hata'}</span>
        <Activity className="h-3.5 w-3.5 opacity-60" aria-hidden="true" />
      </button>

      {showDetails && (
        <div className="absolute right-0 top-full z-40 mt-2 w-72 rounded-xl border border-navy-200 bg-white p-4 shadow-lg">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wide text-navy-500">Bileşenler</p>
            <Link
              to="/system"
              onMouseDown={() => setShowDetails(false)}
              className="text-xs font-medium text-primary hover:underline"
            >
              Detaylar
            </Link>
          </div>
          <ul className="space-y-2">
            <li className="flex items-center justify-between text-sm">
              <span className="text-navy-700">Canlılık</span>
              <span
                className={cn(
                  'rounded-full px-2 py-0.5 text-xs font-medium',
                  live.data?.status === 'alive' ? 'bg-successSubtle text-success' : 'bg-dangerSubtle text-danger',
                )}
              >
                {live.data?.status === 'alive' ? 'Canlı' : 'Kapalı'}
              </span>
            </li>
            {ready.data?.components.map((component) => (
              <li key={component.name} className="flex items-center justify-between text-sm">
                <span className="text-navy-700">{component.name}</span>
                <span
                  className={cn(
                    'rounded-full px-2 py-0.5 text-xs font-medium',
                    component.status === 'ok' ? 'bg-successSubtle text-success' : 'bg-dangerSubtle text-danger',
                  )}
                >
                  {mapComponentStatus(component.status)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function mapComponentStatus(status: string): string {
  switch (status) {
    case 'ok':
      return 'Tamam'
    case 'unavailable':
      return 'Kapalı'
    default:
      return status
  }
}
