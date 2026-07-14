import { useHealthLive, useHealthReady } from '@/api/health'
import { cn } from '@/lib/utils'
import { Activity, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'
import { useState } from 'react'

export function HealthIndicator() {
  const live = useHealthLive()
  const ready = useHealthReady()
  const [showDetails, setShowDetails] = useState(false)

  const isLoading = live.isLoading || ready.isLoading
  const isReady = ready.data?.status === 'ready'

  return (
    <div className="relative">
      <button
        onClick={() => setShowDetails((v) => !v)}
        className={cn(
          'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors',
          isLoading && 'bg-slate-100 text-slate-600 border-slate-200',
          !isLoading && isReady && 'bg-emerald-100 text-emerald-800 border-emerald-200',
          !isLoading && !isReady && 'bg-red-100 text-red-800 border-red-200',
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
        <span>{isLoading ? 'Kontrol ediliyor' : isReady ? 'Sistem Hazır' : 'Sistem Hazır Değil'}</span>
        <Activity className="h-3.5 w-3.5 text-slate-400" aria-hidden="true" />
      </button>

      {showDetails && (
        <div className="absolute right-0 top-full z-40 mt-2 w-72 rounded-xl border border-slate-200 bg-white p-4 shadow-lg">
          <p className="mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Bileşen Durumu
          </p>
          <ul className="space-y-2">
            <li className="flex items-center justify-between text-sm">
              <span className="text-slate-700">Canlılık</span>
              <span className={cn('rounded-full px-2 py-0.5 text-xs font-medium', live.data?.status === 'alive' ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800')}>
                {live.data?.status === 'alive' ? 'Canlı' : 'Kapalı'}
              </span>
            </li>
            {ready.data?.components.map((component) => (
              <li key={component.name} className="flex items-center justify-between text-sm">
                <span className="text-slate-700">{component.name}</span>
                <span
                  className={cn(
                    'rounded-full px-2 py-0.5 text-xs font-medium',
                    component.status === 'ok' ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800',
                  )}
                >
                  {component.status === 'ok' ? 'Tamam' : 'Hata'}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
