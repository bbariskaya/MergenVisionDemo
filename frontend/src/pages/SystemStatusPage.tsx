import { useHealthLive, useHealthReady } from '@/api/health'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn, formatDate } from '@/lib/utils'
import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

function statusTheme(status: string) {
  if (status === 'ready' || status === 'alive' || status === 'ok') {
    return {
      icon: CheckCircle2,
      badge: <Badge status="known">Aktif</Badge>,
      card: 'border-l-4 border-l-success',
      text: 'text-navy-900',
    }
  }
  return {
    icon: AlertCircle,
    badge: <Badge status="failed">Müsait Değil</Badge>,
    card: 'border-l-4 border-l-danger',
    text: 'text-navy-900',
  }
}

export default function SystemStatusPage() {
  const live = useHealthLive()
  const ready = useHealthReady()

  const isLoading = live.isLoading || ready.isLoading
  const isReady = ready.data?.status === 'ready'

  return (
    <div className="space-y-6">
      <Card className={cn(!isLoading && (isReady ? 'border-l-success' : 'border-l-danger'), 'border-l-4')}>
        <CardContent className="flex items-center gap-4 py-6">
          {isLoading ? (
            <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
          ) : isReady ? (
            <CheckCircle2 className="h-8 w-8 text-success" aria-hidden="true" />
          ) : (
            <AlertCircle className="h-8 w-8 text-danger" aria-hidden="true" />
          )}
          <div>
            <p className="text-sm text-navy-500">Genel Sistem Durumu</p>
            <p className="text-xl font-semibold text-navy-900">
              {isLoading ? 'Kontrol ediliyor…' : isReady ? 'Tüm sistemler hazır' : 'Sistem hazır değil'}
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {isLoading ? (
          <>
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
          </>
        ) : (
          <>
            <ComponentCard
              name="API Canlılık"
              status={live.data?.status ?? 'unavailable'}
              updatedAt={live.dataUpdatedAt}
            />
            {ready.data?.components.map((component) => (
              <ComponentCard
                key={component.name}
                name={component.name}
                status={component.status}
                updatedAt={ready.dataUpdatedAt}
                details={component.details}
              />
            ))}
          </>
        )}
      </div>
    </div>
  )
}

function ComponentCard({
  name,
  status,
  updatedAt,
  details,
}: {
  name: string
  status: string
  updatedAt: number
  details?: Record<string, unknown>
}) {
  const theme = statusTheme(status)
  const Icon = theme.icon
  const detailText = details && Object.keys(details).length > 0 ? JSON.stringify(details) : undefined

  return (
    <Card className={theme.card}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Icon className={cn('h-5 w-5', status === 'ok' || status === 'alive' ? 'text-success' : 'text-danger')} aria-hidden="true" />
          {name}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-navy-500">Durum</span>
          {theme.badge}
        </div>
        {detailText && (
          <p className="max-h-24 overflow-auto rounded bg-navy-50 p-2 text-xs font-mono text-navy-600">
            {detailText}
          </p>
        )}
        <p className="text-xs text-navy-400">Son kontrol: {formatDate(new Date(updatedAt).toISOString())}</p>
      </CardContent>
    </Card>
  )
}
