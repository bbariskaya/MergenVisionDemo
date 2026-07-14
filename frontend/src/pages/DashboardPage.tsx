import { useHealthReady } from '@/api/health'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn, mapProcessStatus } from '@/lib/utils'
import { Activity, AlertTriangle, ScanFace, UserPlus } from 'lucide-react'
import { Link } from 'react-router'

export interface DashboardPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

export default function DashboardPage({ onToast }: DashboardPageProps) {
  const ready = useHealthReady()
  void onToast

  const isReady = ready.data?.status === 'ready'

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">Operasyon Merkezi</h1>
        <p className="mt-1 text-slate-500">Yüz tanıma ve kayıt işlemlerinizi yönetin.</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <HealthStatusCard ready={ready} />
        <QuickActionCard
          to="/enroll"
          title="Kayıt Yap"
          description="Yeni kişi ve yüz kaydı oluşturun."
          icon={UserPlus}
          color="bg-primary"
        />
        <QuickActionCard
          to="/identify"
          title="Yüz Tanı"
          description="Bir görseldeki yüzleri tanıyın."
          icon={ScanFace}
          color="bg-accent"
        />
        <QuickActionCard
          to="/search-face"
          title="Kayıtlı Yüzler"
          description="Kayıtlı yüzleri görüntüleyin."
          icon={Activity}
          color="bg-secondary"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Sistem Durumu</CardTitle>
          </CardHeader>
          <CardContent>
            {ready.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : ready.error ? (
              <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                Sağlık durumu alınamadı. Lütfen yeniden deneyin.
              </div>
            ) : (
              <ul className="divide-y divide-slate-100">
                <li className="flex items-center justify-between py-3">
                  <span className="font-medium text-slate-700">Genel Durum</span>
                  <span
                    className={cn(
                      'rounded-full px-2.5 py-1 text-xs font-semibold',
                      isReady ? 'bg-emerald-100 text-emerald-800' : 'bg-red-100 text-red-800',
                    )}
                  >
                    {isReady ? 'Hazır' : 'Hazır Değil'}
                  </span>
                </li>
                {ready.data?.components.map((component) => (
                  <li key={component.name} className="flex items-center justify-between py-3">
                    <span className="text-slate-600">{component.name}</span>
                    <span
                      className={cn(
                        'rounded-full px-2.5 py-1 text-xs font-medium',
                        component.status === 'ok'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-red-100 text-red-800',
                      )}
                    >
                      {mapProcessStatus(component.status)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Hızlı Bilgi</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-slate-600">
            <p>Kayıt yaparak kişileri veritabanına ekleyin. Tanıma ekranında bir görsel yükleyerek eşleşmeleri görün.</p>
            <p className="flex items-center gap-2 rounded-lg bg-amber-50 p-3 text-amber-800">
              <AlertTriangle className="h-5 w-5 shrink-0" aria-hidden="true" />
              <span>Liste/sayfalama endpoint’i olmadığı için yüz listesi şu an kullanılamıyor.</span>
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function HealthStatusCard({ ready }: { ready: ReturnType<typeof useHealthReady> }) {
  return (
    <Card className="bg-navy-900 text-white">
      <CardContent className="p-6">
        <p className="text-sm font-medium text-slate-300">Sistem Sağlığı</p>
        <div className="mt-2 flex items-center gap-2">
          {ready.isLoading ? (
            <Skeleton className="h-8 w-24 bg-slate-700" />
          ) : ready.data?.status === 'ready' ? (
            <>
              <div className="h-3 w-3 rounded-full bg-emerald-400" aria-hidden="true" />
              <span className="text-2xl font-bold">Hazır</span>
            </>
          ) : (
            <>
              <div className="h-3 w-3 rounded-full bg-red-400" aria-hidden="true" />
              <span className="text-2xl font-bold">Kapalı</span>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function QuickActionCard({
  to,
  title,
  description,
  icon: Icon,
  color,
  disabled,
}: {
  to: string
  title: string
  description: string
  icon: typeof UserPlus
  color: string
  disabled?: boolean
}) {
  const content = (
    <>
      <div className={cn('mb-3 inline-flex rounded-lg p-2 text-white', color)}>
        <Icon className="h-6 w-6" aria-hidden="true" />
      </div>
      <CardTitle>{title}</CardTitle>
      <p className="mt-1 text-sm text-slate-500">{description}</p>
    </>
  )

  if (disabled) {
    return (
      <div className={cn('card opacity-60', 'cursor-not-allowed')}>
        <CardContent>{content}</CardContent>
      </div>
    )
  }

  return (
    <Link to={to} className="card card-hover block">
      <CardContent>{content}</CardContent>
    </Link>
  )
}
