import { useFaceList } from '@/api/faces'
import { useHealthReady } from '@/api/health'
import { useEnrollmentStats } from '@/api/stats'
import { Card, CardContent } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn, formatDate } from '@/lib/utils'
import { Activity, ArrowUpRight, ScanFace, ShieldCheck, UserPlus, Users } from 'lucide-react'
import { Link } from 'react-router'

export interface DashboardPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

export default function DashboardPage({ onToast }: DashboardPageProps) {
  void onToast

  const stats = useEnrollmentStats(false)
  const ready = useHealthReady()
  const recent = useFaceList({ limit: 5, offset: 0 })

  const isReady = ready.data?.status === 'ready'

  const statItems = [
    { label: 'Kayıtlı Kişiler', value: stats.data?.personCount ?? 0, href: '/search-face' },
    { label: 'Kayıtlı Fotoğraflar', value: stats.data?.photoCount ?? 0, href: '/search-face' },
    { label: 'Aktif Yüz Örnekleri', value: stats.data?.faceCount ?? 0 },
    { label: 'Tanıma İşlemi', value: stats.data?.recognitionCount ?? 0, href: '/identify' },
  ]

  return (
    <div className="space-y-6">
      <section
        aria-labelledby="hero-heading"
        className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-navy-900 via-navy-800 to-navy-900 text-white shadow-xl"
      >
        <div className="absolute right-0 top-0 h-full w-1/2 bg-[radial-gradient(circle_at_top_right,rgba(59,130,246,0.15),transparent_60%)]" aria-hidden="true" />
        <div className="relative p-6 lg:p-8">
          <div className="max-w-3xl space-y-5">
            <div className="flex items-center gap-2 text-sm font-medium text-navy-200">
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              Interprobe Yüz Tanıma Platformu
            </div>
            <div>
              <h1 id="hero-heading" className="text-2xl font-bold leading-tight tracking-tight sm:text-3xl lg:text-4xl">
                Yüz Tanıma Operasyon Merkezi
              </h1>
              <p className="mt-2 max-w-xl text-sm leading-relaxed text-navy-200 sm:text-base">
                Kayıtlı kimlikleri tanıyın, yönetin ve operasyonel kararlarınızı hızlandırın.
              </p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              <Link to="/identify" className="btn-primary inline-flex justify-center bg-white text-navy-900 hover:bg-navy-50">
                <ScanFace className="h-4 w-4" aria-hidden="true" />
                Görselden Yüz Tanı
              </Link>
              <Link to="/enroll" className="inline-flex justify-center items-center gap-2 rounded-lg border border-white/30 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-white/10">
                <UserPlus className="h-4 w-4" aria-hidden="true" />
                Yeni Kişi Kaydı
              </Link>
            </div>
          </div>

          <div className="mt-8 grid divide-y divide-white/10 rounded-xl border border-white/10 bg-navy-900/50 sm:grid-cols-2 sm:divide-x sm:divide-y-0 lg:grid-cols-4">
            {stats.isLoading ? (
              <>
                <Skeleton className="h-20" />
                <Skeleton className="h-20" />
                <Skeleton className="h-20" />
                <Skeleton className="h-20" />
              </>
            ) : stats.error ? (
              <div className="col-span-full p-4 text-sm text-red-200">
                İstatistikler alınamadı. Lütfen sayfayı yenileyin.
              </div>
            ) : (
              statItems.map((s) => (
                <StatItem key={s.label} label={s.label} value={s.value} href={s.href} />
              ))
            )}
          </div>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-3">
        <section className="lg:col-span-2" aria-labelledby="recent-heading">
          <Card>
            <CardContent className="p-0">
              <div className="flex items-center justify-between border-b border-navy-100 px-5 py-4">
                <h2 id="recent-heading" className="text-base font-semibold text-navy-900">
                  Son Eklenen Kişiler
                </h2>
                <Link to="/search-face" className="text-sm font-medium text-primary hover:underline">
                  Tümünü gör
                </Link>
              </div>
              {recent.isLoading ? (
                <div className="space-y-0 divide-y divide-navy-100">
                  <Skeleton className="h-16 rounded-none" />
                  <Skeleton className="h-16 rounded-none" />
                  <Skeleton className="h-16 rounded-none" />
                </div>
              ) : recent.error ? (
                <div className="border-t border-danger bg-dangerSubtle p-4 text-sm text-danger">
                  Kişi listesi alınamadı.
                </div>
              ) : recent.data && recent.data.items.length > 0 ? (
                <ul className="divide-y divide-navy-100">
                  {recent.data.items.map((person) => (
                    <li key={person.faceId}>
                      <Link
                        to={`/faces/${person.faceId}`}
                        className="group flex items-center gap-4 px-5 py-3 transition-colors hover:bg-navy-50"
                      >
                        {person.photoId ? (
                          <img
                            src={`/api/v1/photos/${person.photoId}`}
                            alt={person.name}
                            className="h-11 w-11 rounded-full object-cover ring-2 ring-navy-100"
                            loading="lazy"
                          />
                        ) : (
                          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-navy-100 text-navy-500">
                            <Users className="h-5 w-5" />
                          </div>
                        )}
                        <div className="min-w-0 flex-1">
                          <p className="truncate font-medium text-navy-900 group-hover:text-primary">{person.name}</p>
                          <p className="text-xs text-navy-500">
                            {person.photoCount} fotoğraf · {formatDate(person.createdAt)}
                          </p>
                        </div>
                        <ArrowUpRight className="h-4 w-4 text-navy-300 transition-colors group-hover:text-primary" aria-hidden="true" />
                      </Link>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="p-6 text-center text-sm text-navy-500">
                  Henüz kayıtlı kişi bulunmuyor.
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        <section aria-labelledby="health-heading">
          <Card className={cn('overflow-hidden', !ready.isLoading && (isReady ? 'border-t-4 border-t-success' : 'border-t-4 border-t-danger'))}>
            <CardContent className="p-0">
              <div className="border-b border-navy-100 px-5 py-4">
                <h2 id="health-heading" className="text-base font-semibold text-navy-900">Sistem Durumu</h2>
              </div>
              <div className="space-y-4 p-5">
                {ready.isLoading ? (
                  <Skeleton className="h-8 w-32" />
                ) : (
                  <>
                    <div className="flex items-center gap-3">
                      <span className={cn('h-3 w-3 rounded-full', isReady ? 'bg-success' : 'bg-danger')} aria-hidden="true" />
                      <div>
                        <p className="text-sm font-medium text-navy-900">
                          {isReady ? 'Hazır' : 'Hazır Değil'}
                        </p>
                        <p className="text-xs text-navy-500">
                          {isReady ? 'Tüm bileşenler çalışıyor.' : 'Bazı bileşenlerde sorun var.'}
                        </p>
                      </div>
                    </div>
                    <Link to="/system" className="btn-secondary w-full justify-center py-2 text-xs">
                      Detaylı Durumu Gör
                    </Link>
                  </>
                )}
              </div>
            </CardContent>
          </Card>

          <div className="mt-6 rounded-xl border border-navy-100 bg-white p-5">
            <div className="flex items-center gap-3">
              <div className="inline-flex rounded-lg bg-primary-50 p-2 text-primary">
                <Activity className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <p className="text-sm font-medium text-navy-900">Interprobe Platform</p>
                <p className="text-xs text-navy-500">Demo Ortamı · v0.1.0</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

function StatItem({ label, value, href }: { label: string; value: number; href?: string }) {
  const content = (
    <div className="p-4 text-center sm:text-left">
      <p className="text-2xl font-bold text-white">{value.toLocaleString('tr-TR')}</p>
      <p className="text-xs font-medium text-navy-300">{label}</p>
    </div>
  )
  if (href) {
    return (
      <Link to={href} className="block transition-colors hover:bg-white/5">
        {content}
      </Link>
    )
  }
  return <div className="transition-colors">{content}</div>
}


