import { useFaceList } from '@/api/faces'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { Input } from '@/components/ui/Input'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn, formatDate, mapFaceStatus } from '@/lib/utils'
import { ChevronLeft, ChevronRight, Grid3X3, List, Search, User, UserPlus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router'

const PAGE_SIZE = 20

type ViewMode = 'grid' | 'list'

export default function FaceSearchPage() {
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const [view, setView] = useState<ViewMode>('grid')
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'inactive'>('all')

  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedSearch(search)
      setOffset(0)
    }, 400)
    return () => clearTimeout(t)
  }, [search])

  const isActiveParam = statusFilter === 'all' ? undefined : statusFilter === 'active'

  const { data, isLoading, error } = useFaceList({
    search: debouncedSearch || undefined,
    isActive: isActiveParam,
    limit: PAGE_SIZE,
    offset,
  })

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="page-title">Kişiler</h1>
          <p className="page-subtitle mt-1">Kayıtlı kişileri arayın ve görüntüleyin.</p>
        </div>
        <Link to="/enroll" className="btn-primary justify-center sm:w-auto">
          <UserPlus className="h-4 w-4" aria-hidden="true" />
          Yeni Kişi Kaydı
        </Link>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-navy-400" aria-hidden="true" />
            <Input
              placeholder="İsim ile ara…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
              aria-label="İsim ile ara"
            />
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="statusFilter" className="sr-only">Durum filtresi</label>
            <select
              id="statusFilter"
              value={statusFilter}
              onChange={(e) => { setStatusFilter(e.target.value as typeof statusFilter); setOffset(0) }}
              className="input h-10 py-0"
            >
              <option value="all">Tümü</option>
              <option value="active">Aktif</option>
              <option value="inactive">Pasif</option>
            </select>
            <div className="flex rounded-lg border border-navy-200 bg-white p-0.5">
              <ViewToggle mode="grid" current={view} icon={Grid3X3} label="Kart görünümü" onChange={setView} />
              <ViewToggle mode="list" current={view} icon={List} label="Liste görünümü" onChange={setView} />
            </div>
          </div>
        </CardContent>
      </Card>

      {isLoading ? (
        view === 'grid' ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-64" />
            ))}
          </div>
        ) : (
          <div className="space-y-3">
            <Skeleton className="h-14" />
            <Skeleton className="h-14" />
            <Skeleton className="h-14" />
          </div>
        )
      ) : error ? (
        <AlertError message={error.message} />
      ) : data && data.items.length > 0 ? (
        <>
          <p className="text-sm text-navy-500">
            {data.total.toLocaleString('tr-TR')} kayıt bulundu
          </p>
          {view === 'grid' ? <GridView items={data.items} /> : <ListView items={data.items} />}
          <Pagination offset={data.offset} limit={data.limit} total={data.total} onChange={setOffset} />
        </>
      ) : (
        <EmptyState
          icon={User}
          title={debouncedSearch ? 'Sonuç bulunamadı' : 'Henüz kayıtlı kişi yok'}
          description={
            debouncedSearch
              ? 'Arama kriterlerini değiştirin veya yeni kayıt ekleyin.'
              : 'İlk kaydı oluşturmak için Yeni Kişi Kaydı sayfasına gidin.'
          }
          action={
            <Link to="/enroll" className="btn-primary text-xs px-3 py-2">
              Yeni Kişi Kaydı
            </Link>
          }
        />
      )}
    </div>
  )
}

function ViewToggle({
  mode,
  current,
  icon: Icon,
  label,
  onChange,
}: {
  mode: ViewMode
  current: ViewMode
  icon: typeof Grid3X3
  label: string
  onChange: (m: ViewMode) => void
}) {
  const active = mode === current
  return (
    <button
      type="button"
      onClick={() => onChange(mode)}
      className={cn(
        'rounded p-1.5 transition-colors focus-visible:ring-2 focus-visible:ring-primary',
        active ? 'bg-navy-100 text-navy-900' : 'text-navy-400 hover:bg-navy-50',
      )}
      aria-label={label}
      aria-pressed={active}
    >
      <Icon className="h-4 w-4" />
    </button>
  )
}

function GridView({ items }: { items: Array<{ faceId: string; name: string; nationalIdMasked: string; photoId: string | null; photoCount: number; createdAt: string; status: string }> }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {items.map((item) => (
        <Link
          key={item.faceId}
          to={`/faces/${item.faceId}`}
          className="card card-hover overflow-hidden"
        >
          <div className="aspect-[4/3] bg-navy-50">
            {item.photoId ? (
              <img
                src={`/api/v1/photos/${item.photoId}`}
                alt={item.name}
                className="h-full w-full object-cover"
                loading="lazy"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-navy-300">
                <User className="h-10 w-10" />
              </div>
            )}
          </div>
          <CardContent className="p-4">
            <p className="truncate font-semibold text-navy-900">{item.name}</p>
            <p className="text-xs text-navy-500">{item.nationalIdMasked}</p>
            <div className="mt-3 flex items-center justify-between text-xs">
              <span className="rounded-full bg-navy-100 px-2 py-0.5 text-navy-600">
                {item.photoCount} fotoğraf
              </span>
              <span className={cn('rounded-full px-2 py-0.5 font-medium', item.status === 'active' ? 'bg-successSubtle text-success' : 'bg-navy-100 text-navy-600')}>
                {mapFaceStatus(item.status)}
              </span>
            </div>
          </CardContent>
        </Link>
      ))}
    </div>
  )
}

function ListView({ items }: { items: Array<{ faceId: string; name: string; nationalIdMasked: string; photoId: string | null; photoCount: number; createdAt: string; status: string }> }) {
  return (
    <div className="overflow-hidden rounded-xl border border-navy-200 bg-white">
      <table className="min-w-full divide-y divide-navy-100">
        <thead className="bg-navy-50">
          <tr>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-navy-500">Kişi</th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-navy-500">Kimlik No</th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-navy-500">Fotoğraf</th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-navy-500">Kayıt Tarihi</th>
            <th scope="col" className="px-4 py-3 text-right text-xs font-semibold uppercase text-navy-500">İşlem</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-navy-100">
          {items.map((item) => (
            <tr key={item.faceId} className="hover:bg-navy-50/50">
              <td className="px-4 py-3">
                <div className="flex items-center gap-3">
                  {item.photoId ? (
                    <img
                      src={`/api/v1/photos/${item.photoId}`}
                      alt={item.name}
                      className="h-12 w-12 rounded-lg object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-navy-50 text-navy-400">
                      <User className="h-5 w-5" />
                    </div>
                  )}
                  <div>
                    <p className="font-medium text-navy-900">{item.name}</p>
                    <p className={cn('text-xs', item.status === 'active' ? 'text-success' : 'text-navy-500')}>
                      {mapFaceStatus(item.status)}
                    </p>
                  </div>
                </div>
              </td>
              <td className="px-4 py-3 text-sm text-navy-600">{item.nationalIdMasked}</td>
              <td className="px-4 py-3 text-sm text-navy-600">{item.photoCount} fotoğraf</td>
              <td className="px-4 py-3 text-sm text-navy-600">{formatDate(item.createdAt)}</td>
              <td className="px-4 py-3 text-right">
                <Link to={`/faces/${item.faceId}`} className="text-sm font-medium text-primary hover:underline">
                  Detay
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function AlertError({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-danger bg-dangerSubtle p-4 text-sm text-danger" role="alert">
      Liste yüklenirken hata oluştu: {message}
    </div>
  )
}

function Pagination({
  offset,
  limit,
  total,
  onChange,
}: {
  offset: number
  limit: number
  total: number
  onChange: (offset: number) => void
}) {
  const currentPage = Math.floor(offset / limit) + 1
  const totalPages = Math.ceil(total / limit)

  return (
    <div className="flex items-center justify-between border-t border-navy-200 pt-4">
      <p className="text-sm text-navy-500">
        {offset + 1}-{Math.min(offset + limit, total)} / {total} kayıt
      </p>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onChange(Math.max(0, offset - limit))}
          disabled={offset === 0}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-sm text-navy-600">
          Sayfa {currentPage} / {totalPages}
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onChange(offset + limit)}
          disabled={offset + limit >= total}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
