import { useFaceList } from '@/api/faces'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapProcessStatus } from '@/lib/utils'
import { ChevronLeft, ChevronRight, Search, User } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router'

const PAGE_SIZE = 20

export default function FaceSearchPage() {
  const [search, setSearch] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedSearch(search)
      setOffset(0)
    }, 400)
    return () => clearTimeout(t)
  }, [search])

  const { data, isLoading, error } = useFaceList({
    search: debouncedSearch || undefined,
    limit: PAGE_SIZE,
    offset,
  })

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="page-title">Kayıtlı Yüzler</h1>
          <p className="mt-1 text-slate-500">Kayıtlı kişileri arayın ve görüntüleyin.</p>
        </div>
        <Link to="/enroll">
          <Button>Yeni Kayıt</Button>
        </Link>
      </div>

      <Card>
        <CardContent className="py-4">
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" />
            <Input
              placeholder="İsim ile ara…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
              aria-label="İsim ile ara"
            />
          </div>
        </CardContent>
      </Card>

      {isLoading ? (
        <div className="space-y-3">
          <Skeleton className="h-14 w-full" />
          <Skeleton className="h-14 w-full" />
          <Skeleton className="h-14 w-full" />
        </div>
      ) : error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          Liste yüklenirken hata oluştu: {error.message}
        </div>
      ) : data && data.items.length > 0 ? (
        <>
          <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-slate-500">Kişi</th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-slate-500">Kimlik No</th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-slate-500">Durum</th>
                  <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase text-slate-500">Kayıt Tarihi</th>
                  <th scope="col" className="px-4 py-3 text-right text-xs font-semibold uppercase text-slate-500">İşlem</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {data.items.map((item) => (
                  <tr key={item.faceId} className="hover:bg-slate-50">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        {item.photoId ? (
                          <img
                            src={`/api/v1/photos/${item.photoId}`}
                            alt={item.name}
                            className="h-9 w-9 rounded-full object-cover"
                            loading="lazy"
                          />
                        ) : (
                          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-100 text-primary">
                            <User className="h-5 w-5" aria-hidden="true" />
                          </div>
                        )}
                        <div>
                          <p className="font-medium text-navy-900">{item.name}</p>
                          <p className="text-xs text-slate-500">{item.faceId.slice(0, 8)}…</p>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-600">{item.nationalIdMasked}</td>
                    <td className="px-4 py-3">
                      <Badge status={item.status}>{mapProcessStatus(item.status)}</Badge>
                    </td>
                    <td className="px-4 py-3 text-sm text-slate-600">{formatDate(item.createdAt)}</td>
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

          <Pagination
            offset={data.offset}
            limit={data.limit}
            total={data.total}
            onChange={setOffset}
          />
        </>
      ) : (
        <div className="rounded-xl border border-slate-200 bg-white p-12 text-center text-slate-500">
          <User className="mx-auto mb-3 h-10 w-10 text-slate-300" aria-hidden="true" />
          <p className="font-medium">Kayıtlı yüz bulunamadı.</p>
          <p className="text-sm">Arama kriterlerini değiştirin veya yeni kayıt ekleyin.</p>
        </div>
      )}
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
    <div className="flex items-center justify-between">
      <p className="text-sm text-slate-500">
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
        <span className="text-sm text-slate-600">
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
