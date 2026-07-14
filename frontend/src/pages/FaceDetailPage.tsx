import {
  useAddPersonPhotoMutation,
  useDeleteFaceMutation,
  useDeletePersonPhotoMutation,
  useFace,
  useFaceHistory,
} from '@/api/faces'
import { PageHeader } from '@/components/PageHeader'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Modal } from '@/components/ui/Modal'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapFaceStatus } from '@/lib/utils'
import { CalendarDays, Plus, Trash2, User } from 'lucide-react'
import { useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

export interface FaceDetailPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

export default function FaceDetailPage({ onToast }: FaceDetailPageProps) {
  const { faceId } = useParams<{ faceId: string }>()
  const navigate = useNavigate()
  const [showDelete, setShowDelete] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const faceQuery = useFace(faceId || '')
  const historyQuery = useFaceHistory(faceId || '')
  const deleteMutation = useDeleteFaceMutation()
  const addPhotoMutation = useAddPersonPhotoMutation(faceId || '')
  const deletePhotoMutation = useDeletePersonPhotoMutation(faceId || '')

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file || !faceId) return
    addPhotoMutation.mutate(file, {
      onSuccess: () => {
        onToast?.({ variant: 'success', title: 'Fotoğraf eklendi' })
      },
      onError: (err) => {
        onToast?.({ variant: 'error', title: 'Fotoğraf eklenemedi', message: err.message })
      },
    })
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  async function handleDelete() {
    if (!faceId) return
    try {
      await deleteMutation.mutateAsync(faceId)
      onToast?.({ variant: 'success', title: 'Kayıt silindi' })
      navigate('/search-face')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Silme işlemi başarısız.'
      onToast?.({ variant: 'error', title: 'Silme hatası', message })
    }
  }

  if (faceQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    )
  }

  if (faceQuery.error || !faceQuery.data) {
    return (
      <Alert variant="error" title="Yüz kaydı bulunamadı">
        İstenen kayda ulaşılamadı. Kayıt silinmiş olabilir.
      </Alert>
    )
  }

  const face = faceQuery.data
  const primaryPhotoId = face.photos.find((p) => p.status === 'active')?.photoId ?? face.photos[0]?.photoId

  return (
    <div className="space-y-6">
      <PageHeader
        title={face.name}
        subtitle="Kişi kaydı detayı, fotoğraf galerisi ve tanıma geçmişi."
        action={
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileSelect}
            />
            <Button
              variant="secondary"
              onClick={() => fileInputRef.current?.click()}
              isLoading={addPhotoMutation.isPending}
            >
              <Plus className="mr-2 h-4 w-4" />
              Fotoğraf Ekle
            </Button>
            <Button variant="danger" onClick={() => setShowDelete(true)}>
              <Trash2 className="mr-2 h-4 w-4" />
              Kaydı Sil
            </Button>
          </div>
        }
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardContent className="p-5">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-navy-200 bg-navy-50">
              {primaryPhotoId ? (
                <img
                  src={`/api/v1/photos/${primaryPhotoId}`}
                  alt={face.name}
                  className="h-full w-full object-cover"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-navy-300">
                  <User className="h-20 w-20" />
                </div>
              )}
            </div>
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">Durum</span>
                <Badge status={face.status}>{mapFaceStatus(face.status)}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">Kimlik No</span>
                <span className="font-medium text-navy-900">{face.nationalIdMasked}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">Kayıt Tarihi</span>
                <span className="font-medium text-navy-900">{formatDate(face.createdAt)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Fotoğraflar</CardTitle>
            </CardHeader>
            <CardContent>
              {face.photos.length > 0 ? (
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
                  {face.photos.map((photo) => (
                    <div
                      key={photo.photoId}
                      className="group relative overflow-hidden rounded-lg border border-navy-200 bg-navy-50"
                    >
                      <button
                        type="button"
                        onClick={() => {
                          if (confirm('Bu fotoğrafı silmek istediğinize emin misiniz?')) {
                            deletePhotoMutation.mutate(photo.photoId, {
                              onSuccess: () => onToast?.({ variant: 'success', title: 'Fotoğraf silindi' }),
                              onError: (err) => onToast?.({ variant: 'error', title: 'Silme hatası', message: err.message }),
                            })
                          }
                        }}
                        className="absolute right-2 top-2 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-white/90 text-danger opacity-0 shadow-sm transition-opacity group-hover:opacity-100 focus-visible:opacity-100 focus-visible:ring-danger"
                        aria-label="Fotoğrafı sil"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                      <img
                        src={`/api/v1/photos/${photo.photoId}`}
                        alt={face.name}
                        className="aspect-square w-full object-cover"
                        loading="lazy"
                      />
                      <div className="flex items-center justify-between bg-white px-2 py-1.5 text-[11px] text-navy-500">
                        <Badge status={photo.status}>{mapFaceStatus(photo.status)}</Badge>
                        <span>{formatDate(photo.createdAt)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-navy-500">Bu kişiye ait fotoğraf bulunmuyor.</p>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Tanıma Geçmişi</CardTitle>
            </CardHeader>
            <CardContent>
              {historyQuery.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : historyQuery.error ? (
                <Alert variant="error" title="Geçmiş alınamadı">
                  {historyQuery.error.message}
                </Alert>
              ) : historyQuery.data && historyQuery.data.length > 0 ? (
                <ul className="divide-y divide-navy-100">
                  {historyQuery.data.map((entry) => (
                    <li key={entry.processId} className="flex items-center justify-between py-3">
                      <div className="flex items-center gap-3">
                        <CalendarDays className="h-4 w-4 text-navy-400" aria-hidden="true" />
                        <div>
                          <p className="text-sm font-medium text-navy-900">{formatDate(entry.timestamp)}</p>
                          <p className="text-xs text-navy-500">İşlem {entry.processId.slice(0, 8)}…</p>
                        </div>
                      </div>
                      <Link
                        to={`/processes/${entry.processId}`}
                        className="text-sm font-medium text-primary hover:underline"
                      >
                        Detay
                      </Link>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-navy-500">Bu kişi için henüz bir tanıma işlemi yok.</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      <Modal
        open={showDelete}
        onClose={() => setShowDelete(false)}
        title="Yüz Kaydını Sil"
        footer={
          <>
            <Button variant="ghost" onClick={() => setShowDelete(false)}>
              İptal
            </Button>
            <Button variant="danger" onClick={handleDelete} isLoading={deleteMutation.isPending}>
              Sil
            </Button>
          </>
        }
      >
        <p>
          <span className="font-semibold">{face.name}</span> kaydını silmek istediğinize emin misiniz? Bu işlem geri alınamaz.
        </p>
      </Modal>
    </div>
  )
}
