import { useDeleteFaceMutation, useFace, useFaceHistory } from '@/api/faces'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Modal } from '@/components/ui/Modal'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapProcessStatus } from '@/lib/utils'
import { Trash2 } from 'lucide-react'
import { useState } from 'react'
import { useNavigate, useParams } from 'react-router'

export interface FaceDetailPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

export default function FaceDetailPage({ onToast }: FaceDetailPageProps) {
  const { faceId } = useParams<{ faceId: string }>()
  const navigate = useNavigate()
  const [showDelete, setShowDelete] = useState(false)

  const faceQuery = useFace(faceId || '')
  const historyQuery = useFaceHistory(faceId || '')
  const deleteMutation = useDeleteFaceMutation()

  async function handleDelete() {
    if (!faceId) return
    try {
      await deleteMutation.mutateAsync(faceId)
      onToast?.({ variant: 'success', title: 'Kayıt silindi' })
      navigate('/')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Silme işlemi başarısız.'
      onToast?.({ variant: 'error', title: 'Silme hatası', message })
    }
  }

  if (faceQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
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

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="page-title">{face.name}</h1>
          <p className="mt-1 text-slate-500">Yüz kaydı detayı ve tanıma geçmişi</p>
        </div>
        <Button variant="danger" onClick={() => setShowDelete(true)}>
          <Trash2 className="mr-2 h-4 w-4" />
          Kaydı Sil
        </Button>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Kişi Bilgisi</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid gap-4 sm:grid-cols-2">
              <div>
                <dt className="text-sm text-slate-500">Ad Soyad</dt>
                <dd className="font-medium text-navy-900">{face.name}</dd>
              </div>
              <div>
                <dt className="text-sm text-slate-500">Kimlik No (Maskeli)</dt>
                <dd className="font-medium text-navy-900">{face.nationalIdMasked}</dd>
              </div>
              <div>
                <dt className="text-sm text-slate-500">Durum</dt>
                <dd>
                  <Badge status={face.status}>{mapProcessStatus(face.status)}</Badge>
                </dd>
              </div>
              <div>
                <dt className="text-sm text-slate-500">Kayıt Tarihi</dt>
                <dd className="font-medium text-navy-900">{formatDate(face.createdAt)}</dd>
              </div>
              <div>
                <dt className="text-sm text-slate-500">Yüz ID</dt>
                <dd className="font-mono text-xs text-slate-600">{face.faceId}</dd>
              </div>
              <div>
                <dt className="text-sm text-slate-500">Kişi ID</dt>
                <dd className="font-mono text-xs text-slate-600">{face.personId}</dd>
              </div>
            </dl>
            {face.metadata && Object.keys(face.metadata).length > 0 && (
              <div className="mt-6">
                <h3 className="mb-2 text-sm font-semibold text-slate-700">Metadata</h3>
                <pre className="max-h-48 overflow-auto rounded-lg bg-slate-100 p-3 text-xs text-slate-700">
                  {JSON.stringify(face.metadata, null, 2)}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Tanıma Geçmişi</CardTitle>
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
              <ul className="divide-y divide-slate-100">
                {historyQuery.data.map((entry) => (
                  <li key={entry.processId} className="py-3">
                    <p className="text-sm font-medium text-navy-900">{formatDate(entry.timestamp)}</p>
                    <p className="text-xs text-slate-500">İşlem: {entry.processId.slice(0, 8)}…</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500">Bu kişi için henüz bir tanıma işlemi yok.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Fotoğraflar</CardTitle>
        </CardHeader>
        <CardContent>
          {face.photos.length > 0 ? (
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
              {face.photos.map((photo) => (
                <div
                  key={photo.photoId}
                  className="overflow-hidden rounded-lg border border-slate-200"
                >
                  <img
                    src={`/api/v1/photos/${photo.photoId}`}
                    alt={face.name}
                    className="h-48 w-full object-cover"
                    loading="lazy"
                  />
                  <div className="flex items-center justify-between p-2 text-xs text-slate-500">
                    <Badge status={photo.status}>{mapProcessStatus(photo.status)}</Badge>
                    <span>{formatDate(photo.createdAt)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Bu kişiye ait fotoğraf bulunmuyor.</p>
          )}
        </CardContent>
      </Card>

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
        <p>Bu yüz kaydını silmek istediğinize emin misiniz? Bu işlem geri alınamaz.</p>
      </Modal>
    </div>
  )
}
