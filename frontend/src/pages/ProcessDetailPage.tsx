import { useProcess } from '@/api/processes'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { SimilarityScore } from '@/components/ui/SimilarityScore'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapProcessStatus, mapRecognizeStatus } from '@/lib/utils'
import { Link, useParams } from 'react-router'

export default function ProcessDetailPage() {
  const { processId } = useParams<{ processId: string }>()
  const processQuery = useProcess(processId || '')

  if (processQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
      </div>
    )
  }

  if (processQuery.error || !processQuery.data) {
    return (
      <Alert variant="error" title="İşlem bulunamadı">
        İstenen tanıma işlemine ulaşılamadı.
      </Alert>
    )
  }

  const proc = processQuery.data

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">İşlem Detayı</h1>
        <p className="mt-1 text-slate-500">{proc.processId}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Genel Bilgiler</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <dt className="text-sm text-slate-500">Durum</dt>
              <dd>
                <Badge status={proc.status}>{mapProcessStatus(proc.status)}</Badge>
              </dd>
            </div>
            <div>
              <dt className="text-sm text-slate-500">Yüz Sayısı</dt>
              <dd className="font-medium text-navy-900">{proc.faceCount ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-sm text-slate-500">Başlangıç</dt>
              <dd className="font-medium text-navy-900">{formatDate(proc.createdAt)}</dd>
            </div>
            <div>
              <dt className="text-sm text-slate-500">Bitiş</dt>
              <dd className="font-medium text-navy-900">{proc.completedAt ? formatDate(proc.completedAt) : '—'}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <h2 className="text-lg font-semibold text-navy-900">Tespit Edilen Yüzler</h2>
      {proc.faces.length === 0 ? (
        <Alert variant="info" title="Yüz bulunamadı">
          Bu işlemde herhangi bir yüz tespit edilmedi.
        </Alert>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {proc.faces.map((face) => (
            <Card key={face.faceIndex}>
              <CardContent className="p-5">
                <div className="mb-3 flex items-center justify-between">
                  <span className="font-semibold text-navy-900">Yüz {face.faceIndex + 1}</span>
                  <Badge status={face.status}>{mapRecognizeStatus(face.status)}</Badge>
                </div>
                {face.score !== null && (
                  <SimilarityScore score={face.score} size="sm" showDecision={false} />
                )}
                {face.faceId ? (
                  <Link
                    to={`/faces/${face.faceId}`}
                    className="mt-3 inline-block text-sm font-medium text-primary hover:underline"
                  >
                    Yüz Detayını Gör
                  </Link>
                ) : (
                  <p className="mt-3 text-sm text-slate-400">Kayıtlı yüz eşleşmesi yok.</p>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
