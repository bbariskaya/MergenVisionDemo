import { useCancelBulkJob, useLatestBulkJob, useStartVggfaceBulkJob } from '@/api/bulkJobs'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Modal } from '@/components/ui/Modal'
import { useEffect, useState } from 'react'
import { Activity, Loader2, Pause, Play, Users } from 'lucide-react'

function formatDuration(seconds: number): string {
  if (!seconds || seconds < 0) return '0s'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const parts: string[] = []
  if (h > 0) parts.push(`${h}s`)
  if (m > 0 || h > 0) parts.push(`${m}d`)
  parts.push(`${s}sn`)
  return parts.join(' ')
}

function formatEta(
  enrolled: number,
  requested: number,
  avgPerSecond: number,
): string {
  if (!avgPerSecond || enrolled <= 0) return '—'
  const remaining = Math.max(0, requested - enrolled)
  return formatDuration(remaining / avgPerSecond)
}

export default function BulkEnrollmentPage() {
  const { data: job, isLoading, error } = useLatestBulkJob()
  const startJob = useStartVggfaceBulkJob()
  const cancelJob = useCancelBulkJob()
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [jobId, setJobId] = useState<string>(job?.jobId ?? '')

  useEffect(() => {
    if (job?.jobId) setJobId(job.jobId)
  }, [job?.jobId])

  const isRunning =
    jobId &&
    job &&
    ['queued', 'running', 'cancel_requested', 'cancelling'].includes(job.status)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Toplu Kayıt İşlemi</h1>
        {isRunning ? (
          <Button
            variant="danger"
            onClick={() => setConfirmOpen(true)}
            disabled={cancelJob.isPending}
          >
            <Pause className="mr-2 h-4 w-4" />
            İşlemi Durdur
          </Button>
        ) : (
          <Button
            onClick={() => startJob.mutate(undefined)}
            disabled={startJob.isPending || isLoading}
          >
            {startJob.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            VGGFace İşlemini Başlat
          </Button>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
          Henüz bir toplu kayıt işlemi başlatılmamış.
        </div>
      )}

      {job && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-500">
                Durum
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2 text-2xl font-bold text-navy-900">
                <Activity className="h-6 w-6 text-primary" />
                {job.status === 'running' && 'Çalışıyor'}
                {job.status === 'queued' && 'Sırada'}
                {job.status === 'cancel_requested' && 'Durdurma İsteği'}
                {job.status === 'cancelling' && 'Durduruluyor'}
                {job.status === 'cancelled' && 'Durduruldu'}
                {job.status === 'completed' && 'Tamamlandı'}
                {job.status === 'failed' && 'Başarısız'}
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {job.assignedWorkers.join(', ')}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-500">
                Hedef / Mevcut Fotoğraf
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-navy-900">
                {job.currentActivePhotos.toLocaleString('tr-TR')} /{' '}
                {job.targetTotalActivePhotos.toLocaleString('tr-TR')}
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Bu işlemde eklendi: {job.photosAddedByJob.toLocaleString('tr-TR')}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-500">
                İlerleme
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-navy-900">
                {job.totalEnrolled.toLocaleString('tr-TR')} /{' '}
                {job.requestedPhotos.toLocaleString('tr-TR')}
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {job.totalDuplicate} yinelenen, {job.totalNoFace} yüz bulunamadı
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {job && (
        <Card>
          <CardHeader>
            <CardTitle className="text-navy-900">Performans</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
              <div>
                <p className="text-xs text-slate-500">Geçen Süre</p>
                <p className="text-lg font-semibold text-navy-900">
                  {formatDuration(job.elapsedSeconds)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Tahmini Kalan</p>
                <p className="text-lg font-semibold text-navy-900">
                  {formatEta(job.totalEnrolled, job.requestedPhotos, job.avgPhotosPerSecond)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Ort. Fotoğraf/sn</p>
                <p className="text-lg font-semibold text-navy-900">
                  {job.avgPhotosPerSecond.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">GPU 0 Probe p95</p>
                <p className="text-lg font-semibold text-navy-900">
                  {job.probeP95Ms ? `${job.probeP95Ms.toFixed(0)} ms` : '—'}
                </p>
              </div>
            </div>

            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-500">
                <span>İşlenen</span>
                <span>
                  {job.requestedPhotos > 0
                    ? Math.min(100, Math.round((job.totalEnrolled / job.requestedPhotos) * 100))
                    : 0}
                  %
                </span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{
                    width: `${
                      job.requestedPhotos > 0
                        ? Math.min(100, (job.totalEnrolled / job.requestedPhotos) * 100)
                        : 0
                    }%`,
                  }}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {job && job.shards.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-navy-900">İşçi Durumu</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {job.shards.map((shard) => (
                <div
                  key={shard.shardIndex}
                  className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 p-3"
                >
                  <div className="flex items-center gap-3">
                    {shard.status === 'running' ? (
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    ) : (
                      <Users className="h-4 w-4 text-slate-400" />
                    )}
                    <div>
                      <p className="text-sm font-medium text-navy-900">{shard.workerId}</p>
                      <p className="text-xs text-slate-500">
                        {shard.status} — shard {shard.shardIndex}
                      </p>
                    </div>
                  </div>
                  <div className="text-right text-sm text-navy-900">
                    {(shard.progress?.faces_enrolled as number) ?? 0} kayıt
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Modal
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title="İşlemi Durdur"
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
              Vazgeç
            </Button>
            <Button
variant="danger"
              onClick={() => {
                if (jobId) cancelJob.mutate(jobId)
                setConfirmOpen(false)
              }}
              disabled={cancelJob.isPending}
            >
              {cancelJob.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Pause className="mr-2 h-4 w-4" />
              )}
              Durdur
            </Button>
          </>
        }
      >
        <p>
          Toplu kayıt işlemi güvenli biçimde durdurulacak. Tamamlanmış kayıtlar korunacaktır.
        </p>
      </Modal>
    </div>
  )
}
