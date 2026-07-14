import {
  useBulkJob,
  useCancelBulkJob,
  useLatestBulkJob,
  useStartCasiaBulkJob,
  useStartLfwBulkJob,
  useStartVggfaceBulkJob,
} from '@/api/bulkJobs'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Modal } from '@/components/ui/Modal'
import { useEffect, useMemo, useState } from 'react'
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
  processed: number,
  requested: number,
  perSecond: number,
): string {
  if (!perSecond || processed <= 0) return '—'
  const remaining = Math.max(0, requested - processed)
  return formatDuration(remaining / perSecond)
}

export default function BulkEnrollmentPage() {
  const { data: latestJob, isLoading, error } = useLatestBulkJob()
  const [candidateJobId, setCandidateJobId] = useState<string>(latestJob?.jobId ?? '')

  useEffect(() => {
    if (latestJob?.jobId) setCandidateJobId(latestJob.jobId)
  }, [latestJob?.jobId])

  const { data: polledJob, error: pollError } = useBulkJob(candidateJobId)

  // Prefer the live polled job while one exists; fall back to the latest lookup.
  const job = useMemo(
    () => (candidateJobId && polledJob ? polledJob : latestJob),
    [candidateJobId, polledJob, latestJob],
  )
  const displayedError = error || pollError

  const startJob = useStartVggfaceBulkJob()
  const startLfwJob = useStartLfwBulkJob()
  const startCasiaJob = useStartCasiaBulkJob()
  const cancelJob = useCancelBulkJob()
  const [confirmOpen, setConfirmOpen] = useState(false)

  useEffect(() => {
    if (startJob.data?.jobId) setCandidateJobId(startJob.data.jobId)
  }, [startJob.data?.jobId])

  useEffect(() => {
    if (startLfwJob.data?.jobId) setCandidateJobId(startLfwJob.data.jobId)
  }, [startLfwJob.data?.jobId])

  useEffect(() => {
    if (startCasiaJob.data?.jobId) setCandidateJobId(startCasiaJob.data.jobId)
  }, [startCasiaJob.data?.jobId])

  const isRunning =
    candidateJobId &&
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
          <div className="flex items-center gap-3">
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
            <Button
              variant="secondary"
              onClick={() => startLfwJob.mutate(undefined)}
              disabled={startLfwJob.isPending || isLoading}
            >
              {startLfwJob.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              LFW İşlemini Başlat
            </Button>
            <Button
              variant="secondary"
              onClick={() => startCasiaJob.mutate(undefined)}
              disabled={startCasiaJob.isPending || isLoading}
            >
              {startCasiaJob.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              CASIA İşlemini Başlat
            </Button>
          </div>
        )}
      </div>

      {displayedError && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">
          Henüz bir toplu kayıt işlemi başlatılmamış veya durum alınamadı.
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
                {job.totalProcessed.toLocaleString('tr-TR')} /{' '}
                {job.requestedPhotos.toLocaleString('tr-TR')}
              </div>
              <p className="mt-1 text-xs text-slate-500">
                {job.totalScanned.toLocaleString('tr-TR')} tarandı,{' '}
                {job.totalEnrolled.toLocaleString('tr-TR')} kaydedildi
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-500">
                Durum Özeti
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-navy-900">
                {job.totalDuplicate.toLocaleString('tr-TR')} /{' '}
                {job.totalNoFace.toLocaleString('tr-TR')} /{' '}
                {job.totalErrors.toLocaleString('tr-TR')}
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Yinelenen / yüz yok / hata
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
            <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-6">
              <div>
                <p className="text-xs text-slate-500">Geçen Süre</p>
                <p className="text-lg font-semibold text-navy-900">
                  {formatDuration(job.elapsedSeconds)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Tahmini Kalan</p>
                <p className="text-lg font-semibold text-navy-900">
                  {formatEta(job.totalProcessed, job.requestedPhotos, job.processedPhotosPerSecond)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Ort. Fotoğraf/sn</p>
                <p className="text-lg font-semibold text-navy-900">
                  {job.avgPhotosPerSecond.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Taradı/sn</p>
                <p className="text-lg font-semibold text-navy-900">
                  {job.scannedPhotosPerSecond.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">İşledi/sn</p>
                <p className="text-lg font-semibold text-navy-900">
                  {job.processedPhotosPerSecond.toFixed(2)}
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
                    ? Math.min(100, Math.round((job.totalProcessed / job.requestedPhotos) * 100))
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
                        ? Math.min(100, (job.totalProcessed / job.requestedPhotos) * 100)
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
                if (candidateJobId) cancelJob.mutate(candidateJobId)
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
