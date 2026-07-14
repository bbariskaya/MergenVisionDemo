import { useRecognizeMutation } from '@/api/faces'
import type { RecognizeResponse, RecognizedFace } from '@/api/types'
import { PageHeader } from '@/components/PageHeader'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { FileDropzone } from '@/components/ui/FileDropzone'
import { SimilarityScore } from '@/components/ui/SimilarityScore'
import { useToast } from '@/hooks/useToast'
import { cn, clamp, formatSimilarity, mapRecognizeStatus } from '@/lib/utils'
import { ArrowLeft, ImageIcon, Loader2, RotateCcw, ScanFace, Search, User, UserCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'

export interface IdentifyPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

const OVERLAY_STYLES = [
  { border: 'border-primary', bg: 'bg-primary/10', text: 'text-primary-700', chipBg: 'bg-primary', chipText: 'text-white' },
  { border: 'border-navy-400', bg: 'bg-navy-100/50', text: 'text-navy-700', chipBg: 'bg-navy-600', chipText: 'text-white' },
  { border: 'border-success', bg: 'bg-success/10', text: 'text-emerald-700', chipBg: 'bg-success', chipText: 'text-white' },
  { border: 'border-warning', bg: 'bg-warning/10', text: 'text-amber-700', chipBg: 'bg-warning', chipText: 'text-white' },
]

export default function IdentifyPage({ onToast }: IdentifyPageProps) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(0.6)
  const [result, setResult] = useState<RecognizeResponse | null>(null)
  const [selectedFace, setSelectedFace] = useState<number | null>(null)

  const { addToast } = useToast()
  const showToast = onToast || addToast
  const recognize = useRecognizeMutation()

  const imageRef = useRef<HTMLImageElement>(null)
  const [imageSize, setImageSize] = useState<{ width: number; height: number; naturalWidth: number; naturalHeight: number } | null>(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      setResult(null)
      setImageSize(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setResult(null)
    try {
      const data = await recognize.mutateAsync({ image: file, topK, threshold })
      setResult(data)
      if (data.faceCount === 0) {
        showToast({ variant: 'info', title: 'Yüz algılanmadı', message: 'Görselde tanımlanabilir yüz bulunamadı.' })
      } else {
        showToast({ variant: 'success', title: 'Tanıma tamamlandı', message: `${data.faceCount} yüz bulundu.` })
      }
      if (data.faces.length > 0) setSelectedFace(0)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Tanıma işlemi başarısız.'
      showToast({ variant: 'error', title: 'Tanıma hatası', message })
    }
  }

  function reset() {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setSelectedFace(null)
    setImageSize(null)
    recognize.reset()
  }

  function updateImageSize() {
    const img = imageRef.current
    if (!img) return
    setImageSize({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    })
  }

  const faceStyles = result?.faces.map((_, i) => OVERLAY_STYLES[i % OVERLAY_STYLES.length]) || []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Yüz Tanıma"
        subtitle="Bir görsel yükleyerek kayıtlı kişilerle eşleştirin."
        action={
          result && (
            <Button type="button" variant="secondary" onClick={reset}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Yeni Görsel
            </Button>
          )
        }
      />

      <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Card className="overflow-hidden">
            <CardContent className="space-y-5 p-5">
              {previewUrl ? (
                <div className="relative overflow-hidden rounded-lg border border-navy-200">
                  <img
                    src={previewUrl}
                    alt="Sorgu görseli"
                    className="max-h-64 w-full object-contain"
                  />
                  <button
                    type="button"
                    onClick={() => setFile(null)}
                    className="absolute right-2 top-2 rounded bg-white/90 px-2 py-1 text-xs font-medium text-navy-700 shadow-sm hover:bg-white focus-visible:ring-primary"
                  >
                    Değiştir
                  </button>
                </div>
              ) : (
                <FileDropzone value={file} onChange={setFile} previewUrl={previewUrl} label="Görsel Yükle" />
              )}

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="topK" className="label">Sonuç Sayısı</label>
                  <input
                    id="topK"
                    type="number"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(e) => setTopK(clamp(Number(e.target.value) || 1, 1, 20))}
                    className="input"
                  />
                </div>
                <div>
                  <label htmlFor="threshold" className="label">Karar Eşiği</label>
                  <input
                    id="threshold"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={threshold}
                    onChange={(e) => setThreshold(clamp(Number(e.target.value) || 0, 0, 1))}
                    className="input"
                  />
                </div>
              </div>

              <Button type="submit" className="w-full" isLoading={recognize.isPending} disabled={!file}>
                <ScanFace className="mr-2 h-4 w-4" />
                {recognize.isPending ? 'Tanıma yapılıyor…' : 'Yüzleri Tanı'}
              </Button>

              <p className="text-xs leading-relaxed text-navy-400">
                Yüklenen görseller sadece tanıma işlemi için kullanılır ve saklanmaz.
              </p>
            </CardContent>
          </Card>

          {result && <ResultSummary result={result} threshold={threshold} />}
        </div>

        <div className="lg:col-span-2">
          <Card className="h-full min-h-[360px]">
            <CardContent className="relative h-full p-0">
              {recognize.isPending ? (
                <div className="relative flex h-full min-h-[360px] items-center justify-center overflow-hidden rounded-xl">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Sorgu görseli"
                      className="absolute inset-0 h-full w-full object-contain opacity-40"
                    />
                  )}
                  <div className="relative z-10 flex flex-col items-center text-navy-600">
                    <Loader2 className="mb-3 h-10 w-10 animate-spin text-primary" aria-hidden="true" />
                    <p className="font-medium">Yüzler tanınıyor…</p>
                    <p className="text-sm">Bu işlem birkaç saniye sürebilir.</p>
                  </div>
                </div>
              ) : previewUrl ? (
                <div className="relative mx-auto inline-block max-w-full p-4">
                  <img
                    ref={imageRef}
                    src={previewUrl}
                    alt="Sorgu görseli"
                    className="block max-h-[70vh] w-auto rounded-lg"
                    onLoad={updateImageSize}
                  />
                  {result && imageSize && result.faces.map((face, i) => (
                    <FaceOverlay
                      key={face.faceIndex}
                      face={face}
                      index={i}
                      style={faceStyles[i]}
                      isSelected={selectedFace === i}
                      onClick={() => setSelectedFace(i)}
                      imageSize={imageSize}
                    />
                  ))}
                </div>
              ) : (
                <div className="flex min-h-[360px] flex-col items-center justify-center text-navy-400">
                  <div className="mb-3 rounded-full bg-navy-50 p-3 text-navy-300">
                    <ImageIcon className="h-8 w-8" aria-hidden="true" />
                  </div>
                  <p className="font-medium text-navy-600">Başlamak için sol taraftan bir görsel yükleyin.</p>
                  <p className="text-sm">JPEG veya PNG, maksimum 10 MB.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </form>

      {result && result.faces.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {result.faces.map((face, i) => (
            <FaceResultCard
              key={face.faceIndex}
              face={face}
              index={i}
              style={faceStyles[i]}
              isSelected={selectedFace === i}
              onClick={() => setSelectedFace(i)}
              threshold={threshold}
            />
          ))}
        </div>
      )}

      {result && result.faces.length === 0 && (
        <Alert variant="info" title="Görselde yüz bulunamadı">
          Lütfen net, ön cephe bir yüz görseli yükleyin. Bu bir hata değil; görselde yüz olmadığı anlamına gelir.
        </Alert>
      )}
    </div>
  )
}

function ResultSummary({ result, threshold }: { result: RecognizeResponse; threshold: number }) {
  const known = result.faces.filter((f) => f.status === 'known').length
  const unknown = result.faces.length - known
  return (
    <Card>
      <CardContent className="space-y-4 p-5">
        <h3 className="font-semibold text-navy-900">Sonuç Özeti</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-navy-50 p-2">
            <p className="text-lg font-bold text-navy-900">{result.faceCount}</p>
            <p className="text-[10px] uppercase tracking-wide text-navy-500">Yüz</p>
          </div>
          <div className="rounded-lg bg-successSubtle p-2">
            <p className="text-lg font-bold text-success">{known}</p>
            <p className="text-[10px] uppercase tracking-wide text-navy-500">Eşleşen</p>
          </div>
          <div className="rounded-lg bg-navy-50 p-2">
            <p className="text-lg font-bold text-navy-900">{unknown}</p>
            <p className="text-[10px] uppercase tracking-wide text-navy-500">Bilinmeyen</p>
          </div>
        </div>
        <div className="flex items-center justify-between border-t border-navy-100 pt-3 text-sm">
          <span className="text-navy-500">Karar eşiği</span>
          <span className="font-medium tabular-nums text-navy-900">{threshold.toFixed(2)}</span>
        </div>
        <Link to={`/processes/${result.processId}`} className="btn-secondary block w-full text-center text-xs px-3 py-2">
          İşlem Detayını Gör
        </Link>
      </CardContent>
    </Card>
  )
}

function FaceOverlay({
  face,
  index,
  style,
  isSelected,
  onClick,
  imageSize,
}: {
  face: RecognizedFace
  index: number
  style: (typeof OVERLAY_STYLES)[number]
  isSelected: boolean
  onClick: () => void
  imageSize: { width: number; height: number; naturalWidth: number; naturalHeight: number }
}) {
  const { x1, y1, x2, y2 } = face.boundingBox
  const scaleX = imageSize.width / imageSize.naturalWidth
  const scaleY = imageSize.height / imageSize.naturalHeight
  const left = x1 * scaleX
  const top = y1 * scaleY
  const width = (x2 - x1) * scaleX
  const height = (y2 - y1) * scaleY

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'absolute border-2 bg-transparent transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2',
        style.border,
        isSelected ? 'ring-2 ring-offset-2 ring-primary' : 'opacity-80 hover:opacity-100',
      )}
      style={{ left, top, width, height }}
      aria-label={`Yüz ${index + 1}: ${mapRecognizeStatus(face.status)}`}
    >
      <span
        className={cn(
          'absolute -top-6 left-0 rounded px-1.5 py-0.5 text-xs font-bold',
          style.chipBg,
          style.chipText,
        )}
      >
        {index + 1}
      </span>
    </button>
  )
}

function FaceResultCard({
  face,
  index,
  style,
  isSelected,
  onClick,
  threshold,
}: {
  face: RecognizedFace
  index: number
  style: (typeof OVERLAY_STYLES)[number]
  isSelected: boolean
  onClick: () => void
  threshold: number
}) {
  const isKnown = face.status === 'known'
  const topCandidate = face.candidates[0]

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick() } }}
      className={cn(
        'card card-hover w-full cursor-pointer text-left transition-shadow focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
        isSelected && 'ring-2 ring-primary ring-offset-2',
      )}
    >
      <CardContent className="p-5">
        <div className="mb-4 flex items-center justify-between">
          <div
            className={cn(
              'flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-semibold',
              style.bg,
              style.border,
              style.text,
            )}
          >
            <span>Yüz {index + 1}</span>
          </div>
          <StatusBadge status={face.status} />
        </div>

        {isKnown && topCandidate ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              {topCandidate.photoId ? (
                <img
                  src={`/api/v1/photos/${topCandidate.photoId}`}
                  alt={topCandidate.name || ''}
                  className="h-14 w-14 rounded-lg object-cover"
                  loading="lazy"
                />
              ) : (
                <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-navy-50 text-navy-400">
                  <User className="h-6 w-6" />
                </div>
              )}
              <div className="min-w-0 flex-1">
                <p className="truncate font-semibold text-navy-900">{topCandidate.name || 'Kayıtlı kişi'}</p>
                <SimilarityScore score={topCandidate.score} threshold={threshold} size="sm" />
              </div>
            </div>
            <Link
              to={`/faces/${topCandidate.faceId}`}
              className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
              onClick={(e) => e.stopPropagation()}
            >
              Kişi detayını gör
              <ArrowLeft className="h-3.5 w-3.5 rotate-180" />
            </Link>
          </div>
        ) : (
          <div className="flex items-center gap-3 text-navy-500">
            <div className="flex h-14 w-14 items-center justify-center rounded-lg bg-navy-50">
              <Search className="h-6 w-6" />
            </div>
            <div>
              <p className="font-medium text-navy-900">Bilinmeyen kişi</p>
              <p className="text-sm">Bu yüz kayıtlı kişilerle eşleşmedi.</p>
            </div>
          </div>
        )}

        {face.candidates.length > 0 && topCandidate && (
          <div className="mt-4 border-t border-navy-100 pt-3">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-navy-400">İlk adaylar</p>
            <ul className="space-y-2">
              {face.candidates.slice(0, 3).map((candidate) => (
                <li key={candidate.faceId} className="flex items-center justify-between text-sm">
                  <span className="truncate text-navy-700">{candidate.name || 'İsimsiz kayıt'}</span>
                  <span className="tabular-nums text-navy-500">{formatSimilarity(candidate.score)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const isKnown = status === 'known'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
        isKnown ? 'bg-successSubtle text-success' : 'bg-navy-100 text-navy-600',
      )}
    >
      {isKnown ? <UserCheck className="h-3 w-3" /> : <Search className="h-3 w-3" />}
      {mapRecognizeStatus(status)}
    </span>
  )
}
