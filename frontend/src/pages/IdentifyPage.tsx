import { useRecognizeMutation } from '@/api/faces'
import type { RecognizeResponse, RecognizedFace } from '@/api/types'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { FileDropzone } from '@/components/ui/FileDropzone'
import { useToast } from '@/hooks/useToast'
import { cn, clamp, mapRecognizeStatus } from '@/lib/utils'
import { Loader2, Search, UserCheck, UserX } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'

export interface IdentifyPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

const PALETTE = [
  'border-emerald-500 bg-emerald-500/10 text-emerald-700',
  'border-blue-500 bg-blue-500/10 text-blue-700',
  'border-amber-500 bg-amber-500/10 text-amber-700',
  'border-purple-500 bg-purple-500/10 text-purple-700',
  'border-pink-500 bg-pink-500/10 text-pink-700',
  'border-cyan-500 bg-cyan-500/10 text-cyan-700',
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

  const faceColors = result?.faces.map((_, i) => PALETTE[i % PALETTE.length]) || []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">Yüz Tanı</h1>
        <p className="mt-1 text-slate-500">Bir görsel yükleyerek kayıtlı yüzlerle eşleştirin.</p>
      </div>

      <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Card>
            <CardContent className="space-y-4">
              <FileDropzone value={file} onChange={setFile} previewUrl={previewUrl} />
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="topK" className="label">Sonuç Sayısı (Top-K)</label>
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
                  <label htmlFor="threshold" className="label">Eşik Değer</label>
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
                <Search className="mr-2 h-4 w-4" />
                Tanı
              </Button>
              {result && (
                <Button type="button" variant="ghost" className="w-full" onClick={reset}>
                  Yeni Görsel
                </Button>
              )}
            </CardContent>
          </Card>

          {result && (
            <ResultSummary result={result} />
          )}
        </div>

        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardContent className="relative">
              {recognize.isPending ? (
                <div className="flex min-h-[320px] flex-col items-center justify-center text-slate-500">
                  <Loader2 className="mb-3 h-10 w-10 animate-spin text-primary" aria-hidden="true" />
                  <p className="font-medium">Yüzler tanınıyor…</p>
                  <p className="text-sm">Bu işlem birkaç saniye sürebilir.</p>
                </div>
              ) : previewUrl ? (
                <div className="relative mx-auto inline-block max-w-full">
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
                      colorClass={faceColors[i]}
                      isSelected={selectedFace === i}
                      onClick={() => setSelectedFace(i)}
                      imageSize={imageSize}
                    />
                  ))}
                </div>
              ) : (
                <div className="flex min-h-[320px] flex-col items-center justify-center text-slate-400">
                  <Search className="mb-2 h-12 w-12" aria-hidden="true" />
                  <p>Başlamak için sol taraftan bir görsel yükleyin.</p>
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
              colorClass={faceColors[i]}
              isSelected={selectedFace === i}
              onClick={() => setSelectedFace(i)}
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

function ResultSummary({ result }: { result: RecognizeResponse }) {
  const known = result.faces.filter((f) => f.status === 'known').length
  const unknown = result.faces.length - known
  return (
    <Card>
      <CardContent className="space-y-3">
        <h3 className="font-semibold text-navy-900">Sonuç Özeti</h3>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-500">Toplam Yüz</span>
          <span className="font-medium">{result.faceCount}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-500">Bulundu</span>
          <span className="font-medium text-emerald-600">{known}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-500">Bulunamadı</span>
          <span className="font-medium text-slate-600">{unknown}</span>
        </div>
        <Link to={`/processes/${result.processId}`} className="block pt-2">
          <Button variant="secondary" className="w-full">
            İşlem Detayını Gör
          </Button>
        </Link>
      </CardContent>
    </Card>
  )
}

function FaceOverlay({
  face,
  index,
  colorClass,
  isSelected,
  onClick,
  imageSize,
}: {
  face: RecognizedFace
  index: number
  colorClass: string
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
        'absolute border-2 transition-all duration-150 focus:outline-none',
        colorClass,
        isSelected ? 'ring-2 ring-offset-2' : 'opacity-80 hover:opacity-100',
      )}
      style={{ left, top, width, height }}
      aria-label={`Yüz ${index + 1}: ${mapRecognizeStatus(face.status)}`}
    >
      <span
        className={cn(
          'absolute -top-6 left-0 rounded px-1.5 py-0.5 text-xs font-bold',
          colorClass.split(' ')[2],
          colorClass.split(' ')[1].replace('/10', '/90'),
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
  colorClass,
  isSelected,
  onClick,
}: {
  face: RecognizedFace
  index: number
  colorClass: string
  isSelected: boolean
  onClick: () => void
}) {
  const isKnown = face.status === 'known'
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'card w-full text-left transition-all duration-150',
        isSelected && 'ring-2 ring-primary ring-offset-2',
      )}
    >
      <CardContent className="p-5">
        <div className="mb-3 flex items-center justify-between">
          <div className={cn('flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-semibold', colorClass)}>
            <span>Yüz {index + 1}</span>
          </div>
          {isKnown ? (
            <UserCheck className="h-5 w-5 text-emerald-600" aria-hidden="true" />
          ) : (
            <UserX className="h-5 w-5 text-slate-400" aria-hidden="true" />
          )}
        </div>
        <p className="text-sm text-slate-500">Durum</p>
        <p className="font-semibold text-navy-900">{mapRecognizeStatus(face.status)}</p>
        {isKnown && face.name && (
          <>
            <p className="mt-2 text-sm text-slate-500">Kişi</p>
            <p className="font-semibold text-navy-900">{face.name}</p>
          </>
        )}
        {isKnown && face.confidence !== null && (
          <>
            <p className="mt-2 text-sm text-slate-500">Benzerlik Skoru</p>
            <p className="text-lg font-bold text-emerald-600">{(face.confidence * 100).toFixed(1)}%</p>
          </>
        )}
        {face.candidates.length > 0 && (
          <div className="mt-4 border-t border-slate-100 pt-3">
            <p className="mb-2 text-xs font-semibold uppercase text-slate-400">Adaylar</p>
            <ul className="space-y-1.5">
              {face.candidates.slice(0, 3).map((candidate) => (
                <li key={candidate.faceId} className="flex items-center justify-between text-sm">
                  <span className="truncate text-slate-600">{candidate.faceId.slice(0, 8)}…</span>
                  <span className="font-medium text-primary">{(candidate.score * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </button>
  )
}
