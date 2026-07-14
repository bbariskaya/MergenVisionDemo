import { useEnrollMutation } from '@/api/faces'
import type { EnrollResponse } from '@/api/types'
import { PageHeader } from '@/components/PageHeader'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { FileDropzone } from '@/components/ui/FileDropzone'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/hooks/useToast'
import { cn } from '@/lib/utils'
import { CheckCircle2, ChevronLeft, ChevronRight, RotateCcw, UserCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router'

export interface EnrollPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

type Step = 'info' | 'photo' | 'review'

const steps: { id: Step; label: string }[] = [
  { id: 'info', label: 'Kişi Bilgileri' },
  { id: 'photo', label: 'Fotoğraf' },
  { id: 'review', label: 'Kontrol ve Kayıt' },
]

function maskNationalId(value: string): string {
  return value.replace(/\d(?=\d{4})/g, '*')
}

export default function EnrollPage({ onToast }: EnrollPageProps) {
  const [step, setStep] = useState<Step>('info')
  const [name, setName] = useState('')
  const [nationalId, setNationalId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [touched, setTouched] = useState<Record<string, boolean>>({})
  const [result, setResult] = useState<EnrollResponse | null>(null)

  const { addToast } = useToast()
  const showToast = onToast || addToast
  const enroll = useEnrollMutation()

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const errors: Record<string, string> = {}
  if (touched.name && !name.trim()) errors.name = 'Ad soyad zorunludur.'
  if (touched.nationalId && !nationalId.trim()) errors.nationalId = 'T.C. kimlik numarası zorunludur.'
  if (touched.photo && !file) errors.photo = 'Kayıt için görsel yükleyin.'

  function canProceed(): boolean {
    if (step === 'info') return name.trim().length > 0 && nationalId.trim().length > 0
    if (step === 'photo') return file !== null
    return true
  }

  function nextStep() {
    if (step === 'info') {
      setTouched((t) => ({ ...t, name: true, nationalId: true }))
      if (name.trim() && nationalId.trim()) setStep('photo')
      return
    }
    if (step === 'photo') {
      setTouched((t) => ({ ...t, photo: true }))
      if (file) setStep('review')
    }
  }

  function prevStep() {
    const idx = steps.findIndex((s) => s.id === step)
    if (idx > 0) setStep(steps[idx - 1].id)
  }

  function reset() {
    setName('')
    setNationalId('')
    setFile(null)
    setPreviewUrl(null)
    setStep('info')
    setTouched({})
    setResult(null)
    enroll.reset()
  }

  async function handleSubmit() {
    if (!file) return
    try {
      const data = await enroll.mutateAsync({
        name: name.trim(),
        nationalId: nationalId.trim(),
        image: file,
        metadata: { source: 'web-ui' },
      })
      setResult(data)
      showToast({ variant: 'success', title: 'Kayıt başarılı', message: data.name })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Kayıt işlemi başarısız oldu.'
      showToast({ variant: 'error', title: 'Kayıt hatası', message })
    }
  }

  if (result) {
    return (
      <div className="mx-auto max-w-xl space-y-6">
        <Card>
          <CardContent className="p-8 text-center">
            <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-successSubtle text-success">
              <UserCheck className="h-8 w-8" aria-hidden="true" />
            </div>
            <h1 className="page-title">Kayıt Tamamlandı</h1>
            <p className="page-subtitle mt-1">Yeni yüz kaydı başarıyla oluşturuldu.</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="space-y-4 p-6">
            <SummaryRow label="Kişi" value={result.name} />
            <SummaryRow label="Kimlik No" value={maskNationalId(nationalId)} />
            <SummaryRow label="Durum" value="Aktif" />
            <SummaryRow label="Kayıt Zamanı" value={new Date(result.createdAt).toLocaleString('tr-TR')} />
            <div className="flex flex-col gap-2 pt-2 sm:flex-row">
              <Button variant="secondary" onClick={reset} className="w-full sm:w-auto">
                <RotateCcw className="mr-2 h-4 w-4" />
                Yeni Kayıt
              </Button>
              <Link to={`/faces/${result.faceId}`} className="btn-primary w-full justify-center sm:w-auto">
                Detayı Gör
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <PageHeader
        title="Yeni Kişi Kaydı"
        subtitle="Kişi bilgilerini ve yüz görselini ekleyin."
      />

      <Stepper active={step} />

      <Card>
        <CardContent className="p-6">
          {step === 'info' && (
            <div className="space-y-5">
              <Input
                label="Ad Soyad"
                value={name}
                onChange={(e) => setName(e.target.value)}
                onBlur={() => setTouched((t) => ({ ...t, name: true }))}
                placeholder="Örn: Ahmet Yılmaz"
                autoComplete="name"
                error={errors.name}
              />
              <Input
                label="T.C. Kimlik Numarası"
                value={nationalId}
                onChange={(e) => setNationalId(e.target.value)}
                onBlur={() => setTouched((t) => ({ ...t, nationalId: true }))}
                placeholder="11 haneli kimlik numarası"
                inputMode="numeric"
                autoComplete="off"
                error={errors.nationalId}
              />
              <p className="text-xs text-navy-400">
                Kimlik numarası maskeleme ile gösterilir ve asla açık metin olarak saklanmaz.
              </p>
            </div>
          )}

          {step === 'photo' && (
            <div className="space-y-4">
              <FileDropzone
                label="Yüz Görseli"
                value={file}
                onChange={setFile}
                previewUrl={previewUrl}
                error={errors.photo}
              />
              {file && previewUrl && (
                <p className="text-xs text-navy-500">
                  Seçilen dosya: <span className="font-medium text-navy-700">{file.name}</span> ({(file.size / 1024).toFixed(1)} KB)
                </p>
              )}
              <ul className="space-y-1 text-xs text-navy-500">
                <li>• Net, ön cephe ve tek kişilik görsel tercih edilir.</li>
                <li>• Gölgesiz, ışıklı bir ortamda çekilmiş olmalı.</li>
                <li>• JPEG veya PNG, maksimum 10 MB.</li>
              </ul>
            </div>
          )}

          {step === 'review' && (
            <div className="space-y-5">
              <div className="rounded-lg border border-navy-200 bg-navy-50 p-4">
                <dl className="grid gap-3 sm:grid-cols-2">
                  <div>
                    <dt className="text-xs uppercase tracking-wide text-navy-500">Ad Soyad</dt>
                    <dd className="font-medium text-navy-900">{name}</dd>
                  </div>
                  <div>
                    <dt className="text-xs uppercase tracking-wide text-navy-500">Kimlik No</dt>
                    <dd className="font-medium text-navy-900">{maskNationalId(nationalId)}</dd>
                  </div>
                </dl>
              </div>
              {previewUrl && (
                <div>
                  <p className="mb-2 text-xs uppercase tracking-wide text-navy-500">Kayıt Fotoğrafı</p>
                  <img
                    src={previewUrl}
                    alt="Kayıt için seçilen yüz görseli"
                    className="max-h-64 rounded-lg border border-navy-200 object-contain"
                  />
                </div>
              )}
              <p className="text-sm text-navy-600">
                Kaydet düğmesine basarak kişiyi veritabanına ekleyeceksiniz. Bu işlem geri alınamaz.
              </p>
            </div>
          )}

          {enroll.error && (
            <Alert variant="error" title="Kayıt başarısız" className="mt-5">
              {enroll.error.message || 'Bilinmeyen hata oluştu.'}
            </Alert>
          )}

          <div className="mt-8 flex items-center justify-between border-t border-navy-100 pt-5">
            <Button type="button" variant="ghost" onClick={prevStep} disabled={step === 'info'}>
              <ChevronLeft className="mr-1 h-4 w-4" />
              Geri
            </Button>
            {step === 'review' ? (
              <Button onClick={handleSubmit} isLoading={enroll.isPending}>
                <CheckCircle2 className="mr-2 h-4 w-4" />
                Kaydet
              </Button>
            ) : (
              <Button type="button" onClick={nextStep} disabled={!canProceed()}>
                İleri
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function Stepper({ active }: { active: Step }) {
  const activeIndex = steps.findIndex((s) => s.id === active)
  return (
    <nav aria-label="Kayıt adımları">
      <ol className="flex items-center">
        {steps.map((s, i) => {
          const isActive = s.id === active
          const isCompleted = i < activeIndex
          return (
            <li key={s.id} className="flex flex-1 items-center">
              <div className="flex flex-col items-center gap-2">
                <span
                  className={cn(
                    'flex h-8 w-8 items-center justify-center rounded-full border-2 text-sm font-semibold transition-colors',
                    isActive
                      ? 'border-primary bg-primary text-white'
                      : isCompleted
                        ? 'border-primary bg-primary-50 text-primary'
                        : 'border-navy-200 bg-white text-navy-400',
                  )}
                  aria-current={isActive ? 'step' : undefined}
                >
                  {isCompleted ? <CheckCircle2 className="h-4 w-4" /> : i + 1}
                </span>
                <span
                  className={cn(
                    'hidden text-xs font-medium sm:block',
                    isActive ? 'text-navy-900' : 'text-navy-400',
                  )}
                >
                  {s.label}
                </span>
              </div>
              {i < steps.length - 1 && (
                <div
                  className={cn(
                    'mx-2 h-0.5 flex-1 rounded-full transition-colors',
                    i < activeIndex ? 'bg-primary' : 'bg-navy-200',
                  )}
                  aria-hidden="true"
                />
              )}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-navy-100 py-2 last:border-b-0">
      <span className="text-sm text-navy-500">{label}</span>
      <span className="text-sm font-medium text-navy-900">{value}</span>
    </div>
  )
}
