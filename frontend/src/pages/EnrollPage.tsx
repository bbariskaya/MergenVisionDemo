import { useEnrollMutation } from '@/api/faces'
import type { EnrollResponse } from '@/api/types'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { FileDropzone } from '@/components/ui/FileDropzone'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/hooks/useToast'
import { CheckCircle2, RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router'

export interface EnrollPageProps {
  onToast?: (toast: { variant: 'success' | 'error' | 'info'; title: string; message?: string }) => void
}

export default function EnrollPage({ onToast }: EnrollPageProps) {
  const [name, setName] = useState('')
  const [nationalId, setNationalId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
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

  function validate(): boolean {
    const next: Record<string, string> = {}
    if (!name.trim()) next.name = 'Ad soyad zorunludur.'
    if (!nationalId.trim()) next.nationalId = 'T.C. kimlik numarası zorunludur.'
    if (!file) next.file = 'Kayıt için görsel yükleyin.'
    setErrors(next)
    return Object.keys(next).length === 0
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!validate() || !file) return

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

  function reset() {
    setName('')
    setNationalId('')
    setFile(null)
    setPreviewUrl(null)
    setErrors({})
    setResult(null)
    enroll.reset()
  }

  if (result) {
    return (
      <div className="mx-auto max-w-2xl space-y-6">
        <div className="text-center">
          <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
            <CheckCircle2 className="h-8 w-8" aria-hidden="true" />
          </div>
          <h1 className="page-title">Kayıt Tamamlandı</h1>
          <p className="mt-1 text-slate-500">Yeni yüz kaydı başarıyla oluşturuldu.</p>
        </div>
        <Card>
          <CardContent className="space-y-4">
            <ResultRow label="Kişi" value={result.name} />
            <ResultRow label="Maskeleme" value="•••••••••••••" />
            <ResultRow label="Durum" value="Aktif" />
            <ResultRow label="Kayıt Zamanı" value={new Date(result.createdAt).toLocaleString('tr-TR')} />
            <div className="flex gap-3 pt-2">
              <Button variant="secondary" onClick={reset}>
                <RotateCcw className="mr-2 h-4 w-4" />
                Yeni Kayıt
              </Button>
              <Link to={`/faces/${result.faceId}`}>
                <Button>Detayı Gör</Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="page-title">Yüz Kaydı Yap</h1>
        <p className="mt-1 text-slate-500">Kişi bilgilerini ve yüz görselini girin.</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <Card>
          <CardContent className="space-y-5">
            <Input
              label="Ad Soyad"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Örn: Ahmet Yılmaz"
              autoComplete="name"
              error={errors.name}
            />
            <Input
              label="T.C. Kimlik Numarası"
              value={nationalId}
              onChange={(e) => setNationalId(e.target.value)}
              placeholder="11 haneli kimlik numarası"
              inputMode="numeric"
              autoComplete="off"
              error={errors.nationalId}
            />
            <FileDropzone
              label="Yüz Görseli"
              value={file}
              onChange={setFile}
              previewUrl={previewUrl}
              error={errors.file}
            />
            {enroll.error && (
              <Alert variant="error" title="Kayıt başarısız">
                {enroll.error.message || 'Bilinmeyen hata oluştu.'}
              </Alert>
            )}
            <div className="flex items-center justify-end gap-3 pt-2">
              <Button type="button" variant="ghost" onClick={reset} disabled={enroll.isPending}>
                Temizle
              </Button>
              <Button type="submit" isLoading={enroll.isPending}>
                Kaydet
              </Button>
            </div>
          </CardContent>
        </Card>
      </form>
    </div>
  )
}

function ResultRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-slate-100 py-2 last:border-b-0">
      <span className="text-sm text-slate-500">{label}</span>
      <span className="text-sm font-medium text-navy-900">{value}</span>
    </div>
  )
}
