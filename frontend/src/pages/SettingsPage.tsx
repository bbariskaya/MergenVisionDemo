import { Card, CardContent } from '@/components/ui/Card'
import { Info } from 'lucide-react'

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="page-title">Ayarlar</h1>
      <p className="page-subtitle mt-1">Uygulama bilgileri.</p>
      <Card className="mt-6">
        <CardContent className="flex items-start gap-4 py-8">
          <Info className="mt-1 h-6 w-6 text-primary" aria-hidden="true" />
          <div>
            <h2 className="font-semibold text-navy-900">Interprobe Yüz Tanıma Platformu</h2>
            <p className="mt-1 text-sm text-navy-600">
              Bu ön yüz canlı backend servislerine bağlıdır. Demo ortamında çalışmaktadır.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
