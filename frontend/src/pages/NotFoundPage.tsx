import { Button } from '@/components/ui/Button'
import { Home } from 'lucide-react'
import { Link } from 'react-router'

export default function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
      <h1 className="text-6xl font-extrabold text-navy-900">404</h1>
      <p className="mt-4 text-lg text-slate-600">Sayfa bulunamadı.</p>
      <Link to="/" className="mt-6">
        <Button>
          <Home className="mr-2 h-4 w-4" />
          Ana Sayfaya Dön
        </Button>
      </Link>
    </div>
  )
}
