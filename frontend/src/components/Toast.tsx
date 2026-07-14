import { cn } from '@/lib/utils'
import type { Toast as ToastType } from '@/hooks/useToast'
import { AlertCircle, CheckCircle2, Info, X, XCircle } from 'lucide-react'

export interface ToastContainerProps {
  toasts: ToastType[]
  onRemove: (id: string) => void
}

export function ToastContainer({ toasts, onRemove }: ToastContainerProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-3" role="region" aria-live="polite" aria-label="Bildirimler">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onRemove={onRemove} />
      ))}
    </div>
  )
}

function ToastItem({ toast, onRemove }: { toast: ToastType; onRemove: (id: string) => void }) {
  const icons = {
    info: Info,
    success: CheckCircle2,
    warning: AlertCircle,
    error: XCircle,
  }
  const styles = {
    info: 'bg-blue-600 text-white',
    success: 'bg-emerald-600 text-white',
    warning: 'bg-amber-500 text-white',
    error: 'bg-red-600 text-white',
  }
  const Icon = icons[toast.variant]
  return (
    <div
      className={cn(
        'flex w-80 items-start gap-3 rounded-lg p-4 shadow-lg transition-all duration-200',
        styles[toast.variant],
      )}
      role="alert"
    >
      <Icon className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />
      <div className="flex-1">
        <p className="font-semibold">{toast.title}</p>
        {toast.message && <p className="mt-0.5 text-sm opacity-90">{toast.message}</p>}
      </div>
      <button
        onClick={() => onRemove(toast.id)}
        className="rounded p-1 hover:bg-white/20"
        aria-label="Bildirimi kapat"
      >
        <X className="h-4 w-4" />
      </button>
    </div>
  )
}
