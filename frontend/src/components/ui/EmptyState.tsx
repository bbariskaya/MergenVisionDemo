import { cn } from '@/lib/utils'
import type { LucideIcon } from 'lucide-react'

export interface EmptyStateProps {
  icon: LucideIcon
  title: string
  description?: string
  action?: React.ReactNode
  className?: string
}

export function EmptyState({ icon: Icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center rounded-xl border border-dashed border-navy-200 bg-white p-10 text-center',
        className,
      )}
    >
      <div className="mb-3 inline-flex rounded-full bg-navy-50 p-3 text-navy-400">
        <Icon className="h-6 w-6" aria-hidden="true" />
      </div>
      <p className="font-medium text-navy-900">{title}</p>
      {description && <p className="mt-1 max-w-xs text-sm text-navy-500">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
