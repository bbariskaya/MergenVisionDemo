import { cn, statusColor } from '@/lib/utils'
import { type HTMLAttributes } from 'react'

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  status?: string
  children: React.ReactNode
}

export function Badge({ status, children, className, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium',
        status ? statusColor(status) : 'bg-slate-100 text-slate-700 border-slate-200',
        className,
      )}
      {...props}
    >
      {children}
    </span>
  )
}
