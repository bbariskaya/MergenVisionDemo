import { cn, formatSimilarity, matchMargin } from '@/lib/utils'

export interface SimilarityScoreProps {
  score: number | null
  threshold?: number
  size?: 'sm' | 'md' | 'lg'
  showDecision?: boolean
  className?: string
}

export function SimilarityScore({
  score,
  threshold,
  size = 'md',
  showDecision = true,
  className,
}: SimilarityScoreProps) {
  const formatted = formatSimilarity(score)
  const hasThreshold = threshold !== undefined && score !== null
  const matched = hasThreshold && score >= threshold!

  const sizes = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-2xl',
  }

  return (
    <div className={cn('inline-flex flex-col', className)}>
      <div className="flex items-baseline gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-navy-500">Benzerlik</span>
        <span className={cn('font-semibold tabular-nums text-navy-900', sizes[size])}>{formatted}</span>
      </div>
      {hasThreshold && showDecision && (
        <div className="mt-1 flex items-center gap-2 text-xs">
          <span
            className={cn(
              'inline-flex items-center rounded-full px-2 py-0.5 font-medium',
              matched ? 'bg-successSubtle text-success' : 'bg-dangerSubtle text-danger',
            )}
          >
            {matched ? 'Eşleşme' : 'Eşik altı'}
          </span>
          {score !== null && (
            <span className="tabular-nums text-navy-400">{matchMargin(score, threshold!)}</span>
          )}
        </div>
      )}
    </div>
  )
}
