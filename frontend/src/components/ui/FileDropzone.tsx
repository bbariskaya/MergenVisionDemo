import { cn } from '@/lib/utils'
import { Upload, X } from 'lucide-react'
import { type ChangeEvent, type DragEvent, useRef, useState } from 'react'
import { Button } from './Button'

export interface FileDropzoneProps {
  accept?: string
  maxSizeBytes?: number
  value: File | null
  onChange: (file: File | null) => void
  previewUrl: string | null
  label?: string
  helperText?: string
  error?: string
}

export function FileDropzone({
  accept = 'image/jpeg,image/png,image/jpg',
  maxSizeBytes = 10 * 1024 * 1024,
  value,
  onChange,
  previewUrl,
  label = 'Görsel Yükle',
  helperText = 'Sürükleyip bırakın veya tıklayın. JPEG/PNG, max 10 MB.',
  error,
}: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  function validateAndSet(file: File) {
    if (!file.type.startsWith('image/')) {
      setLocalError('Yalnızca görsel dosyaları kabul edilir.')
      return
    }
    if (file.size > maxSizeBytes) {
      setLocalError('Dosya boyutu 10 MB’ı geçemez.')
      return
    }
    setLocalError(null)
    onChange(file)
  }

  function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) validateAndSet(file)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) validateAndSet(file)
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(true)
  }

  function handleDragLeave() {
    setIsDragging(false)
  }

  function clear() {
    onChange(null)
    setLocalError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const displayedError = error || localError

  return (
    <div className="w-full">
      {label && <p className="label">{label}</p>}
      {previewUrl ? (
        <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-white">
          <img
            src={previewUrl}
            alt="Yüklenecek görsel önizlemesi"
            className="max-h-80 w-full object-contain"
          />
          <div className="absolute right-2 top-2">
            <Button type="button" variant="danger" size="sm" onClick={clear} aria-label="Görseli kaldır">
              <X className="h-4 w-4" />
            </Button>
          </div>
          {value && (
            <p className="border-t border-slate-100 px-4 py-2 text-xs text-slate-500">
              {value.name} ({(value.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </div>
      ) : (
        <div
          onClick={() => inputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={cn(
            'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed bg-slate-50 p-8 transition-colors duration-150 hover:bg-slate-100',
            isDragging ? 'border-primary bg-primary-50' : 'border-slate-300',
            displayedError && 'border-danger bg-red-50',
          )}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
          }}
          aria-label={label}
        >
          <Upload className={cn('mb-2 h-8 w-8', displayedError ? 'text-danger' : 'text-slate-400')} aria-hidden="true" />
          <p className="text-sm font-medium text-slate-700">{isDragging ? 'Buraya bırakın' : 'Görsel seçin veya sürükleyin'}</p>
          <p className="mt-1 text-xs text-slate-500">{helperText}</p>
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            className="sr-only"
            onChange={handleFileSelect}
            aria-hidden="true"
          />
        </div>
      )}
      {displayedError && (
        <p className="mt-1.5 text-xs text-danger" role="alert">
          {displayedError}
        </p>
      )}
    </div>
  )
}
