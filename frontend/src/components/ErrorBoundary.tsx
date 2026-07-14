import { Component, type ErrorInfo, type ReactNode } from 'react'
import { Alert } from './ui/Alert'
import { Button } from './ui/Button'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught error:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center p-6">
          <div className="w-full max-w-md">
            <Alert variant="error" title="Beklenmeyen bir hata oluştu">
              <p className="mb-4">Lütfen sayfayı yenileyin veya destek ekibine başvurun.</p>
              <Button onClick={() => window.location.reload()}>Sayfayı Yenile</Button>
            </Alert>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
