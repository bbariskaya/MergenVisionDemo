import { ErrorBoundary } from '@/components/ErrorBoundary'
import { Layout } from '@/components/Layout'
import { ToastContainer } from '@/components/Toast'
import { useToast } from '@/hooks/useToast'
import DashboardPage from '@/pages/DashboardPage'
import EnrollPage from '@/pages/EnrollPage'
import FaceDetailPage from '@/pages/FaceDetailPage'
import FaceSearchPage from '@/pages/FaceSearchPage'
import IdentifyPage from '@/pages/IdentifyPage'
import NotFoundPage from '@/pages/NotFoundPage'
import ProcessDetailPage from '@/pages/ProcessDetailPage'
import SettingsPage from '@/pages/SettingsPage'
import { Route, Routes } from 'react-router'

function AppContent() {
  const { toasts, addToast, removeToast } = useToast()

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage onToast={addToast} />} />
        <Route path="/enroll" element={<EnrollPage onToast={addToast} />} />
        <Route path="/identify" element={<IdentifyPage onToast={addToast} />} />
        <Route path="/search-face" element={<FaceSearchPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/faces/:faceId" element={<FaceDetailPage onToast={addToast} />} />
        <Route path="/processes/:processId" element={<ProcessDetailPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </Layout>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  )
}
