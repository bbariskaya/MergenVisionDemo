import { render, screen } from '@testing-library/react'
import { Search } from 'lucide-react'
import { describe, expect, it } from 'vitest'
import { EmptyState } from '../EmptyState'

describe('EmptyState', () => {
  it('renders title and description', () => {
    render(<EmptyState title="Sonuç yok" description="Arama kriterlerini değiştirin." icon={Search} />)
    expect(screen.getByText('Sonuç yok')).toBeInTheDocument()
    expect(screen.getByText('Arama kriterlerini değiştirin.')).toBeInTheDocument()
  })

  it('exposes the icon as a decorative image', () => {
    render(<EmptyState title="Boş" icon={Search} />)
    const svg = document.querySelector('svg')
    expect(svg).toBeInTheDocument()
  })
})
