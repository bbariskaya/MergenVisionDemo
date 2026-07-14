import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SimilarityScore } from '../SimilarityScore'

describe('SimilarityScore', () => {
  it('renders formatted similarity as decimal', () => {
    render(<SimilarityScore score={0.55123} />)
    expect(screen.getByText('0.55')).toBeInTheDocument()
  })

  it('renders an em-dash when score is null', () => {
    render(<SimilarityScore score={null} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('displays match and margin above threshold', () => {
    render(<SimilarityScore score={0.65} threshold={0.5} />)
    expect(screen.getByText('Eşleşme')).toBeInTheDocument()
    expect(screen.getByText('+0.15')).toBeInTheDocument()
  })

  it('displays below threshold when score is too low', () => {
    render(<SimilarityScore score={0.45} threshold={0.5} />)
    expect(screen.getByText('Eşik altı')).toBeInTheDocument()
  })

  it('does not mislabel score as confidence percentage', () => {
    render(<SimilarityScore score={0.65} />)
    expect(screen.queryByText(/güven/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/%65/)).not.toBeInTheDocument()
  })
})
