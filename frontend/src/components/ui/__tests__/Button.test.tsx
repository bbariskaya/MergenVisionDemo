import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Button } from '../Button'

describe('Button', () => {
  it('renders children', () => {
    render(<Button>Kaydet</Button>)
    expect(screen.getByRole('button', { name: 'Kaydet' })).toBeInTheDocument()
  })

  it('is disabled when isLoading', () => {
    render(<Button isLoading>Yükle</Button>)
    expect(screen.getByRole('button')).toBeDisabled()
  })

  it('calls onClick when clicked', async () => {
    const onClick = vi.fn()
    render(<Button onClick={onClick}>Tıkla</Button>)
    screen.getByRole('button', { name: 'Tıkla' }).click()
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})
