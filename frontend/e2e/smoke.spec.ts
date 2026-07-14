import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const lfw = (relative: string) =>
  path.resolve(__dirname, '../../lfw/lfw-deepfunneled/lfw-deepfunneled', relative)

const fixtures = (name: string) => path.resolve(__dirname, 'fixtures', name)

async function uploadFile(page: ReturnType<typeof test>, filePath: string) {
  const input = page.locator('input[type="file"]').first()
  await input.setInputFiles(filePath)
}

async function deleteE2EPeople(request: ReturnType<typeof test>['request']) {
  try {
    const response = await request.get('/api/v1/faces?search=E2E&limit=100')
    if (!response.ok()) return
    const data = await response.json()
    for (const person of data.items ?? []) {
      await request.delete(`/api/v1/faces/${person.faceId}`)
    }
  } catch {
    // ignore cleanup failures
  }
}

test.describe('Interprobe UI smoke', () => {
  test('dashboard shows logo, health indicator and primary CTA', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.goto('/')
    await expect(page.locator('img[alt="Interprobe"]')).toBeVisible()
    await expect(page.getByRole('main').getByRole('heading', { name: 'Yüz Tanıma Operasyon Merkezi' })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Görselden Yüz Tanı' })).toBeVisible()
    await expect(page.getByRole('button', { name: /Hazır/ }).first()).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: 'e2e/screenshots/dashboard.png', fullPage: false })
  })

  test.describe('enroll and identify', () => {
    test.beforeEach(async ({ page }) => {
      await deleteE2EPeople(page.request)
    })

    test.afterEach(async ({ page }) => {
      await deleteE2EPeople(page.request)
    })

    test('enrolls a face through the stepper and shows masked national ID', async ({ page }) => {
      const nationalId = '11111111111'
      await page.goto('/enroll')
      await page.getByLabel('Ad Soyad').fill('E2E Test Person')
      await page.getByLabel('T.C. Kimlik Numarası').fill(nationalId)
      await page.getByRole('button', { name: 'İleri' }).click()

      await uploadFile(page, lfw('Lino_Oviedo/Lino_Oviedo_0001.jpg'))
      await page.getByRole('button', { name: 'İleri' }).click()

      await page.getByRole('button', { name: 'Kaydet' }).click()
      await expect(page.getByText('Kayıt Tamamlandı')).toBeVisible({ timeout: 60_000 })

      const pageContent = await page.content()
      expect(pageContent).not.toContain(nationalId)
    })

    test('identifies a known face', async ({ page }) => {
      await page.goto('/identify')
      await uploadFile(page, lfw('Jessica_Capshaw/Jessica_Capshaw_0001.jpg'))
      await page.getByRole('button', { name: 'Yüzleri Tanı' }).click()

      await expect(page.getByText('Bulundu').first()).toBeVisible({ timeout: 60_000 })

      await page.screenshot({ path: 'e2e/screenshots/identify-known.png', fullPage: false })
    })

    test('identifies an unknown face', async ({ page }) => {
      await page.goto('/identify')
      await uploadFile(page, fixtures('unknown-face.jpg'))
      await page.fill('#threshold', '0.7')
      await page.getByRole('button', { name: 'Yüzleri Tanı' }).click()
      await expect(page.getByText('Bulunamadı').first()).toBeVisible({ timeout: 20_000 })
      await page.screenshot({ path: 'e2e/screenshots/identify-unknown.png', fullPage: false })
    })

    test('returns no-face result without error', async ({ page }) => {
      await page.goto('/identify')
      await uploadFile(page, fixtures('no-face.jpg'))
      await page.getByRole('button', { name: 'Yüzleri Tanı' }).click()
      await expect(page.getByText(/görselde yüz bulunamadı/i).first()).toBeVisible({ timeout: 60_000 })
      const content = await page.content()
      expect(content).not.toContain('Hata')
      await page.screenshot({ path: 'e2e/screenshots/identify-no-face.png', fullPage: false })
    })
  })

  test('registered faces list loads and person detail opens', async ({ page }) => {
    await page.goto('/search-face')
    await expect(page.getByRole('main').getByRole('heading', { name: 'Kişiler' })).toBeVisible()
    const firstCard = page.locator('a[href^="/faces/"]').first()
    await expect(firstCard).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: 'e2e/screenshots/face-list.png', fullPage: false })

    await firstCard.click()
    await expect(page.getByRole('main').getByRole('heading', { name: 'Fotoğraflar' })).toBeVisible()
  })

  test('no raw national ID leaks in network responses', async ({ page }) => {
    const rawIds: string[] = []
    page.on('response', async (response) => {
      const url = response.url()
      if (url.includes('/api/v1/')) {
        try {
          const body = await response.text()
          if (body.includes('12345678901') || body.includes('11111111111')) {
            rawIds.push(url)
          }
        } catch {
          // ignore binary or unreadable bodies
        }
      }
    })
    await page.goto('/')
    await page.goto('/enroll')
    await page.goto('/identify')
    await page.goto('/search-face')

    expect(rawIds).toHaveLength(0)
  })

  test('mobile navigation opens', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/')
    const menu = page.getByRole('button', { name: 'Menüyü aç' })
    await expect(menu).toBeVisible()
    await menu.click()
    const drawer = page.getByTestId('mobile-drawer')
    await expect(drawer.getByRole('link', { name: 'Yüz Tanıma', exact: true })).toBeVisible()
    await expect(drawer.getByRole('link', { name: 'Kişiler', exact: true })).toBeVisible()
    await page.screenshot({ path: 'e2e/screenshots/mobile-nav.png', fullPage: false })
  })

  test('dashboard at 1440x900', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await page.screenshot({ path: 'e2e/screenshots/final-dashboard-1440x900.png', fullPage: false })
  })

  test('dashboard at 390x844', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await page.goto('/')
    await page.waitForLoadState('networkidle')
    await page.screenshot({ path: 'e2e/screenshots/final-dashboard-390x844.png', fullPage: false })
  })

  test('keyboard navigation shows visible focus', async ({ page }) => {
    await page.goto('/')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    const focused = await page.evaluate(() => document.activeElement?.tagName)
    expect(focused).not.toBe('BODY')
    const focusVisible = await page.evaluate(() => {
      const el = document.activeElement
      if (!el || el === document.body) return false
      const s = window.getComputedStyle(el)
      return s.outlineWidth !== '0px' || /box-shadow/.test(s.boxShadow)
    })
    expect(focusVisible).toBe(true)
  })

  test('all routes load without severe console or request errors', async ({ page }) => {
    const consoleErrors: string[] = []
    const failedRequests: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => consoleErrors.push(err.message))
    page.on('response', (resp) => {
      if (resp.status() >= 400 && !resp.url().includes('/api/v1/health')) {
        failedRequests.push(`${resp.status()} ${resp.url()}`)
      }
    })
    for (const route of ['/', '/identify', '/enroll', '/search-face', '/processes', '/system', '/settings']) {
      await page.goto(route)
      await page.waitForLoadState('networkidle')
    }
    expect(consoleErrors).toEqual([])
    expect(failedRequests).toEqual([])
  })
})
