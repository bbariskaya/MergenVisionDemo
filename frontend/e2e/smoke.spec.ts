import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const lfw = (relative: string) =>
  path.resolve(__dirname, '../../lfw/lfw-deepfunneled/lfw-deepfunneled', relative)

const fixtures = (name: string) => path.resolve(__dirname, 'fixtures', name)

function uniqueId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
}

async function uploadFile(page: ReturnType<typeof test>, filePath: string) {
  const input = page.locator('input[type="file"]').first()
  await input.setInputFiles(filePath)
}

test.describe('Interprobe UI smoke', () => {
  test('dashboard shows logo and health indicator', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('img[alt="Interprobe"]')).toBeVisible()
    await expect(page.getByText('Sistem Hazır').first()).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: 'e2e/screenshots/dashboard.png', fullPage: false })
  })

  test.describe('enroll and identify', () => {
    test('enrolls a face and shows masked national ID', async ({ page }) => {
      const nationalId = uniqueId('enroll')
      await page.goto('/enroll')
      await page.getByLabel('Ad Soyad').fill('E2E Test Person')
      await page.getByLabel('T.C. Kimlik Numarası').fill(nationalId)
      await uploadFile(page, lfw('Lino_Oviedo/Lino_Oviedo_0001.jpg'))
      await page.getByRole('button', { name: 'Kaydet' }).click()
      await expect(page.getByText('Kayıt Tamamlandı')).toBeVisible({ timeout: 60_000 })

      const pageContent = await page.content()
      expect(pageContent).not.toContain(nationalId)
    })

    test('identifies a known face', async ({ page }) => {
      const nationalId = uniqueId('known')
      const name = 'E2E Known Person'
      await page.goto('/enroll')
      await page.getByLabel('Ad Soyad').fill(name)
      await page.getByLabel('T.C. Kimlik Numarası').fill(nationalId)
      await uploadFile(page, lfw('Lino_Oviedo/Lino_Oviedo_0001.jpg'))
      await page.getByRole('button', { name: 'Kaydet' }).click()
      await expect(page.getByText('Kayıt Tamamlandı')).toBeVisible({ timeout: 60_000 })

      await page.goto('/identify')
      await uploadFile(page, lfw('Lino_Oviedo/Lino_Oviedo_0002.jpg'))
      await page.getByRole('button', { name: 'Tanı' }).click()

      await expect(page.getByText('Bulundu').first()).toBeVisible({ timeout: 60_000 })

      const content = await page.content()
      expect(content).not.toContain(nationalId)
      await page.screenshot({ path: 'e2e/screenshots/identify-known.png', fullPage: false })
    })

    test('identifies an unknown face', async ({ page }) => {
      await page.goto('/identify')
      await uploadFile(page, lfw('Jessica_Capshaw/Jessica_Capshaw_0001.jpg'))
      await page.getByRole('button', { name: 'Tanı' }).click()
      await expect(page.getByText('Bulunamadı').first()).toBeVisible({ timeout: 60_000 })
      await page.screenshot({ path: 'e2e/screenshots/identify-unknown.png', fullPage: false })
    })

    test('returns no-face result without error', async ({ page }) => {
      await page.goto('/identify')
      await uploadFile(page, fixtures('no-face.jpg'))
      await page.getByRole('button', { name: 'Tanı' }).click()
      await expect(page.getByText(/görselde yüz bulunamadı/i).first()).toBeVisible({ timeout: 60_000 })
      const content = await page.content()
      expect(content).not.toContain('Hata')
      await page.screenshot({ path: 'e2e/screenshots/identify-no-face.png', fullPage: false })
    })
  })

  test('registered faces list loads', async ({ page }) => {
    await page.goto('/search-face')
    await expect(page.getByRole('heading', { name: 'Kayıtlı Yüzler' })).toBeVisible()
    await expect(page.locator('table tbody tr').first()).toBeVisible({ timeout: 20_000 })
    await page.screenshot({ path: 'e2e/screenshots/face-list.png', fullPage: false })
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
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/')
    const menu = page.getByRole('button', { name: 'Menüyü aç' })
    await expect(menu).toBeVisible()
    await menu.click()
    await expect(page.locator('nav').getByRole('link', { name: 'Yüz Tanı', exact: true })).toBeVisible()
    await page.screenshot({ path: 'e2e/screenshots/mobile-nav.png', fullPage: false })
  })
})
