# Interprobe UI Production Polish — Implementation Plan

> **Goal:** Transform the current functional admin-template UI into a polished, credible enterprise biometric identity operations console while using only real backend endpoints and preserving all existing user flows.

## Global Constraints
- No subagents or parallel agents.
- No fake data, metrics, charts or backend capabilities.
- Do not change ML behavior.
- No major new UI framework or animation dependency.
- No git add/commit/push/reset.
- UI language: professional Turkish.
- Use the real Interprobe logo at `/frontend/interprobe_logo.jpeg` — do not redraw or replace.
- Score is raw cosine similarity, not probability; label as “Benzerlik” / “Eşleşme Skoru”.
- WCAG AA contrast, visible focus, reduced-motion support, responsive 390×844 → 1920×1080.

## File Map

### Asset / token foundation
- `frontend/public/interprobe_logo.jpeg` — copy real logo here so Vite serves it.
- `frontend/index.html` — keep Plus Jakarta Sans, add preconnect already present.
- `frontend/tailwind.config.js` — refine palette: deep navy, cobalt primary, emerald success, warm off-white bg, charcoal text.
- `frontend/src/index.css` — update CSS variables and utility classes for typography, buttons, inputs, cards.
- `frontend/src/lib/utils.ts` — add `formatSimilarity`, `similarityPercent` helpers; keep status mappers.

### Application shell
- `frontend/src/components/Sidebar.tsx` — larger logo, product descriptor, refined active state, organized nav.
- `frontend/src/components/Layout.tsx` — contextual title/breadcrumb, Demo Ortamı badge, compact health indicator.
- `frontend/src/components/HealthIndicator.tsx` — keep compact popover; fine-tune colors.
- `frontend/src/App.tsx` — add `/system` route for full Sistem Durumu.

### Shared UI primitives
- `frontend/src/components/ui/Button.tsx` — keep variants, adjust primary to cobalt from orange.
- `frontend/src/components/ui/Card.tsx` — no major change.
- `frontend/src/components/ui/Badge.tsx` — add `size` prop option, keep status colors.
- `frontend/src/components/ui/Alert.tsx` — keep.
- `frontend/src/components/ui/Input.tsx` — keep.
- `frontend/src/components/ui/FileDropzone.tsx` — improve drag/drop visuals, keyboard focus.
- `frontend/src/components/ui/Modal.tsx` — keep, verify focus.
- `frontend/src/components/ui/Skeleton.tsx` — no change.
- **Create** `frontend/src/components/ui/EmptyState.tsx` — reusable centered empty state.
- **Create** `frontend/src/components/ui/SimilarityScore.tsx` — display similarity as decimal + optional threshold/margin.
- **Create** `frontend/src/components/PageHeader.tsx` — page title + subtitle + optional actions.

### Pages (redesign)
- `frontend/src/pages/DashboardPage.tsx` — hero metrics, primary CTAs, recent people from real list, compact health.
- `frontend/src/pages/IdentifyPage.tsx` — upload/demo stage, processing, result with overlay boxes, result cards, similarity language.
- `frontend/src/pages/EnrollPage.tsx` — stepper flow: Kişi Bilgileri → Fotoğraf → Kontrol/Kayıt; preview, masked ID.
- `frontend/src/pages/FaceSearchPage.tsx` — card grid/list toggle, search, filter, pagination, empty/error states.
- `frontend/src/pages/FaceDetailPage.tsx` — profile header, masked ID, photo gallery, recognition history.
- `frontend/src/pages/ProcessDetailPage.tsx` — improved score language.
- **Create** `frontend/src/pages/SystemStatusPage.tsx` — full health detail; moved from dashboard dominance.
- `frontend/src/pages/SettingsPage.tsx` — remove endpoint/phase notes; keep basic info only.
- `frontend/src/pages/NotFoundPage.tsx` — keep simple.

### Tests
- `frontend/src/components/ui/__tests__/Button.test.tsx` — update if variant colors asserted.
- **Create** `frontend/src/components/ui/__tests__/SimilarityScore.test.tsx` — TDD component.
- **Create** `frontend/src/components/ui/__tests__/EmptyState.test.tsx` — TDD component.
- `frontend/e2e/smoke.spec.ts` — update labels to new page titles and add dashboard CTA tests.

## Task Checklist

- [ ] **Task 1 — Brand assets & design tokens**
  - Copy `frontend/interprobe_logo.jpeg` → `frontend/public/interprobe_logo.jpeg`.
  - Update `tailwind.config.js` palette and `index.css` variables/buttons.
  - Add utility helpers in `lib/utils.ts`.

- [ ] **Task 2 — Application shell**
  - Redesign `Sidebar` with product descriptor and refined active indicator.
  - Redesign `Layout` topbar with breadcrumb, Demo Ortamı badge, compact health.
  - Add `/system` route in `App.tsx`; create `SystemStatusPage`.

- [ ] **Task 3 — Shared primitives**
  - TDD create `SimilarityScore` component.
  - TDD create `EmptyState` component.
  - Create `PageHeader` component.
  - Adjust `Button` primary color.
  - Polish `FileDropzone` keyboard/accessibility.

- [ ] **Task 4 — Dashboard**
  - Hero title/subtitle, primary/secondary CTA.
  - Real metrics from `/stats`.
  - “Son Eklenen Kişiler” from real face list (limit 5).
  - Compact system health summary linking to `/system`.

- [ ] **Task 5 — Recognition flow**
  - Before upload: refined dropzone, privacy note, supported formats.
  - During: spinner with image preview preserved.
  - After: bounding-box overlay, numbered faces, result cards, similarity language, match margin.

- [ ] **Task 6 — Enrollment flow**
  - Stepper UI: Kişi Bilgileri, Fotoğraf, Kontrol ve Kayıt.
  - Inline validation, masked national ID display, photo preview.
  - Success summary with links.

- [ ] **Task 7 — People list & detail**
  - Card grid with useful thumbnails, filters, pagination.
  - Person detail profile with primary portrait, gallery, history.

- [ ] **Task 8 — Quality pipeline**
  - Run `npm run lint` and fix issues.
  - Run `npm run typecheck` and fix issues.
  - Run `npm run test` for unit/component tests.
  - Run `npm run build`.
  - Run Playwright `npx playwright test` against running backend.
  - Capture 1440×900 and 390×844 final screenshots.

## Test Strategy
- TDD: write `SimilarityScore.test.tsx` and `EmptyState.test.tsx` first.
- E2E: update smoke tests to match new labels; add dashboard CTA, person-search and face-detail coverage.
