# Interprobe Modern Frontend UI Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a real-endpoint-bound, production-quality React SPA for the Interprobe biometric operations console in Turkish.

**Architecture:** Client-side Vite React TypeScript SPA with React Router (declarative) and TanStack Query. Tailwind CSS for styling. API calls isolated in a typed client. Domain pages: Dashboard, Enroll, Identify, Face Detail/History, Process Detail.

**Tech Stack:** React 19.2.7, React Router 7.5.3, TanStack Query 5.84.1, Vite 6.0.0, TypeScript 5.x, Tailwind CSS 3.4.x, Lucide React, Vitest, Playwright.

## Global Constraints
- Backend contract is immutable; only existing endpoints are used.
- Raw `national_id` must never be rendered; only `nationalIdMasked`.
- No internal storage keys, embeddings, or stack traces shown in UI or console.
- UI language: Turkish.
- Icons: Lucide only; no emojis.
- Light mode primary; dark mode acceptable but completeness required if added.
- `prefers-reduced-motion` respected.
- Responsive: 375px, 768px, 1024px, 1440px.
- No git commit/push; no backend changes.

---

### Task 1: Scaffold Vite React TS Project

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/vite-env.d.ts`
- Modify: `frontend/nginx.conf`

**Interfaces:**
- Produces: buildable Vite app, dev proxy to `/api/v1` and `/health/`.

- [ ] Step 1: Write package.json with exact dependencies and devDependencies.
- [ ] Step 2: Write tsconfig files with strict React settings.
- [ ] Step 3: Write Vite config with React plugin and proxy.
- [ ] Step 4: Write Tailwind + PostCSS config referencing `src/**/*.{ts,tsx}`.
- [ ] Step 5: Write root HTML and entrypoint mounting `<App />`.
- [ ] Step 6: Update nginx.conf to proxy `/api/` and `/health/` to api:8000 and serve `/index.html` fallback.
- [ ] Step 7: Run `npm install`.

Expected verification: `npm install` exits 0.

---

### Task 2: Design System + Global Styles

**Files:**
- Create: `frontend/src/index.css`
- Create: `frontend/src/lib/utils.ts`

**Interfaces:**
- Consumes: `design-system/interprobe/MASTER.md` tokens.
- Produces: Tailwind theme extension, global CSS variables, `cn()` utility.

- [ ] Step 1: Add CSS variables for primary/secondary/accent/background/foreground and spacing.
- [ ] Step 2: Import Plus Jakarta Sans font.
- [ ] Step 3: Write `cn()` tailwind-merge + clsx utility.
- [ ] Step 4: Create basic `.btn`, `.input`, `.card` classes.

Expected verification: dev server builds without CSS error.

---

### Task 3: Typed API Client and Query Factory

**Files:**
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/queryKeys.ts`
- Create: `frontend/src/api/faces.ts`
- Create: `frontend/src/api/processes.ts`
- Create: `frontend/src/api/health.ts`

**Interfaces:**
- Produces: typed request/response types mirroring OpenAPI schemas; `apiFetch`, `apiUpload` helpers; TanStack Query hooks.

- [ ] Step 1: Mirror OpenAPI schemas in TypeScript (`RecognizeResponse`, `EnrollResponse`, `FaceDetail`, `ProcessDetail`, `BulkEnrollResponse`, etc.).
- [ ] Step 2: Write `apiFetch` for JSON GET/DELETE and `apiUpload` for multipart POST.
- [ ] Step 3: Write query key factory (`health`, `face`, `process`).
- [ ] Step 4: Write hooks: `useHealthLive`, `useHealthReady`, `useFace`, `useFaceHistory`, `useProcess`, `useEnrollMutation`, `useRecognizeMutation`, `useDeleteFaceMutation`.

Expected verification: TypeScript compiles types file.

---

### Task 4: Application Shell

**Files:**
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/components/Layout.tsx`
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/components/Header.tsx`
- Create: `frontend/src/components/HealthIndicator.tsx`
- Create: `frontend/src/components/MobileNav.tsx`

**Interfaces:**
- Consumes: `useHealthLive`, `useHealthReady`.
- Produces: responsive shell with logo, nav, breadcrumb, health badge.

- [ ] Step 1: Implement Layout with sidebar (desktop collapsible) and mobile drawer.
- [ ] Step 2: Add Interprobe logo in sidebar/header using existing asset.
- [ ] Step 3: Add Turkish nav labels: Ana Sayfa, Kayıt, Tanıma, Yüz Detayı placeholder.
- [ ] Step 4: Add health indicator component showing ready/unavailable with component details.
- [ ] Step 5: Add breadcrumb based on current route.

Expected verification: Playwright screenshot shows logo + nav + health badge.

---

### Task 5: Common UI Components

**Files:**
- Create: `frontend/src/components/ui/Button.tsx`
- Create: `frontend/src/components/ui/Input.tsx`
- Create: `frontend/src/components/ui/Card.tsx`
- Create: `frontend/src/components/ui/Skeleton.tsx`
- Create: `frontend/src/components/ui/Alert.tsx`
- Create: `frontend/src/components/ui/Badge.tsx`
- Create: `frontend/src/components/ui/Modal.tsx`
- Create: `frontend/src/components/ui/FileDropzone.tsx`

**Interfaces:**
- Produces: reusable accessible primitives.

- [ ] Step 1: Implement each component with Tailwind tokens, focus states, reduced-motion.
- [ ] Step 2: FileDropzone supports drag-drop, click, MIME/size preview validation.

Expected verification: components render in Vitest smoke tests.

---

### Task 6: Dashboard Page

**Files:**
- Create: `frontend/src/pages/DashboardPage.tsx`

**Interfaces:**
- Consumes: `useHealthReady`, navigation actions.
- Produces: dashboard UI with status cards and quick actions.

- [ ] Step 1: Show health status card with component breakdown.
- [ ] Step 2: Show quick action cards: Kayıt Yap, Yüz Tanı, Sonuç Görüntüle.
- [ ] Step 3: Implement empty state if readiness unavailable.

Expected verification: Dashboard screenshot with logo, health, quick actions.

---

### Task 7: Enroll Page

**Files:**
- Create: `frontend/src/pages/EnrollPage.tsx`
- Create: `frontend/src/components/EnrollmentResult.tsx`

**Interfaces:**
- Consumes: `useEnrollMutation`.
- Produces: form + upload + result display.

- [ ] Step 1: Implement controlled form (name, nationalId fields) with validation.
- [ ] Step 2: Integrate FileDropzone with image preview.
- [ ] Step 3: Call `POST /api/v1/faces/enroll`.
- [ ] Step 4: Display success with masked ID only.
- [ ] Step 5: Display errors: no face detected, duplicate photo, validation.
- [ ] Step 6: Add reset flow.

Expected verification: Playwright flow enrolls a person and asserts masked ID shown, raw ID hidden.

---

### Task 8: Identify Page

**Files:**
- Create: `frontend/src/pages/IdentifyPage.tsx`
- Create: `frontend/src/components/RecognitionResult.tsx`
- Create: `frontend/src/components/FaceBoundingBox.tsx`

**Interfaces:**
- Consumes: `useRecognizeMutation`.
- Produces: identify workspace, results, face overlays.

- [ ] Step 1: Large drag-drop workspace with preview.
- [ ] Step 2: Top-k and threshold controls.
- [ ] Step 3: Call `POST /api/v1/faces/recognize`.
- [ ] Step 4: Render loading/processing state.
- [ ] Step 5: Render result cards with Turkish status mapping (known=bulundu, unknown=bulunamadı, no-face=yüz algılanmadı).
- [ ] Step 6: Overlay bounding boxes scaled to image natural dimensions.
- [ ] Step 7: Multi-face results use color/index mapping.
- [ ] Step 8: Show candidate list per face when known.

Expected verification: Playwright tests unknown face, no-face image, known face if enrolled.

---

### Task 9: Face Detail and History Page

**Files:**
- Create: `frontend/src/pages/FaceDetailPage.tsx`

**Interfaces:**
- Consumes: `useFace`, `useFaceHistory`, `useDeleteFaceMutation`.
- Produces: detail/history view.

- [ ] Step 1: Fetch face by UUID from route param.
- [ ] Step 2: Show person info, masked national ID, status, metadata.
- [ ] Step 3: Show recognition history list.
- [ ] Step 4: Implement delete with confirmation modal.
- [ ] Step 5: Handle not-found and error states.

Expected verification: Detail page renders with masked ID; deletion redirects.

---

### Task 10: Process Detail Page

**Files:**
- Create: `frontend/src/pages/ProcessDetailPage.tsx`

**Interfaces:**
- Consumes: `useProcess`.
- Produces: process detail view.

- [ ] Step 1: Fetch process by UUID.
- [ ] Step 2: Show status, timestamp, face count, known/unknown summary.
- [ ] Step 3: Link to face detail for known matches.

Expected verification: Playwright opens process detail after identify.

---

### Task 11: Error Boundary + Toast

**Files:**
- Create: `frontend/src/components/ErrorBoundary.tsx`
- Create: `frontend/src/hooks/useToast.ts`
- Create: `frontend/src/components/Toast.tsx`

**Interfaces:**
- Produces: global error boundary and toast notifications.

- [ ] Step 1: Implement route-level error boundary.
- [ ] Step 2: Implement toast hook and container.
- [ ] Step 3: Show sanitized API errors in toast/alert.

---

### Task 12: Unit and Component Tests

**Files:**
- Create: `frontend/src/setupTests.ts`
- Create: `frontend/src/components/ui/__tests__/Button.test.tsx`
- Create: `frontend/src/api/__tests__/types.test.ts`

**Interfaces:**
- Consumes: components and API types.

- [ ] Step 1: Configure Vitest with jsdom.
- [ ] Step 2: Write tests for Button and Alert components.
- [ ] Step 3: Write type-level tests ensuring masked fields exist.

Expected verification: `npm run test` passes.

---

### Task 13: Playwright Smoke Tests

**Files:**
- Create: `frontend/e2e/smoke.spec.ts`
- Create: `frontend/playwright.config.ts`

**Interfaces:**
- Consumes: running Vite dev server + backend.

- [ ] Step 1: Configure Playwright to test against `http://localhost:8080` or `http://localhost:5173` with proxy.
- [ ] Step 2: Test dashboard shell, logo, health indicator.
- [ ] Step 3: Test enroll flow with LFW image through real backend.
- [ ] Step 4: Test identify flow: known, unknown, and no-face if backend + fixtures available.
- [ ] Step 5: Verify no raw nationalId in DOM or responses.
- [ ] Step 6: Screenshot desktop and mobile viewports.

Expected verification: `npx playwright test` passes with screenshots.

---

### Task 14: Production Build + Nginx

**Files:**
- Modify: `frontend/package.json` scripts.
- Modify: `frontend/nginx.conf`.

- [ ] Step 1: Ensure build script compiles.
- [ ] Step 2: Verify nginx config with `nginx -t`.
- [ ] Step 3: Typecheck and lint pass.

Expected verification: `npm run build && npm run typecheck && npm run lint` exit 0; `nginx -t` with config exits 0.

---

### Task 15: Review Package

**Files:**
- Create: `backend/docs/implementation/review_packages/SPRINT-001-CODE-REVIEW-PACKAGE.md`

- [ ] Step 1: Summarize changed files, API mapping, blocked items.
- [ ] Step 2: Attach validation outputs and screenshot paths.
- [ ] Step 3: Update `IMPLEMENTATION_DETAILS.md`.

Expected verification: package file exists and is self-contained.
