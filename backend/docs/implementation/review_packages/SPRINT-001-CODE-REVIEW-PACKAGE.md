# SPRINT-001 — Modern Interprobe UI/UX Code Review Package

**Sprint:** `MODERN_INTERPROBE_UI_UX_SPRINT`  
**Scope:** Frontend-only; backend contract consumed but not modified.  
**Date:** 2026-07-14  
**Status:** PASS (with originally blocked People list now unblocked by backend agent)

---

## 1. What Changed

### New / Rewritten Frontend Stack
All files under `frontend/` were rewritten from the placeholder `package.json` into a production Vite React TypeScript SPA.

Key files:
- `frontend/package.json` — dependencies, scripts
- `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tsconfig.node.json`
- `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/index.css`
- `frontend/nginx.conf` — SPA fallback + `/api/` + `/health/` proxy
- `frontend/src/main.tsx` — root, QueryClient, BrowserRouter
- `frontend/src/App.tsx` — routes + toast provider

### Typed API Client
- `frontend/src/api/types.ts` — TypeScript mirrors of OpenAPI schemas, including mid-sprint `FaceListItem`/`FaceListResponse`.
- `frontend/src/api/client.ts` — `apiFetch`, `apiUpload`, `ApiError`.
- `frontend/src/api/queryKeys.ts` — query key factory.
- `frontend/src/api/health.ts`, `faces.ts`, `processes.ts` — TanStack Query hooks and mutations.

### UI Components
- `frontend/src/components/ui/{Button,Input,Card,Skeleton,Alert,Badge,Modal,FileDropzone}.tsx`
- `frontend/src/components/{Layout,Sidebar,Header,HealthIndicator,ErrorBoundary,Toast}.tsx`
- `frontend/src/hooks/useToast.ts`
- `frontend/src/lib/utils.ts` — `cn`, formatting, status mapping, clamp

### Pages
- `frontend/src/pages/DashboardPage.tsx`
- `frontend/src/pages/EnrollPage.tsx`
- `frontend/src/pages/IdentifyPage.tsx`
- `frontend/src/pages/FaceSearchPage.tsx` — initially blocked, now live with list/search/pagination
- `frontend/src/pages/FaceDetailPage.tsx`
- `frontend/src/pages/ProcessDetailPage.tsx`
- `frontend/src/pages/SettingsPage.tsx`
- `frontend/src/pages/NotFoundPage.tsx`

### Tests
- `frontend/src/api/__tests__/types.test.ts`
- `frontend/src/lib/__tests__/utils.test.ts`
- `frontend/src/components/ui/__tests__/Button.test.tsx`
- `frontend/e2e/smoke.spec.ts`
- `frontend/e2e/fixtures/no-face.jpg`
- `frontend/playwright.config.ts`
- `frontend/vitest.config.ts`
- `frontend/src/setupTests.ts`

### Docs
- `backend/docs/implementation/CURRENT_SPRINT.md` — updated to reflect new list endpoint
- `backend/docs/implementation/IMPLEMENTATION_DETAILS.md` — added frontend section
- This review package

---

## 2. API Mapping

| UI Screen | Endpoint(s) Used |
|---|---|
| Dashboard | `GET /health/live`, `GET /health/ready` |
| Enroll | `POST /api/v1/faces/enroll` |
| Identify | `POST /api/v1/faces/recognize?top_k={k}&threshold={t}` |
| Face Search (People) | `GET /api/v1/faces?search={q}&is_active={bool}&limit={n}&offset={o}` |
| Face Detail | `GET /api/v1/faces/{face_id}`, `GET /api/v1/faces/{face_id}/history`, `DELETE /api/v1/faces/{face_id}` |
| Process Detail | `GET /api/v1/processes/{process_id}` |

**Discovered / verified live:** `http://localhost:8000/openapi.json`

---

## 3. Validation Evidence

### 3.1 Static Checks
```bash
cd frontend
npm run typecheck   # pass
npm run lint        # pass
npm run build       # pass
npm run test        # 9 passed
```

### 3.2 Nginx Config Test
```bash
docker run --rm -v /home/user/MergenVisionDemo/frontend/nginx.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:1.27-alpine nginx -t
# nginx: configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful
```

### 3.3 Playwright Smoke (real backend)
```bash
cd frontend
npx playwright test --project=chromium
```

Result:
```
Running 8 tests using 1 worker
  ✓  1 dashboard shows logo and health indicator
  ✓  2 enrolls a face and shows masked national ID
  ✓  3 identifies a known face
  ✓  4 identifies an unknown face
  ✓  5 returns no-face result without error
  ✓  6 registered faces list loads
  ✓  7 no raw national ID leaks in network responses
  ✓  8 mobile navigation opens
  8 passed (8.1s)
```

Screenshots saved under `frontend/e2e/screenshots/`:
- `dashboard.png`
- `face-list.png`
- `identify-known.png`
- `identify-unknown.png`
- `identify-no-face.png`
- `mobile-nav.png`

---

## 4. Security / Compliance Checklist

| Requirement | Evidence |
|---|---|
| Raw `national_id` hidden | Playwright asserts no raw ID in DOM; API detail only returns `nationalIdMasked` |
| No internal object keys / embeddings exposed | UI never renders `object_key`, vector, or raw errors |
| Turkish UI | All labels and status mapping use Turkish |
| No emoji icons | Only Lucide React icons used |
| Responsive | Mobile nav and desktop layout tested; screenshots captured at 375px and 1280px |
| Accessibility | Focus rings, `aria-label`, semantic roles on交互 controls |
| Reduced motion | `prefers-reduced-motion` disables animations |

---

## 5. Known Blockers / Resolved

| Item | Initial Status | Resolution |
|---|---|---|
| `GET /api/v1/faces` list endpoint | BLOCKED | Backend agent added it mid-sprint; FaceSearchPage implemented |
| People listing / search / pagination | BLOCKED | Now working via live endpoint |

Remaining non-goals (not implemented, by design):
- Fake authentication
- Phase 2 / video / RTSP screens
- Oracle online UI
- Photo thumbnail fetching (no dedicated image URL endpoint yet)

---

## 6. MCP / Tool Accountability

| Tool | Usage |
|---|---|
| `codebase-memory-mcp` | Indexed project status, backend route/schema discovery |
| `context7` | React 19, React Router 7, TanStack Query 5, Vite 6 API verification |
| `postman` | Not required; contract verified via live OpenAPI + curl |
| `playwright` | E2E smoke tests and screenshot evidence |
| `exa` | Not used |
| `deepwiki` | Not used |
| `21st` | FORBIDDEN_NOT_USED |

Skills used: `using-superpowers`, `brainstorming`, `writing-plans`, `executing-plans`, `verification-before-completion`, `ui-ux-pro-max`, `codebase-memory`, `context7-mcp`.

---

## 7. Git Scope

Only `frontend/` and `backend/docs/implementation/` files modified. No backend source, GPU, engine, Docker volume, model, or git operations performed. No commits/pushes made.

---

## 8. Recommended Next Sprint

**SPRINT-002: Photo Viewer + Enrollment Enhancements**
- Add dedicated `GET /photos/{photo_id}/url` or presigned URL endpoint so FaceSearch/FaceDetail can show actual thumbnails.
- Implement bulk-enroll UI using existing `POST /faces/enroll/bulk`.
- Add dark mode toggle with full token parity and screenshot regression.
