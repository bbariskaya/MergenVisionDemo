# Modern Interprobe UI/UX Sprint

## Identity
- Sprint name: `MODERN_INTERPROBE_UI_UX_SPRINT`
- Scope: Frontend-only implementation; backend contract unchanged.
- Base context: Backend `/api/v1` contract is live and confirmed.
- Design system: `design-system/interprobe/MASTER.md` (ui-ux-pro-max generated).

## Goal
Build a production-quality, responsive Turkish-language React application shell and domain screens for the Interprobe biometric operations console, bound only to existing backend endpoints.

## Backend Contract (Verified)

Live OpenAPI: `http://localhost:8000/openapi.json`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health/live` | Liveness |
| GET | `/health/ready` | Readiness + component status |
| POST | `/api/v1/faces/recognize?top_k&threshold` | Identify faces in query image |
| POST | `/api/v1/faces/enroll` | Enroll a named face/person |
| POST | `/api/v1/faces/enroll/bulk` | Bulk enroll |
| GET | `/api/v1/faces/{face_id}` | Face/person detail |
| DELETE | `/api/v1/faces/{face_id}` | Soft-delete a face |
| GET | `/api/v1/faces/{face_id}/history` | Recognition history for face |
| GET | `/api/v1/processes/{process_id}` | Recognition process detail |
| GET | `/api/v1/faces?search&is_active&limit&offset` | List enrolled faces |

**Initially blocked:** `GET /api/v1/faces` was absent. Backend agent later added it, so People list feature was implemented.

## Tech Stack
- React 19.2.7 (client-side SPA)
- React Router 7.5.3 (library declarative mode)
- TanStack Query 5.84.1
- Vite 6.0.0
- TypeScript 5.x
- Tailwind CSS 3.4.x
- Lucide React
- Playwright (E2E smoke on real backend)
- Vitest + React Testing Library (unit/component)

## Design Decisions
- Clean light-primary theme, with navy sidebar matching Interprobe logo.
- 8px spacing rhythm, WCAG AA contrast.
- Turkish UI labels; API enum/status mapped to Turkish.
- Lucide icons only; no emoji structural icons.
- `national_id` never rendered raw; only `nationalIdMasked` shown.
- Internal storage object keys and embeddings never rendered.

## Implementation Blocks
1. Scaffold Vite React TS frontend, Tailwind, routing, query client.
2. Typed API client and query key factory.
3. App shell: sidebar (desktop/mobile), header, breadcrumb, health indicator.
4. Dashboard: system health, quick actions, shortcuts.
5. Enroll page: form with preview, upload, no-face/multi-face API validation.
6. Identify page: drag-drop workspace, preview, top-K/threshold controls, result cards, responsive bounding-box overlay.
7. Face detail + history page.
8. Process detail page.
9. Loading/error/empty/no-face/unknown state handling.
10. Playwright smoke and screenshot evidence.
11. Build, typecheck, lint, vitest verification.
12. Review package generation.

## Acceptance Criteria
- [ ] Interprobe logo asset used in shell.
- [ ] All implemented screens bound to real endpoints.
- [ ] No raw national ID or object key leakage in DOM, responses, or console.
- [ ] Loading/error/empty/no-face/unknown/multi-face states implemented.
- [ ] Responsive layout verified at 375px and 1280px.
- [ ] Playwright smoke passes against real backend.
- [ ] `npm run build`, `npm run typecheck`, `npm run lint`, `npm run test` pass.
- [ ] SPRINT code review package created.
- [ ] No backend, GPU, engine, Docker volume, model, or git changes.

## Expected Verdict
- `UI_SPRINT_PASS=true` when all acceptance criteria above are met.
