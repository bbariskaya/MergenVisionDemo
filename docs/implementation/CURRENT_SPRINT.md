# Sprint: VGGFace Bulk Enrollment Performans ve Regresyon Düzeltmesi

## Amaç

MergenVision VGGFace bulk enrollment hattındaki performans, batching, canlı ilerleme ve worker runtime regresyonlarını kaynak seviyesinde düzeltmek. Hiçbir veri/volume silmeden, mevcut dirty worktree’yi koruyarak çalışılmaktadır.

## Temel Kaynak Nedenler

1. `model_pack` varsayılanı `antelopev2` → `extract_batch()` fiilen batch-1 inference yapıyor.
2. `maxPhotos` sınırı, ~197K fotoğrafın tamamı hash’lendikten ve materialize edildikten sonra uygulanıyor.
3. Aynı seçilmiş JPEG 4 defa okunuyor (API preflight, worker manifest, GPU extraction, MinIO upload).
4. GPU extraction ve persistence seri çalışıyor; eski pipeline’daki bounded queue ile örtüşme kaldırılmış.
5. Worker global `ThreadPoolExecutor(128)` kullanıyor; GPU inference Uvicorn event loop üzerinde; status/cancel gecikiyor.
6. RetinaFace geçişinde sıfır-candidate/no-face görüntü `DeviceTensor(ptr=0)` ile batch’i düşürüyor.
7. `MODEL_PACK` detector seçimi, embedding model version ve Qdrant `modelVersion` birbirine karışmış.
8. UI `useLatestBulkJob()` ile polling yapmıyor; ilerleme sadece `totalEnrolled` üzerinden hesaplanıyor.

## Deliverables

1. Runtime yetenek logu ve hata kayıtları toplanacak.
2. RetinaFace batch doğruluğu düzeltilecek (zero-candidate, no-face, recognizer stream).
3. Detector / embedding model version ayrılacak.
4. Manifest/bütçe akışı düzeltilecek; `maxPhotos` hash/materialize öncesine çekilecek.
5. Her fotoğraf bir kez okunacak ve byte buffer inference ile persistence arasında taşınacak.
6. Bounded reader → GPU → persistence pipeline geri gelecek.
7. Worker’a dedicated GPU executor ve bounded I/O executor eklenecek.
8. Database/Qdrant/MinIO idempotent batch upsertleri korunacak.
9. UI job-specific polling + ayrı metriklerle güncellenecek.
10. Gerçek GPU/container runtime kanıtıyla benchmark raporlanacak.

## Acceptance Komutları

```bash
# 1. Runtime doğrulama
docker compose ps
docker compose logs --tail=50 gpu-worker-1
docker compose exec api python -m backend.tools.capability_log

# 2. Retina batch smoke testleri
docker compose exec gpu-worker-1 pytest /app/backend/tests/gpu/test_retinaface_batch.py -v

# 3. Manifest bütçe testi
docker compose exec api pytest /app/backend/tests/test_manifest_budget.py -v

# 4. 5K VGGFace job
curl -s -X POST http://localhost:8000/bulk-jobs/vggface \
  -H 'Content-Type: application/json' \
  -d '{"maxPhotos":5000}' | jq .

# 5. UI polling ve metrik doğrulama (terminalde ya da Playwright)
# BulkEnrollmentPage 2 sn aralıklarla güncellenmeli, scanned/sec + enrolled/sec ayrı gösterilmeli.

# 6. Karşılaştırmalı benchmark
docker compose exec gpu-worker-1 python /app/backend/benchmarks/bulk_e2e_benchmark.py --max-photos 20000
```

## Non-Goals

- Yeni model/engine indirme.
- PostgreSQL/MinIO/Qdrant volume silme.
- Mevcut active kayıtları silme.
- Kubernetes/microservice/tracker/video/OPC dağıtım.
- Yeni DB tablosu (onay olmadan).

## Bilinen Blokörler

- Yok. RetinaFace sıfır-candidate crash Gate 2’de çözülecek.

## Son Güncelleme

2026-07-14
