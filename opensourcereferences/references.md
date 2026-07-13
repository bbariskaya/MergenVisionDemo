# MergenVisionDemo Open-Source Reference Map

Bu dosya requirement veya implementation planı değildir.

Amacı, agent bir teknik problemi çözerken önce doğru yerel donor kodu, resmî dokümantasyonu ve upstream source implementasyonunu incelemesini sağlamaktır.

Bir repository’nin bu listede bulunması:

- dependency olarak eklenmesini,
- kodunun aynen kopyalanmasını,
- bütün mimarisinin benimsenmesini,
- production için seçildiğini

ifade etmez.

## 1. Kaynak kullanma sırası

Teknik kararlar şu sırayla alınır:

1. Kullanıcının güncel açık kararı.
2. `requirements/phase1requirements.md`.
3. `requirements/ProjectRequirements.md` içindeki güncel kararla çelişmeyen maddeler.
4. Aktif sprint hedefi ve acceptance kriterleri.
5. Eski MergenVision repository’lerindeki çalışan source.
6. Framework/vendor resmî dokümantasyonu.
7. Bu dosyadaki upstream repository source’u.
8. Issue, discussion ve blog içerikleri; yalnız yardımcı kanıt olarak.

Agent yalnız “bu listede var” diye bir framework veya dependency ekleyemez.

Bir kaynak kullanılmadan önce:

- Çözülen problem açıkça belirlenir.
- İlgili gerçek source path ve symbol bulunur.
- Mümkünse release/tag/commit kaydedilir.
- Kullanılan runtime ve dependency sürümüyle uyumluluk kontrol edilir.
- Kaynak yaklaşımının mevcut requirement ile uyumu doğrulanır.
- Gereksiz framework veya runtime dependency eklenmez.
- Kopyalanan veya adapte edilen source varsa kaynak path kaydedilir.
- Uygulanan çözüm gerçek test/runtime ile doğrulanır.

## 2. Kaynak sınıfları

- `PRIMARY`: İlk bakılacak resmî kaynak.
- `CANDIDATE`: Kullanılabilir implementation adayı; doğrulama gerekir.
- `REFERENCE_ONLY`: Davranış karşılaştırması içindir.
- `OPTIMIZATION_ONLY`: Çalışan demo profillendikten sonra değerlendirilir.
- `FUTURE_PHASE_2`: Phase 1 sırasında kullanılmaz.
- `OUT_OF_SCOPE`: Mevcut ürün için kullanılmaz.

---

# 3. Face detection, alignment ve recognition

## 3.1 InsightFace

**Classification:** `PRIMARY_REFERENCE_ONLY`

- Repository: https://github.com/deepinsight/insightface
- Python package: https://github.com/deepinsight/insightface/tree/master/python-package
- Detection: https://github.com/deepinsight/insightface/tree/master/detection
- Recognition: https://github.com/deepinsight/insightface/tree/master/recognition

İncelenecek konular:

- SCRFD ve RetinaFace preprocessing.
- Detector output tensor decode.
- Bounding-box mapping.
- Five-point landmark mapping.
- NMS.
- Landmark order.
- ArcFace canonical alignment template.
- Similarity transform.
- ArcFace preprocessing.
- 512-D embedding.
- L2 normalization.
- Cosine similarity.
- Batch inference.
- Accuracy/evaluation yöntemleri.

Kurallar:

- `FaceAnalysis` production mimarisini belirlemek için körlemesine kullanılmaz.
- InsightFace source, preprocessing/alignment/correctness referansı olabilir.
- Demo sırasında çalışan ve doğruluğu doğrulanan mevcut model artifact’ları kullanılabilir.
- Model artifact adı, kaynağı, checksum’ı, input/output shape’leri ve preprocessing contract’ı kaydedilir.
- Model provenance incelemesi demo implementationını bloke etmez.
- Model otomatik indirilmeden önce kullanıcıdan izin alınır.

## 3.2 ONNX Runtime

**Classification:** `PRIMARY_INITIAL_RUNTIME_CANDIDATE`

- Repository: https://github.com/microsoft/onnxruntime
- Documentation: https://onnxruntime.ai/docs/
- CUDA Execution Provider:
  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- TensorRT Execution Provider:
  https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html

İncelenecek konular:

- CUDAExecutionProvider.
- TensorRTExecutionProvider.
- Provider ordering.
- I/O binding.
- Dynamic batch.
- Session reuse.
- Model warmup.
- Pinned memory.
- CPU/GPU transferleri.
- Provider fallback.

Kurallar:

- Provider listesinde CUDA görülmesi GPU hot-path kanıtı değildir.
- Sessiz CPU fallback demo acceptance sayılmaz.
- Session request başına oluşturulmaz.
- Demo12 pipeline’ı taşınırken preprocessing ve output mapping doğrulanır.

## 3.3 ONNX

**Classification:** `REFERENCE_ONLY`

- Repository: https://github.com/onnx/onnx
- Documentation: https://onnx.ai/onnx/

İncelenecek konular:

- Opset.
- Dynamic axes.
- Input/output names.
- Shape inference.
- Model checker.
- Export/runtime uyumu.

## 3.4 OpenCV

**Classification:** `REFERENCE_ONLY`

- Repository: https://github.com/opencv/opencv
- Documentation: https://docs.opencv.org/

Kullanım alanları:

- Reference image decode.
- Color conversion.
- Bounding-box görselleştirme.
- Alignment correctness oracle.
- Test image işlemleri.

Kurallar:

- Reference/correctness yolunda kullanılabilir.
- Optimize GPU hot-path otomatik olarak OpenCV CPU decode’a düşürülmez.
- `cv2.VideoCapture` Phase 1 fotoğraf demosu için kullanılmaz.

---

# 4. NVIDIA GPU inference ve optimizasyon

## 4.1 TensorRT

**Classification:** `PRIMARY_OPTIMIZATION_CANDIDATE`

- Repository: https://github.com/NVIDIA/TensorRT
- Samples: https://github.com/NVIDIA/TensorRT/tree/main/samples
- Python samples:
  https://github.com/NVIDIA/TensorRT/tree/main/samples/python
- Polygraphy:
  https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy
- Documentation: https://docs.nvidia.com/deeplearning/tensorrt/

İncelenecek konular:

- ONNX parser.
- Explicit batch.
- Dynamic optimization profiles.
- Execution contexts.
- Tensor address binding.
- CUDA stream ownership.
- Asynchronous inference.
- Engine serialization.
- FP16.
- INT8 calibration.
- `trtexec`.
- Polygraphy ONNX/TensorRT output parity.
- Engine/runtime/version compatibility.
- Worker başına context/stream.

Kurallar:

- İlk çalışan demo oluşmadan TensorRT refactor başlatılmaz.
- Engine build success inference correctness kanıtı değildir.
- Request başına engine veya execution context oluşturulmaz.
- Output sırf post-process için gereksiz biçimde CPU’ya taşınmaz.
- INT8 yalnız accuracy ve benchmark kanıtı varsa kullanılır.
- C++ geçişi profiling sonucuna göre yapılır.

## 4.2 CV-CUDA

**Classification:** `OPTIMIZATION_ONLY`

- Repository: https://github.com/CVCUDA/CV-CUDA
- Documentation: https://cvcuda.github.io/CV-CUDA/

İncelenecek konular:

- GPU resize.
- Normalize.
- Color conversion.
- Warp affine.
- Variable-shape batch.
- Stream interoperability.
- Python/C++ API farkları.

Kurallar:

- İlk demo için zorunlu değildir.
- Preprocess/alignment darboğazı ölçülürse değerlendirilir.

## 4.3 nvImageCodec

**Classification:** `OPTIMIZATION_ONLY`

- Repository: https://github.com/NVIDIA/nvImageCodec
- Documentation: https://docs.nvidia.com/cuda/nvimagecodec/

İncelenecek konular:

- GPU JPEG decode.
- Batch decode.
- CUDA stream kullanımı.
- Device-output buffers.
- Format desteği.
- CPU fallback.

Kurallar:

- İlk demo için zorunlu değildir.
- CPU image decode darboğazı ölçülürse değerlendirilir.

## 4.4 DeepStream Libraries

**Classification:** `FUTURE_PHASE_2`

- Repository:
  https://github.com/NVIDIA-AI-IOT/deepstream_libraries

Yalnız gelecekte:

- video,
- RTSP,
- GStreamer,
- PyNvVideoCodec,
- stream processing

için incelenir.

Phase 1 implementationına eklenmez.

---

# 5. Vector search

## 5.1 Qdrant

**Classification:** `PRIMARY`

- Repository: https://github.com/qdrant/qdrant
- Documentation: https://qdrant.tech/documentation/
- Concepts: https://qdrant.tech/documentation/concepts/

İncelenecek konular:

- Cosine distance.
- 512-D vector collection.
- HNSW.
- Payload filtering.
- Payload indexes.
- Batch upsert.
- Query/search.
- Point lifecycle.
- Snapshot/backup.
- Quantization.
- Replication/sharding.

Kurallar:

- Qdrant derived ve rebuildable face index’idir.
- Point ID, PostgreSQL face-sample ID ile deterministik ilişkilendirilir.
- Qdrant payload’a raw national ID, ad, soyad veya geniş PII yazılmaz.
- HNSW/quantization parametreleri benchmark olmadan değiştirilmez.
- Replication/sharding demo ihtiyacı oluşmadan eklenmez.

## 5.2 Qdrant Python client

**Classification:** `PRIMARY`

- Repository: https://github.com/qdrant/qdrant-client

İncelenecek konular:

- `AsyncQdrantClient`.
- Client lifecycle.
- `close`.
- Collection creation.
- Payload indexes.
- Batch upsert.
- Query API.
- Filters.
- Timeouts.
- Server/client version compatibility.

Kurallar:

- Client request başına oluşturulmaz.
- Application lifespan boyunca paylaşılır.
- Test server ve client sürümleri uyumlu tutulur.

## 5.3 Qdrant vector benchmark

**Classification:** `REFERENCE_ONLY`

- Repository:
  https://github.com/qdrant/vector-db-benchmark

Kullanım alanları:

- Recall/latency trade-off.
- Warmup.
- Search latency percentiles.
- Index build süresi.
- Dataset hazırlama fikirleri.

Bu repository MergenVision benchmark’ının yerine geçmez.

## 5.4 FAISS

**Classification:** `REFERENCE_ONLY`

- Repository: https://github.com/facebookresearch/faiss

Kullanım alanları:

- Küçük dataset için exact search oracle.
- Qdrant approximate-search recall karşılaştırması.
- Threshold/calibration deneyleri.

Production vector store olarak otomatik seçilmez.

---

# 6. Object storage

## 6.1 MinIO Python SDK

**Classification:** `PRIMARY`

- Repository: https://github.com/minio/minio-py
- Documentation:
  https://min.io/docs/minio/linux/developers/python/API.html

İncelenecek konular:

- Bucket initialization.
- `put_object`.
- `get_object`.
- `stat_object`.
- Idempotent upload.
- Content type.
- Metadata.
- Connection reuse.
- Response close/release.
- Error mapping.
- Sync SDK çağrılarının async uygulamadaki etkisi.

Kurallar:

- Client request başına oluşturulmaz.
- Object key PII içermez.
- Metadata allowlist kullanılır.
- UI internal object key görmez.
- Fotoğraf backend content endpointinden sunulur.
- Duplicate upload davranışı açık olmalıdır.

## 6.2 MinIO server

**Classification:** `PRIMARY_RUNTIME`

- Repository: https://github.com/minio/minio
- Documentation: https://min.io/docs/minio/container/index.html

İncelenecek konular:

- Container deployment.
- Persistent volume.
- Healthcheck.
- Bucket bootstrap.
- Credentials.
- Image version pinning.
- Restart persistence.

---

# 7. PostgreSQL, ORM ve migration

## 7.1 PostgreSQL

**Classification:** `PRIMARY`

- Repository: https://github.com/postgres/postgres
- Documentation: https://www.postgresql.org/docs/

İncelenecek konular:

- UUID.
- JSONB.
- Indexes.
- Unique constraints.
- Partial indexes.
- Transactions.
- Row locking.
- Batch insert.
- Pagination.
- Query plans.
- Connection limits.

Kurallar:

- PostgreSQL relational/business source of truth’tür.
- Embedding veya image binary PostgreSQL’e yazılmaz.
- İndeksler gerçek query pattern’lerine göre oluşturulur.

## 7.2 SQLAlchemy

**Classification:** `PRIMARY`

- Repository: https://github.com/sqlalchemy/sqlalchemy
- Documentation: https://docs.sqlalchemy.org/en/20/
- AsyncIO:
  https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

İncelenecek konular:

- SQLAlchemy 2.x mapping.
- `AsyncEngine`.
- `async_sessionmaker`.
- Transaction boundaries.
- Eager loading.
- Bulk operations.
- Connection pooling.
- Application lifespan.

Kurallar:

- Engine/sessionmaker request başına oluşturulmaz.
- Aynı `AsyncSession` eşzamanlı task’larda paylaşılmaz.
- Uzun ML inference sırasında gereksiz DB transaction açık tutulmaz.
- Online API ve bulk runner aynı schema contract’ını kullanır.

## 7.3 Alembic

**Classification:** `PRIMARY`

- Repository: https://github.com/sqlalchemy/alembic
- Documentation: https://alembic.sqlalchemy.org/

İncelenecek konular:

- Initial schema.
- Upgrade/downgrade.
- Async environment.
- Autogenerate.
- Constraint/index migration.
- Migration review.

Kurallar:

- Autogenerate sonucu körlemesine kabul edilmez.
- Migration source gözden geçirilir.
- Uygulama başlangıcında `create_all` migration yerine kullanılmaz.

## 7.4 Psycopg

**Classification:** `PRIMARY`

- Repository: https://github.com/psycopg/psycopg
- Documentation: https://www.psycopg.org/psycopg3/docs/

İncelenecek konular:

- Async driver.
- Pooling.
- COPY.
- Batch insert.
- PostgreSQL error handling.
- Bulk persistence.

---

# 8. Oracle import boundary

## 8.1 python-oracledb

**Classification:** `FUTURE_INTEGRATION_PRIMARY`

- Repository: https://github.com/oracle/python-oracledb
- Documentation:
  https://python-oracledb.readthedocs.io/

İncelenecek konular:

- Thin mode.
- Connection pooling.
- Fetch batching.
- LOB/image retrieval.
- Datatype mapping.
- Retry/resume.
- Read-only import query.
- Secret/config handling.

Kurallar:

- Oracle online recognition hot-path’e bağlanmaz.
- Oracle bulk import source adapter’ıdır.
- Gerçek schema, credentials ve network erişimi olmadan entegrasyon tamamlandı denmez.
- Demo için folder/CSV importer kullanılabilir.
- Oracle adapter ve demo importer aynı bulk input contract’ını uygular.

---

# 9. FastAPI ve backend

## 9.1 FastAPI

**Classification:** `PRIMARY`

- Repository: https://github.com/fastapi/fastapi
- Documentation: https://fastapi.tiangolo.com/
- Lifespan: https://fastapi.tiangolo.com/advanced/events/
- Upload: https://fastapi.tiangolo.com/tutorial/request-files/
- Response models:
  https://fastapi.tiangolo.com/tutorial/response-model/

İncelenecek konular:

- Typed request/response.
- `UploadFile`.
- Lifespan.
- Dependency injection.
- Exception handlers.
- OpenAPI.
- Streaming content.
- Upload validation.
- Health/readiness.

Kurallar:

- Router’da SQL, MinIO, Qdrant veya inference bulunmaz.
- Router HTTP parsing, service çağrısı ve typed response dönüşümü yapar.
- Application service FastAPI `UploadFile` veya HTTP exception’a bağımlı olmaz.
- Public response için `dict[str, Any]` kullanılmaz.
- Bulk enrollment online request içinde çalıştırılmaz.

## 9.2 Pydantic

**Classification:** `PRIMARY`

- Repository: https://github.com/pydantic/pydantic
- Settings: https://github.com/pydantic/pydantic-settings
- Documentation: https://docs.pydantic.dev/

İncelenecek konular:

- Request/response validation.
- Settings.
- Secret types.
- Environment variables.
- Aliases.
- UUID/datetime serialization.
- Strict validation.

## 9.3 Cryptography

**Classification:** `PRIMARY_SECURITY`

- Repository: https://github.com/pyca/cryptography
- Documentation: https://cryptography.io/

İncelenecek konular:

- AES-GCM.
- Secure random nonce.
- Key validation.
- HMAC-SHA256.
- Constant-time comparison.
- Versioned ciphertext.

National-ID korunacaksa custom/home-grown cryptography yazılmaz.

---

# 10. Nginx, Docker ve multi-GPU

## 10.1 Nginx

**Classification:** `PRIMARY_POST_DEMO_OPTIMIZATION`

- Repository: https://github.com/nginx/nginx
- Documentation: https://nginx.org/en/docs/
- Load balancing:
  https://nginx.org/en/docs/http/load_balancing.html
- Proxy module:
  https://nginx.org/en/docs/http/ngx_http_proxy_module.html

İncelenecek konular:

- Upstream GPU workers.
- Worker health/failure.
- Request body size.
- Upload timeout.
- Connection reuse.
- Retry safety.
- Static React serving.
- API reverse proxy.

Kurallar:

- İlk demo için tek backend/GPU worker yeterlidir.
- Çoklu GPU’da GPU başına bir model-owning worker/container değerlendirilir.
- Nginx online istekleri dağıtır.
- Bulk runner milyonlarca HTTP request göndermez.

## 10.2 Docker Compose

**Classification:** `PRIMARY`

- Repository: https://github.com/docker/compose
- Documentation: https://docs.docker.com/compose/
- GPU support:
  https://docs.docker.com/compose/how-tos/gpu-support/

İncelenecek konular:

- Healthchecks.
- Service readiness.
- Persistent volumes.
- Environment/secrets.
- NVIDIA GPU reservation.
- Image tag pinning.
- Backend/frontend dependency flow.

Kurallar:

- Container started durumu readiness kanıtı değildir.
- PostgreSQL, MinIO ve Qdrant healthcheck kullanır.
- Kalıcı veriler restart sonrası korunur.
- Hardcoded GPU UUID kullanılmaz.

---

# 11. React ve polished demo UI

## 11.1 React

**Classification:** `PRIMARY`

- Repository: https://github.com/facebook/react
- Documentation: https://react.dev/

## 11.2 Vite

**Classification:** `PRIMARY`

- Repository: https://github.com/vitejs/vite
- Documentation: https://vite.dev/

## 11.3 React Router

**Classification:** `PRIMARY`

- Repository: https://github.com/remix-run/react-router
- Documentation: https://reactrouter.com/

## 11.4 TanStack Query

**Classification:** `PRIMARY`

- Repository: https://github.com/TanStack/query
- Documentation: https://tanstack.com/query/

İncelenecek konular:

- Server-state cache.
- Loading/error state.
- Mutation invalidation.
- Request cancellation.
- Bulk status polling.
- Retry control.

## 11.5 shadcn/ui

**Classification:** `PRIMARY_UI_CANDIDATE`

- Repository: https://github.com/shadcn-ui/ui
- Documentation: https://ui.shadcn.com/

Demo bileşenleri:

- cards
- data tables
- dialogs
- tabs
- sliders
- progress
- skeleton
- toast
- forms

## 11.6 Radix UI

**Classification:** `PRIMARY_UI_PRIMITIVES`

- Repository: https://github.com/radix-ui/primitives
- Documentation: https://www.radix-ui.com/primitives

Accessibility ve interaction davranışları için referanstır.

## 11.7 Tailwind CSS

**Classification:** `PRIMARY_UI_STYLING`

- Repository: https://github.com/tailwindlabs/tailwindcss
- Documentation: https://tailwindcss.com/docs

## 11.8 Recharts

**Classification:** `CANDIDATE`

- Repository: https://github.com/recharts/recharts
- Documentation: https://recharts.org/

Dashboard ve benchmark grafikleri için kullanılabilir.

## 11.9 Lucide

**Classification:** `CANDIDATE`

- Repository: https://github.com/lucide-icons/lucide
- Documentation: https://lucide.dev/

UI ikonları için kullanılabilir.

---

# 12. Test ve benchmark

## 12.1 Pytest

**Classification:** `PRIMARY`

- Repository: https://github.com/pytest-dev/pytest
- Documentation: https://docs.pytest.org/

Test önceliği:

1. Gerçek JPEG ile enrollment.
2. Aynı kişinin farklı fotoğrafıyla known identify.
3. Farklı kişinin unknown sonucu.
4. No-face.
5. Multi-face.
6. Multiple photos per person.
7. Top-K photo/person resolution.
8. Gerçek PostgreSQL/MinIO/Qdrant.
9. Docker restart persistence.
10. Küçük bulk enrollment.
11. Demo sonrasında 1M benchmark.

Fake inference gerçek demo acceptance değildir.

## 12.2 Playwright

**Classification:** `PRIMARY_UI_ACCEPTANCE`

- Repository: https://github.com/microsoft/playwright
- Python: https://github.com/microsoft/playwright-python
- Documentation: https://playwright.dev/

Polished UI hazır olduğunda:

- page load
- create person
- upload photo
- identify
- top-K render
- loading/error/empty state
- fatal console errors

doğrulanır.

## 12.3 k6

**Classification:** `PRIMARY_ONLINE_LOAD_TEST`

- Repository: https://github.com/grafana/k6
- Documentation: https://grafana.com/docs/k6/

Online API için ölçülür:

- latency percentiles
- concurrency
- errors
- Nginx load distribution
- worker saturation

Bulk enrollment throughput’u k6 ile ölçülmez.

## 12.4 NVIDIA profiling

**Classification:** `OPTIMIZATION_ONLY`

- TensorRT `trtexec`:
  https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec
- Nsight Systems:
  https://docs.nvidia.com/nsight-systems/
- Nsight Compute:
  https://docs.nvidia.com/nsight-compute/

Demo çalıştıktan sonra ölçülür:

- GPU utilization.
- Kernel timeline.
- H2D/D2H copy.
- CUDA synchronization.
- Batch occupancy.
- Detector latency.
- Embedding latency.
- Decode/persistence bottleneck.

Profiling olmadan C++/CUDA yeniden yazımı başlatılmaz.

---

# 13. Genel vision kaynakları

## 13.1 Segment Anything

**Classification:** `OUT_OF_SCOPE`

- Repository: https://github.com/facebookresearch/segment-anything

Segmentation içindir. Phase 1 face-recognition pipeline’ına eklenmez.

## 13.2 Ultralytics

**Classification:** `OUT_OF_SCOPE`

- Repository: https://github.com/ultralytics/ultralytics

General object detection, segmentation ve tracking framework’üdür.

Kurallar:

- Face detector yerine otomatik seçilmez.
- YOLO-face fork’ları kanıt olmadan kullanılmaz.
- Phase 1 dependency’si olarak eklenmez.

## 13.3 Roboflow Supervision

**Classification:** `REFERENCE_ONLY`

- Repository: https://github.com/roboflow/supervision

Bounding-box annotation ve visualization pattern’leri için incelenebilir.

Production face inference veya storage mimarisinin kaynağı değildir.

## 13.4 PaddlePaddle

**Classification:** `REFERENCE_ONLY`

- Repository: https://github.com/PaddlePaddle/Paddle
- PaddleDetection:
  https://github.com/PaddlePaddle/PaddleDetection
- PaddleClas:
  https://github.com/PaddlePaddle/PaddleClas

Model/preprocessing davranışını karşılaştırmak için incelenebilir.

Production runtime dependency’si olarak otomatik eklenmez.

---

# 14. Yerel donor repository’ler

## 14.1 Workspace/mergenvision

**Classification:** `PRIMARY_LOCAL_DONOR`

Path:

`/home/user/Workspace/mergenvision`

İncelenecek:

- FastAPI lifespan.
- Shared engine/client lifecycle.
- TensorRT SCRFD/ArcFace pipeline.
- React/Vite UI.
- Docker Compose.
- Person/photo/identify flow.
- Typed API responses.
- Bulk enrollment.
- Image-content endpointleri.

Taşınmayacak:

- Video/Phase 2 kodu.
- Hardcoded path/secret/GPU.
- Router direct SQL.
- Gereksiz scope.
- Kanıtsız performance claim.

## 14.2 MergenVision

**Classification:** `PRIMARY_SCALE_DONOR`

Path:

`/home/user/MergenVision`

İncelenecek:

- 1M enrollment.
- Three-GPU worker yaklaşımı.
- Nginx load balancing.
- Batch persistence.
- Benchmark scripts.
- Worker warmup.
- Qdrant batch davranışı.

Taşınmayacak:

- DALI dependency’si.
- Hardcoded GPU identity.
- API contract drift.
- Migration’sız schema yaklaşımı.
- Test/type hataları.
- Kanıtı bulunmayan benchmark iddiaları.

## 14.3 Demo12

**Classification:** `PRIMARY_CORRECTNESS_DONOR`

Path:

`/home/user/Demo/Demo12`

İncelenecek:

- ONNX/InsightFace reference pipeline.
- Reproducibility.
- Alignment.
- Accuracy tests.
- LFW/VGGFace2 harness.
- Docker setup.

Taşınmayacak:

- Hardcoded GPU UUID.
- Plaintext secret.
- Kaynağı, checksum’ı veya input/output contract’ı bilinmeyen model artifact’ı.

## 14.4 Demo/MergenVision

**Classification:** `SECONDARY_LOCAL_DONOR`

Path:

`/home/user/Demo/MergenVision`

Yalnız diğer donor’larda eksik kalan çalışan davranışlar için incelenir.

## 14.5 FaceRecognitionProject

**Classification:** `API_CONTRACT_REFERENCE`

Path:

`/home/user/Workspace/FaceRecognitionProject`

İncelenecek:

- API contract.
- Multi-face response.
- Process/history davranışı.
- Known/anonymous/unknown modeli.

Bu repository’nin API-only kararı MergenVisionDemo için geçerli değildir.

## 14.6 MergenVisionFinalVersion

**Classification:** `SELECTIVE_SECURITY_REFERENCE`

Path:

`/home/user/Workspace/MergenVisionFinalVersion`

Yalnız seçici olarak incelenecek:

- National-ID protection.
- Migration/constraint örnekleri.
- Deterministic storage key.
- Qdrant payload privacy.
- Partial-failure test fikirleri.

Taşınmayacak:

- Büyük application service’ler.
- Giant reconciliation state machine.
- Per-request engine/client üretimi.
- Fake acceptance yaklaşımı.
- Full-source documentation dump.
- Governance ağırlığı.
- Unavailable inference mimarisi.

---

# 15. Kaynak karar şablonu

Bir upstream veya donor yaklaşımı uygulanırken kısa kayıt:

```text
Problem:
Local target path:
Source repository:
Source commit/tag:
Source path/symbol:
Model artifact/provenance:
Observed behavior:
Chosen approach:
Rejected alternative:
Local adaptation:
Validation:
Known limitation:
Bu kayıt implementationı yavaşlatacak rapora dönüştürülmez.

16. Yasak çıkarımlar

Aşağıdaki çıkarımlar yapılamaz:

“GitHub’da var, production-ready’dir.”
“Test geçti, gerçek inference çalışıyordur.”
“CUDA provider görünüyor, bütün hot-path GPU’dadır.”
“TensorRT engine oluştu, sonuç doğrudur.”
“1M çalıştı, 10M garanti edilir.”
“Qdrant kullanıyor, otomatik olarak ölçeklenebilir.”
“Container başladı, bütün servisler hazırdır.”
“UI açıldı, backend E2E çalışıyordur.”
“Eski repoda çalıştı, yeni runtime sürümünde de çalışır.”
“Fake inference known sonucu üretti, face recognition tamamlandı.”
“Benchmark output dosyası var, ölçüm geçerlidir.”
