# MergenVisionDemo Agent Guide

Bu dosya MergenVisionDemo için kalıcı çalışma kurallarını tanımlar.

Öncelik sırası:

1. Kullanıcının güncel açık talimatı.
2. Bu `AGENTS.md`.
3. `requirements/phase1requirements.md`.
4. `requirements/ProjectRequirements.md` içindeki güncel kararla çelişmeyen maddeler.
5. Onaylanmış aktif milestone planı.
6. `opensourcereferences/references.md`.
7. Eski repository source’ları.

## 1. Proje hedefi

MergenVisionDemo, müşteriye gösterilecek fotoğraf tabanlı yüz tanıma demosudur.

Phase 1 sonunda sistem:

- Docker Compose ile ayağa kalkar.
- Polished React + TypeScript UI sunar.
- Kişi oluşturur, günceller, listeler ve siler.
- Bir kişiye birden fazla referans fotoğrafı ekler.
- Gerçek face detection, five-point alignment ve ArcFace embedding çalıştırır.
- Görüntüdeki bütün yüzleri bağımsız işler.
- Qdrant üzerinden sample/photo seviyesinde top-K arama yapar.
- Eşleşen referans fotoğrafları, kişi bilgisini ve similarity score’u UI’da gösterir.
- No-face sonucunu başarılı iş sonucu olarak döner.
- PostgreSQL, MinIO ve Qdrant ile kalıcı çalışır.
- Demo oturduktan sonra bulk enrollment, 1M benchmark, üç GPU worker ve Nginx optimizasyonu kazanır.

Video, RTSP, GStreamer, DeepStream, tracker ve object detection Phase 1 kapsamında değildir.

## 2. Repository sınırları

Tek yazılabilir repository:

`/home/user/Workspace/MergenVisionDemo`

Salt-okunur donor repository’ler:

- `/home/user/Workspace/mergenvision`
- `/home/user/MergenVision`
- `/home/user/Demo/Demo12`
- `/home/user/Demo/Demo12_VGGFace2Lab`
- `/home/user/Demo/MergenVision`
- `/home/user/Workspace/FaceRecognitionProject`
- `/home/user/Workspace/MergenVisionFinalVersion`

Eski repository’lerde hiçbir dosya değiştirme.

Kullanıcı açıkça istemedikçe:

- git add
- commit
- push
- branch/merge
- history rewrite
- model veya dataset indirme
- sistem CUDA/driver değişikliği
- destructive Docker/volume işlemi

yapma.

Mevcut kullanıcı değişikliklerini koru.

## 3. Çalışma biçimi

Her milestone veya multi-file görev başlangıcında:

1. Repository root ve `git status --short` doğrulanır.
2. `AGENTS.md` tamamen okunur.
3. İki requirement dosyası ve aktif milestone planı okunur.
4. İlgili mevcut source ve testler incelenir.
5. `opensourcereferences/references.md` içinden ilgili kaynaklar belirlenir.
6. Yerel donor source gerçek path ve symbol üzerinden incelenir.
7. Exact deliverable ve acceptance komutları netleştirilir.
8. Ardından implementation yapılır.

Aynı foundation tekrar tekrar tasarlanmaz.

Blocker olmayan küçük iyileştirmeler ayrı sprint açmadan aktif milestone içinde ele alınır.

Her milestone çalışan, kullanıcı tarafından görülebilen bir dikey sonuç üretir.

## 4. Onaylanmış ürün kararları

### UI

`ProjectRequirements.md` içindeki “UI olmayacak/API-only” kararı bu proje için geçersizdir.

React + TypeScript UI zorunludur.

### Recognition status

Phase 1 demo sonucu yalnızca:

- `known`
- `unknown`

olur.

Unknown yüz:

- otomatik person oluşturmaz,
- otomatik anonymous identity oluşturmaz,
- PostgreSQL’e enrollment olarak yazılmaz,
- Qdrant’a yeni sample olarak eklenmez.

`anonymous/new_anonymous` lifecycle bu demo kapsamında uygulanmaz.

### Top-K

Top-K sonuçları sample/photo seviyesindedir.

Aynı kişiye ait farklı kayıtlı fotoğraflar ayrı candidate olarak dönebilir. Örneğin ilk üç sonuç aynı Jennifer Aniston kişisinin üç farklı referans fotoğrafı olabilir.

Her candidate en az şunları içerir:

- `rank`
- `sampleId`
- `photoId`
- `personId`
- `firstName`
- `lastName`
- `score`

UI fotoğrafı `photoId` üzerinden güvenli content endpointinden alır.

MinIO object key veya internal URL public response’a konulmaz.

Default top-K `5`, izin verilen aralık `1–20` olur.

Recognition threshold environment/config üzerinden gelir. Donor pipeline’daki doğrulanmış değer başlangıç noktasıdır; ölçüm olmadan rastgele değiştirilmez.

### Multiple faces

Identify görüntüsündeki bütün yüzler bağımsız işlenir.

Her yüz için:

- face index
- bounding box
- score
- known/unknown
- top-K candidates

döner.

No-face:

- HTTP 200 döner,
- `faceCount: 0` içerir,
- hata olarak raporlanmaz.

### National ID

Demo davranışı:

- Request boundary’de kabul edilir.
- Keyed HMAC lookup değeri saklanır.
- Masked display değeri saklanır/döndürülür.
- Raw national ID kalıcı olarak saklanmaz.
- Raw national ID loglanmaz.
- MinIO object key/metadata’ya yazılmaz.
- Qdrant payload’a yazılmaz.
- UI listesi ve detail response’unda yalnız masked değer görünür.

HMAC key environment secret’tan gelir; hardcoded veya boş default kullanılmaz.

### Oracle

Oracle online recognition hot-path dependency’si değildir.

Phase 1’de:

- ortak bulk input contract,
- folder/CSV demo adapter,
- gelecekteki Oracle adapter sınırı

tasarlanabilir.

Gerçek Oracle entegrasyonu ancak müşteri schema, credentials ve network bilgisi sağladığında tamamlanmış sayılır.

### Ölçek iddiası

Ölçüm olmadan:

- 10M-ready
- production-ready
- linear scaling
- full GPU utilization
- 3× speedup

ifadeleri kullanılmaz.

Yalnız çalıştırılmış benchmark sonucu raporlanır.

## 5. KISS mimarisi

Yalnız şu ana bölümler kullanılır:

- `api`
- `services`
- `infrastructure`
- `ml`
- `bulk`

### API

Router:

- HTTP request parse eder.
- Typed Pydantic schema kullanır.
- Service çağırır.
- Typed response döner.

Router içinde:

- SQL
- repository query
- MinIO
- Qdrant
- inference
- business rule

bulunmaz.

Public response için `dict[str, Any]` kullanılmaz.

Bütün public endpointler `/api/v1` altında bulunur. Health endpointleri istisna olabilir.

### Services

Service:

- kişi/fotoğraf/enrollment/identification workflow’unu yürütür,
- transaction sınırını yönetir,
- infrastructure client’larını orchestration için çağırır,
- FastAPI `UploadFile`, `Depends`, `HTTPException` veya API schema’ya bağımlı olmaz.

### Infrastructure

Infrastructure:

- SQLAlchemy/PostgreSQL,
- MinIO,
- Qdrant

uygulamalarını içerir.

Generic repository framework veya gereksiz base class yazılmaz.

Açık ve küçük query/repository fonksiyonları tercih edilir.

### ML

ML modülü:

- image bytes/tensor kabul eder,
- detection,
- alignment,
- embedding,
- L2 normalization

yapar.

HTTP, UI, PostgreSQL, MinIO veya Qdrant bilmez.

### Bulk

Bulk runner online API’den ayrı process/CLI’dır.

Milyonlarca fotoğraf online HTTP endpointine tek tek gönderilmez.

Bulk runner aynı ML ve persistence bileşenlerini batch olarak kullanır.

## 6. Runtime lifecycle

Aşağıdaki resource’lar application lifespan’da bir kez oluşturulur:

- SQLAlchemy engine
- sessionmaker
- MinIO client
- Qdrant client
- TensorRT engine/runtime
- model execution context/worker resources

Request başına engine, client veya model oluşturulmaz.

Startup sırasında:

- config validation,
- PostgreSQL readiness,
- MinIO bucket readiness,
- Qdrant collection contract,
- model load,
- model warmup

doğrulanır.

Readiness yalnız gerçekten hazırsa HTTP 200 döner; aksi durumda HTTP 503 döner.

Shutdown sırasında resource’lar güvenli biçimde kapatılır/dispose edilir.

## 7. İlk inference yaklaşımı

İlk donor:

`/home/user/Workspace/mergenvision`

İncelenecek ana parçalar:

- SCRFD TensorRT detector
- five-point alignment
- ArcFace TensorRT recognizer
- FacePipeline
- model registry/config
- lifespan wiring

Source körlemesine kopyalanmaz. Şunlar doğrulanır:

- model artifact path
- checksum
- TensorRT/CUDA compatibility
- input/output tensor names
- shapes
- dtype
- preprocessing
- landmark order
- alignment template
- embedding dimension
- L2 normalization
- batch-1 smoke
- same-person/different-person davranışı

TensorRT engine hedef GPU/runtime’da açılamazsa hata gizlenmez.

ONNX Runtime’a sessiz fallback yapılmaz.

Gerekirse kullanıcıya durum bildirilir ve açık config kararıyla ONNX donor’una geçilir:

`/home/user/Demo/Demo12`

Fake inference yalnız unit testte kullanılabilir.

## 8. Veri sahipliği

### PostgreSQL

Sahibi olduğu veriler:

- person
- photo metadata
- face sample metadata
- recognition request/result history
- lifecycle/status
- model/version referansı

PostgreSQL’e image binary veya embedding yazılmaz.

### MinIO

Sahibi olduğu veriler:

- original person photos
- gerekiyorsa aligned reference crops
- demo kararıyla query image/crop

Object key yalnız sistem UUID’leri ve teknik segmentler içerir.

### Qdrant

Sahibi olduğu veriler:

- 512-D normalized embedding
- rebuildable search payload

Minimum payload:

- `sampleId`
- `photoId`
- `personId`
- `modelVersion`
- `active`

PII payload’a yazılmaz.

Qdrant candidate final known kararından önce PostgreSQL’de doğrulanır:

- sample active
- photo active
- person active
- sample/photo/person ilişkileri tutarlı
- model/profile uyumlu

## 9. Enrollment consistency

PostgreSQL, MinIO ve Qdrant tek transaction paylaşmaz.

İlk sürümde minimum, açık workflow:

1. Deterministic IDs oluştur.
2. PostgreSQL photo/sample kaydını `staged` oluştur.
3. MinIO’ya idempotent object put yap.
4. Qdrant’a aynı `sampleId` ile idempotent upsert yap.
5. PostgreSQL sample/photo kaydını `active` yap.
6. Hata durumunda kayıt `staged/failed` kalır.
7. Aynı işlem güvenli biçimde retry edilebilir.
8. Küçük bir retry/reconcile komutu yalnız failed/staged kayıtları onarır.

Giant reconciliation framework veya global state machine yazılmaz.

MinIO ve Qdrant tamamlanmadan kayıt UI’da active görünmez.

## 10. API minimum kapsamı

Milestone 1 minimum endpointleri:

- `GET /health/live`
- `GET /health/ready`
- `POST /api/v1/people`
- `GET /api/v1/people`
- `GET /api/v1/people/{personId}`
- `POST /api/v1/people/{personId}/photos`
- `GET /api/v1/people/{personId}/photos`
- `GET /api/v1/photos/{photoId}/content`
- `POST /api/v1/identify`

Milestone 2’de:

- person update/delete
- photo delete
- recognition request list/detail
- dashboard summaries

eklenir.

Endpoint ve tablo yalnız gerçek UI/demo davranışına hizmet ediyorsa eklenir.

## 11. UI minimum kapsamı

Milestone 1:

- App shell/navigation.
- People list.
- Create person.
- Person detail.
- Photo upload/gallery.
- Identify workspace.
- Top-K result cards.
- Health/readiness indication.

Milestone 2:

- Dashboard.
- Person edit/delete.
- Photo delete.
- Multi-face bounding-box overlay.
- Request/history pages.
- Loading, empty ve error states.
- Responsive polish.
- Demo settings.

UI:

- raw national ID göstermez,
- MinIO object key görmez,
- secret görmez,
- backend error detail’ini ham biçimde göstermez.

## 12. Milestone sırası

### Milestone 1 — Working single-GPU vertical demo

Amaç:

`Docker Compose → create person → upload/enroll photo → real inference → PostgreSQL/MinIO/Qdrant → identify → sample-level top-K UI`

Bu akış gerçek image ve gerçek inference ile tamamlanmadan Milestone 2’ye geçilmez.

### Milestone 2 — Polished client demo

- UI polish.
- Multiple photos.
- Multi-face.
- History.
- Privacy cleanup.
- Restart persistence.
- Demo acceptance.

### Milestone 3 — Performance and scale

- Tek-GPU batch bulk runner.
- Ölçülmüş benchmark.
- 1M dataset run.
- Üç GPU worker.
- Nginx online load balancing.
- Profiling-guided TensorRT tuning.
- C++/CUDA kararı.

Milestone 1 tamamlanmadan Milestone 3 kodu yazılmaz.

## 13. Test ve acceptance

Testlerin sayısı başarı ölçütü değildir.

Milestone 1’in zorunlu gerçek kanıtları:

1. Docker Compose health.
2. Gerçek PostgreSQL migration.
3. Gerçek MinIO upload/read.
4. Gerçek Qdrant upsert/search.
5. Gerçek model load/warmup.
6. Gerçek JPEG enrollment.
7. Aynı kişinin başka fotoğrafıyla known identify.
8. Farklı kişinin unknown sonucu.
9. No-face sonucu.
10. UI’dan create/enroll/identify akışı.

Mock test bunların yerine geçmez.

Integration test eksik environment yüzünden sessizce başarıyla skip edilmez.

Çalıştırılmayan doğrulama:

- `NOT_RUN`
- `BLOCKED`
- `SKIPPED`

olarak dürüstçe raporlanır.

Test başarısız olduğunda sırf gate geçsin diye silinmez veya skip edilmez. Önce testin doğru contract’ı ölçtüğü doğrulanır; ardından production root cause düzeltilir.

## 14. Reference kullanımı

İlgili kod yazılmadan önce:

`opensourcereferences/references.md`

okunur.

Öncelikli donor rolleri:

- Primary application/UI/runtime donor:
  `/home/user/Workspace/mergenvision`
- Initial TensorRT inference donor:
  `/home/user/Workspace/mergenvision`
- Correctness/ONNX fallback donor:
  `/home/user/Demo/Demo12`
- Bulk/benchmark donor:
  `/home/user/Demo/Demo12_VGGFace2Lab`
- Multi-GPU/Nginx donor:
  `/home/user/MergenVision`
- Selective security reference:
  `/home/user/Workspace/MergenVisionFinalVersion`

FinalVersion’ın application/reconciliation mimarisi donor değildir.

## 15. MCP ve skill kullanımı

Relevant olduğunda:

- `codebase-memory-mcp`: repository/call-path discovery.
- `context7`: version-sensitive framework API’leri.
- `deepwiki`: upstream repository behavior.
- `exa`: resmî source/provenance gerektiğinde.
- `postman`: gerçek API acceptance aşamasında.
- `playwright`: gerçek UI mevcut olduğunda.
- `21st`: kullanılmaz.
- Ruflo: kullanılmaz.

Bütün MCP’leri göstermelik çağırma.

Multi-file implementation öncesinde kısa plan yapılır.

Production behavior ve bug fix için test-first yaklaşımı kullanılır; fakat test üretmek ürün dikeyinin önüne geçmez.

Completion iddiası öncesi gerçek komutlar çalıştırılır.

## 16. Dokümantasyon

Aktif milestone planı kısa tutulur:

`docs/CURRENT_MILESTONE.md`

İçerik:

- objective
- exact deliverables
- acceptance commands
- non-goals
- blocker

Milestone sonunda kısa implementation log güncellenir:

`docs/IMPLEMENTATION_LOG.md`

İçerik:

- outcome
- çalışan davranış
- değişen dosyalar ve önemli symbol’ler
- çalıştırılan komutlar ve sonuçlar
- bilinen sınırlamalar
- sonraki milestone

Full source code, test source’u, diff, cache veya raw log dokümana kopyalanmaz.

Yeni report package oluşturulmaz.

## 17. Final cevap formatı

Her implementation görevi sonunda:

1. `PASS`, `PARTIAL`, `BLOCKED` veya `NOT_TESTED`.
2. Gerçekten çalışan sonuç.
3. Değişen dosya grupları.
4. Çalıştırılan validation komutları ve sonuçları.
5. Çalışmayan/çalıştırılmayan kontroller.
6. Bilinen sınırlamalar.
7. Tek önerilen sonraki adım.

Kanıt olmadan:

- production-ready
- 10M-ready
- fully optimized
- GPU-only
- accuracy verified

ifadeleri kullanılmaz.

## 18. Yasaklar

Phase 1 sırasında ekleme:

- video
- RTSP
- GStreamer
- DeepStream
- tracker
- object detection
- segmentation
- Kubernetes
- microservice platformu
- generic DDD framework
- dev port/interface hiyerarşisi
- büyük reconciliation state machine
- fake production inference
- request başına engine/client/model
- online request içinde Oracle
- online HTTP üzerinden milyonluk bulk enrollment
- hardcoded secret
- hardcoded GPU UUID
- absolute machine path
- model/dataset/engine artifact’ını Git’e ekleme
- full-source documentation dump
