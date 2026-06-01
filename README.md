# İK Odaklı Açıklanabilir Özgeçmiş-İş Eşleştirme Sistemi

Bu proje, insan kaynakları senaryoları için geliştirilmiş açıklanabilir NLP tabanlı bir özgeçmiş ve iş ilanı eşleştirme sistemidir.

Sistem bir aday özgeçmişini iş ilanı metniyle karşılaştırır ve şu çıktıları üretir:

- Genel uygunluk skoru
- Semantik benzerlik skoru
- Ağırlıklı beceri eşleşme skoru
- Eşleşen beceriler
- Eksik veya belirsiz zorunlu beceriler
- İK odaklı değerlendirme metni
- Mülakat odak noktaları
- Aday geliştirme önerileri
- Önceki encoder sürümleriyle model karşılaştırması
- Özgeçmiş ve iş ilanı için PDF/DOCX/TXT yükleme desteği
- Pozitif, olumsuz ve belirsiz beceri ifadeleri için cümle düzeyinde beceri kanıtı doğrulaması

Bu sistem bir karar destek prototipidir. Otomatik işe alım kararı vermek için tasarlanmamıştır.

## Projeyi Çalıştırma

FastAPI backend ve React Vite frontend için iki ayrı terminal kullanın.

### 1. Python Ortamını Oluşturma ve Aktifleştirme

Proje kök dizininde:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sanal ortam zaten varsa yalnızca aktifleştirin:

```powershell
.\.venv\Scripts\activate
```

### 2. Backend'i Başlatma

Proje kök dizininde:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

Backend şu adreste çalışır:

```text
http://localhost:8000
```

Sağlık kontrolü:

```text
http://localhost:8000/health
```

### 3. Frontend'i Başlatma

İkinci bir terminal açın ve çalıştırın:

```powershell
cd frontend
npm install
npm run dev
```

React arayüzü şu adreste çalışır:

```text
http://localhost:5173
```

Tarayıcıda bu adresi açıp özgeçmiş ve iş ilanı analizini kullanabilirsiniz.

## Ana Çalışma Akışı

```text
Yüklenen dosyalar veya yapıştırılan metin
        |
        v
PDF/DOCX için MarkItDown metin çıkarımı
        |
        v
Özgeçmiş metni + iş ilanı metni
        |
        v
Encoder-only Sentence Transformer
        |
        v
Kosinüs benzerlik skoru
        |
        v
Sözlük tabanlı beceri çıkarımı
        |
        v
Özgeçmişteki beceri ifadeleri için beceri kanıtı sınıflandırıcısı
        |
        v
Ağırlıklı beceri eşleşme skoru
        |
        v
Genel uygunluk skoru
        |
        v
Açıklanabilir İK sonucu
```

## Model Yaklaşımı

Proje semantik eşleştirme için encoder-only Sentence Transformer modeli kullanır. Yerel model, özgeçmiş-iş eşleştirme görevi için fine-tune edilmiştir; bu nedenle aday profilleri ve iş gereksinimleri aynı embedding uzayında karşılaştırılır.

Proje ayrıca beceri kanıtı doğrulaması için ikinci bir encoder-only sınıflandırıcı kullanır. Bu sınıflandırıcı, özgeçmişteki bir cümlenin ilgili beceri için pozitif kanıt sağlayıp sağlamadığını, beceriyi olumsuzlayıp olumsuzlamadığını veya belirsiz şekilde bahsedip bahsetmediğini kontrol eder.

Çalışma zamanında backend `models/` altındaki yerel model klasörlerini kullanır. Karşılaştırma baseline modeli `models/base-minilm/` altında tutulur; bu sayede proje başka bir bilgisayarda çalıştırıldığında `sentence-transformers/all-MiniLM-L6-v2` modelini Hugging Face üzerinden indirmek zorunda kalmaz.

Fine-tuning proje metodolojisinin bir parçasıdır, ancak bu README tamamlanmış sistemi çalıştırmaya odaklanır. Deney sonuçları `RESULTS.md` dosyasında belgelenmiştir.

Final skor semantik benzerlik ve ağırlıklı beceri eşleşmesini birleştirir:

```text
overall_score =
  0.80 * semantic_similarity_score +
  0.20 * weighted_skill_match_score
```

Model, genel sosyal becerilere kıyasla teknik ve role özgü becerilere daha fazla ağırlık verir. Böylece communication veya teamwork gibi genel beceriler zayıf bir adayı olduğundan güçlü göstermez.

## Backend API

FastAPI backend şu endpoint'leri sağlar:

```text
GET  /health
GET  /sample
POST /analyze
POST /compare
POST /extract-text
```

Ana endpoint:

```text
POST /analyze
```

Beklenen JSON gövdesi:

```json
{
  "resume_text": "Aday özgeçmiş metni...",
  "job_description_text": "İş ilanı metni..."
}
```

Cevap; skorları, beceri listelerini, eşleşme kategorisini, İK değerlendirmesini, mülakat odak noktalarını ve aday önerilerini içerir.

`POST /compare`, aynı özgeçmiş-iş ilanı çiftini mevcut model sürümleriyle skorlar ve semantik, beceri ve genel skorları karşılaştırır.

`POST /extract-text`, multipart `file` alanıyla `.pdf`, `.docx`, `.txt` veya `.md` dosyalarından metin çıkarır. React arayüzü hem özgeçmiş hem de iş ilanı yüklemeleri için bu endpoint'i kullanır.

## Proje Yapısı

```text
NLP-Proje/
|-- README.md
|-- IDEA.md
|-- PLAN.md
|-- PROJECT_SUMMARY.md
|-- RESULTS.md
|-- requirements.txt
|-- api.py
|-- app.py
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- skill_evidence/
|   |-- skills.json
|   `-- examples/
|-- frontend/
|   |-- package.json
|   |-- index.html
|   `-- src/
|       |-- App.tsx
|       |-- api.ts
|       |-- styles.css
|       `-- types.ts
|-- models/
|   |-- base-minilm/
|   |-- resume-job-minilm-finetuned/
|   |-- resume-job-minilm-finetuned-2epoch/
|   `-- skill-evidence-minilm-classifier/
|-- scripts/
|-- src/
|   |-- similarity.py
|   |-- skill_extractor.py
|   |-- skill_weights.py
|   |-- matcher.py
|   |-- skill_evidence.py
|   |-- recommender.py
|   `-- evaluation.py
`-- tests/
```

## Önemli Dosyalar

- `api.py`: React arayüzünün kullandığı FastAPI backend
- `frontend/src/App.tsx`: ana React arayüzü
- `src/similarity.py`: encoder-only model yükleme ve kosinüs benzerliği
- `src/skill_extractor.py`: beceri tespiti ve alias eşleştirme
- `src/skill_evidence.py`: pozitif, olumsuz ve belirsiz beceri kanıtı sınıflandırması
- `src/skill_weights.py`: role özgü beceri ağırlıkları
- `src/matcher.py`: final eşleştirme pipeline'ı
- `src/recommender.py`: eşleşme kategorisi, İK açıklaması, mülakat odağı ve aday önerileri
- `data/skill_evidence/skill_evidence_dataset.csv`: elle hazırlanmış beceri kanıtı sınıflandırıcı veri seti
- `data/skills.json`: açıklanabilirlik için kullanılan beceri sözlüğü
- `models/`: sistemin kullandığı yerel Sentence Transformer model dosyaları

## Testleri Çalıştırma

Proje kök dizininde:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Opsiyonel Streamlit Demo

Ana arayüz React Vite uygulamasıdır. İkincil Python-only demo olarak Streamlit sürümü de bulunur:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notlar

- Ana demo için `http://localhost:5173` adresini kullanın.
- Frontend'i kullanırken backend çalışır durumda olmalıdır.
- Backend yerel modeli bir kez yükler ve analiz isteklerinde yeniden kullanır.
- Detaylı deney sonuçları `RESULTS.md` dosyasında belgelenmiştir.
