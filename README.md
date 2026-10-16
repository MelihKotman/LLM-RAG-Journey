# LLM-RAG-Journey

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](<https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black>)
![PEFT](<https://img.shields.io/badge/PEFT%20(LoRA)-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black>)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-3B6EA5?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-4285F4?style=for-the-badge&logo=meta&logoColor=white)

![GPT-2](https://img.shields.io/badge/GPT--2-10A37F?style=for-the-badge)
![BERT](https://img.shields.io/badge/BERT-4285F4?style=for-the-badge&logo=google&logoColor=white)
![T5](https://img.shields.io/badge/T5-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Qwen2.5](https://img.shields.io/badge/Qwen2.5-615CED?style=for-the-badge)
![SmolLM2](https://img.shields.io/badge/SmolLM2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![multilingual-e5](https://img.shields.io/badge/multilingual--e5-0A66C2?style=for-the-badge)

Büyük Dil Modellerinin (LLM) ve Retrieval-Augmented Generation'ın (RAG) gerçekte nasıl çalıştığını, yüzeysel "kullanmayı öğrenmek" yerine algoritmik temeliyle kavramak için tutulan, Türkçe ağırlıklı, uygulamalı bir öğrenim deposu. Önce bir LLM'in içi (Faz 1), sonra onun üzerine kurulan RAG sistemleri (Faz 2) numaralı Jupyter notebook'larıyla işleniyor. [DS-Mastery-Journey](../DS-Mastery-Journey) deposuyla aynı çalışma disiplinini izliyor.

## Bu Depo Ne İşe Yarıyor

Depodaki her alt proje ("lab"), bir konu alanını kapsayan numaralı notebook'lardan oluşuyor. Her notebook sezgiyle başlar, küçük bir elle örnekten kodla doğrulamaya geçer, gerçek metin/veri üzerinde (mümkün olduğunca Türkçe de) uygulanır, çözümü okuyucuya bırakılan egzersizlerle devam eder ve doğrulanmış kaynaklarla biter. Notebook'lardaki iddialar kod çalıştırılarak doğrulanmıştır. Amaç ezber değil, "neden böyle" sorusunu deneyle yanıtlamak.

## Genel Yapı

- **`Transformer-Foundations-lab/`** — Faz 1: bir LLM'in "kaputun altında" nasıl çalıştığı
- **`RAG-Systems-lab/`** — Faz 2: RAG sistemlerinin nasıl inşa edildiği
- **`03-tez-degerlendirme/`** — notebook örneklerinde kullanılan senaryo/konu havuzu

## Faz 1: Transformer Temelleri (`Transformer-Foundations-lab`)

Ham metnin token'lara bölünmesinden bir sonraki kelimenin üretilmesine kadar bir dil modelinin tüm hattı sekiz notebook'ta işlendi.

- **01_tokenization** — Byte Pair Encoding sıfırdan kurulup Türkçe ile İngilizce'nin token maliyeti karşılaştırıldı.
- **02_embeddings** — Token'ların vektörlere dönüşümü, word2vec'in geometrisi ve statik embedding'lerin çok anlamlılık sınırı.
- **03_self_attention** — Query/Key/Value ve scaled dot-product attention'ın sıfırdan kurulumu.
- **04_multi_head_attention** — Tek attention kafasının neden yetmediği ve kafaların nasıl birleştiği.
- **05_positional_encoding** — Sıra bilgisinin nasıl enjekte edildiği: sinüzoidal kodlama ve RoPE'a bakış.
- **06_full_architecture** — Hazır GPT-2, BERT ve T5'in içi açılıp tam Transformer bloğu ve üç mimari ailesi incelendi.
- **07_pretraining_finetuning** — Pretraining hedefleri, base vs Instruct, LoRA ve fine-tune ile prompt'a bağlam vermenin karşılaştırması.
- **08_decoding_strategies** — Greedy, beam search, temperature, top-k/top-p/min-p ve tekrar cezalarının üretime etkisi.

Ayrıntılı kapsam ve kaynaklar: [`Transformer-Foundations-lab/README.md`](Transformer-Foundations-lab/README.md)

## Faz 2: RAG Sistemleri (`RAG-Systems-lab`)

Sekiz notebook planlandı: vektör arama, chunking, retrieval yöntemleri, sıfırdan RAG pipeline'ı, RAG değerlendirmesi, Self-RAG/CRAG, GraphRAG ve Türkçe bir RAG asistanı. İlk notebook (`01_vector_search_embeddings`) tamamlandı (egzersizler dahil), sırada `02_chunking_strategies` var. Ayrıntı: [`RAG-Systems-lab/README.md`](RAG-Systems-lab/README.md)

## Kullanılan Kütüphaneler ve Araçlar

**Python** serinin ortak dili. Faz 1'in ilk notebook'larında mekanizmalar **NumPy** ve **Matplotlib** ile sıfırdan kuruluyor; 06'dan itibaren **PyTorch**, **Hugging Face transformers** ve **PEFT** ile hazır modeller kullanılıyor. Faz 2'de 01'den itibaren **sentence-transformers** (embedding modelleri) kullanılıyor; vektör arama için **FAISS**/**Chroma** (03) ve **LangChain**/**LlamaIndex** (04'ten itibaren) eklenecek. API çağrıları (Gemini embedding, ücretsiz katman) standart `urllib` ile yapılıyor; anahtarlar `.env` içinde (`HF_TOKEN`, `GEMINI_API_KEY`), yanıtlar diske önbelleğe alınıyor.

## Kullanılan Modeller

Modeller Hugging Face'ten indiriliyor (ilk çalıştırmada internet gerekir) ve küçük tutuldu, böylece her şey CPU'da çalışabiliyor.

- **GPT-2 (small)** ve **DistilGPT-2** — decoder-only temel model, mimari incelemesi, decoding deneyleri
- **BERT** (`bert-base-uncased`, `bert-base-multilingual-cased`, `dbmdz/bert-base-turkish-cased`) — encoder-only model, boşluk doldurma ve Türkçe karşılaştırma
- **T5 (small)** — encoder-decoder mimarisi ve cross-attention
- **Qwen2.5** (0.5B ve 1.5B; base ve Instruct) — fine-tuning, LoRA ve RAG okuyucusu deneyleri
- **SmolLM2 (360M)** — farklı tokenizer ve küçük model karşılaştırması
- **multilingual-e5-small** — Türkçe dahil çok dilli embedding ve retrieval (Faz 2'nin ana embedding modeli)
- **multilingual-e5-base**, **paraphrase-multilingual-MiniLM-L12-v2**, **all-MiniLM-L6-v2** (İngilizce-only kontrol) — 01'de embedding modeli karşılaştırması; egzersizde **BAAI/bge-m3**, **LaBSE**, **distiluse-base-multilingual-cased-v2**
- **gemini-embedding-001** (API, ücretsiz katman) — 01'de API embedding karşılaştırması; vektörler `data/gemini_onbellek.json` içinde önbellekli

Not: Modelleri indirmeyi gerektiren bazı hücreler opsiyonel olarak işaretlenmiş olabilir; sıfırdan yazılan çekirdek kodlar internetsiz çalışır.

## Notebook Formatı

1. Başlık ve özet — kapsam, önkoşullar, kullanılan veri
2. Teori — sezgi, küçük elle örnek, kodla doğrulama; her bölümün sonunda "Ne gördük?"
3. Uygulama — gerçek veri üzerinde, Türkçe dahil
4. Egzersizler — çözümü verilmeyen, her biri kendi hücresinde
5. Kaynaklar — doğrulanmış kitap/kurs/makale önerileri

## Sırada Ne Var

Faz 1 bitti. Faz 2'de `01_vector_search_embeddings` tamamlandı (egzersizler çözüldü ve kontrol edildi). Sırada: `RAG-Systems-lab/02_chunking_strategies`.
