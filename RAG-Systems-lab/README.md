# RAG-Systems-lab

Bu lab, Transformer-Foundations-lab'da kurulan temel (bir LLM'in nasıl çalıştığı) üzerine, Retrieval-Augmented Generation (RAG) sistemlerinin nasıl inşa edildiğini ele alacak: bir sorgunun nasıl bir vektöre dönüştüğü, ilgili belgelerin nasıl bulunup getirildiği, bu belgelerin bir LLM'in bağlamına nasıl yerleştirildiği, ve sonucun nasıl değerlendirildiği.

Henüz notebook yazılmadı — Transformer-Foundations-lab tamamlandıktan sonra buraya geçilecek. Aşağıdaki liste planlanan kapsamı gösteriyor.

## Planlanan Notebook'lar

### 01_vector_search_embeddings
Metin embedding'lerinin anlamsal arama için nasıl kullanıldığı; kosinüs benzerliği; sentence-transformer tabanlı embedding modelleri; Türkçe embedding modelleri (arama kalitesine doğrudan etkisi).

### 02_chunking_strategies
Uzun belgelerin retrieval için nasıl parçalara (chunk) bölüneceği; sabit boyut vs anlamsal (semantic) chunking; chunk boyutunun retrieval kalitesine etkisi.

### 03_retrieval_methods
Yoğun (dense/embedding tabanlı) ve seyrek (sparse — BM25 gibi) retrieval yöntemleri, hibrit arama, FAISS/Chroma ile küçük ölçekli bir vektör indeksi kurmak.

### 04_rag_pipeline_from_scratch
Retrieval + generation adımlarını LangChain/LlamaIndex gibi framework'ler olmadan, sıfırdan birleştirip uçtan uca çalışan minimal bir RAG sistemi kurmak.

### 05_rag_evaluation
Bir RAG sisteminin çıktısının nasıl değerlendirileceği (faithfulness, answer relevance, context precision/recall); RAGAS gibi değerlendirme çerçeveleri.

### 06_self_rag_crag
Kendini düzeltme mekanizmaları: Self-RAG ve CRAG (Corrective RAG) — getirilen belgenin kalitesi düşükse sistemin bunu nasıl fark edip düzelttiği. (SAYZEK aday konu #7 ile doğrudan ilişkili.)

### 07_graphrag
Graf tabanlı bilgi erişimi — ilişkisel ve çok adımlı sorularda klasik RAG'in neden yetersiz kaldığı, GraphRAG'in bunu nasıl çözdüğü. (SAYZEK aday konu #13 ile doğrudan ilişkili.)

### 08_turkce_rag_asistani
Önceki notebook'larda öğrenilenleri birleştirip kaynak gösteren, Türkçe bir soru-cevap asistanı prototipi. (SAYZEK aday konu #15 ile doğrudan ilişkili — muhtemel tez omurgası adaylarından biri.)

## Kaynaklar (genel, notebook'lar ilerledikçe genişletilecek)

- Lewis ve ark. (2020) — orijinal RAG makalesi — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Gao ve ark. — RAG Survey — [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
- "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers" — [arXiv:2506.00054](https://arxiv.org/abs/2506.00054)
- "A Survey on Knowledge-Oriented Retrieval-Augmented Generation" — [arXiv:2503.10677](https://arxiv.org/abs/2503.10677)
- "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" — [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
- Yan ve ark. — Corrective Retrieval Augmented Generation (CRAG) — [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) · [kod](https://github.com/HuskyInSalt/CRAG)
- Jay Alammar — [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/)
- DeepLearning.AI — "Building and Evaluating Advanced RAG", "LangChain for LLM Application Development"
