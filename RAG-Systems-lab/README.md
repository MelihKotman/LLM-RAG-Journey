# RAG-Systems-lab

Bu lab, Transformer-Foundations-lab'da kurulan temel (bir LLM'in nasıl çalıştığı) üzerine, Retrieval-Augmented Generation (RAG) sistemlerinin nasıl inşa edildiğini ele alacak: bir sorgunun nasıl bir vektöre dönüştüğü, ilgili belgelerin nasıl bulunup getirildiği, bu belgelerin bir LLM'in bağlamına nasıl yerleştirildiği, ve sonucun nasıl değerlendirildiği.

Notebook'lar Transformer-Foundations-lab bittikçe buraya ekleniyor. Aşağıdaki liste planlanan kapsamı gösteriyor.

**Çerçeve (2026-10-12):** Tez konusu #18 seçildi. Her notebook #18'i omurga alır (sonunda "#18'e bağlantı" bölümü) ve kavramı anlatırken diğer SAYZEK konularından kısa örnekler verir. Aşağıdaki "SAYZEK örnekleri" satırları taslak eşlemedir; notebook yazılırken netleşir.

## Planlanan Notebook'lar

### 01_vector_search_embeddings
**SAYZEK örnekleri:** #15 (Türkçe doküman asistanı — embedding seçimi arama kalitesini belirler), #11 (insan/makine metni ayrımı — embedding üstüne sınıflayıcı), #19 (prosedür asistanı — kısa komuttan doğru prosedür adımına). **#18 bağlantısı:** aynı olayı anlatan farklı haberleri/raporları eşleştirmek.

Metin embedding'lerinin anlamsal arama için nasıl kullanıldığı; kosinüs benzerliği; sentence-transformer tabanlı embedding modelleri; Türkçe embedding modelleri (arama kalitesine doğrudan etkisi).

### 02_chunking_strategies
**SAYZEK örnekleri:** #8 (tablolar parçalanır mı), #10 (uzun bağlam vs RAG), #19 (prosedür adımları bölünmemeli). **#18 bağlantısı:** tutanak/rapor gibi uzun belgelerde olay sınırına göre parçalama.

Uzun belgelerin retrieval için nasıl parçalara (chunk) bölüneceği; sabit boyut vs anlamsal (semantic) chunking; chunk boyutunun retrieval kalitesine etkisi.

### 03_retrieval_methods
**SAYZEK örnekleri:** #1 (OSINT sorgu üretimi ve çapraz doğrulama), #15, #39 (kapalı ağda çalışan hafif asistan). **#18 bağlantısı:** kişi/yer adlarında sparse (BM25), anlam sorularında dense; hibrit.

Yoğun (dense/embedding tabanlı) ve seyrek (sparse — BM25 gibi) retrieval yöntemleri, hibrit arama, FAISS/Chroma ile küçük ölçekli bir vektör indeksi kurmak.

### 04_rag_pipeline_from_scratch
**SAYZEK örnekleri:** #15, #9 (konuşma belleği), #6 (bellek mimarisi), #39. **#18 bağlantısı:** her cümleyi kaynak belgeye bağlayan çıktı biçimi.

Retrieval + generation adımlarını LangChain/LlamaIndex gibi framework'ler olmadan, sıfırdan birleştirip uçtan uca çalışan minimal bir RAG sistemi kurmak.

### 05_rag_evaluation
**SAYZEK örnekleri:** #12 (Türkçe benchmark tasarımı), #2 (hata paylı değerlendirme protokolü), #1 (güvenilirlik damgası). **#18 bağlantısı:** değerlendirme veri seti ve insan değerlendirmeli başarım raporu.

Bir RAG sisteminin çıktısının nasıl değerlendirileceği (faithfulness, answer relevance, context precision/recall); RAGAS gibi değerlendirme çerçeveleri.

### 06_self_rag_crag
**SAYZEK örnekleri:** #7 (doğrudan), #1 (kaynak çapraz doğrulaması), #29 (siber olay değerlendirmesinde kanıt kalitesi). **#18 bağlantısı:** düşük güvenli ifadeleri ve çelişkili/eksik bilgiyi işaretlemek.

Kendini düzeltme mekanizmaları: Self-RAG ve CRAG (Corrective RAG) — getirilen belgenin kalitesi düşükse sistemin bunu nasıl fark edip düzelttiği. (SAYZEK aday konu #7 ile doğrudan ilişkili.)

### 07_graphrag
**SAYZEK örnekleri:** #13 (doğrudan), #45 (çok kaynaklı karar destek), #19 (prosedürler arası bağımlılık). **#18 bağlantısı:** kişi-yer-olay grafı ve olayların zaman çizelgesi.

Graf tabanlı bilgi erişimi — ilişkisel ve çok adımlı sorularda klasik RAG'in neden yetersiz kaldığı, GraphRAG'in bunu nasıl çözdüğü. (SAYZEK aday konu #13 ile doğrudan ilişkili.)

### 08_turkce_rag_asistani
Önceki notebook'larda öğrenilenleri birleştirip kaynak gösteren, Türkçe bir soru-cevap asistanı prototipi. (SAYZEK aday konu #15 ile doğrudan ilişkili; #18'in Türkçe prototipine zemin hazırlayan notebook.)

## Kaynaklar (genel, notebook'lar ilerledikçe genişletilecek)

- Lewis ve ark. (2020) — orijinal RAG makalesi — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Gao ve ark. — RAG Survey — [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
- "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers" — [arXiv:2506.00054](https://arxiv.org/abs/2506.00054)
- "A Survey on Knowledge-Oriented Retrieval-Augmented Generation" — [arXiv:2503.10677](https://arxiv.org/abs/2503.10677)
- "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" — [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
- Yan ve ark. — Corrective Retrieval Augmented Generation (CRAG) — [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) · [kod](https://github.com/HuskyInSalt/CRAG)
- Jay Alammar — [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/)
- DeepLearning.AI — "Building and Evaluating Advanced RAG", "LangChain for LLM Application Development"
