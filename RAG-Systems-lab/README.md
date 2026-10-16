# RAG-Systems-lab

Bu lab, Transformer-Foundations-lab'da kurulan temel (bir LLM'in nasıl çalıştığı) üzerine, Retrieval-Augmented Generation (RAG) sistemlerinin nasıl inşa edildiğini ele alacak: bir sorgunun nasıl bir vektöre dönüştüğü, ilgili belgelerin nasıl bulunup getirildiği, bu belgelerin bir LLM'in bağlamına nasıl yerleştirildiği, ve sonucun nasıl değerlendirildiği.

Faz 1 bitti; **01 tamamlandı**, 02-08 planlı. Aşağıdaki listede tamamlanan notebook'lar "neler öğrendik" özetiyle, kalanlar planlanan kapsamla yer alıyor.

## Çerçeve (revize: 2026-10-12)

- **Çalışma senaryosu #18.** SAYZEK başvurusu #18 (açık kaynak metinden LLM ile karar destek brifingi) ile yapıldı; sonuç Kasım ortasında belli olacak. Lab bu sonuca bağlı değil: #18 yalnızca "somut senaryo" olarak kullanılır, her notebook'un sonundaki "#18'e bağlantı" bölümü kısa (2-3 paragraf) tutulur. Başvuru olumsuz çıksa da bitirme tezi zorunlu olduğu için buradaki her şey (RAG hattı, değerlendirme, API/yerel model bilgisi) başka bir LLM/RAG konusunda aynen geçerli kalır.
- **Okuyucu modeli değiştirilebilir.** Faz 1'de deneyler 0.5B-1.5B yerel modellerle yapıldı (CPU'da öğrenmek için doğru seçim), ama tezde büyük olasılıkla API'deki güçlü bir model kullanılacak. Bu yüzden 04'te tek bir yardımcı fonksiyon yazılır (`llm(prompt, kaynak="yerel" | "api")`), sonraki notebook'lar aynı fonksiyonu kullanır ve kritik deneyler **iki okuyucuyla** (yerel Qwen vs API modeli) tekrarlanır. Amaç "hangisi iyi" demek değil, küçük modelde gördüğümüz bulguların (örn. yanlış bağlamda "Bilmiyorum" diyememe) büyük modelde nasıl değiştiğini ölçmek.
- **API kuralları (kısa).** Anahtar `.env` içinde durur, depoya girmez; API yanıtları diske önbelleğe alınır (maliyet + tekrar üretilebilirlik); değerlendirme deneylerinde `temperature=0`; model sağlayıcısı notebook içinde tek satırda değiştirilebilir olur.
- **Önce elle, sonra framework.** Her çekirdek mekanizma önce framework'süz kurulur (ne yaptığını görmek için); ardından aynısı bir framework'le yeniden yazılır ve "framework neyi gizliyor, neyi kazandırıyor?" sorusu kısa bir tabloyla cevaplanır. Ana framework **LangChain** (gerekirse LangGraph); LlamaIndex 08'de kısa bir karşılaştırma olarak bakılır. Framework API'leri hızlı değiştiği için sürümler sabitlenir (`requirements`), kod yazılmadan önce güncel dokümana bakılır ve notebook temiz kernel'de çalıştırılarak doğrulanır.
- **Egzersiz biçimi.** Her egzersizde mutlaka bir TODO listesi bulunur, ama kısa tutulur: en fazla 3-4 madde, her madde tek cümle ve tek bir eylem (ne yapılacağı söylenir, nasıl yapılacağı söylenmez). Amaç yönlendirmek, çözümü adım adım vermek değil.
- **Boyut ilkesi.** Yeni notebook eklenmedi, sayı 8 kaldı. Her notebook Faz 1'dekiyle aynı ölçüde kalır: en fazla ~6 teori bölümü, 5-6 egzersiz. Yeni içerik (prompt tasarımı, framework karşılaştırması, sadakat değerlendirmesi, API vs yerel) yalnızca 04, 05, 06 ve 08'e girer; dengelemek için 07 (GraphRAG) kasıtlı olarak hafif tutulur.
- Diğer SAYZEK konularından kısa örnekler vermeye devam edilir; "SAYZEK örnekleri" satırları taslak eşlemedir.

## Notebook'lar

### 01_vector_search_embeddings ✅
**SAYZEK örnekleri:** #15 (Türkçe doküman asistanı — embedding seçimi arama kalitesini belirler), #11 (insan/makine metni ayrımı — embedding üstüne sınıflayıcı), #19 (prosedür asistanı — kısa komuttan doğru prosedür adımına). **#18 bağlantısı:** aynı olayı anlatan farklı haberleri/raporları eşleştirmek.

Kullanılan: `sentence-transformers`, `transformers`, `numpy`, `matplotlib`. Modeller: `multilingual-e5-small` (ana), `multilingual-e5-base`, `paraphrase-multilingual-MiniLM-L12-v2`, `all-MiniLM-L6-v2` (İngilizce-only kontrol) ve **API modeli `gemini-embedding-001`** (ücretsiz katman; her yanıt `data/gemini_onbellek.json`'a önbelleklenir, oturum başına en fazla 90 yeni çağrı, 429'da durulur; önbellek depoda olduğu için anahtarsız da çalışır). Veri: elle yazılmış 40 Türkçe pasaj (yönetmelik, bakım prosedürü, haber, genel) ve 22 soru (örtüşme / parafraz / zor) + 4 cevapsız soru; dağıtıcı olarak 02'deki Türkçe Wikipedia verisinden 1000 cümle.

Neler öğrendik:

- **Arama = metni vektöre çevir, en yakın vektörleri bul.** Pasajlar önceden bir kez, sorular her seferinde vektörleştirilir; aynı model, aynı uzay (pipeline şeması).
- **Embedding modeli = encoder + ortalama havuzlama + normalize.** 4 adım elle yazıldı, `sentence-transformers` ile birebir aynı çıktı (fark 0).
- **Normalize vektörlerde kosinüs, nokta çarpım ve öklid aynı sıralamayı verir** (22/22 soru). Normalize etmeyince nokta çarpım ilk sonucu bile değiştirebiliyor.
- **Kelime örtüşmesi baseline'ı zayıf değil** (MRR 0.84; e5-small 0.95); embedding'in kazancı parafrazda (0.79 → 0.96).
- **Dil uyumu belirleyici:** İngilizce-only model R@1 0.36, e5-small 0.91, e5-base 0.95, Gemini 1.00. 22 soruda tek soru = 4.5 puan; e5-small ↔ e5-base farkı gürültü seviyesinde.
- **`query:`/`passage:` ön eki bu sette fark yaratmadı** (ön eksiz 0.95, ön ekli 0.91; fark tek soru); ters kullanım zararlı (0.82).
- **Embedding konuyu yakalar, cevabı değil:** "yangın söndürüldü mü?" sorusunda yangının başladığını anlatan haber öne geçti.
- **Skorun mutlak değeri güvenilmez:** e5'te alakasız cümle bile ~0.8 alıyor; "cevap yok" eşiği modele göre değişiyor ve kırılgan (boşluk: e5-small −0.006, e5-base +0.012, Gemini +0.119).
- **Kaba kuvvet arama** 1 milyon x 384'te ~70 ms ve 1.5 GB; vektör veritabanı ihtiyacı beklenenden geç başlıyor, asıl maliyet bellek.
- **#18 bağlantısı:** 10/10 haberin en yakın komşusu aynı olayı anlatan haber (benzerlik 0.91-0.94).
- 5 TODO'lu egzersiz: normalize etmemenin etkisi, kendi sorularını yazmak, "cevap yok" eşiği, kendi modelini eklemek (bge-m3 MRR 1.00 > LaBSE 0.95 > distiluse 0.77), vektör kırpma (Matryoshka). Kontrol notları: E1'de tam sıralama eşitliği çok katı bir ölçüt (ilk sonuca/ilk 3'e bakmak daha anlamlı; ham vektör normu karakter sayısıyla ilişkisiz, r≈0.00) ve ön ek TODO'da istendiği halde kullanılmadı; E2'de "zor" sorular aslında kolay çıktı (ortak kök çok), tek hata parafraz sorusunda (doğru pasaj 3., ilk üç skor arası 0.01) ve yorumdaki "zor sorularda yanlış pasaj öne çıkıyor" çıkarımı kendi sonucuyla uyuşmuyor; E3'te eşik aynı 26 soruda seçildi (iyimser) ve 1.00 doğruluğun marjı e5-base/MiniLM'de çok dar; E5'te `kirp` fonksiyonu matrisi yerinde bozuyordu (çıktıdaki e5 değerleri 0.932/0.955 bayat, doğrusu 0.902/0.920/0.911 — yorumdaki rakamlar doğru; `.copy()` ile düzeltilip yeniden çalıştırılmalı). Ek kontrol: 1040 pasajlı korpusta e5-small kırpma MRR'ı 0.955 → 0.902 (256) → 0.853 (128) → 0.737 (64); 40 pasajdaki "128'de toparlanma" gürültü.

### 02_chunking_strategies
**SAYZEK örnekleri:** #8 (tablolar parçalanır mı), #10 (uzun bağlam vs RAG), #19 (prosedür adımları bölünmemeli). **#18 bağlantısı:** tutanak/rapor gibi uzun belgelerde olay sınırına göre parçalama.

Uzun belgelerin retrieval için nasıl parçalara (chunk) bölüneceği; sabit boyut vs anlamsal (semantic) chunking; chunk boyutunun retrieval kalitesine etkisi. (Değişiklik yok.)

### 03_retrieval_methods
**SAYZEK örnekleri:** #1 (OSINT sorgu üretimi ve çapraz doğrulama), #15, #39 (kapalı ağda çalışan hafif asistan). **#18 bağlantısı:** kişi/yer adlarında sparse (BM25), anlam sorularında dense; hibrit.

Yoğun (dense/embedding tabanlı) ve seyrek (sparse — BM25 gibi) retrieval yöntemleri, hibrit arama, FAISS/Chroma ile küçük ölçekli bir vektör indeksi kurmak. (Değişiklik yok.)

### 04_rag_pipeline_from_scratch  *(kapsam genişledi)*
**SAYZEK örnekleri:** #15, #9 (konuşma belleği), #6 (bellek mimarisi), #39. **#18 bağlantısı:** her cümleyi kaynak belgeye bağlayan çıktı biçimi.

Retrieval + generation adımlarını LangChain/LlamaIndex gibi framework'ler olmadan, sıfırdan birleştirip uçtan uca çalışan minimal bir RAG sistemi kurmak. Dört ek bölüm, hepsi bu notebook'un içinde:

1. **Prompt tasarımı:** rol + kural ("yalnızca verilen kaynaktan yaz, yoksa bilmediğini söyle") + kaynak gösterme biçimi; few-shot'ın etkisi; kötü prompt / iyi prompt aynı bağlamda karşılaştırması.
2. **Yapılandırılmış çıktı:** cevabı JSON (iddia + kaynak numarası) olarak istemek ve ayrıştırmak; model bozuk biçim üretirse ne yapılır.
3. **Okuyucu seçimi:** `llm()` yardımcı fonksiyonu ve aynı pipeline'ı yerel Qwen ile API modeliyle koşturmak (kalite, hız, maliyet — kısa tablo).
4. **Aynı pipeline LangChain ile:** elle kurduğumuz hattın (retriever → prompt → LLM → çıktı ayrıştırma) LangChain karşılığı; sonuçlar elle kurulanla karşılaştırılır, framework'ün gizlediği adımlar (chunk birleştirme, prompt şablonu, ayrıştırıcı) tek tek açılır. Elle ve framework sürümünün satır sayısı/esnekliği kısa bir tabloyla özetlenir.

### 05_rag_evaluation  *(kapsam genişledi)*
**SAYZEK örnekleri:** #12 (Türkçe benchmark tasarımı), #2 (hata paylı değerlendirme protokolü), #1 (güvenilirlik damgası). **#18 bağlantısı:** değerlendirme veri seti ve insan değerlendirmeli başarım raporu.

Bir RAG sisteminin çıktısının nasıl değerlendirileceği (faithfulness, answer relevance, context precision/recall); RAGAS gibi değerlendirme çerçeveleri. Tek ek bölüm: **uzun çıktı (özet/brifing) sadakati** — ROUGE'un neden yetmediği, iddia-kaynak atıf kontrolü (her cümle kaynakta destekleniyor mu), LLM-as-judge ve sınırları (hakem de yanılır; küçük elle etiketli sette hakem-insan uyumu), tek seed/küçük sette küçük farkları yorumlamama (Faz 1 dersi).

### 06_self_rag_crag
**SAYZEK örnekleri:** #7 (doğrudan), #1 (kaynak çapraz doğrulaması), #29 (siber olay değerlendirmesinde kanıt kalitesi). **#18 bağlantısı:** düşük güvenli ifadeleri ve çelişkili/eksik bilgiyi işaretlemek.

Kendini düzeltme mekanizmaları: Self-RAG ve CRAG (Corrective RAG) — getirilen belgenin kalitesi düşükse sistemin bunu nasıl fark edip düzelttiği. (SAYZEK aday konu #7 ile doğrudan ilişkili.) Çekirdek döngü (getir → belgeyi puanla → yetersizse yeniden ara/düzelt) önce elle yazılır; sonra aynı akış **LangGraph** ile düğüm/kenar olarak kısaca kurulur (döngülü akışlar için framework'ün gerçekten işe yaradığı yer). Ek bölüm küçük tutulur.

### 07_graphrag  *(kasıtlı olarak hafif)*
**SAYZEK örnekleri:** #13 (doğrudan), #45 (çok kaynaklı karar destek), #19 (prosedürler arası bağımlılık). **#18 bağlantısı:** kişi-yer-olay grafı ve olayların zaman çizelgesi.

Graf tabanlı bilgi erişimi — ilişkisel ve çok adımlı sorularda klasik RAG'in neden yetersiz kaldığı, GraphRAG'in bunu nasıl çözdüğü. Kapsam: kavram + küçük elle kurulmuş bir graf üzerinde "klasik RAG'in cevaplayamadığı çok adımlı soru" deneyi; tam GraphRAG hattı kurulmaz (tez konusu GraphRAG olmadıkça gereksiz). Kısa notebook, ~3-4 egzersiz.

### 08_turkce_rag_asistani  *(kapsam netleşti)*
Önceki notebook'larda öğrenilenleri birleştirip kaynak gösteren, Türkçe bir soru-cevap ve **kısa brifing üreten** asistan prototipi (ana iskelet framework'le — LangChain — kurulur, kritik adımlar elle yazdığımız parçalarla değiştirilebilir; LlamaIndex'te aynı iş kısaca denenir) (SAYZEK aday konu #15 ile doğrudan ilişkili; #18'in Türkçe prototipine zemin hazırlayan notebook). Son bölüm: aynı prototipin yerel okuyucu ve API okuyucuyla çıktıları 05'teki sadakat ölçütleriyle karşılaştırılır; "tezde hangisini neden seçerdim" sorusuna kanıta dayalı kısa bir cevap yazılır.

## Kaynaklar (genel, notebook'lar ilerledikçe genişletilecek)

- Lewis ve ark. (2020) — orijinal RAG makalesi — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Gao ve ark. — RAG Survey — [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
- "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers" — [arXiv:2506.00054](https://arxiv.org/abs/2506.00054)
- "A Survey on Knowledge-Oriented Retrieval-Augmented Generation" — [arXiv:2503.10677](https://arxiv.org/abs/2503.10677)
- "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG" — [arXiv:2501.09136](https://arxiv.org/abs/2501.09136)
- Yan ve ark. — Corrective Retrieval Augmented Generation (CRAG) — [arXiv:2401.15884](https://arxiv.org/abs/2401.15884) · [kod](https://github.com/HuskyInSalt/CRAG)
- Jay Alammar — [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/)
- DeepLearning.AI — "Building and Evaluating Advanced RAG", "LangChain for LLM Application Development"
