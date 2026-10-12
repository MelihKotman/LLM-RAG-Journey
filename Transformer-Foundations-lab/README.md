# Transformer-Foundations-lab

Bu lab, bir LLM'in "kaputun altında" gerçekte ne yaptığını — metnin token'lara bölünmesinden, bu token'ların anlamlı vektörlere dönüşmesine, attention mekanizmasıyla bağlam kurmasına ve sonunda bir sonraki kelimeyi üretmesine kadar — sıfırdan, adım adım kurarak öğrenmeyi hedefliyor. Amaç kütüphaneleri "kullanmayı öğrenmek" değil; her mekanizmanın *neden* böyle tasarlandığını küçük, elle takip edilebilir örneklerle görüp sonra kodla doğrulamak. 06'dan itibaren sıfırdan kod en aza iniyor ve hazır, eğitilmiş modeller kullanılıyor (sıfırdan kod yalnızca mekanizmanın çekirdeği için).

## Notebook Formatı

Her notebook aynı kalıbı izliyor:

1. **Başlık ve özet** — notebook'un kapsamı, önkoşulları, kullanılan veriler
2. **Teori** — kavramın neden gerektiği; önce sezgi/benzetme, sonra küçük örnek, ardından kodla doğrulama; her bölüm "Ne gördük?" ile biter ve her ana mekanizmanın akış şeması vardır
3. **Uygulama** — gerçek metin/veri üzerinde (mümkün olduğunca Türkçe örnekler dahil, çünkü nihai hedef Türkçe'ye duyarlı bir tez konusu)
4. **Egzersizler** — çözümü verilmeyen, TODO adımlı sorular
5. **Kaynaklar** — o notebook'a özel, doğrulanmış kitap/makale/kurs linkleri

## Kullanılan Kütüphaneler ve Modeller

### Kütüphaneler

| Kütüphane | Ne için kullanıldı | Notebook |
|---|---|---|
| `numpy` | Sıfırdan kurulan mekanizmalar: skip-gram, attention, PE | 02-05 |
| `matplotlib` | Tüm grafikler ve akış şemaları | 01-08 |
| `collections.Counter`, `re` | BPE'nin sıfırdan kurulumu (harici kütüphane yok) | 01 |
| `torch` | Hazır modellerin çalıştırılması, logit/olasılık hesapları | 06-08 |
| `transformers` (Hugging Face) | `AutoModel*`, tokenizer, `pipeline` (fill-mask), `Trainer`, `generate`, `GenerationConfig`, top-k/top-p/min-p logits warper'ları; e5 embedding'i de buradan (`AutoModel`) | 06-08 |
| `peft` | Gerçek LoRA eğitimi (çekirdek elle yazılıp bununla doğrulandı) | 07 |
| `unicodedata` | Üretilen metinde Latin dışı harf (alfabe sızıntısı) ölçümü | 07-08 |
| `python-dotenv`, `psutil`, `itertools` | Ortam değişkenleri (HF ayarları), bellek takibi, kombinasyonlar (tutarlılık ölçümü) | 06-08 |

Python 3.13; modeller ilk çalıştırmada Hugging Face'ten iner (toplam birkaç GB). Veri: 02-05'te Tiny Shakespeare ve Turkish-Wiki-NER-Dataset (CC-BY-SA-4.0); 06-08 veri dosyası kullanmaz (07-08'deki 10 bilgilik uydurma "Karaçam Teknoloji Enstitüsü" verisi notebook'un içindedir).

### Modeller (özet)

| Model | Aile | Parametre | Yapı | Bağlam | Tokenizer (sözlük) | Notebook |
|---|---|---|---|---|---|---|
| GPT-2 small | decoder-only | 124M | 12 blok, 12 kafa, d=768 | 1024 | byte-level BPE (50.257) | 06, 07, 08 |
| BERT-base (uncased) | encoder-only | ~110M | 12 blok, 12 kafa, d=768 | 512 | WordPiece (30.522) | 06, 07 |
| T5-small | encoder-decoder | ~60M | 6+6 blok, 8 kafa, d=512 | göreli konum | SentencePiece (32.128) | 06, 07 (tokenizer) |
| Qwen2.5-0.5B (base) | decoder-only | ~0,49B | 24 blok, 14 sorgu / 2 KV kafa (GQA), d=896 | 32.768 | byte-level BPE (151.936 embedding satırı) | 07, 08 |
| Qwen2.5-0.5B-Instruct | decoder-only | ~0,49B | base ile aynı mimari | 32.768 | aynı | 07, 08 |
| Qwen2.5-1.5B | decoder-only | ~1,5B | 28 blok, d=1536 | 32.768 | aynı | 07, 08 (okuyucu denemeleri) |
| bert-base-multilingual-cased | encoder-only | ~178M | BERT-base mimarisi | 512 | WordPiece, ~120 bin | 07 |
| dbmdz Türkçe BERT | encoder-only | ~110M | BERT-base mimarisi | 512 | WordPiece, 32 bin | 07 |
| multilingual-e5-small | embedding (encoder) | ~118M | 12 blok, d=384 | 512 | SentencePiece, ~250 bin | 07 |
| DistilGPT-2, SmolLM2 | decoder-only | 82M / (egzersiz) | küçültülmüş / kendi tokenizer'lı | — | — | 07 egzersizleri |

(GPT-2, Qwen2.5-0.5B ve Instruct değerleri modellerin `config.json` dosyalarından doğrulandı; diğerleri model kartlarından yaklaşık değerlerdir.)

### Modellerin özellikleri

- **GPT-2 small:** İngilizce ağırlıklı (OpenAI'ın WebText'i, ~40 GB web metni). Pre-LayerNorm, **öğrenilmiş** konum embedding'i, `lm_head` = `wte`'nin tersi (ağırlık paylaşımı). Türkçe'yi çok parçalar (06'da Türkçe İngilizce'ye göre çok daha fazla token, 07'de bit/karakter ~4 kat). Decoding deneylerinin ana modeli, çünkü saf "sonraki token" modeli.
- **BERT-base:** Çift yönlü (sağ bağlamı görür), masked language modeling ile eğitildi (İngilizce Wikipedia + BookCorpus), post-LayerNorm. Üretim için tasarlanmadı; fill-mask ve temsil çıkarma için.
- **T5-small:** Encoder-decoder, "silineni üret" (span corruption) ile eğitildi (C4), her görevi metinden metne çeviri olarak ele alır; cross-attention buradan görüldü.
- **Qwen2.5-0.5B (base):** Çok dilli (Türkçe dahil ~30 dil), ~18 trilyon token ile eğitildi; modern bileşenler: RoPE, RMSNorm, SwiGLU, GQA (14 sorgu kafası 2 KV kafasını paylaşır), embedding paylaşımı. Base model sadece devam ettirir, talimat izlemez ("Translate this to English?" gibi metin sızdırabilir).
- **Qwen2.5-0.5B-Instruct:** Aynı ağırlıkların talimat/sohbet ince ayarlı hali (fine-tuning **biçim** kazandırır, bilgi değil). **Gizli tuzak:** kendi `generation_config`'i `temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1` ile geliyor; `generate` bunları sessizce uygular (07'deki `sohbet()` de bunu kullandı, 08'de fark edildi).
- **Qwen2.5-1.5B:** Aynı ailenin üç kat büyüğü; RAG okuyucusu olarak 0.5B ile kıyaslandı (yanlış bağlamda "Bilmiyorum" diyebiliyor).
- **bert-base-multilingual-cased / dbmdz Türkçe BERT:** Çok dilli vs Türkçe'ye özel BERT karşılaştırması için; İngilizce BERT Türkçe fill-mask'te 0/12 kaldı.
- **multilingual-e5-small:** Metni 384 boyutlu vektöre çeviren embedding modeli; sorguya `query: `, pasaja `passage: ` ön eki istiyor; 07'de retrieval için kullanıldı.
- **DistilGPT-2:** GPT-2'nin damıtılmış küçük hali (6 blok); küçülmenin Türkçe'de ~4 kat fazla bit/karakter kaybettirdiğini göstermek için egzersizde kullanıldı.

## Notebook'lar

### 01_tokenization
Kullanılan: `collections.Counter`, `re`, `matplotlib` (model yok). Veri: küçük bir Türkçe ve bir İngilizce corpus.

Neler öğrendik:

- Byte Pair Encoding'i sıfırdan kurmak: en sık çifti bul → birleştir → tekrarla; küçük örnekte elle türetip koda döktük.
- Türkçe'nin eklemeli (agglutinative) yapısı BPE'yi İngilizce'ye göre daha çok "yoruyor": merge sayısına karşı kelime-başına-token grafiğiyle ölçtük.
- Karakter, kelime ve BPE yaklaşımlarının dengesi (`Avrupalılaştıramadıklarımızdanmışsınızcasına` örneği).
- Egzersizler çözüldü.

### 02_embeddings
Kullanılan: `numpy`, `matplotlib`. Veri: Tiny Shakespeare (İngilizce), Turkish-Wiki-NER-Dataset cümleleri (Türkçe).

Neler öğrendik:

- Skip-gram + negatif örneklemeyi sıfırdan kurmak; gradyanları elle türetip sayısal gradyan kontrolüyle doğruladık.
- Embedding'lerin geometrik sezgisi: `king - man + woman` analojisi küçük corpus'ta kısmen çalıştı (neden, tartışıldı).
- Statik embedding'in çok anlamlılık (polysemy) sınırı: `kara`, `top`, `ocak` tek vektörle temsil ediliyor.
- 5 egzersiz çözüldü.

### 03_self_attention
Kullanılan: `numpy`, `matplotlib`; 02'nin Türkçe embedding'leri yeniden kullanıldı.

Neler öğrendik:

- Query/Key/Value mantığı ve `scaled_dot_product_attention`; "kara kutu düştü" cümlesinde elle hesap, sonra fonksiyonla doğrulama.
- `√d_k` ölçeklemesi keyfi değil: nokta çarpım varyansı boyutla büyüyor, ölçeksiz softmax körelir (sayısal deney).
- Statik `"kara"` vektörü iki farklı cümlede attention sonrası farklı çıktı üretir; 02'nin polysemy sorusuna somut cevap.
- Egzersiz bulguları: tohum değişince ağırlıkların keyfiliği, causal mask, gerçek veriyle permütasyon testi, ölçekli/ölçeksiz `d_k` karşılaştırması, iki kafanın farklı ilişkiler yakalayıp `concat` ile birleşmesi.
- Süreçten ders: paylaşılan `rng`/global değişkenlerin sırasız çalıştırmada bayatlaması (gerçek Jupyter reprodüksiyon hatası).

### 04_multi_head_attention
Kullanılan: `numpy`, `matplotlib`. İlk sürüm fazla matematik ağırlıklı olduğu için 2026-09-27'de sadeleştirildi (eski sürüm `multi_head_attention_v1_eski.ipynb`).

Neler öğrendik:

- Dikkat bütçesi: tek kafa iki kelimeden en fazla 0.5'er pay alabiliyor, iki kafa ~1'er.
- Tarif: böl → her parçada ayrı attention → concat + `W_O` (akış şemasıyla).
- Parametre sayısı kafa sayısından bağımsız sabit kalıyor.
- `W_O` olmadan kafa 1 sadece kendi 10 sayısını etkiliyor, onunla 50'sini.
- Gerçek Türkçe embedding'lerde 5 kafa farklı şeylere bakıyor.
- Egzersiz bulguları: kafa sayısı artınca ortalama keskinlik değişmiyor ama kafalar arası fark büyüyor; bütçe kuralı (h kafa → toplam pay ≤ h); causal mask broadcasting ile tüm kafalara uygulanır; kopya kafalar çeşitlilik getirmez; "en önemli kafa" tohuma göre değişir; multi-head hâlâ sıraya kör. 6 egzersiz çözüldü.

### 05_positional_encoding
Kullanılan: `numpy`, `matplotlib`. Hafif matematikle, sezgi → kod → "Ne gördük?" düzeninde yazıldı.

Neler öğrendik:

- Attention sıraya kördür: `"büyük ev"` = `"ev büyük"`.
- İki naif fikir neden bozuluyor: sıra numarasını eklemek kelimeyi boğuyor, 0-1'e sıkıştırmak cümle uzunluğuna bağımlı.
- Sinüzoidal PE = "farklı hızlarda dönen 25 saat ibresi" (`pe_clocks.png`, `pe_heatmap.png`, `positional_pipeline.png`).
- 4 özellik deneyle doğrulandı: sınırlı değerler, her konum uzunluğu 5, 2000 konumda çakışma yok, benzerlik sadece mesafeye bağlı, "k adım ileri = sabit döndürme matrisi".
- Gerçek Türkçe embedding'lerle permütasyon testi PE ile artık `False` (sıra görülüyor).
- RoPE'a kısa bakış; tez konusu #10 (uzun bağlam) bağlantısı.
- Egzersiz bulguları (6 egzersiz, 2026-09-29'da hatalar düzeltildi, eski sürüm `positional_encoding_v1_eski.ipynb`): base küçükse komşular iyi ayrılır ama uzak mesafe karışır, büyükse uzak konumlar benzeşir; sadece-sinüste hiçbir lineer matris "k adım ileri" yapamaz; 3×PE kelime kimliğini %1-3'e düşürür; d=512'de mesafe profili pürüzsüzleşir; causal mask PE olmadan da sıra sızdırır; RoPE'da skor tablosu kaymaya karşı sabit, sinüzoidal toplamada değişir.

### 06_full_architecture
İlk kez **hazır modellerle**: GPT-2 small, BERT-base, kısaca T5-small (`torch` + `transformers`).

Neler öğrendik:

- Tam blok anatomisi: LayerNorm + attention + FFN + residual, ×12 (`transformer_pipeline.png`).
- Parametre dağılımı: 124,4M; FFN bloğun üçte ikisi; `lm_head` = `wte`'nin tersi; GPT-2'nin konum embedding'i öğrenilmiş.
- GPT-2'nin attention'ını elle yeniden kurduk, ağırlıklar birebir tuttu.
- Causal mask deneyi: son kelimeyi değiştirince GPT-2'de önceki konumlar sabit, BERT'te hepsi değişiyor.
- FFN kelime başına çalışır (karıştır-geri sırala testi); residual/LayerNorm, ilk token'ın devasa vektörü, blok atlama (blok 0 kritik).
- Kendi ~30 satırlık `forward_manual`'ımız GPT-2 logit'lerini ~6e-5 farkla yeniden üretti.
- Pre-LN (GPT-2) vs post-LN (BERT); üç aile (encoder-only / decoder-only / encoder-decoder), BERT boşluk doldurma vs GPT-2 devam, T5 cross-attention.
- Egzersiz bulguları (7 egzersiz çözüldü): 144 kafanın sadece 2'si ağırlıklı önceki token'a bakıyor, 97'si dikkatinin yarısından fazlasını ilk token'a veriyor (attention sink, tez #10); BERT sağ bağlamı görüyor, GPT-2 bilgi sola taşınınca düzeliyor; T5 cross-attention haritası dikdörtgen (7×11).

### 07_pretraining_finetuning
Hazır modellerle, sıfırdan kod en aza indirilmiş (tek istisna ~10 satırlık LoRA katmanı). `torch` + `transformers` + `peft`; ilk çalıştırmada ~4 GB model iner. Türkçe ağırlıklı, İngilizce karşılaştırmalı.

Neler öğrendik:

- Pretraining'in üç "ödevi": GPT sonrakini tahmin eder, BERT maskeyi bulur, T5 silineni üretir; token-başı sürpriz ve fill-mask çıktılarıyla gördük.
- Farklı tokenizer'lı modelleri adil kıyaslamak için bit/karakter: GPT-2'de Türkçe İngilizce'nin ~4 katı, Qwen'de ~2 katı.
- Base vs Instruct: fine-tuning **biçim** kazandırır, bilgi değil.
- LoRA'nın çekirdeği: elle yazılan katman `peft` ile birebir aynı sonucu verdi (fark 0.0); gerçek eğitim `peft` + `Trainer` ile (%1,75 parametre).
- **RAG'e göz kırpan deney** (uydurma "Karaçam Teknoloji Enstitüsü", 10 bilgi): LoRA görülen sorularda 20/20 ama ters yönlü sorularda 0/6, alakasız sorulara sızıyor, bilgiyi güncellemek yeniden eğitim istiyor; RAG %60-85, ters yönde 3/6, güncelleme saniyeler.
- Egzersiz bulguları (6 egzersiz çözüldü, Claude kontrol edip düzeltti; yedekler `Claude outputs/`): SmolLM2'nin kendi tokenizer'ı var ve Türkçeyi GPT-2 gibi ~0.54 token/karakter ile bölüyor (Qwen 0.34); DistilGPT-2 Türkçe'de ~4 kat fazla bpc kaybediyor; BERT fill-mask isabetleri tek token'lık cevaplardan; LoRA ezberi r=1'de bile tam, ters yön her rank'ta 0; karma (replay) veri sızıntıyı silmiyor, kaynağını değiştiriyor; RAG'de 26 sorunun 25'inde bulucu doğru, hatalar okuyucudan, 1.5B yanlış bağlamda 9/10 "Bilmiyorum" derken 0.5B hiç demiyor; soru çeşitliliği arttıkça paraphrase doğruluğu artıyor.

### 08_decoding_strategies
Faz-1'in son notebook'u. `gpt2`, `Qwen2.5-0.5B` (base + Instruct), `torch` + `transformers`. Hazır modellerle çalışır; çekirdek mekanizmalar (greedy, beam, filtre zinciri) 3-10 satırda elle yazılıp Hugging Face'le doğrulandı.

Neler öğrendik:

- Model token seçmez, **olasılık dağılımı** verir; seçimi decoding kuralı yapar (döngü şeması: prompt → logits → softmax → seçim → ekle).
- **Greedy** kendini pekiştiren tekrar döngüsüne düşer (tekrar oranı 0.30; Türkçe Qwen'de 0.88).
- **Beam search** log-olasılık toplamını maksimize eder, k büyüdükçe tekil iyileşmez; elle yazılan sürüm HF `num_beams` ile eşleşti. En olası metin iyi metin değil: insan metni modelinkinden daha "sürprizli" (Holtzman 2019).
- **Temperature** logit'leri T'ye böler; kuyruğu **kesmez** (T=1.5'te metin çöker).
- **Top-k / top-p / min-p** kuyruğu keser; nucleus boyutu 1 ile ~14 bin token arasında değişir, bu yüzden sabit k yetmez. HF sırası: temperature → top-k → top-p → yeniden normalize → örnekle.
- **Gizli `generation_config`:** `do_sample=True` sessizce `top_k=50` getirir; Qwen Instruct `temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1` ile gelir.
- Strateji karşılaştırması (tekrar, çeşitlilik, bağımsız Qwen "hakem" ile bit/karakter): tatlı nokta `T=0.7 + top-p 0.9`; greedy/beam çeşitliliği ~0.1.
- **Repetition penalty / `no_repeat_ngram_size`** semptomu bastırır, nedeni değil; 1.3'te metin bozulur; beam + `no_repeat_ngram_size=3` en küçük zararı verir.
- Türkçe saf T=1.0'da 10 örneğin 7'sinde Latin dışı harf sızıyor; kuyruk kesmek sızıntıyı siliyor ama modelin Türkçe bilgisini artırmıyor.
- **RAG bağlantısı:** bağlam dağılımı keskinleştirir (entropi 2.00 → 0.90); yüksek sıcaklık RAG okuyucusunu çökertir (T=0.5/1.0/1.5 → %80/47/8); ceza "bilgiyi kopyalamak" gereken RAG'de ters teper; örnekler arası tutarlılık halüsinasyon sinyali olabilir (SelfCheckGPT fikri).
- Karar rehberi tez konuları #2, #7, #1, #15 ile bağlandı.
- 5 TODO'lu egzersiz: dağılımı oku (entropi/nucleus), sıcaklık × top-p ızgarası, min-p'yi elle yaz, beam + tekrar çareleri (EN/TR), tutarlılık bir halüsinasyon dedektörü olabilir mi. Kontrol notları: E3 ve E5 düzeltildi (yeniden çalıştırılmalı), E4/E1 yorumları düzeltildi; "olgusal ≠ düşük entropi" gözlemi E1'e eklendi.

## Sırada Ne Var

01–07 tamamen bitti (egzersizler dahil), 08'in egzersizleri çözüldü ve kontrol edildi (E3/E5 yeniden çalıştırma ve bazı yorum maddeleri kullanıcıda). Faz-1 (Transformer temelleri) bununla tamamlanıyor; sıradaki: `RAG-Systems-lab/01_vector_search_embeddings`.
