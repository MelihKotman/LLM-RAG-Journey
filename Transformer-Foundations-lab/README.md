# Transformer-Foundations-lab

Bu lab, bir LLM'in "kaputun altında" gerçekte ne yaptığını — metnin token'lara bölünmesinden, bu token'ların anlamlı vektörlere dönüşmesine, attention mekanizmasıyla bağlam kurmasına ve sonunda bir sonraki kelimeyi üretmesine kadar — sıfırdan, adım adım kurarak öğrenmeyi hedefliyor. Amaç kütüphaneleri "kullanmayı öğrenmek" değil; her mekanizmanın *neden* böyle tasarlandığını küçük, elle takip edilebilir örneklerle görüp sonra kodla doğrulamak.

## Notebook Formatı

Her notebook aynı kalıbı izliyor:

1. **Başlık ve özet** — notebook'un kapsamı, önkoşulları, kullanılan veriler
2. **Teori** — kavramın neden gerektiği, küçük bir örnek üzerinde elle türetme, ardından kodla doğrulama
3. **Uygulama** — gerçek metin/veri üzerinde (mümkün olduğunca Türkçe örnekler dahil, çünkü nihai hedef Türkçe'ye duyarlı bir tez konusu)
4. **Egzersizler** — çözümü verilmeyen, okuyucunun kendisinin tamamlaması beklenen sorular
5. **Kaynaklar** — o notebook'a özel, doğrulanmış kitap/makale/kurs linkleri

## Notebook'lar

### 01_tokenization — tamamlandı
Byte Pair Encoding'i sıfırdan (Counter tabanlı, harici kütüphane olmadan) kurduk: küçük bir örnek üzerinde elle türetme, ardından gerçek bir Türkçe ve bir İngilizce corpus üzerinde eğitim. Türkçe'nin eklemeli (agglutinative) yapısının BPE'yi İngilizce'ye göre nasıl daha çok "yorduğunu" ölçülebilir şekilde gösterdik (merge sayısına karşı kelime-başına-token grafiği) ve ünlü uzun Türkçe kelime örneğiyle (`Avrupalılaştıramadıklarımızdanmışsınızcasına`) karakter/kelime/BPE yaklaşımlarını karşılaştırdık.

### 02_embeddings — tamamlandı
Skip-gram + negatif örneklemeyi sıfırdan (numpy) kurduk: gradyanları elle türetip sayısal gradyan kontrolüyle doğruladık, gerçek İngilizce (Tiny Shakespeare) ve Türkçe (Wikipedia cümleleri) verisi üzerinde eğittik. `king - man + woman` analojisini test ettik (corpus küçük olduğu için kısmen çalıştı, bunu tartıştık) ve statik embedding'lerin çok anlamlılık (polysemy) sınırını `kara`/`top`/`ocak` gibi gerçek örneklerle gösterdik. 5 egzersiz de çözüldü.

### 03_self_attention — tamamlandı
Query/Key/Value'yu sıfırdan (numpy) kurduk: küçük bir cümle üzerinde ("kara kutu düştü") elle hesaplayıp genel bir `scaled_dot_product_attention` fonksiyonuyla doğruladık; `√d_k` ölçeklemesinin keyfi olmadığını (nokta çarpım varyansının boyutla orantılı büyüdüğünü, ölçeksiz softmax'ın köreldiğini) sayısal deneyle gösterdik. 02'de eğittiğimiz gerçek Türkçe embedding uzayını (aynı korpus, aynı kod, aynı tohum) yeniden kullanarak, statik `"kara"` vektörünün iki farklı gerçek cümlede (rastgele/eğitilmemiş ağırlıklarla) attention sonrası farklı çıktılar ürettiğini gösterdik — 02'nin bitirdiği polysemy sorusuna somut bir cevap. 5 egzersiz de çözüldü: rastgele tohum değişince ağırlıkların keyfiliği, causal mask, gerçek veriyle permütasyon testi, ölçekli/ölçeksiz `d_k` karşılaştırması, ve iki bağımsız kafanın `"kara"` için farklı ilişkileri öne çıkarıp `concat` ile birleşmesi (multi-head önizlemesi). Çözüm sürecinde birkaç gerçek Jupyter reprodüksiyon hatası (paylaşılan `rng`/global değişkenlerin sırasız çalıştırmada bayatlaması) canlı olarak bulunup teşhis edildi.

### 04_multi_head_attention — tamamlandı
İlk sürüm fazla matematik ağırlıklı olduğu için 2026-09-27'de sadeleştirildi (eski sürüm: `multi_head_attention_v1_eski.ipynb`). Her bölüm artık sezgi → kod → "Ne gördük?" sırasını izliyor. İçerik: "dikkat bütçesi" deneyi (tek kafa iki kelimeden en fazla 0.5'er pay alabiliyor, iki kafa ~1'er); 3 adımlık tarif (böl → her parçada ayrı attention → concat + `W_O`) ve akış şeması; parametre sayısının sabit kalması; iki kafanın elle hesabı; döngülü fonksiyon (vektörize sürüm isteğe bağlı); `W_O` deneyi (onsuz kafa 1 sadece kendi 10 sayısını etkiliyor, onunla 50'sini); gerçek Türkçe embedding'lerde 5 kafa. 6 egzersizin tamamı çözüldü: kafa sayısı artınca ortalama keskinlik değişmiyor ama kafalar arası fark büyüyor (300 tohum ortalaması), bütçe kuralı (h kafa → toplam pay ≤ h, paylaşılan kafada kelime başına ≤ 1/k), causal mask'ın broadcasting ile tüm kafalara uygulanması, kopya kafaların çeşitlilik getirmemesi, rastgele ağırlıklarda 'en önemli kafa'nın tohuma göre değişmesi, multi-head'in hâlâ sıraya kör olması.

### 05_positional_encoding — tamamlandı
04'ten öğrenilen dersle sezgi → kod → "Ne gördük?" düzeninde, hafif matematikle yazıldı. İçerik: attention'ın sıraya körlüğü (`"büyük ev"` = `"ev büyük"`); iki naif fikrin neden bozulduğu (sıra numarasını eklemek kelimeyi boğuyor, 0-1'e sıkıştırmak cümle uzunluğuna bağımlı); sinüzoidal PE'nin "farklı hızlarda dönen 25 saat ibresi" olarak anlatımı (`pe_clocks.png`, `pe_heatmap.png`) ve akış şeması (`positional_pipeline.png`); d=4 elle hesap; 4 özelliğin deneyle kontrolü (sınırlı değerler / her konum uzunluğu 5, 2000 konumda çakışma yok, benzerliğin sadece mesafeye bağlı olması `pe_similarity.png`, "k adım ileri = sabit döndürme matrisi"); gerçek Türkçe embedding'lerle permütasyon testi (PE ile artık `False`); RoPE'a kısa bakış ve tez konusu #10 (uzun bağlam) bağlantısı. 6 egzersiz çözüldü (2026-09-29'da hatalar düzeltildi, eski sürüm `positional_encoding_v1_eski.ipynb`): base küçükse komşular iyi ayrılıyor ama uzak mesafe bilgisi karışıyor, büyükse uzak konumlar birbirine benziyor (`base_profile.png`); sadece-sinüste hiçbir lineer matris 'k adım ileri' yapamıyor; 3×PE kelime kimliğini %1-3'e düşürüyor (ölçek önemli); d=512'de mesafe profili daha pürüzsüz; causal mask PE olmadan da sıra bilgisi sızdırıyor; RoPE'da skor tablosu cümle kaydırılınca değişmiyor, sinüzoidal toplamada değişiyor.

### 06_full_architecture — tamamlandı (egzersizler dahil)
İlk kez **hazır, eğitilmiş modellerle** çalışıyor (GPT-2 small, BERT-base, kısaca T5-small; `torch` + `transformers` gerekir, ilk çalıştırmada modelleri indirir). Amaç: 01-05'te sıfırdan kurduğumuz parçaların gerçek bir modelde aynı işi yaptığını sayıyla göstermek. İçerik: tam blok şeması (`transformer_pipeline.png`: LayerNorm + attention + FFN + residual, ×12); parametrelerin dağılımı (124,4M; FFN bloğun üçte ikisi; `lm_head` = `wte`'nin tersi; GPT-2 konum embedding'i öğrenilmiş); GPT-2'nin attention'ının elle yeniden kurulup ağırlıklarının birebir tutması; causal mask deneyi (son kelimeyi değiştirince GPT-2'de önceki konumlar sabit, BERT'te hepsi değişiyor; `causal_vs_bidirectional.png`); FFN'in kelime başına çalıştığı (karıştır-geri sırala testi); residual/LayerNorm, vektör boyunun bloklarla büyümesi, ilk token'ın devasa vektörü, blok atlama deneyi; kendi ~30 satırlık `forward_manual`'ımızın GPT-2 logit'lerini ~6e-5 farkla yeniden üretmesi; pre-LN (GPT-2) vs post-LN (BERT); üç aile (`architecture_families.png`), BERT boşluk doldurma vs GPT-2 devam, T5 cross-attention. 7 egzersiz (çözümsüz): model büyüklüğü/parametre sayımı, attention vs FFN katkısı, blok ablasyonu, Türkçe token maliyeti, kafa avı (144 kafa), sağ bağlam, T5 cross-attention (isteğe bağlı).

### 07_pretraining_finetuning — planlandı
Pretraining hedefleri (causal language modeling vs masked language modeling), fine-tuning'in ne işe yaradığı, parametre-verimli fine-tuning'e (LoRA) giriş — bu, RAG sistemlerinde retrieval edilen bağlamın modele nasıl "öğretildiğini" anlamak için önkoşul.

### 08_decoding_strategies — planlandı
Bir LLM bir sonraki token'ı nasıl "seçer": greedy decoding, beam search, temperature/top-k/top-p sampling — RAG çıktılarının neden bazen tutarsız/yaratıcı olabildiğini bu bölüm açıklayacak.

## Sırada Ne Var

01–06 tamamen bitti (egzersizler dahil). Sıradaki: `07_pretraining_finetuning`.
