# LLM-RAG-Journey

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-4285F4?style=for-the-badge&logo=meta&logoColor=white)

Bu depo, [SAYZEK Akademik Tez programı](03-tez-degerlendirme/aday-konular.md) kapsamında bir bitirme tezi konusuna karar vermeden önce, Büyük Dil Modellerinin (LLM) ve Retrieval-Augmented Generation'ın (RAG) gerçekte ne iş yaptığını ve nasıl çalıştığını — yüzeysel "kullanmayı öğrenmek" değil, matematiksel ve algoritmik temeliyle — kavramak için tutulan yapılandırılmış bir öğrenim sürecinin ana merkezidir. [DS-Mastery-Journey](../DS-Mastery-Journey) deposuyla aynı çalışma disiplinini izliyor: her notebook teori + elle türetme + kodla doğrulama + gerçek veri + egzersiz kalıbını takip ediyor.

## Bu Depo Ne İşe Yarıyor

Depodaki her alt proje ("lab"), belirli bir konu alanını kapsayan, numaralandırılmış Jupyter notebook'larından oluşuyor. Her notebook aynı sabit kalıbı izliyor: önce teorik özet (formüllerin sadece yazılması değil, küçük bir örnek üzerinde elle takip edilerek türetilmesi), ardından gerçek metin/veri üzerinde uygulama, ardından çözümü okuyucuya bırakılan egzersizler, ve son olarak ileri okuma için doğrulanmış kaynak listesi. Notebook'lardaki iddialar (bir tokenizer'ın gerçekten daha az token ürettiği, bir grafiğin doğru sonucu verdiği gibi) kod çalıştırılarak doğrulanmış durumda.

Amaç, konuları ezberlemek değil "neden böyle" mantığıyla adım adım anlamak. Bu anlayış oturduktan sonra `03-tez-degerlendirme/aday-konular.md` içindeki SAYZEK tez konularından (özellikle LLM/RAG ağırlıklı olanlardan) hangisinin uygun bir tez omurgası olabileceğine karar vermek.

## Genel Yapı

- **`Transformer-Foundations-lab/`** — Faz 1: bir LLM'in "kaputun altında" nasıl çalıştığı (tokenization → embedding → attention → tam mimari → pretraining → decoding)
- **`RAG-Systems-lab/`** — Faz 2: retrieval-augmented generation sistemlerinin nasıl inşa edildiği (henüz başlanmadı — Faz 1 tamamlandıktan sonra)
- **`03-tez-degerlendirme/`** — SAYZEK aday tez konularının listesi ve bunların LLM/RAG öğrenimiyle nasıl eşleştiğine dair değerlendirme; öğrenim ilerledikçe düzenli aralıklarla gözden geçiriliyor

## Faz 1: Transformer Temelleri (`Transformer-Foundations-lab`)

Bir dil modelinin ham metni token'lara bölmesinden, bu token'ları anlamlı vektörlere dönüştürmesine, attention ile bağlam kurmasına ve sonunda bir sonraki kelimeyi üretmesine kadar sekiz notebook'luk bir seri:

- **01_tokenization** ✅ tamamlandı — Byte Pair Encoding'i sıfırdan kurup Türkçe'nin eklemeli yapısının tokenization'ı İngilizce'ye göre nasıl zorlaştırdığını ölçülebilir şekilde gösterdik.
- **02_embeddings** — token ID'lerinin öğrenilebilir vektörlere dönüşümü, geometrik anlam ilişkileri
- **03_self_attention** — Query/Key/Value, scaled dot-product attention
- **04_multi_head_attention** — neden tek attention "kafası" yetmiyor
- **05_positional_encoding** — sıra bilgisinin nasıl enjekte edildiği, sinusoidal formül, RoPE'a bakış
- **06_full_architecture** — encoder-decoder vs decoder-only, causal mask, tam bir Transformer bloğu
- **07_pretraining_finetuning** — pretraining hedefleri, fine-tuning ve LoRA'ya giriş
- **08_decoding_strategies** — greedy, beam search, temperature/top-k/top-p sampling

Ayrıntılı kapsam ve kaynaklar için: [`Transformer-Foundations-lab/README.md`](Transformer-Foundations-lab/README.md)

## Faz 2: RAG Sistemleri (`RAG-Systems-lab`)

Faz 1 tamamlandıktan sonra başlanacak; planlanan sekiz notebook (vektör arama, chunking, retrieval yöntemleri, sıfırdan RAG pipeline'ı, RAG değerlendirmesi, Self-RAG/CRAG, GraphRAG, Türkçe RAG asistanı) [`RAG-Systems-lab/README.md`](RAG-Systems-lab/README.md) içinde listelendi. SAYZEK aday konularından #7 (Self-RAG/CRAG), #10 (uzun bağlam vs RAG), #13 (GraphRAG) ve #15 (Türkçe RAG asistanı) bu fazın notebook'larıyla doğrudan örtüşüyor.

## Kullanılan Kütüphaneler ve Araçlar

**Python**, tüm serinin ortak dili. Faz 1'de ağırlıklı olarak **NumPy** ve **Matplotlib** ile, mekanizmalar harici bir derin öğrenme kütüphanesine sarılmadan sıfırdan kuruluyor — amaç "neden çalıştığını" görmek. Faz 2'de bunun üzerine **Hugging Face transformers/tokenizers**, **LangChain**/**LlamaIndex**, ve vektör veritabanları olarak **FAISS**/**Chroma** eklenecek. Türkçe deneylerde Hugging Face üzerindeki Türkçe'ye özel sentence-transformer embedding modelleri kullanılacak.

Not: Bazı notebook hücreleri (örn. gerçek bir Hugging Face tokenizer/modelini indirip karşılaştırma) internet erişimi gerektiriyor ve bu nedenle "bonus/opsiyonel egzersiz" olarak işaretleniyor; notebook'un ana akışı her zaman internet gerektirmeden (sıfırdan yazılmış kodla) tam olarak çalışıyor.

## Notebook Formatı

Her notebook, konudan bağımsız olarak aynı yapıyı izliyor:

1. Başlık ve özet — notebook'un kapsamı, önkoşulları, kullanılan veriler
2. Teori — formüllerin/algoritmaların küçük bir örnek üzerinde elle türetilmesi, ardından kodla doğrulanması
3. Uygulama — gerçek metin/veri üzerinde (mümkün olduğunca Türkçe örnekler dahil edilerek)
4. Egzersizler — çözümü verilmeyen, okuyucunun kendisinin çözmesi beklenen sorular
5. Kaynaklar — konuyla ilgili, doğrulanmış kitap/kurs/makale önerileri

## Sırada Ne Var

Transformer-Foundations-lab/02_embeddings ile devam: token embedding'lerinin geometrik anlamı ve neden bağlama duyarlı (contextual) hale getirilmeleri gerektiği.
