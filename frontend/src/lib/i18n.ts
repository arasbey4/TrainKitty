/** UI strings — Turkish by default; structure ready for an English pack. */

const tr = {
  appName: "TrainKit",
  tagline: "Describe it. Train it. Run it.",
  nav: {
    overview: "Genel Bakış",
    projects: "Projeler",
    builder: "Model Oluşturucu",
    datasets: "Veri Kümeleri",
    models: "Modeller",
    jobs: "Eğitim İşleri",
    evaluations: "Değerlendirmeler",
    artifacts: "Artefaktlar",
    ollama: "Ollama Bağlantısı",
    settings: "Ayarlar",
  },
  common: {
    loading: "Yükleniyor...",
    error: "Hata",
    save: "Kaydet",
    cancel: "İptal",
    confirm: "Onayla",
    delete: "Sil",
    create: "Oluştur",
    refresh: "Yenile",
    back: "Geri",
    unknown: "Bilinmiyor",
    unavailable: "Kullanılamıyor",
    estimate: "tahminî",
    measured: "ölçülen",
    approve: "Onayla",
    details: "Ayrıntılar",
    actions: "İşlemler",
    name: "Ad",
    status: "Durum",
    empty: "Kayıt yok",
  },
  jobs: {
    statusQueued: "Sırada",
    statusPreparing: "Hazırlanıyor",
    statusDownloading: "İndiriliyor",
    statusValidating: "Doğrulanıyor",
    statusTraining: "Eğitim",
    statusEvaluating: "Değerlendirme",
    statusPackaging: "Paketleme",
    statusCompleted: "Tamamlandı",
    statusFailed: "Başarısız",
    statusCancelled: "İptal Edildi",
    logs: "Canlı Günlük",
    metrics: "Gerçek Metrikler",
    loss: "Eğitim kaybı",
    evalLoss: "Doğrulama kaybı",
    step: "Adım",
    totalSteps: "Toplam adım",
    cancel: "İşi İptal Et",
    canceling: "İptal ediliyor...",
    empty: "Henüz eğitim işi yok. Bir proje seçip veri kümesi hazırladıktan sonra eğitim başlatın.",
  },
  ollama: {
    notConnected: "Ollama'a bağlanılamadı",
    connected: "Bağlı",
    endpoint: "Uç Adres",
    testConnection: "Bağlantıyı Sına",
    orchestrator: "Orkestrasyon Modeli",
    orchestratorHint:
      "Orkestrasyon modeli, TrainKit'i yöneten yerel agent'tır. Çok genellikle eğittiğiniz temel model farklıdır.",
    pull: "Model İndir",
    pullHint: "Uzun sürebilir; ilerleme canlıdır ve Ollama API'sinden gelir.",
    inferenceSmoke: "Çıkarım Dene",
    privacyNote:
      "Tüm çıkarım bu makinedeki Ollama'ya gider; hiçbir veri dışarı yüklenmez ve telemetri yoktur.",
  },
  datasets: {
    generate: "Ollama ile Üret",
    targetCount: "Hedef örnek sayısı",
    prepare: "Hazırla ve Doğrula",
    importHF: "Hugging Face'ten İçe Aktar",
    preview: "Önizleme",
    generatedHint:
      "Üretilen örnekler şema doğrulamasından geçmiştir; olgusal doğruluk bağımsız olarak henüz kanıtlanmadı.",
    licenseWarning: "Lisans",
  },
  training: {
    plan: "Eğitim Planı",
    preflight: "Ön Kontrol",
    start: "Eğitimi Başlat",
    approvalNeeded: "Bu işlem kaynak kullanır; onayınız gerekir.",
    configOnlyNote: "Yapılandırma modu: hiçbir ağırlık eğitilmez, yalnızca çıkarım yapılandırması doğrulanır.",
    comparisonNote: "Baz modeli ve eğitilmiş model yanları gerçek çıkarımla karşılaştırılır.",
  },
  errors: {
    ollamaDown:
      "Ollama çalışmıyor gibi görünüyor. Terminalde `ollama serve` komutunu çalıştırıp yeniden deneyin.",
    missingDeps:
      "Eğitim bağımlılıkları eksik: pip install 'trainkit[training]'",
  },
} as const;

export type Lang = "tr" | "en";
let currentLang: Lang = "tr";

export function t(key: string): string {
  const parts = key.split(".");
  let node: unknown = tr;
  for (const p of parts) {
    if (node && typeof node === "object" && p in (node as Record<string, unknown>)) {
      node = (node as Record<string, unknown>)[p];
    } else {
      return key;
    }
  }
  return typeof node === "string" ? node : key;
}

export function setLang(lang: Lang): void {
  currentLang = lang;
}

export function getLang(): Lang {
  return currentLang;
}
