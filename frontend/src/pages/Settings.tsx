import { useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, ErrorBox, Input, KeyValue, SectionTitle } from "../components/ui";

const EDITABLE: { key: string; label: string; type: "number" | "text" }[] = [
  { key: "ollama.base_url", label: "Ollama uç adresi", type: "text" },
  { key: "ollama.request_timeout_s", label: "İstek zaman aşımı (sn)", type: "number" },
  { key: "ollama.num_ctx", label: "Orkestratör bağlam penceresi (num_ctx)", type: "number" },
  { key: "ollama.temperature", label: "Orkestratör sıcaklık", type: "text" },
];

export default function SettingsPage() {
  const { data, error, refresh } = useApi<{ settings: Record<string, unknown>; allowed_keys: string[] }>("/settings");
  const [values, setValues] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  async function save(keys: string[]) {
    setErr(null);
    setSaved(false);
    const payload: Record<string, unknown> = {};
    for (const k of keys) {
      const raw = values[k];
      const field = EDITABLE.find((f) => f.key === k);
      payload[k] = field?.type === "number" ? Number(raw) : raw;
    }
    try {
      await api.patch("/settings", { values: payload });
      setSaved(true);
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  const s = data?.settings ?? {};

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Ayarlar</h1>
        <p className="text-sm text-ink-faint">Bağlantı ve davranış parametreleri veritabanında saklanır; .env dosyası gerekmez.</p>
      </header>
      <ErrorBox message={err ?? error} />
      {saved && <div className="rounded-md border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700">Kaydedildi.</div>}

      <Card className="space-y-3">
        <SectionTitle>Backend Ayarları</SectionTitle>
        {EDITABLE.map((f) => (
          <label key={f.key} className="block text-xs text-ink-faint">
            {f.label}
            <Input
              type={f.type}
              step="any"
              className="mt-1 max-w-md"
              value={String(values[f.key] ?? s[f.key] ?? "")}
              onChange={f.type === "number" ? (e) => setValues({ ...values, [f.key]: String(Number(e.target.value)) }) : (e) => setValues({ ...values, [f.key]: e.target.value })}
            />
          </label>
        ))}
        <Button variant="primary" onClick={() => save(EDITABLE.map((f) => f.key))}>Kaydet</Button>
      </Card>

      <Card>
        <SectionTitle>Gizlilik</SectionTitle>
        <div className="text-sm">
          <KeyValue k={"Veri/değiştir: üretilen ve içe aktarılan veri setleri"} v="yerel veri dizininizde" />
          <KeyValue k="Ollama çıkarımı" v="yalnızca bu makine" />
          <KeyValue k="Telemetri/analitik" v="yok" />
          <KeyValue k="Hugging Face erişimi" v="yalnızca açık istekle (veri kümesi indirme / model metadata)" />
        </div>
      </Card>

      <Card>
        <SectionTitle>İzinli ayar anahtarları</SectionTitle>
        <div className="mono text-xs">{(data?.allowed_keys ?? []).join(", ")}</div>
      </Card>
    </div>
  );
}
