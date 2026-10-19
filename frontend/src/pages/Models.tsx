import { useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, ErrorBox, Input, KeyValue, SectionTitle, Spinner, StatusBadge } from "../components/ui";
import { fmtBytes } from "../lib/api";

type OllamaModel = {
  name: string;
  size_bytes: number | null;
  parameter_size: string | null;
  family: string | null;
  quantization_level: string | null;
};

export default function Models() {
  const ollama = useApi<{ connected: boolean; models?: OllamaModel[]; error?: string; base_url?: string }>("/ollama/status");
  const [repoId, setRepoId] = useState("hf-internal-testing/tiny-random-LlamaForCausalLM");
  const [meta, setMeta] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function inspect() {
    setErr(null);
    setBusy(true);
    try {
      const m = await api.post<Record<string, unknown>>("/models/inspect", { repo_id: repoId });
      setMeta(m);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Modeller</h1>
        <p className="text-sm text-ink-faint">
          Solda: gerçek Ollama kurulumu. Sağda: Hugging Face kimliği gerçek gerçekten sorgulanır; uyumluluk yalnızca doğrulanabilir bilgiye dayanır.
        </p>
      </header>

      <Card>
        <SectionTitle hint="Orkestratör model olmayabilir — roller farklidir">Yerel Ollama Modelleri</SectionTitle>
        {ollama.loading ? (
          <Spinner />
        ) : !ollama.data?.connected ? (
          <ErrorBox message={ollama.data?.error ?? ollama.error ?? "Ollama'a ulaşılamadı."} />
        ) : (ollama.data.models ?? []).length === 0 ? (
          <div className="text-sm text-ink-faint">Kurulu model yok. Ollama Bağlantısı sayfasından `ollama pull` yapabilirsiniz.</div>
        ) : (
          <div className="space-y-1">
            {ollama.data.models!.map((m) => (
              <div key={m.name} className="flex items-baseline justify-between border-b border-black/5 py-1 text-sm last:border-0 dark:border-white/5">
                <span className="mono">{m.name}</span>
                <span className="text-xs text-ink-faint">
                  {m.parameter_size ?? "?"} · {fmtBytes(m.size_bytes) ?? "boyut bilinmiyor"} · {m.quantization_level ?? ""}
                </span>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card className="space-y-3">
        <SectionTitle>Hugging Face Model İnceleme</SectionTitle>
        <ErrorBox message={err} />
        <div className="flex gap-2">
          <Input className="max-w-md" value={repoId} onChange={(e) => setRepoId(e.target.value)} />
          <Button variant="primary" onClick={inspect} disabled={busy}>İncele</Button>
        </div>
        {meta && (
          <div className="max-w-xl">
            <KeyValue k="Mimari" v={String(meta.architecture ?? "bilinmiyor")} />
            <KeyValue k="Lisans" v={String(meta.license ?? "bilinmiyor")} />
            <KeyValue k="Disk boyutu" v={meta.total_disk_bytes ? fmtBytes(Number(meta.total_disk_bytes))! : "bilinmiyor"} />
            <KeyValue k="Gated" v={String(meta.gated)} />
            <div className="mt-2">
              Eğitim modları:
              {Object.entries((meta.compatibility as { modes?: Record<string, { available?: boolean; note?: string }> }).modes ?? {}).map(
                ([k, v]) => (
                  <div key={k} className="flex gap-2 text-sm">
                    <StatusBadge status={v.available ? "completed" : "draft"} label={v.available ? "uygun" : "uygun değil"} />
                    <span>{k}</span>
                    {v.note && <span className="text-xs text-ink-faint">— {v.note}</span>}
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
