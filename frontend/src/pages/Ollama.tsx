import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, ErrorBox, Input, KeyValue, SectionTitle, Spinner, StatusBadge } from "../components/ui";

type Model = { name: string; size_bytes: number | null; parameter_size: string | null };

export default function OllamaPage() {
  const status = useApi<{
    connected: boolean;
    base_url?: string;
    models?: Model[];
    orchestrator_model?: string | null;
    orchestrator_installed?: boolean;
    error?: string;
  }>("/ollama/status");
  const settings = useApi<{ settings: Record<string, unknown> }>("/settings");
  const [base, setBase] = useState("");
  const [testMsg, setTestMsg] = useState<string | null>(null);
  const [testErr, setTestErr] = useState<string | null>(null);
  const [pullModel, setPullModel] = useState("");
  const [pullLogs, setPullLogs] = useState<string[]>([]);
  const [pulling, setPulling] = useState(false);
  const [smoke, setSmoke] = useState<string | null>(null);

  useEffect(() => {
    if (settings.data) setBase(String(settings.data.settings["ollama.base_url"] ?? ""));
  }, [settings.data]);

  async function saveEndpoint() {
    setTestErr(null);
    try {
      await api.patch("/settings", { values: { "ollama.base_url": base } });
      status.refresh();
    } catch (e) {
      setTestErr((e as Error).message);
    }
  }

  async function test() {
    setTestErr(null);
    setTestMsg(null);
    try {
      const res = await api.post<{ ok?: boolean; models_found?: number; sample_response?: string; inference_ok?: boolean; error?: string }>("/ollama/test", {});
      if (res.ok) {
        setTestMsg(`Bağlantı OK · ${res.models_found} model · ${res.inference_ok ? "çıkarım OK: " + (res.sample_response ?? "") : "çıkarım yalnızca model seçiminde test edilir"}`);
      } else {
        setTestErr(res.error ?? "Sına başarısız.");
      }
    } catch (e) {
      setTestErr((e as Error).message);
    }
  }

  async function pull() {
    if (!pullModel.trim()) return;
    setPulling(true);
    setPullLogs([]);
    setTestErr(null);
    try {
      // Real SSE streaming pull; progress comes from Ollama's API.
      const res = await fetch(`/api/ollama/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: pullModel.trim() }),
      });
      if (!res.body) throw new Error("SSE gövde yok (sunucu desteklemiyor olabilir).");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const p of parts) {
          if (!p.startsWith("data: ")) continue;
          try {
            const ev = JSON.parse(p.slice(6));
            if (ev.type === "pull") {
              setPullLogs((prev) => [
                ...prev.slice(-30),
                `${ev.status ?? ""}${ev.total != null && ev.completed != null ? ` ${Math.round((ev.completed / ev.total) * 100)}%` : ""}`,
              ]);
            } else if (ev.type === "error") {
              setTestErr(ev.error);
            } else if (ev.type === "done") {
              status.refresh();
            }
          } catch {
            /* skip */
          }
        }
      }
    } catch (e) {
      setTestErr((e as Error).message);
    } finally {
      setPulling(false);
    }
  }

  async function selectOrchestrator(name: string) {
    try {
      await api.post("/ollama/orchestrator", { model: name });
      status.refresh();
    } catch (e) {
      setTestErr((e as Error).message);
    }
  }

  async function smokeInference(model: string) {
    setSmoke(null);
    setTestErr(null);
    try {
      const res = await api.post<{ response: string }>("/ollama/generate", { prompt: "Kendini tek cümleyle tanıt." });
      setSmoke(`${model}: ${res.response}`);
    } catch (e) {
      setTestErr((e as Error).message);
    }
  }

  const st = status.data;

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Ollama Bağlantısı</h1>
        <p className="text-sm text-ink-faint">{`Yerel agent'ın çalıştığı servis; uç adres değiştirilebilir.`}</p>
      </header>
      <ErrorBox message={testErr} />

      <Card className="space-y-3">
        <div className="flex items-center justify-between">
          <SectionTitle>Durum</SectionTitle>
          {st?.connected ? (
            <StatusBadge status="completed" label="Bağlı" />
          ) : (
            <StatusBadge status="failed" label="Bağlı değil" />
          )}
        </div>
        {status.loading ? <Spinner /> : st?.connected ? (
          <KeyValue k="Uç adres" v={<span className="mono">{st.base_url}</span>} />
        ) : (
          <div className="text-sm text-red-500">{st?.error ?? "Bağlantı yok."}</div>
        )}
        <div className="flex flex-wrap items-end gap-2">
          <Input className="max-w-md" value={base} onChange={(e) => setBase(e.target.value)} placeholder="http://127.0.0.1:11434" />
          <Button onClick={saveEndpoint}>Kaydet</Button>
          <Button variant="primary" onClick={test}>Bağlantıyı Sına</Button>
        </div>
        {testMsg && <div className="rounded-md border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700 dark:text-green-300">{testMsg}</div>}
      </Card>

      <Card className="space-y-3">
        <SectionTitle hint="Agent modeli ≠ eğitilecek temel model — roller ayrıdır">Orkestrasyon Modeli</SectionTitle>
        {st?.connected ? (
          <div className="flex flex-col gap-1">
            {(st.models ?? []).map((m) => (
              <div key={m.name} className="flex items-center justify-between border-b border-black/5 py-1 text-sm last:border-0 dark:border-white/5">
                <span className="mono">{m.name}</span>
                <div className="flex items-center gap-2">
                  {st.orchestrator_model === m.name ? (
                    <StatusBadge status="completed" label="Seçili" />
                  ) : (
                    <>
                      <Button onClick={() => smokeInference(m.name)}>Çıkarım Dene</Button>
                      <Button variant="primary" onClick={() => selectOrchestrator(m.name)}>Seç</Button>
                    </>
                  )}
                </div>
              </div>
            ))}
            {smoke && <div className="rounded-md border border-black/10 px-3 py-2 text-sm dark:border-white/10">{smoke}</div>}
            {st.orchestrator_installed === false && (
              <div className="text-sm text-amber-600">Seçili orkestratör kurulu değil; listeden bir model seçin.</div>
            )}
          </div>
        ) : (
          <div className="text-sm text-ink-faint">Bağlantı kurulana kadar model listesi alınamaz.</div>
        )}
      </Card>

      <Card className="space-y-3">
        <SectionTitle hint="İndirme ilerlemesi Ollama API'sinden gerçekten akar">Model İndir (pull)</SectionTitle>
        <div className="flex gap-2">
          <Input className="max-w-md" value={pullModel} onChange={(e) => setPullModel(e.target.value)} placeholder="örn. qwen2.5:0.5b" />
          <Button onClick={pull} disabled={pulling}>İndir</Button>
        </div>
        {pullLogs.length > 0 && (
          <pre className="mono max-h-40 overflow-auto rounded bg-black/5 p-2 dark:bg-black/40">{pullLogs.join("\n")}</pre>
        )}
      </Card>
    </div>
  );
}
