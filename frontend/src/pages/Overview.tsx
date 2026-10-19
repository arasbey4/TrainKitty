import { Link } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { Card, EmptyState, KeyValue, Spinner, StatusBadge } from "../components/ui";
import { fmtBytes } from "../lib/api";

type ProjectsResp = { items: { id: string; name: string; status: string }[]; total: number };
type JobsResp = {
  items: { id: string; status: string; mode: string; progress_step: number; total_steps: number; created_at: string }[];
  total: number;
};
type Hardware = {
  os: string;
  cpu: { model: string; logical_cores: number | null };
  ram: { total_bytes: number | null; available_bytes: number | null };
  disk: { free_bytes: number | null };
  gpu: { cuda: { available: boolean; devices: { name: string }[] } };
  training_ready: boolean;
};
type OllamaStatus = {
  connected: boolean;
  base_url?: string;
  models?: { name: string }[];
  orchestrator_model?: string | null;
  error?: string;
};

export default function Overview() {
  const projects = useApi<ProjectsResp>("/projects?limit=5");
  const jobs = useApi<JobsResp>("/training/jobs?limit=5");
  const hw = useApi<Hardware>("/system/hardware");
  const ollama = useApi<OllamaStatus>("/ollama/status");

  const isEmpty = !projects.loading && (projects.data?.total ?? 0) === 0;

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Genel Bakış</h1>
        <p className="text-sm text-ink-faint">Gerçek veriler — istatistikler doğrudan backend'den gelir.</p>
      </header>

      {isEmpty && (
        <EmptyState
          title="TrainKit'e hoş geldiniz"
          hint="Başlamak için: 1) Ollama bağlantısını kurun ve bir model seçin. 2) Bir proje oluşturun ve modelinizi tanımlayın. 3) Veri kümesi üretin veya içe aktarın. 4) Eğitim planını onaylayıp çalıştırın."
        />
      )}

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <div className="text-xs text-ink-faint">Projeler</div>
          <div className="mt-1 text-2xl font-semibold">{projects.data?.total ?? "—"}</div>
          <div className="mt-2 space-y-1">
            {projects.data?.items.slice(0, 3).map((p) => (
              <Link key={p.id} to={`/projects/${p.id}`} className="block truncate text-sm hover:text-accent">
                {p.name} <StatusBadge status={p.status} />
              </Link>
            ))}
          </div>
        </Card>
        <Card>
          <div className="text-xs text-ink-faint">Ollama</div>
          {ollama.loading ? (
            <Spinner />
          ) : ollama.data?.connected ? (
            <>
              <div className="mt-1 flex items-center gap-2 text-sm">
                <span className="h-2 w-2 rounded-full bg-green-500" /> Bağlı
              </div>
              <div className="mt-1 text-xs text-ink-faint">{ollama.data.base_url}</div>
              <div className="mt-1 text-xs">Kurulu model: {ollama.data.models?.length ?? 0}</div>
              <div className="mt-1 truncate text-xs">Orkestratör: {ollama.data.orchestrator_model ?? "-"}</div>
            </>
          ) : (
            <>
              <div className="mt-1 flex items-center gap-2 text-sm text-red-500">
                <span className="h-2 w-2 rounded-full bg-red-500" /> Bağlı değil
              </div>
              <Link to="/ollama" className="mt-2 text-xs text-accent underline">
                Bağlantıyı düzelt →
              </Link>
            </>
          )}
        </Card>
        <Card>
          <div className="text-xs text-ink-faint">Eğitim İşleri</div>
          <div className="mt-1 text-2xl font-semibold">{jobs.data?.items?.length ?? 0}</div>
          <div className="mt-2 space-y-1">
            {jobs.data?.items.slice(0, 3).map((j) => (
              <Link key={j.id} to={`/training/${j.id}`} className="flex items-baseline justify-between text-sm hover:text-accent">
                <span className="mono truncate">{j.id}</span>
                <StatusBadge status={j.status} />
              </Link>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-soft">Gerçek Donanım</h2>
        {hw.loading ? (
          <Spinner />
        ) : hw.error ? (
          <div className="text-sm text-red-500">{hw.error}</div>
        ) : hw.data ? (
          <div className="grid gap-x-8 sm:grid-cols-2">
            <div>
              <KeyValue k="İşletim sistemi" v={hw.data.os} />
              <KeyValue k="CPU" v={hw.data.cpu.model} />
              <KeyValue k="Çekirdek" v={String(hw.data.cpu.logical_cores ?? "—")} />
              <KeyValue
                k="RAM"
                v={`${fmtBytes(hw.data.ram.available_bytes)} kullanılabilir / ${fmtBytes(hw.data.ram.total_bytes)}`}
              />
            </div>
            <div>
              <KeyValue k="Boş disk" v={fmtBytes(hw.data.disk.free_bytes)} />
              <KeyValue
                k="GPU"
                v={
                  hw.data.gpu.cuda.available
                    ? hw.data.gpu.cuda.devices.map((d) => d.name).join(", ")
                    : "Yok (GPU yok veya torch kurulu değil)"
                }
              />
              <KeyValue
                k="Eğitim motoru"
                v={
                  <StatusBadge
                    status={hw.data.training_ready ? "completed" : "draft"}
                    label={hw.data.training_ready ? "Hazır" : "Bağımlılık eksik"}
                  />
                }
              />
            </div>
          </div>
        ) : null}
      </Card>
    </div>
  );
}
