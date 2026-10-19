import { useApi } from "../hooks/useApi";
import { Card, EmptyState, ErrorBox, KeyValue, Spinner, StatusBadge } from "../components/ui";
import { fmtBytes } from "../lib/api";

type Artifact = {
  id: string;
  kind: string;
  path: string;
  size_bytes: number;
  project_id: string | null;
  job_id: string | null;
  created_at: string;
};

export default function Artifacts() {
  const { data, error, loading } = useApi<{ items: Artifact[] }>("/artifacts?limit=100");
  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Artefaktlar</h1>
        <p className="text-sm text-ink-faint">Eğitim çıktıları, dışa aktarımlar ve raporlar yerel diskte; kayıt defteri doğrulanmış boyutları gösterir.</p>
      </header>
      {loading ? (
        <Spinner />
      ) : error ? (
        <ErrorBox message={error} />
      ) : (data?.items ?? []).length === 0 ? (
        <EmptyState title="Artefakt yok" hint="Eğitim ve dışa aktarım yürüttükçe kayıtlar burada listelenir." />
      ) : (
        <div className="space-y-2">
          {data!.items.map((a) => (
            <Card key={a.id}>
              <div className="flex items-baseline justify-between">
                <div className="mono text-xs">{a.id}</div>
                <StatusBadge status="draft" label={a.kind} />
              </div>
              <div className="mt-1">
                <KeyValue k="Yol" v={<span className="mono text-xs">{a.path}</span>} />
                <KeyValue k="Boyut" v={fmtBytes(a.size_bytes) ?? "—"} />
                <KeyValue k="İş" v={<span className="mono text-xs">{a.job_id ?? "—"}</span>} />
                <KeyValue k="Tarih" v={new Date(a.created_at).toLocaleString("tr-TR")} />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
