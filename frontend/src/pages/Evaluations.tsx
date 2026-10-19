import { useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, EmptyState, ErrorBox, KeyValue, SectionTitle, Spinner, StatusBadge } from "../components/ui";

type EvalRow = {
  instruction: string;
  reference: string;
  trained_model_output: string | null;
  base_model_output: string | null;
  non_empty: boolean;
  reference_align_prefix: boolean;
};
type EvalReport = {
  status: string;
  counts?: { examples_evaluated?: number; trained_model_responses?: number; base_model_responses?: number; format_ok?: number };
  rows?: EvalRow[];
  note?: string;
  judge_note?: string | null;
};

export default function Evaluations() {
  const { data, error, loading } = useApi<{ items: { id: string; status: string; job_id: string; created_at: string }[] }>(
    "/projects/none-unknown/evaluations"
  );
  const jobs = useApi<{ items: { id: string; status: string }[] }>("/training/jobs?limit=20");
  const [jobId, setJobId] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [report, setReport] = useState<(EvalReport & { id?: string }) | null>(null);

  const completedJobs = (jobs.data?.items ?? []).filter((j) => j.status === "completed");

  async function evaluate() {
    if (!jobId) return setErr("Tamamlanmış bir eğitim işi seçin.");
    setErr(null);
    setBusy(true);
    try {
      const res = await api.post<{ id: string; report: EvalReport }>("/evaluations", {
        job_id: jobId,
        max_examples: 3,
        compare_with_base: true,
      });
      setReport({ ...res.report, id: res.id });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Değerlendirmeler</h1>
        <p className="text-sm text-ink-faint">
          Gerçek çıkarım: eğitilmiş model ve temel model aynı istemlerle karşılaştırılır. Değerlendirme tamamlaması ile eğitim tamamlaması ayrı durumlar içindir.
        </p>
      </header>
      <ErrorBox message={err ?? error} />

      <Card className="space-y-3">
        <SectionTitle>Değerlendirme Çalıştır</SectionTitle>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-ink-faint">
            Tamamlanmış iş
            <select className="input mt-1 block max-w-md" value={jobId} onChange={(e) => setJobId(e.target.value)}>
              <option value="">— seç —</option>
              {completedJobs.map((j) => (
                <option key={j.id} value={j.id}>{j.id}</option>
              ))}
            </select>
          </label>
          <Button variant="primary" onClick={evaluate} disabled={busy}>Değerlendir</Button>
        </div>
        {completedJobs.length === 0 && !loading && <div className="text-sm text-ink-faint">Henüz tamamlanmış eğitim işi yok.</div>}
      </Card>

      {busy && <Spinner label="Gerçek çıkarım yapılıyor (her örnek hem temel hem eğitilmiş modelle çalışır)..." />}

      {report && (
        <Card className="space-y-3">
          <div className="flex items-center gap-2">
            <StatusBadge status={report.status} label={report.status === "completed" ? "Tamam" : "Başarısız"} />
            <span className="mono text-xs">{report.id}</span>
          </div>
          <KeyValue k="Örnek sayısı" v={String(report.counts?.examples_evaluated ?? 0)} />
          <KeyValue k="Eğitilmiş model yanıt verdi" v={String(report.counts?.trained_model_responses ?? 0)} />
          <KeyValue k="Temel model yanıt verdi" v={String(report.counts?.base_model_responses ?? 0)} />
          <KeyValue k="Biçim uygun" v={String(report.counts?.format_ok ?? 0)} />
          {report.note && <div className="text-xs text-ink-faint">{report.note}</div>}
          {(report.rows ?? []).map((row, i) => (
            <div key={i} className="rounded-md border border-black/10 p-3 text-sm dark:border-white/10">
              <div className="mb-1 font-medium">Görev: {row.instruction}</div>
              <div className="mb-1"><span className="text-xs text-ink-faint">Beklenen:</span> {row.reference}</div>
              <div className="mb-1"><span className="text-xs font-medium text-accent">Eğitilmiş:</span> {row.trained_model_output ?? "—"}</div>
              <div><span className="text-xs text-ink-faint">Temel:</span> {row.base_model_output ?? "—"}</div>
            </div>
          ))}
        </Card>
      )}

      {!report && !loading && <EmptyState title="Değerlendirme geçmişi bu ekranda proje bazında listelenir" />}
    </div>
  );
}
