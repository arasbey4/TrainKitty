import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { api, DatasetMeta, Job, fmtBytes } from "../lib/api";
import { useApi, useJobStream } from "../hooks/useApi";
import {
  Button, Card, EmptyState, ErrorBox, Input, KeyValue, SectionTitle, Spinner, StatusBadge,
} from "../components/ui";
import { t } from "../lib/i18n";

type JobFull = Job & {
  process?: { cpu_percent: number | null; memory_rss_bytes: number | null; gpu_utilization: number | null; gpu_memory_bytes: number | null };
  events?: { ts: string; level: string; message: string; operation?: string }[];
};

export default function Training() {
  const { jobId } = useParams();
  return jobId ? <JobDetail jobId={jobId} /> : <JobList />;
}

function JobList() {
  const { data, error, loading, refresh } = useApi<{ items: Job[] }>("/training/jobs?limit=50");
  return (
    <div className="space-y-5">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-xl font-bold">Eğitim İşleri</h1>
          <p className="text-sm text-ink-faint">Gerçek süreçler; kayıp eğrileri gerçek eğitici günlüklerinden gelir.</p>
        </div>
        <Button onClick={refresh}>Yenile</Button>
      </header>
      {loading ? <Spinner /> : error ? <ErrorBox message={error} /> : (data?.items ?? []).length === 0 ? (
        <EmptyState title={t("jobs.empty")} />
      ) : (
        <div className="space-y-2">
          {data!.items.map((j) => (
            <Card key={j.id} className="flex items-baseline justify-between">
              <div>
                <Link to={`/training/${j.id}`} className="mono font-medium hover:text-accent">{j.id}</Link>
                <div className="text-xs text-ink-faint">
                  {j.mode} · {j.progress_step}/{j.total_steps} adım
                </div>
              </div>
              <StatusBadge status={j.status} label={statusTr(j.status)} />
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function statusTr(s: string) {
  const map: Record<string, string> = {
    queued: t("jobs.statusQueued"),
    preparing: t("jobs.statusPreparing"),
    downloading: t("jobs.statusDownloading"),
    validating: t("jobs.statusValidating"),
    training: t("jobs.statusTraining"),
    evaluating: t("jobs.statusEvaluating"),
    packaging: t("jobs.statusPackaging"),
    completed: t("jobs.statusCompleted"),
    failed: t("jobs.statusFailed"),
    cancelled: t("jobs.statusCancelled"),
  };
  return map[s] ?? s;
}

function JobDetail({ jobId }: { jobId: string }) {
  const nav = useNavigate();
  const job = useApi<JobFull>(`/training/jobs/${jobId}`);
  const stream = useJobStream(jobId);
  const [msg, setMsg] = useState<string | null>(null);

  const allMetrics = useMemo(() => sortMetrics(stream.metrics), [stream.metrics]);

  async function cancelJob() {
    if (!window.confirm(t("jobs.cancel"))) return;
    try {
      await api.post(`/training/jobs/${jobId}/cancel`);
    } catch (e) {
      setMsg((e as Error).message);
    }
  }

  const activeJob = job.data?.status && ["queued", "preparing", "downloading", "validating", "training", "evaluating", "packaging"].includes(job.data.status);

  return (
    <div className="space-y-5">
      <header className="flex items-baseline justify-between">
        <div>
          <Link to="/training" className="text-xs text-ink-faint">← İşler</Link>
          <h1 className="mono text-lg font-bold">{jobId}</h1>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={job.data?.status ?? ""} label={statusTr(job.data?.status ?? "")} />
          {activeJob && (
            <Button variant="danger" onClick={cancelJob}>{t("jobs.cancel")}</Button>
          )}
        </div>
      </header>
      <ErrorBox message={msg ?? job.error ?? lastError(stream.events) ?? null} />

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <div className="text-xs text-ink-faint">İlerleme {(job.data?.total_steps ?? 0) > 0 ? `(tahminî toplam ${job.data!.total_steps})` : ""}</div>
          <div className="mt-1 text-2xl font-semibold">
            {Math.max(job.data?.progress_step ?? 0, ...allMetrics.map((m) => m.step))}
            <span className="text-sm text-ink-faint"> / {job.data?.total_steps ?? "?"}</span>
          </div>
          {allMetrics.length > 0 && <KeyValue k="Son kayıp" v={lastNum(allMetrics.map((m) => m.loss).filter((x): x is number => x != null))} />}
        </Card>
        <Card>
          <div className="text-xs text-ink-faint">Süreç (gerçek; GPU ölçümü kullanılamıyorsa gösterilmez)</div>
          <KeyValue k="CPU %" v={job.data?.process?.cpu_percent != null ? String(job.data!.process!.cpu_percent) : "—"} />
          <KeyValue k="Bellek (RSS)" v={fmtBytes(job.data?.process?.memory_rss_bytes ?? null) ?? "—"} />
          <KeyValue k="GPU" v="ölçüm kullanılamıyor" />
        </Card>
        <Card>
          <div className="text-xs text-ink-faint">Yapılandırma</div>
          <KeyValue k="Mod" v={job.data?.mode ?? "—"} />
          <KeyValue k="Temel model" v={job.data?.config?.base_model_id ?? "—"} />
          <KeyValue k="Veri kümesi" v={job.data?.config?.dataset_id ?? "—"} />
        </Card>
      </div>

      <Card>
        <SectionTitle hint="transformers Trainer gerçek loss değerleri">{t("jobs.metrics")}</SectionTitle>
        {allMetrics.length === 0 ? (
          <div className="text-sm text-ink-faint">Henüz metrik yok.</div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={allMetrics}>
              <XAxis dataKey="step" fontSize={11} />
              <YAxis fontSize={11} domain={["auto", "auto"]} />
              <Tooltip />
              <Line type="monotone" dataKey="loss" stroke="#2563eb" dot={false} name={t("jobs.loss")} />
              <Line type="monotone" dataKey="eval_loss" stroke="#16a34a" dot={false} name={t("jobs.evalLoss")} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Card>

      <Card>
        <SectionTitle hint="stdout'tan gerçek akış — uydurma değer yok">{t("jobs.logs")}</SectionTitle>
        <pre className="mono max-h-80 overflow-auto whitespace-pre-wrap rounded bg-black/5 p-3 dark:bg-black/40">
          {stream.logs.slice(-200).join("\n") || "—"}
        </pre>
      </Card>

      <Card>
        <SectionTitle>Olaylar</SectionTitle>
        {(job.data?.events ?? []).slice(-30).map((e, i) => (
          <div key={i} className="border-b border-black/5 py-1 text-sm last:border-0 dark:border-white/5">
            <span className="text-xs text-ink-faint">{new Date(e.ts).toLocaleTimeString("tr-TR")}</span>{" "}
            <span className={e.level === "error" ? "text-red-500" : ""}>{e.message}</span>
          </div>
        ))}
        {(job.data?.events ?? []).length === 0 && <div className="text-sm text-ink-faint">Kayıt yok.</div>}
      </Card>
    </div>
  );
}

function sortMetrics(m: { step: number; loss: number | null; eval_loss: number | null; epoch: number | null }[]) {
  return [...m].filter((v, i, a) => i === 0 || v.step !== a[i - 1].step).map((v, i) => ({ ...v, idx: i }));
}

function lastNum(xs: number[]): string {
  return xs.length ? xs[xs.length - 1].toFixed(4) : "—";
}

function lastError(events: { type: string;[k: string]: unknown }[]): string | null {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].type === "error" && typeof events[i]["error"] === "string") return events[i]["error"] as string;
  }
  return null;
}
