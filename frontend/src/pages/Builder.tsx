import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, Project } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, ErrorBox, SectionTitle, Spinner } from "../components/ui";

const DEFAULT_DESC =
  "Atlas adında, Python hatalarını net biçimde açıklayan, kısa ve öz yanıt veren, profesyonel ama samimi tonlu bir Türkçe programlama asistanı oluştur.";

export default function Builder() {
  const params = useParams();
  const nav = useNavigate();
  const { data: projects } = useApi<{ items: Project[] }>("/projects?limit=50");
  const [projectId, setProjectId] = useState<string>(params.id ?? "");
  const [desc, setDesc] = useState(DEFAULT_DESC);
  const [draft, setDraft] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [missing, setMissing] = useState<string[]>([]);
  const [applied, setApplied] = useState(false);

  async function draftSpec() {
    if (!projectId) return setErr("Önce bir proje seçin veya oluşturun.");
    setErr(null);
    setApplied(false);
    setBusy(true);
    try {
      const res = await api.post<{ draft: Record<string, unknown>; note: string }>(
        `/projects/${projectId}/spec/draft`,
        { description: desc }
      );
      setDraft(res.draft);
      setMissing(Array.isArray(res.draft.missing_information) ? (res.draft.missing_information as string[]) : []);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function apply() {
    if (!draft || !projectId) return;
    setBusy(true);
    setErr(null);
    try {
      await api.patch(`/projects/${projectId}`, { spec: draft, spec_approved: true, status: "spec_ready" });
      setApplied(true);
      nav(`/projects/${projectId}`);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const rows = draft
    ? Object.entries(draft).filter(([k]) => k !== "missing_information")
    : [];

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Model Oluşturucu</h1>
        <p className="text-sm text-ink-faint">
          Doğal dili yazın; yerel Ollama modeli yapılandırılmış bir spec taslağı üretir. Taslak bir öneridir — oluşturulan içerik kullanıcı onayına sunulur.
        </p>
      </header>
      <ErrorBox message={err} />

      <Card className="space-y-3">
        <SectionTitle>1. Proje</SectionTitle>
        {projects && projects.items.length === 0 ? (
          <p className="text-sm text-ink-faint">
            Henüz proje yok.{" "}
            <Link to="/projects" className="text-accent underline">
              Bir proje oluştur
            </Link>
          </p>
        ) : (
          <select className="input max-w-md" value={projectId} onChange={(e) => setProjectId(e.target.value)}>
            <option value="">— proje seç —</option>
            {projects?.items.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        )}
      </Card>

      <Card className="space-y-3">
        <SectionTitle>2. Tanım</SectionTitle>
        <textarea className="input h-28" value={desc} onChange={(e) => setDesc(e.target.value)} />
        <Button variant="primary" onClick={draftSpec} disabled={busy}>
          {busy ? "Ollama çalışıyor..." : "Spec Taslağı Üret"}
        </Button>
      </Card>

      {busy && !draft && <Spinner label="Yerel model yanıtı bekleniyor (ilk istek modelin yüklenmesiyle biraz sürebilir)..." />}

      {draft && (
        <Card className="space-y-2">
          <SectionTitle hint="Yapılandırılmış tahmin">3. Taslak</SectionTitle>
          {rows.map(([k, v]) => (
            <div key={k} className="flex gap-4 border-b border-black/5 py-1 text-sm last:border-0 dark:border-white/5">
              <span className="w-44 shrink-0 font-mono text-xs text-ink-faint">{k}</span>
              <span>{Array.isArray(v) ? v.join(", ") : typeof v === "string" ? v : JSON.stringify(v)}</span>
            </div>
          ))}
          {missing.length > 0 && (
            <div className="mt-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-300">
              <div className="font-medium">Eksik bilgi soruları:</div>
              <ul className="ml-4 list-disc">
                {missing.map((m, i) => (
                  <li key={i}>{m}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="mt-3 flex gap-2">
            <Button variant="primary" onClick={apply} disabled={busy || !projectId}>
              Onayla ve Projeye Uygula
            </Button>
            <Button onClick={draftSpec} disabled={busy}>Yeniden Üret</Button>
          </div>
          {applied && <div className="text-sm text-green-600">Uygulandı.</div>}
        </Card>
      )}
    </div>
  );
}
