import { useEffect, useRef, useState } from "react";
import { api, DatasetMeta } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, EmptyState, ErrorBox, Input, SectionTitle, Spinner, StatusBadge } from "../components/ui";

type Project = { id: string; name: string; spec_approved: boolean };

export default function Datasets() {
  const { data, error, loading, refresh } = useApi<{ items: DatasetMeta[] }>("/datasets?limit=50");
  const projects = useApi<{ items: Project[] }>("/projects?limit=50");

  // generation state
  const [projectId, setProjectId] = useState("");
  const [targetCount, setTargetCount] = useState(10);
  const [batchSize, setBatchSize] = useState(3);
  const [genRunId, setGenRunId] = useState<string | null>(null);
  const [genStatus, setGenStatus] = useState<Record<string, unknown> | null>(null);
  const [genErr, setGenErr] = useState<string | null>(null);

  // HF import state
  const [repoId, setRepoId] = useState("tatsu-lab/alpaca");
  const [hfInfo, setHfInfo] = useState<Record<string, unknown> | null>(null);
  const [hfErr, setHfErr] = useState<string | null>(null);
  const [hfBusy, setHfBusy] = useState(false);
  const confirmedRef = useRef(false);

  // preview
  const [previewId, setPreviewId] = useState<string | null>(null);
  const preview = useApi<{ records: Record<string, unknown>[]; limit: number }>(
    previewId ? `/datasets/${previewId}/preview?limit=5` : null
  );

  useEffect(() => {
    if (!genRunId || genStatus && (genStatus as { status: string }).status !== "running") return;
    const iv = setInterval(async () => {
      try {
        const s = await api.get<Record<string, unknown>>(`/datasets/generation-status/${genRunId}`);
        setGenStatus(s);
        const st = (s as { status?: string }).status;
        if (st !== "running") setGenRunId(null);
      } catch {
        /* keep */
      }
    }, 1500);
    return () => clearInterval(iv);
  }, [genRunId, genStatus]);

  async function generate() {
    setGenErr(null);
    setGenStatus(null);
    try {
      const res = await api.post<{ run_id: string; status: string }>("/datasets/generate", {
        project_id: projectId || null,
        target_count: targetCount,
        batch_size: batchSize,
        schema_type: "instruction",
        async_run: true,
      });
      setGenRunId(res.run_id);
      if (res.status === "running") setGenStatus({ status: "running" });
    } catch (e) {
      setGenErr((e as Error).message);
    }
  }

  async function discover() {
    setHfErr(null);
    setHfInfo(null);
    setHfBusy(true);
    try {
      const meta = await api.post<Record<string, unknown>>("/datasets/hf/discover", {
        repo_id: repoId,
        download_confirmed: confirmedRef.current,
      });
      setHfInfo(meta);
    } catch (e) {
      setHfErr((e as Error).message);
    } finally {
      setHfBusy(false);
    }
  }

  async function import_() {
    setHfErr(null);
    setHfBusy(true);
    try {
      await api.post("/datasets/hf/import", {
        repo_id: repoId,
        max_records: targetCount,
        project_id: projectId || null,
      });
      confirmedRef.current = false;
      setHfInfo(null);
      refresh();
    } catch (e) {
      setHfErr((e as Error).message);
    } finally {
      setHfBusy(false);
    }
  }

  async function prepare(id: string) {
    try {
      await api.post("/datasets/prepare", { dataset_id: id, validation_split: 0.1, seed: 42, dedupe: true });
      refresh();
    } catch (e) {
      setGenErr((e as Error).message);
    }
  }

  async function remove(id: string) {
    if (!window.confirm("Bu veri kümesi silinsin mi? (onay gerekli)")) return;
    try {
      await api.delete(`/datasets/${id}?confirm=true`);
      refresh();
    } catch (e) {
      setGenErr((e as Error).message);
    }
  }

  return (
    <div className="space-y-5">
      <header>
        <h1 className="text-xl font-bold">Veri Kümeleri</h1>
        <p className="text-sm text-ink-faint">Yerel Ollama ile üretim veya Hugging Face içe aktarma; hazırlama gerçek dönüşümler yapar.</p>
      </header>
      <ErrorBox message={genErr ?? hfErr} />

      <Card className="space-y-3">
        <SectionTitle hint="Örnekler şemaya doğrulanır; olgusal doğruluk ayrıca kanıtlanmaz">Ollama ile Üret</SectionTitle>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-xs text-ink-faint">
            Proje
            <select className="input mt-1 block max-w-xs" value={projectId} onChange={(e) => setProjectId(e.target.value)}>
              <option value="">— genel üretim (spec olmadan) —</option>
              {projects.data?.items.map((p) => (
                <option key={p.id} value={p.id} disabled={!p.spec_approved}>
                  {p.name} {p.spec_approved ? "✓" : "(spec onaysız)"}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-ink-faint">
            {`Hedef örnek sayısı`}
            <Input type="number" min={1} max={2000} className="mt-1 w-28" value={targetCount} onChange={(e) => setTargetCount(Number(e.target.value))} />
          </label>
          <label className="text-xs text-ink-faint">
            Batch boyutu
            <Input type="number" min={1} max={20} className="mt-1 w-20" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
          </label>
          <Button variant="primary" onClick={generate} disabled={!!genRunId}>Üret</Button>
        </div>
        {genStatus && (
          <div className="rounded-md border border-black/10 px-3 py-2 text-sm dark:border-white/10">
            Durum: {(genStatus as { status?: string }).status ?? "—"} · Kabul edilen: {(genStatus as { accepted_total?: number }).accepted_total ?? 0}
            {(genStatus as { result?: { generated?: number; rejected?: number; duplicates?: number } }).result && (
              <span className="ml-2 text-xs text-ink-faint">
                üretilen {((genStatus as { result?: { generated?: number } }).result?.generated) ?? 0}, reddedilen{" "}
                {((genStatus as { result?: { rejected?: number } }).result?.rejected) ?? 0}, yinelenen{" "}
                {((genStatus as { result?: { duplicates?: number } }).result?.duplicates) ?? 0}
              </span>
            )}
          </div>
        )}
      </Card>

      <Card className="space-y-3">
        <SectionTitle>Hugging Face İçe Aktarma</SectionTitle>
        <div className="flex flex-wrap items-end gap-3">
          <Input className="max-w-md" value={repoId} onChange={(e) => setRepoId(e.target.value)} placeholder="org/dataset" />
          <Button onClick={discover} disabled={hfBusy}>Bilgi Getir</Button>
        </div>
        {hfInfo && (
          <div className="rounded-md border border-black/10 px-3 py-2 text-sm dark:border-white/10">
            <div>Kimlik: {String(hfInfo.id)} · Lisans: {(hfInfo as { license?: string }).license ?? "bilinmiyor"}</div>
            <div className="text-xs text-ink-faint">
              Boyut: {(hfInfo as { size_note?: string }).size_note} · Gated: {String(hfInfo.gated)}
              {(hfInfo as { license_check?: { warning?: string } }).license_check?.warning && (
                <div className="mt-1 text-amber-600">{(hfInfo as { license_check?: { warning?: string } }).license_check?.warning}</div>
              )}
            </div>
            {(hfInfo as { requires_confirmation?: boolean }).requires_confirmation ? (
              <div className="mt-2 flex items-center gap-2">
                <Button
                  variant="danger"
                  onClick={() => {
                    confirmedRef.current = true;
                    import_();
                  }}
                >
                  İndirmeyi Onayla ve İçe Aktar
                </Button>
                <span className="text-xs text-ink-faint">büyük veri kümesi indirmesi — açık onay gerekir.</span>
              </div>
            ) : (
              <div className="mt-2">
                <Button variant="primary" onClick={import_} disabled={hfBusy}>İçe Aktar (ilk 1000 kayıt)</Button>
              </div>
            )}
          </div>
        )}
      </Card>

      {loading ? (
        <Spinner />
      ) : error ? (
        <ErrorBox message={error} />
      ) : (data?.items ?? []).length === 0 ? (
        <EmptyState title="Veri kümesi yok" hint="Yukarıdaki iki gerçek kaynağı kullanın." />
      ) : (
        <div className="space-y-2">
          {(data!.items ?? []).map((d) => (
            <Card key={d.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <span className="font-medium">{d.name}</span>{" "}
                  <StatusBadge status={d.status} label={{ raw: "Ham", validated: "Hazırlanmış" }[d.status] ?? d.status} />
                  <div className="text-xs text-ink-faint">
                    kaynak: {d.source} · {d.example_count} örnek · şema: {d.schema_type ?? "bilinmiyor"}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button onClick={() => setPreviewId(previewId === d.id ? null : d.id)}>Önizle</Button>
                  {d.status !== "validated" && (
                    <Button variant="primary" onClick={() => prepare(d.id)}>Hazırla</Button>
                  )}
                  <Button variant="danger" onClick={() => remove(d.id)}>Sil</Button>
                </div>
              </div>
              {previewId === d.id && (
                preview.loading ? <Spinner /> : (
                  <pre className="mono max-h-64 overflow-auto rounded bg-black/5 p-2 dark:bg-black/40">
                    {JSON.stringify(preview.data?.records ?? [], null, 2)}
                  </pre>
                )
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
