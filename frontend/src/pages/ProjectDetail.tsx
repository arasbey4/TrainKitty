import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, DatasetMeta, Project as ProjectType } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, ErrorBox, KeyValue, SectionTitle, Spinner, StatusBadge } from "../components/ui";

type Preset = Record<string, unknown>;

const SPEC_FIELDS: { key: string; label: string; long?: boolean }[] = [
  { key: "target_model_name", label: "Model adı" },
  { key: "purpose", label: "Amaç" },
  { key: "intended_users", label: "Hedef kullanıcılar" },
  { key: "languages", label: "Diller (virgülle)" },
  { key: "personality", label: "Kişilik/ton" },
  { key: "system_instructions", label: "Sistem talimatları", long: true },
  { key: "desired_behavior", label: "İstenen davranışlar (virgülle)" },
  { key: "unwanted_behavior", label: "İstenmeyen davranışlar (virgülle)" },
  { key: "example_inputs", label: "Örnek kullanıcı girdileri (virgülle)" },
  { key: "output_style", label: "Çıktı biçimi" },
  { key: "dataset_requirements", label: "Veri gereksinimleri" },
];

export default function ProjectDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const { data: project, error, loading, refresh } = useApi<ProjectType>(`/projects/${id}`);
  const datasets = useApi<{ items: DatasetMeta[] }>("/datasets?limit=50");
  const [spec, setSpec] = useState<Preset | null>(null);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const confirmed = useRef(false);

  useEffect(() => {
    if (project?.spec) setSpec(project.spec as Preset);
  }, [project?.id, project?.spec]);

  if (loading) return <Spinner />;
  if (error || !project) return <ErrorBox message={error ?? "Proje bulunamadı"} />;

  async function save(approved?: boolean) {
    if (!spec) return;
    setSaving(true);
    setErr(null);
    try {
      await api.patch(`/projects/${project!.id}`, {
        name: spec.target_model_name || project!.name,
        spec,
        spec_approved: approved ?? project!.spec_approved,
        status: approved ? "spec_ready" : project!.status,
      });
      confirmed.current = true;
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function linkDataset(datasetId: string) {
    try {
      await api.patch(`/projects/${project!.id}`, { dataset_id: datasetId, status: "dataset_ready" });
      refresh();
    } catch (e) {
      setErr((e as Error).message);
    }
  }

  return (
    <div className="space-y-5">
      <header className="flex items-center justify-between">
        <div>
          <Link to="/projects" className="text-xs text-ink-faint">← Projeler</Link>
          <h1 className="text-xl font-bold">{project.name}</h1>
        </div>
        <StatusBadge
          status={project.spec_approved ? "approved" : "draft"}
          label={project.spec_approved ? "Spec onaylı" : "Taslak"}
        />
      </header>
      <ErrorBox message={err} />

      <Card>
        <SectionTitle hint="Alanlar model tarafından önerilmiş olabilir — onaylayana kadar bağlayıcı değildir">
          Model Tanımı
        </SectionTitle>
        {!spec ? (
          <p className="text-sm text-ink-faint">
            {"Henüz spec yok."}{" "}
            <Link to={`/builder/${project.id}`} className="text-accent underline">
              Model Oluşturucu
            </Link>{" "}
            ile doğal dilden üretin.
          </p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {SPEC_FIELDS.map((f) => {
              const value = spec[f.key];
              const strVal = Array.isArray(value)
                ? value.join(", ")
                : typeof value === "string"
                  ? value
                  : "";
              return f.long ? (
                <textarea
                  key={f.key}
                  className="input h-24 sm:col-span-2"
                  placeholder={f.label}
                  value={strVal}
                  onChange={(e) => setSpec({ ...spec, [f.key]: e.target.value })}
                />
              ) : (
                <label key={f.key} className="text-xs text-ink-faint">
                  {f.label}
                  <input
                    className="input mt-1"
                    value={strVal}
                    onChange={(e) =>
                      setSpec({
                        ...spec,
                        [f.key]:
                          f.key === "languages" ||
                          f.key === "desired_behavior" ||
                          f.key === "unwanted_behavior" ||
                          f.key === "example_inputs"
                            ? e.target.value
                                .split(",")
                                .map((s) => s.trim())
                                .filter(Boolean)
                            : e.target.value,
                      })
                    }
                  />
                </label>
              );
            })}
            <div className="flex gap-2 sm:col-span-2">
              <Button variant="primary" onClick={() => save(true)} disabled={saving}>
                {project.spec_approved ? "Onaylı — Güncelle" : "Kaydet ve Onayla"}
              </Button>
              <Button onClick={() => save(false)} disabled={saving}>
                Yalnızca Kaydet
              </Button>
            </div>
          </div>
        )}
      </Card>

      <Card>
        <SectionTitle>Veri Kümesi Seçimi</SectionTitle>
        <select
          className="input max-w-md"
          value={project.dataset_id ?? ""}
          onChange={(e) => linkDataset(e.target.value)}
        >
          <option value="">— seçilmedi —</option>
          {datasets.data?.items.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} ({d.example_count} örnek, {d.status})
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-ink-faint">
          Veri Kümeleri sayfasından üretip hazırlayın; burada hazır (validated) kümeler kullanılır.
        </p>
      </Card>

      <Card>
        <SectionTitle>Etkinlik Günlüğü</SectionTitle>
        {(project.activity ?? []).slice(-15).reverse().map((a, i) => (
          <div key={i} className="border-b border-black/5 py-1 text-sm last:border-0 dark:border-white/5">
            <span className="text-xs text-ink-faint">{new Date(a.ts).toLocaleString("tr-TR")}</span> — {a.message}
          </div>
        ))}
        {(project.activity ?? []).length === 0 && <div className="text-sm text-ink-faint">Kayıt yok.</div>}
      </Card>

      <Card className="flex justify-between">
        <div className="text-sm text-ink-faint">Eğitim işi başlatmak için Eğitim İşleri bölümüne gidin.</div>
        <Button onClick={() => nav("/training")}>Eğitim İşleri →</Button>
      </Card>
    </div>
  );
}
