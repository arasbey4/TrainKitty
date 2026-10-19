import { useState } from "react";
import { Link } from "react-router-dom";
import { api, Project as ProjectType } from "../lib/api";
import { useApi } from "../hooks/useApi";
import { Button, Card, EmptyState, ErrorBox, Input, Spinner, StatusBadge } from "../components/ui";

export default function Projects() {
  const { data, error, loading, refresh } = useApi<{ items: ProjectType[]; total: number }>("/projects");
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function create() {
    if (!name.trim()) return setErr("Proje adı gerekli.");
    setErr(null);
    setCreating(true);
    try {
      const p = await api.post<ProjectType>("/projects", { name: name.trim(), description: desc });
      setName("");
      setDesc("");
      window.location.href = `/projects/${p.id}`;
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="space-y-5">
      <header className="flex items-baseline justify-between">
        <div>
          <h1 className="text-xl font-bold">Projeler</h1>
          <p className="text-sm text-ink-faint">Her proje, eğitilecek özel bir modeli tanımlar.</p>
        </div>
        <Button onClick={refresh}>Yenile</Button>
      </header>

      <Card className="space-y-3">
        <h2 className="text-sm font-semibold">Yeni Proje</h2>
        <ErrorBox message={err} />
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input placeholder="Proje adı (örn. Atlas)" value={name} onChange={(e) => setName(e.target.value)} />
          <Input
            placeholder="Kısa açıklama (isteğe bağlı)"
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
            className="flex-1"
          />
          <Button variant="primary" onClick={create} disabled={creating}>
            Oluştur
          </Button>
        </div>
      </Card>

      {loading ? (
        <Spinner />
      ) : error ? (
        <ErrorBox message={error} />
      ) : (data?.items ?? []).length === 0 ? (
        <EmptyState title="Henüz proje yok" hint="Model Oluşturucu sekmesinden doğal dilde bir tanım başlatabilirsiniz." />
      ) : (
        <div className="space-y-2">
          {data!.items.map((p) => (
            <Card key={p.id} className="flex items-center justify-between">
              <div>
                <Link to={`/projects/${p.id}`} className="font-medium hover:text-accent">
                  {p.name}
                </Link>
                <div className="text-xs text-ink-faint">
                  {p.description || "—"} · {p.created_at ? new Date(p.created_at).toLocaleDateString("tr-TR") : ""}
                </div>
              </div>
              <StatusBadge status={p.status} label={{ draft: "Taslak" }[p.status] ?? p.status} />
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
