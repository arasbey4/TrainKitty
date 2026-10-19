import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview";
import Projects from "./pages/Projects";
import ProjectDetail from "./pages/ProjectDetail";
import Builder from "./pages/Builder";
import Datasets from "./pages/Datasets";
import Models from "./pages/Models";
import Training from "./pages/Training";
import Evaluations from "./pages/Evaluations";
import Artifacts from "./pages/Artifacts";
import OllamaPage from "./pages/Ollama";
import SettingsPage from "./pages/Settings";
import { t } from "./lib/i18n";

const NAV = [
  { to: "/", label: t("nav.overview") },
  { to: "/projects", label: t("nav.projects") },
  { to: "/builder", label: t("nav.builder") },
  { to: "/datasets", label: t("nav.datasets") },
  { to: "/models", label: t("nav.models") },
  { to: "/training", label: t("nav.jobs") },
  { to: "/evaluations", label: t("nav.evaluations") },
  { to: "/artifacts", label: t("nav.artifacts") },
  { to: "/ollama", label: t("nav.ollama") },
  { to: "/settings", label: t("nav.settings") },
];

function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-black/8 bg-white px-3 py-5 dark:border-white/10 dark:bg-[#15181e]">
      <div className="mb-6 px-2">
        <div className="text-lg font-bold tracking-tight">TrainKit</div>
        <div className="text-[11px] text-ink-faint">{t("tagline")}</div>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) =>
              `rounded-md px-3 py-1.5 text-sm transition-colors ${
                isActive
                  ? "bg-accent/10 font-medium text-accent dark:bg-[#1c2b4a]"
                  : "text-ink-soft hover:bg-black/4 dark:text-[#9aa5b4] dark:hover:bg-white/5"
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <div className="px-2 text-[10px] leading-4 text-ink-faint">
        Yerel çalışır. Veriler bu makinede kalır; telemetri veya bulut çıkarımı yoktur.
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-5xl">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/projects" element={<Projects />} />
            <Route path="/projects/:id" element={<ProjectDetail />} />
            <Route path="/builder" element={<Builder />} />
            <Route path="/builder/:id" element={<Builder />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/models" element={<Models />} />
            <Route path="/training" element={<Training />} />
            <Route path="/training/:jobId" element={<Training />} />
            <Route path="/evaluations" element={<Evaluations />} />
            <Route path="/artifacts" element={<Artifacts />} />
            <Route path="/ollama" element={<OllamaPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}
