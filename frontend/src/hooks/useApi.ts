import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../lib/api";

/** Simple one-shot fetch with loading/error state. */
export function useApi<T>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!path) {
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    api
      .get<T>(path)
      .then((d) => {
        if (alive) {
          setData(d);
          setError(null);
        }
      })
      .catch((e: ApiError) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, nonce, ...deps]);

  return { data, error, loading, refresh };
}

/** Subscribe to a job's real SSE stream (state/log/metric events), with
 *  polling fallback: the SSE endpoint also emits DB-backed job state each
 *  second and closes shortly after terminal state. */
export function useJobStream(jobId: string | null) {
  type StreamEvent = { type: string; [k: string]: unknown };
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [metrics, setMetrics] = useState<{ step: number; loss: number | null; eval_loss: number | null; epoch: number | null; learning_rate: number | null }[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) return;
    setEvents([]);
    setMetrics([]);
    setLogs([]);
    setStatus(null);
    const es = new EventSource(`/api/training/jobs/${jobId}/stream`);
    esRef.current = es;
    es.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data) as StreamEvent;
        if (data.type === "done") {
          es.close();
          return;
        }
        setEvents((prev) => [...prev.slice(-200), data]);
        if (data.type === "job" && typeof data.status === "string") setStatus(data.status);
        if (data.type === "metric") {
          setMetrics((prev) => [
            ...prev,
            {
              step: Number(data.step ?? 0),
              loss: typeof data.loss === "number" ? data.loss : null,
              eval_loss: typeof data.eval_loss === "number" ? data.eval_loss : null,
              epoch: typeof data.epoch === "number" ? data.epoch : null,
              learning_rate: typeof data.learning_rate === "number" ? data.learning_rate : null,
            },
          ]);
        }
        if (data.type === "log" && typeof data.message === "string") {
          setLogs((prev) => [...prev.slice(-300), data.message as string]);
        }
        if (data.type === "error" && typeof data.error === "string") {
          setLogs((prev) => [...prev, `[hata] ${data.error}`]);
        }
      } catch {
        /* skip malformed */
      }
    };
    es.onerror = () => {
      // fallback: EventSource auto-reconnects; terminal states close the loop
    };
    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId]);

  return { events, metrics, logs, status };
}
