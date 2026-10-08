// service/perf.tsx — performance-trace panel. Reads measured values only: the shell
// scope timings, the engine phase timings, and live resource counters. Tracing is
// opt-in (--profileperf at launch, or the toggle here), so this panel is inert until
// switched on.
import { useCallback, useEffect, useState } from "react";

import { perfReport, perfReset, perfSet, type PerfReport } from "../api/client";
import { t } from "../lib/t";

function us(v: number): string {
  return v.toFixed(2);
}

function gb(used: number | null | undefined, total: number | null | undefined): string {
  if (used == null) return t("service.perf.unavailable");
  if (total == null) return `${used.toFixed(1)} GiB`;
  return `${used.toFixed(1)}/${total.toFixed(1)} GiB`;
}

function pct(v: number | null | undefined): string {
  return v == null ? t("service.perf.unavailable") : `${v.toFixed(1)}%`;
}

export function PerfSection() {
  const [report, setReport] = useState<PerfReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setReport(await perfReport());
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function toggle() {
    setBusy(true);
    try {
      const next = !(report?.enabled ?? false);
      await perfSet(next);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    setBusy(true);
    try {
      await perfReset();
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const r = report;
  const res = r?.resources;

  return (
    <section className="e0-section overflow-hidden p-5" data-testid="svc-perf">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold">{t("service.perf.title")}</h3>
          <p className="text-xs text-muted">{t("service.perf.kicker")}</p>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" data-testid="svc-perf-toggle" disabled={busy} onClick={() => void toggle()} className="e0-btn e0-btn-secondary">
            {r?.enabled ? t("service.perf.disable") : t("service.perf.enable")}
          </button>
          <button type="button" data-testid="svc-perf-refresh" disabled={busy} onClick={() => void refresh()} className="e0-btn e0-btn-secondary">
            {t("service.perf.refresh")}
          </button>
          <button type="button" data-testid="svc-perf-reset" disabled={busy} onClick={() => void reset()} className="e0-btn e0-btn-secondary">
            {t("service.perf.reset")}
          </button>
        </div>
      </div>

      <p className="mb-2 text-xs">
        <span data-testid="svc-perf-state" className={r?.enabled ? "text-accent" : "text-muted"}>
          {r?.enabled ? t("service.perf.on") : t("service.perf.off")}
        </span>
        {r ? <span className="text-muted"> · {t("service.perf.samples", { n: r.resource_samples })}</span> : null}
      </p>
      {error && (
        <p role="alert" className="mb-2 rounded-lg bg-err/10 px-3 py-2 text-xs text-err">{error}</p>
      )}

      {r && r.components.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs tabular-nums" data-testid="svc-perf-table">
            <thead>
              <tr className="text-left text-muted">
                <th className="py-1 pr-3 font-medium">{t("service.perf.th.component")}</th>
                <th className="py-1 pr-3 font-medium">{t("service.perf.th.scope")}</th>
                <th className="py-1 pr-3 text-right font-medium">{t("service.perf.th.ops")}</th>
                <th className="py-1 pr-3 text-right font-medium">{t("service.perf.th.pct")}</th>
                <th className="py-1 pr-3 text-right font-medium">{t("service.perf.th.dev")}</th>
                <th className="py-1 pr-3 text-right font-medium">{t("service.perf.th.idle")}</th>
                <th className="py-1 text-right font-medium">{t("service.perf.th.host")}</th>
              </tr>
            </thead>
            <tbody>
              {r.components.map((c) => (
                <tr key={c.name} className="border-t border-line">
                  <td className="py-1 pr-3">{c.name}</td>
                  <td className="py-1 pr-3 text-muted">{c.scope}</td>
                  <td className="py-1 pr-3 text-right">{c.ops}</td>
                  <td className="py-1 pr-3 text-right">{c.scope === "app" ? "—" : `${c.pct_dev.toFixed(1)}%`}</td>
                  <td className="py-1 pr-3 text-right">{c.scope === "app" ? "—" : us(c.dev_us)}</td>
                  <td className="py-1 pr-3 text-right">{us(c.idle_us)}</td>
                  <td className="py-1 text-right">{us(c.host_us)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-xs text-muted" data-testid="svc-perf-empty">{t("service.perf.empty")}</p>
      )}

      {r?.floor ? (
        <p className="mt-2 text-xs text-muted" data-testid="svc-perf-floor">
          {t("service.perf.floor", { us: r.floor.host_us_each.toFixed(2), n: r.floor.n })}
        </p>
      ) : null}

      <div className="mt-4 border-t border-line pt-3">
        <h4 className="mb-1 text-xs font-semibold">{t("service.perf.resources")}</h4>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs" data-testid="svc-perf-resources">
          <dt className="text-muted">{t("service.perf.cpu")}</dt>
          <dd className="tabular-nums">{pct(res?.cpu_pct)}</dd>
          <dt className="text-muted">{t("service.perf.ram")}</dt>
          <dd className="tabular-nums">{gb(res?.ram_used_gb, res?.ram_total_gb)}</dd>
          <dt className="text-muted">{t("service.perf.gpu")}</dt>
          <dd className="tabular-nums">{pct(res?.gpu_util_pct)}</dd>
          <dt className="text-muted">{t("service.perf.vram")}</dt>
          <dd className="tabular-nums">{gb(res?.vram_used_gb, res?.vram_total_gb)}</dd>
          <dt className="text-muted">{t("service.perf.rebar")}</dt>
          <dd className="tabular-nums" title={res?.rebar?.source ?? undefined}>
            {res?.rebar ? res.rebar.state : t("service.perf.unavailable")}
          </dd>
        </dl>
        <p className="mt-2 text-xs text-muted">{t("service.perf.note")}</p>
        {r?.engine_log ? (
          <p className="mt-1 text-xs text-muted break-all">{t("service.perf.engineLog")}: {r.engine_log}</p>
        ) : null}
      </div>
    </section>
  );
}
