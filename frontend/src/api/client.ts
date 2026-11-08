import { DashboardStateResponse } from "../types/telemetry";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchDashboardState(
  step?: number,
  cycle: number = 0
): Promise<DashboardStateResponse> {
  const params = new URLSearchParams();
  if (step !== undefined) {
    params.set("step", String(step));
  }
  params.set("cycle", String(cycle));
  const res = await fetch(`${API_BASE}/api/dashboard/state?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Eroare API (${res.status}): Nu s-a putut încărca starea SCADA.`);
  }
  return res.json();
}

export async function acceptOffer(
  nodeId: string,
  perStep: Record<string, number>,
  step: number,
  cycle: number
): Promise<DashboardStateResponse> {
  const numericPerStep: Record<number, number> = {};
  for (const [k, v] of Object.entries(perStep)) {
    numericPerStep[Number(k)] = v;
  }
  const res = await fetch(`${API_BASE}/api/dashboard/offer/accept`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      node_id: nodeId,
      per_step: numericPerStep,
      step,
      cycle,
    }),
  });
  if (!res.ok) {
    throw new Error(`Eroare API la aprobarea ofertei (${res.status})`);
  }
  return res.json();
}

export async function refuseOffer(
  nodeId: string,
  steps: number[],
  step: number,
  cycle: number
): Promise<DashboardStateResponse> {
  const res = await fetch(`${API_BASE}/api/dashboard/offer/refuse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      node_id: nodeId,
      steps,
      step,
      cycle,
    }),
  });
  if (!res.ok) {
    throw new Error(`Eroare API la refuzarea ofertei (${res.status})`);
  }
  return res.json();
}

export async function resetOffers(
  steps: number[],
  step: number,
  cycle: number
): Promise<DashboardStateResponse> {
  const res = await fetch(`${API_BASE}/api/dashboard/offer/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      steps,
      step,
      cycle,
    }),
  });
  if (!res.ok) {
    throw new Error(`Eroare API la resetarea ofertelor (${res.status})`);
  }
  return res.json();
}
