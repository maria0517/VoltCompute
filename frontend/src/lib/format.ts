/** Utilitare de formatare (locale ro-RO) folosite în toată interfața. */

export const nf = (value: number, decimals = 2): string =>
  value.toLocaleString("ro-RO", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

/** Backend-ul trimite timestamp-uri "YYYY-MM-DD HH:MM". */
export const parseTimestamp = (ts: string): Date => new Date(ts.replace(" ", "T"));

export const formatDateTime = (ts: string): string =>
  new Intl.DateTimeFormat("ro-RO", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parseTimestamp(ts));

export const formatShortDateTime = (ts: string): string =>
  new Intl.DateTimeFormat("ro-RO", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parseTimestamp(ts));

/** "2026-11" -> "Noiembrie 2026" */
export const formatMonth = (label: string): string => {
  const text = new Intl.DateTimeFormat("ro-RO", { month: "long", year: "numeric" }).format(
    new Date(`${label}-01T00:00:00`)
  );
  return text.charAt(0).toUpperCase() + text.slice(1);
};
