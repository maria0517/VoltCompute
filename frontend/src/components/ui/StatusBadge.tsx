import React from "react";

export type Tone = "neutral" | "info" | "success" | "warning" | "danger";

interface StatusBadgeProps {
  tone: Tone;
  label: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ tone, label }) => (
  <span className={`badge badge-${tone}`}>
    <span className="badge-dot" />
    {label}
  </span>
);
