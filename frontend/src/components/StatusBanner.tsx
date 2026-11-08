import React from "react";
import { BannerInfo } from "../types/telemetry";

const TITLES: Record<BannerInfo["level"], string> = {
  red: "Alertă",
  amber: "Atenție",
  green: "Funcționare normală",
};

export const StatusBanner: React.FC<{ banner: BannerInfo }> = ({ banner }) => (
  <div className={`banner banner-${banner.level}`} role="status">
    <span className="banner-title">{TITLES[banner.level]}</span>
    <span className="banner-text">{banner.text}</span>
  </div>
);
