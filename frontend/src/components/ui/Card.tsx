import React from "react";

interface CardProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ title, subtitle, actions, children, className }) => (
  <section className={`card ${className ?? ""}`}>
    <header className="card-head">
      <div>
        <h2 className="card-title">{title}</h2>
        {subtitle && <p className="card-sub">{subtitle}</p>}
      </div>
      {actions && <div className="card-actions">{actions}</div>}
    </header>
    <div className="card-body">{children}</div>
  </section>
);
