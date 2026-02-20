/**
 * AI Token Meter — visual token / context usage indicator.
 *
 * Shows estimated token usage relative to the model's context window.
 */
import type { CSSProperties } from 'react';

interface Props {
  /** Estimated tokens used in the prompt+response */
  used?: number;
  /** Model context window size */
  limit?: number;
}

const DEFAULT_LIMIT = 65_536; // gpt-oss-120b context window

export default function AITokenMeter({ used, limit = DEFAULT_LIMIT }: Props) {
  if (used == null || used <= 0) return null;

  const pct = Math.min((used / limit) * 100, 100);
  const color =
    pct > 80 ? 'bg-red-500' :
    pct > 50 ? 'bg-yellow-500' :
    'bg-accent';

  return (
    <div className="flex items-center gap-2 text-[10px] text-sidebar-muted">
      <span>~{used.toLocaleString()} tokens</span>
      <div className="w-16 h-1.5 rounded-full bg-sidebar-border overflow-hidden">
        <div
          className={`h-full rounded-full transition-all token-progress-bar ${color}`}
          style={{ '--progress': `${pct}%` } as CSSProperties}
        />
      </div>
      <span className="opacity-50">{pct.toFixed(0)}%</span>
    </div>
  );
}
