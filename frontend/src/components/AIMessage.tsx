/**
 * AIMessage — replaces the old Message component.
 *
 * Surfaces backend intelligence:
 *  - Streaming phase indicators (classifying → retrieving → generating)
 *  - AI Status Bar with event chips
 *  - Intent badge with confidence dot
 *  - Expandable retrieval panel
 *  - Debug panel (when debug mode active)
 *  - Token estimation meter
 *  - Profile injection badge
 */

import { useState, useCallback, useMemo } from 'react';
import { Copy, Check, Bot, UserIcon, ChevronDown, Bug } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import type { Message } from 'ai';

import { AIStatusBar, AIIntentBadge, AIRetrievalPanel, AITokenMeter, AIDebugPanel } from './ai';
import type { StageEvent } from './ai';
import { useChatStore } from '../store';

interface Props {
  message: Message;
  isStreaming?: boolean;
}

/** Rough token estimate: ~4 chars per token */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export default function AIMessage({ message, isStreaming }: Props) {
  const [copied, setCopied] = useState(false);
  const [showRetrieval, setShowRetrieval] = useState(false);
  const { memoryPanelOpen, toggleMemoryPanel, debugMode } = useChatStore();
  const isUser = message.role === 'user';

  const copyText = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  // ── Parse annotations into stages + final metadata ──────────────────
  const { stages, metadata } = useMemo(() => {
    const annotations: Record<string, any>[] = (message as any).annotations ?? [];
    const stageEvents: StageEvent[] = [];
    let meta: Record<string, any> = {};

    for (const ann of annotations) {
      if (ann.stage) {
        stageEvents.push(ann as StageEvent);
      } else if (ann.intent) {
        // Final metadata annotation
        meta = ann;
      }
    }
    return { stages: stageEvents, metadata: meta };
  }, [(message as any).annotations]);

  const hasMetadata = Object.keys(metadata).length > 0;
  const ri = metadata.retrieval_info || stages.find(s => s.stage === 'retrieved')?.retrieval_info;
  const intent = metadata.intent || stages.find(s => s.stage === 'classified')?.intent;
  const confidence = metadata.confidence ?? stages.find(s => s.stage === 'classified')?.confidence ?? 0;

  // ── Streaming phase indicator (before content arrives) ──────────────
  const streamingPhase = useMemo(() => {
    if (!isStreaming || message.content) return null;
    if (stages.length === 0) return 'Thinking…';
    const last = stages[stages.length - 1];
    switch (last.stage) {
      case 'classified': return 'Retrieving context…';
      case 'retrieved': return 'Preparing response…';
      case 'generating': return 'Generating…';
      default: return 'Thinking…';
    }
  }, [isStreaming, message.content, stages]);

  if (isUser) {
    return (
      <div className="group flex gap-3 py-4 fade-in">
        <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs bg-accent/20 text-accent">
          <UserIcon size={14} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-sidebar-muted mb-1 font-medium">You</div>
          <p className="text-gray-200 whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex gap-3 py-4 fade-in">
      {/* Avatar */}
      <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs bg-sidebar-hover text-gray-300">
        <Bot size={14} />
      </div>

      {/* Content column */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs text-sidebar-muted font-medium">Assistant</span>
          {intent && <AIIntentBadge intent={intent} confidence={confidence} />}
        </div>

        {/* Status bar (stage timeline) — only for assistant messages with stages */}
        {stages.length > 0 && (
          <AIStatusBar
            stages={stages}
            isStreaming={!!isStreaming}
            onChipClick={ri ? () => setShowRetrieval(v => !v) : undefined}
          />
        )}

        {/* Streaming phase (before content) */}
        {streamingPhase && (
          <div className="flex items-center gap-2 py-2 text-sm text-sidebar-muted">
            <div className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce [animation-delay:300ms]" />
            </div>
            {streamingPhase}
          </div>
        )}

        {/* Message body (markdown) */}
        {message.content && (
          <div className="prose max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                pre: ({ children, ...props }) => (
                  <CodeBlock {...props}>{children}</CodeBlock>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-accent animate-pulse rounded-sm ml-0.5" />
            )}
          </div>
        )}

        {/* Actions bar */}
        {!isStreaming && message.content && (
          <div className="flex items-center gap-1.5 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={copyText}
              className="flex items-center gap-1 px-2 py-1 text-xs text-sidebar-muted
                         rounded hover:bg-sidebar-hover hover:text-white transition-colors"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </button>

            {/* Memory toggle */}
            {ri && (
              <button
                onClick={() => toggleMemoryPanel(message.id)}
                className="flex items-center gap-1 px-2 py-1 text-xs text-sidebar-muted
                           rounded hover:bg-sidebar-hover hover:text-white transition-colors"
              >
                <ChevronDown
                  size={12}
                  className={`transition-transform ${memoryPanelOpen[message.id] ? 'rotate-180' : ''}`}
                />
                Memory
              </button>
            )}

            {/* Token estimate */}
            <AITokenMeter used={estimateTokens(message.content)} />
          </div>
        )}

        {/* Retrieval panel (toggle from status bar "Details" button) */}
        {showRetrieval && ri && (
          <AIRetrievalPanel info={{ ...ri, intent, confidence }} />
        )}

        {/* Full retrieval panel (toggle from "Memory" action) */}
        {memoryPanelOpen[message.id] && ri && !showRetrieval && (
          <AIRetrievalPanel info={{ ...ri, intent, confidence }} />
        )}

        {/* Debug panel — only in debug mode */}
        {debugMode && hasMetadata && (
          <AIDebugPanel metadata={metadata} />
        )}
      </div>
    </div>
  );
}

/* ── Code block with copy & language label ──────────────────────────────── */

function CodeBlock({ children, ...props }: React.HTMLAttributes<HTMLPreElement>) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const text = (children as any)?.props?.children || '';
    navigator.clipboard.writeText(typeof text === 'string' ? text : String(text));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const childClass = (children as any)?.props?.className || '';
  const lang = childClass.replace('hljs language-', '').replace('language-', '') || 'code';

  return (
    <div className="relative rounded-lg overflow-hidden my-3 border border-sidebar-border">
      <div className="flex items-center justify-between px-4 py-1.5 bg-sidebar-bg text-xs text-sidebar-muted">
        <span>{lang}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 hover:text-white transition-colors"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre {...props} className="!mt-0 !rounded-t-none">
        {children}
      </pre>
    </div>
  );
}
