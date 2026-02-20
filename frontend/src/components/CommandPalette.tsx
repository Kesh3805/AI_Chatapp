/**
 * Command Palette — Ctrl+K / Cmd+K trigger.
 *
 * Quick-access to: new chat, search, profile, debug toggle, clear memory, etc.
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import {
  Plus,
  Search,
  User,
  Bug,
  Trash2,
  MessageSquare,
  Command,
  X,
} from 'lucide-react';
import { useChatStore } from '../store';

interface PaletteCommand {
  id: string;
  icon: React.ReactNode;
  label: string;
  description?: string;
  action: () => void;
}

export default function CommandPalette() {
  const {
    commandPaletteOpen,
    setCommandPaletteOpen,
    setConversationId,
    setProfileModalOpen,
    debugMode,
    toggleDebugMode,
    conversations,
  } = useChatStore();

  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when opened
  useEffect(() => {
    if (commandPaletteOpen) {
      setQuery('');
      setSelectedIdx(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [commandPaletteOpen]);

  // Commands list
  const commands: PaletteCommand[] = useMemo(() => {
    const cmds: PaletteCommand[] = [
      {
        id: 'new-chat',
        icon: <Plus size={16} />,
        label: 'New Chat',
        description: 'Start a new conversation',
        action: () => { setConversationId(null); close(); },
      },
      {
        id: 'profile',
        icon: <User size={16} />,
        label: 'View Profile',
        description: 'Open profile & memory manager',
        action: () => { setProfileModalOpen(true); close(); },
      },
      {
        id: 'debug',
        icon: <Bug size={16} />,
        label: debugMode ? 'Disable Debug Mode' : 'Enable Debug Mode',
        description: 'Show raw AI internals on messages',
        action: () => { toggleDebugMode(); close(); },
      },
    ];

    // Add recent conversations as jump targets
    conversations.slice(0, 8).forEach(c => {
      cmds.push({
        id: `conv-${c.id}`,
        icon: <MessageSquare size={16} />,
        label: c.title || 'Untitled',
        description: 'Switch to conversation',
        action: () => { setConversationId(c.id); close(); },
      });
    });

    return cmds;
  }, [conversations, debugMode, setConversationId, setProfileModalOpen, toggleDebugMode]);

  // Filter by query
  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    const q = query.toLowerCase();
    return commands.filter(
      c => c.label.toLowerCase().includes(q) || c.description?.toLowerCase().includes(q)
    );
  }, [commands, query]);

  // Clamp selection
  useEffect(() => {
    setSelectedIdx(i => Math.min(i, Math.max(0, filtered.length - 1)));
  }, [filtered]);

  const close = () => setCommandPaletteOpen(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIdx(i => Math.min(i + 1, filtered.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIdx(i => Math.max(i - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        filtered[selectedIdx]?.action();
        break;
      case 'Escape':
        close();
        break;
    }
  };

  if (!commandPaletteOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]" onClick={close}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-lg bg-sidebar-bg border border-sidebar-border rounded-xl
                    shadow-2xl overflow-hidden fade-in"
        onClick={e => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-sidebar-border">
          <Command size={16} className="text-sidebar-muted" />
          <input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command…"
            className="flex-1 bg-transparent text-sm text-white placeholder:text-sidebar-muted outline-none"
          />
          <button onClick={close} title="Close" className="text-sidebar-muted hover:text-white transition-colors">
            <X size={14} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto py-2">
          {filtered.length === 0 && (
            <p className="text-center text-sidebar-muted text-xs py-4">No matching commands</p>
          )}
          {filtered.map((cmd, i) => (
            <button
              key={cmd.id}
              onClick={cmd.action}
              onMouseEnter={() => setSelectedIdx(i)}
              className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors
                ${i === selectedIdx ? 'bg-sidebar-hover text-white' : 'text-sidebar-text hover:bg-sidebar-hover/50'}`}
            >
              <span className="text-sidebar-muted">{cmd.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate">{cmd.label}</div>
                {cmd.description && (
                  <div className="text-[10px] text-sidebar-muted truncate">{cmd.description}</div>
                )}
              </div>
              {i === selectedIdx && (
                <span className="text-[10px] text-sidebar-muted px-1.5 py-0.5 rounded bg-sidebar-bg border border-sidebar-border">
                  ↵
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Footer hint */}
        <div className="border-t border-sidebar-border px-4 py-2 flex items-center gap-4 text-[10px] text-sidebar-muted">
          <span><kbd className="px-1 py-0.5 rounded bg-sidebar-hover">↑↓</kbd> Navigate</span>
          <span><kbd className="px-1 py-0.5 rounded bg-sidebar-hover">↵</kbd> Select</span>
          <span><kbd className="px-1 py-0.5 rounded bg-sidebar-hover">Esc</kbd> Close</span>
        </div>
      </div>
    </div>
  );
}
