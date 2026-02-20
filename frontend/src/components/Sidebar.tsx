import { useState } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  User,
  Pencil,
  Check,
  X,
  Bug,
  Command,
  Brain,
  Shield,
  Database,
  Code2,
  BookOpen,
  Globe,
  HelpCircle,
  Sparkles,
} from 'lucide-react';
import { useChatStore } from '../store';

/* ── Category icon detection from conversation title ───────────────────── */

const categoryRules: { test: RegExp; icon: React.ReactNode; color: string }[] = [
  { test: /\b(privacy|private|secure|security|delete|forget)\b/i, icon: <Shield size={12} />, color: 'text-red-400' },
  { test: /\b(machine learning|ML|neural|model|train|AI|LLM|GPT)\b/i, icon: <Brain size={12} />, color: 'text-purple-400' },
  { test: /\b(database|SQL|postgres|pgvector|query|table)\b/i, icon: <Database size={12} />, color: 'text-cyan-400' },
  { test: /\b(code|python|javascript|typescript|react|function|class|API)\b/i, icon: <Code2 size={12} />, color: 'text-green-400' },
  { test: /\b(RAG|retrieval|embedding|FAISS|vector|chunk)\b/i, icon: <BookOpen size={12} />, color: 'text-amber-400' },
  { test: /\b(web|HTTP|URL|fetch|deploy|docker)\b/i, icon: <Globe size={12} />, color: 'text-blue-400' },
  { test: /\b(help|how|what|why|explain)\b/i, icon: <HelpCircle size={12} />, color: 'text-gray-400' },
];

function getCategoryIcon(title: string): { icon: React.ReactNode; color: string } {
  for (const rule of categoryRules) {
    if (rule.test.test(title)) return { icon: rule.icon, color: rule.color };
  }
  return { icon: <MessageSquare size={12} />, color: 'text-sidebar-muted' };
}

export default function Sidebar() {
  const {
    conversations,
    conversationId,
    setConversationId,
    removeConversation,
    renameConversation,
    toggleSidebar,
    setProfileModalOpen,
    debugMode,
    toggleDebugMode,
    setCommandPaletteOpen,
  } = useChatStore();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const handleNew = () => setConversationId(null);

  const startEdit = (id: string, title: string) => {
    setEditingId(id);
    setEditTitle(title);
  };

  const confirmEdit = async () => {
    if (editingId && editTitle.trim()) {
      await renameConversation(editingId, editTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <div className="flex flex-col h-full bg-sidebar-bg border-r border-sidebar-border">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-sidebar-border">
        <button
          onClick={handleNew}
          className="flex items-center gap-2 px-3 py-2 text-sm text-sidebar-text rounded-lg
                     hover:bg-sidebar-hover transition-colors flex-1"
        >
          <Plus size={16} />
          New Chat
        </button>
        <button
          onClick={toggleSidebar}
          title="Close sidebar"
          className="p-2 text-sidebar-muted hover:text-sidebar-text rounded-lg
                     hover:bg-sidebar-hover transition-colors"
        >
          <PanelLeftClose size={16} />
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2">
        {conversations.length === 0 && (
          <p className="text-sidebar-muted text-xs text-center py-4">
            No conversations yet
          </p>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`group flex items-center gap-2 mx-2 px-3 py-2.5 rounded-lg cursor-pointer
              transition-colors text-sm ${
                c.id === conversationId
                  ? 'bg-sidebar-active text-white'
                  : 'text-sidebar-text hover:bg-sidebar-hover'
              }`}
            onClick={() => {
              if (editingId !== c.id) setConversationId(c.id);
            }}
          >
            {(() => {
              const cat = getCategoryIcon(c.title);
              return <span className={`flex-shrink-0 ${cat.color}`}>{cat.icon}</span>;
            })()}

            {editingId === c.id ? (
              <div className="flex items-center gap-1 flex-1 min-w-0">
                <input
                  title="Rename conversation"
                  placeholder="Conversation name"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') confirmEdit();
                    if (e.key === 'Escape') setEditingId(null);
                  }}
                  className="flex-1 bg-transparent border-b border-accent outline-none text-sm min-w-0"
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                />
                <button onClick={(e) => { e.stopPropagation(); confirmEdit(); }}
                  title="Confirm" className="p-0.5 hover:text-accent"><Check size={12} /></button>
                <button onClick={(e) => { e.stopPropagation(); setEditingId(null); }}
                  title="Cancel" className="p-0.5 hover:text-danger"><X size={12} /></button>
              </div>
            ) : (
              <>
                <span className="truncate flex-1">{c.title}</span>
                <div className="hidden group-hover:flex items-center gap-0.5">
                  <button
                    title="Rename"
                    onClick={(e) => { e.stopPropagation(); startEdit(c.id, c.title); }}
                    className="p-1 rounded hover:bg-sidebar-bg text-sidebar-muted hover:text-sidebar-text"
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    title="Delete"
                    onClick={(e) => { e.stopPropagation(); removeConversation(c.id); }}
                    className="p-1 rounded hover:bg-sidebar-bg text-sidebar-muted hover:text-danger"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </>
            )}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="border-t border-sidebar-border p-3 space-y-1">
        <button
          onClick={() => setProfileModalOpen(true)}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm text-sidebar-muted
                     rounded-lg hover:bg-sidebar-hover hover:text-sidebar-text transition-colors"
        >
          <User size={16} />
          Profile & Memory
        </button>
        <button
          onClick={toggleDebugMode}
          className={`flex items-center gap-2 w-full px-3 py-2 text-sm rounded-lg
                     hover:bg-sidebar-hover transition-colors
                     ${debugMode ? 'text-yellow-400' : 'text-sidebar-muted hover:text-sidebar-text'}`}
        >
          <Bug size={16} />
          Debug Mode
          {debugMode && <span className="ml-auto text-[10px] bg-yellow-500/20 px-1.5 py-0.5 rounded-full">ON</span>}
        </button>
        <button
          onClick={() => setCommandPaletteOpen(true)}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm text-sidebar-muted
                     rounded-lg hover:bg-sidebar-hover hover:text-sidebar-text transition-colors"
        >
          <Command size={16} />
          Commands
          <span className="ml-auto text-[10px] text-sidebar-muted/60">Ctrl+K</span>
        </button>
      </div>
    </div>
  );
}
