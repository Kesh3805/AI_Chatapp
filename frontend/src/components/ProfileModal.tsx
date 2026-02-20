import { useState } from 'react';
import { X, Plus, Trash2, Loader2 } from 'lucide-react';
import { useChatStore } from '../store';

export default function ProfileModal() {
  const {
    profileEntries,
    setProfileModalOpen,
    addProfileEntry,
    deleteProfileEntry,
  } = useChatStore();

  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [category, setCategory] = useState('general');
  const [adding, setAdding] = useState(false);

  const handleAdd = async () => {
    if (!key.trim() || !value.trim()) return;
    setAdding(true);
    await addProfileEntry(key.trim(), value.trim(), category);
    setKey('');
    setValue('');
    setCategory('general');
    setAdding(false);
  };

  const categories = ['personal', 'professional', 'preferences', 'health', 'education', 'other'];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-chat-bg border border-sidebar-border rounded-2xl shadow-2xl w-full max-w-lg max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-sidebar-border">
          <h2 className="text-lg font-semibold text-white">Profile & Memory</h2>
          <button
            onClick={() => setProfileModalOpen(false)}
            title="Close"
            className="p-1.5 rounded-lg hover:bg-sidebar-hover text-sidebar-muted
                       hover:text-white transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Entries list */}
        <div className="flex-1 overflow-y-auto p-6">
          {profileEntries.length === 0 ? (
            <p className="text-sidebar-muted text-sm text-center py-8">
              No profile data stored yet.  Share facts about yourself in conversation
              and they'll appear here automatically.
            </p>
          ) : (
            <div className="space-y-2">
              {profileEntries.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-center gap-3 p-3 rounded-lg bg-sidebar-bg group"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline gap-2">
                      <span className="text-xs font-medium text-accent">
                        {entry.key.replace(/_/g, ' ')}
                      </span>
                      <span className="text-[10px] text-sidebar-muted">{entry.category}</span>
                    </div>
                    <p className="text-sm text-gray-200 truncate">{entry.value}</p>
                  </div>
                  <button
                    onClick={() => deleteProfileEntry(entry.id)}
                    title="Delete entry"
                    className="p-1.5 rounded-lg opacity-0 group-hover:opacity-100
                               text-sidebar-muted hover:text-danger transition-all"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Add entry form */}
        <div className="border-t border-sidebar-border p-6">
          <h3 className="text-sm font-medium text-sidebar-text mb-3">Add manually</h3>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <input
              value={key}
              onChange={(e) => setKey(e.target.value)}
              placeholder="Key (e.g. name)"
              className="px-3 py-2 text-sm bg-input-bg border border-input-border rounded-lg
                         text-gray-200 placeholder:text-sidebar-muted outline-none
                         focus:border-accent/50"
            />
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              title="Category"
              aria-label="Category"
              className="px-3 py-2 text-sm bg-input-bg border border-input-border rounded-lg
                         text-gray-200 outline-none focus:border-accent/50"
            >
              {categories.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="Value"
              onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
              className="flex-1 px-3 py-2 text-sm bg-input-bg border border-input-border rounded-lg
                         text-gray-200 placeholder:text-sidebar-muted outline-none
                         focus:border-accent/50"
            />
            <button
              onClick={handleAdd}
              disabled={!key.trim() || !value.trim() || adding}
              className="px-4 py-2 text-sm rounded-lg bg-accent text-white
                         hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed
                         transition-colors flex items-center gap-1"
            >
              {adding ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
              Add
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
