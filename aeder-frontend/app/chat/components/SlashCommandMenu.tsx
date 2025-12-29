"use client";
import React from "react";
import { Map, Calendar, Sparkles } from "lucide-react";

interface SlashCommand {
  command: string;
  description: string;
  icon: React.ReactNode;
}

interface SlashCommandMenuProps {
  isOpen: boolean;
  onSelect: (command: string) => void;
  onClose: () => void;
  filterText?: string;
}

const commands: SlashCommand[] = [
  {
    command: "/roadmap",
    description: "Generate a new technical plan",
    icon: <Map className="w-4 h-4" />,
  },
  {
    command: "/schedule",
    description: "Move current plan to Calendar",
    icon: <Calendar className="w-4 h-4" />,
  },
  {
    command: "/refine",
    description: "Add more detail to the last step",
    icon: <Sparkles className="w-4 h-4" />,
  },
];

const SlashCommandMenu: React.FC<SlashCommandMenuProps> = ({
  isOpen,
  onSelect,
  onClose,
  filterText = "",
}) => {
  if (!isOpen) return null;

  const filteredCommands = commands.filter((cmd) =>
    cmd.command.toLowerCase().includes(filterText.toLowerCase())
  );

  if (filteredCommands.length === 0) return null;

  return (
    <div className="absolute bottom-full left-0 mb-2 w-72 bg-white border-3 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] z-50">
      <div className="p-2 border-b-2 border-black bg-gray-50">
        <span className="text-xs font-bold text-black/50 uppercase tracking-wide">
          Commands
        </span>
      </div>
      <div className="py-1">
        {filteredCommands.map((cmd) => (
          <button
            key={cmd.command}
            onClick={() => {
              onSelect(cmd.command);
              onClose();
            }}
            className="w-full flex items-center gap-3 px-4 py-3 hover:bg-emerald-100 transition-colors text-left"
          >
            <div className="p-2 bg-emerald-400 border-2 border-black">
              {cmd.icon}
            </div>
            <div>
              <span className="font-bold text-black block">{cmd.command}</span>
              <span className="text-xs text-black/60">{cmd.description}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default SlashCommandMenu;
