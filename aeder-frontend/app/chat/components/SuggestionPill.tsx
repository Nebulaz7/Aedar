"use client";
import React from "react";
import { ArrowRight } from "lucide-react";

interface SuggestionPillProps {
  text: string;
  onClick: (text: string) => void;
  index: number;
  isLoaded: boolean;
}

const SuggestionPill: React.FC<SuggestionPillProps> = ({
  text,
  onClick,
  index,
  isLoaded,
}) => {
  return (
    <button
      onClick={() => onClick(text)}
      className={`
        group flex items-center gap-2 px-5 py-3
        bg-white border-3 border-black
        font-bold text-sm text-black
        shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]
        hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]
        hover:translate-x-0.5 hover:translate-y-[2px]
        hover:bg-emerald-100
        transition-all duration-150
        ${isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}
      `}
      style={{
        transitionDelay: `${index * 100}ms`,
      }}
    >
      <span>{text}</span>
      <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
    </button>
  );
};

export default SuggestionPill;
