"use client";
import React from "react";
import { Rocket, BookOpen, Clock, ArrowRight } from "lucide-react";

interface QuickSuggestion {
  icon: React.ReactNode;
  title: string;
  subtext: string;
  prompt: string;
}

interface ZeroStateProps {
  userName?: string;
  onSuggestionClick: (prompt: string) => void;
}

const ZeroState: React.FC<ZeroStateProps> = ({
  userName = "there",
  onSuggestionClick,
}) => {
  const suggestions: QuickSuggestion[] = [
    {
      icon: <Rocket className="w-6 h-6" />,
      title: "Launch SaaS MVP",
      subtext: "4-week timeline • Backend + Frontend",
      prompt:
        "Build a SaaS MVP with user authentication, payment integration, and a dashboard. Timeline: 4 weeks.",
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Learn a new stack",
      subtext: "Study plan • Evenings & Weekends",
      prompt:
        "Create a learning roadmap for mastering React and Node.js, studying only on evenings and weekends.",
    },
    {
      icon: <Clock className="w-6 h-6" />,
      title: "2-Day Hackathon",
      subtext: "High intensity • Hourly breakdown",
      prompt:
        "Plan a 2-day hackathon project to build a real-time chat application with hourly milestones.",
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 py-12">
      {/* Decorative shapes */}
      <div className="absolute top-32 left-20 w-12 h-12 bg-emerald-400 border-3 border-black rotate-12 hidden lg:block" />
      <div className="absolute top-48 right-24 w-8 h-8 bg-yellow-300 border-3 border-black -rotate-6 hidden lg:block" />
      <div className="absolute bottom-32 left-32 w-6 h-6 bg-pink-400 border-3 border-black rounded-full hidden lg:block" />

      {/* Quick Suggestions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl w-full">
        {suggestions.map((suggestion, index) => (
          <button
            key={index}
            onClick={() => onSuggestionClick(suggestion.prompt)}
            className="group bg-white border-4 border-black p-6 shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[3px] hover:translate-y-[3px] transition-all duration-150 text-left"
          >
            <div className="flex items-start gap-4">
              <div className="p-3 bg-emerald-400 border-3 border-black group-hover:bg-emerald-500 transition-colors">
                {suggestion.icon}
              </div>
              <div className="flex-1">
                <h3 className="font-black text-lg text-black mb-1">
                  {suggestion.title}
                </h3>
                <p className="text-sm text-black/60 font-medium">
                  {suggestion.subtext}
                </p>
              </div>
              <ArrowRight className="w-5 h-5 text-black/40 group-hover:text-black group-hover:translate-x-1 transition-all" />
            </div>
          </button>
        ))}
      </div>

      {/* Tip */}
      <div className="mt-12 flex items-center gap-3 text-sm text-black/50 font-medium">
        <div className="w-3 h-3 bg-emerald-400 border-2 border-black rounded-full animate-pulse" />
        <span>Start typing to begin your personalized journey</span>
      </div>
    </div>
  );
};

export default ZeroState;
