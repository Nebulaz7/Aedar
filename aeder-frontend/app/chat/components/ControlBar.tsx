"use client";
import React, { useState } from "react";
import {
  Zap,
  Layers,
  Building2,
  Calendar,
  CalendarCheck,
  ChevronDown,
} from "lucide-react";
import Link from "next/link";

type PlanningDepth = "sprint" | "standard" | "architect";

interface User {
  id: string;
}

interface Profile {
  calendar_connected?: boolean;
}

interface ControlBarProps {
  onDepthChange?: (depth: PlanningDepth) => void;
  onAutoScheduleChange?: (enabled: boolean) => void;
  isCalendarConnected?: boolean;
  user?: User | null;
  profile?: Profile | null;
  onCalendarConnected?: () => void;
}

const ControlBar: React.FC<ControlBarProps> = ({
  onDepthChange,
  onAutoScheduleChange,
  isCalendarConnected = false,
  user,
  profile,
  onCalendarConnected,
}) => {
  const [depth, setDepth] = useState<PlanningDepth>("standard");
  const [autoSchedule, setAutoSchedule] = useState(false);
  const [showDepthDropdown, setShowDepthDropdown] = useState(false);
  const [hoveredDepth, setHoveredDepth] = useState<PlanningDepth | null>(null);

  const handleConnectCalendar = (userId: string) => {
    // Opens Google's consent screen in a popup
    const width = 500;
    const height = 600;
    const left = window.screen.width / 2 - width / 2;
    const top = window.screen.height / 2 - height / 2;

    const authWindow = window.open(
      `${process.env.NEXT_PUBLIC_API_URL}/mcp/calendar/auth?userId=${userId}`,
      'calendar-auth',
      `width=${width},height=${height},left=${left},top=${top}`
    );

    const handleMessage = (event: MessageEvent) => {
      // Only process calendar_connected messages
      if (event.data?.type === 'calendar_connected' && event.data?.success) {
        console.log('Calendar connected successfully!');
        onCalendarConnected?.();
        authWindow?.close();
        window.removeEventListener('message', handleMessage);
      }
    };

    window.addEventListener('message', handleMessage);
  };

  const depthOptions: {
    value: PlanningDepth;
    label: string;
    icon: React.ReactNode;
    tooltip: string;
  }[] = [
    {
      value: "sprint",
      label: "Sprint",
      icon: <Zap className="w-4 h-4 text-black" />,
      tooltip: "Best for simple tasks. Generates 3-5 key milestones.",
    },
    {
      value: "standard",
      label: "Standard",
      icon: <Layers className="w-4 h-4 text-black" />,
      tooltip: "Default. Covers architecture, dev, and launch phases.",
    },
    {
      value: "architect",
      label: "Architect",
      icon: <Building2 className="w-4 h-4 text-black" />,
      tooltip:
        "Detailed breakdown. Includes risk analysis, tech stack choices, and QA steps.",
    },
  ];

  const handleDepthChange = (newDepth: PlanningDepth) => {
    setDepth(newDepth);
    setShowDepthDropdown(false);
    onDepthChange?.(newDepth);
  };

  const handleAutoScheduleToggle = () => {
    const newValue = !autoSchedule;
    setAutoSchedule(newValue);
    onAutoScheduleChange?.(newValue);
  };

  const currentDepthOption = depthOptions.find((opt) => opt.value === depth);

  return (
    <div className="w-full bg-white border-b-4 border-black px-4 py-3">
      <div className="max-w-4xl mx-auto flex flex-wrap items-center justify-between gap-4">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <Link
            href="/"
            className="text-2xl font-black uppercase tracking-tight"
          >
            <span className="bg-linear-to-r from-emerald-500 to-black text-transparent bg-clip-text">
              Aedar
            </span>
          </Link>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Planning Depth Selector */}
          <div className="relative">
            <label className="text-xs font-bold text-black/60 uppercase tracking-wide block mb-1">
              Planning Depth
            </label>
            <button
              onClick={() => setShowDepthDropdown(!showDepthDropdown)}
              className="flex text-black cursor-pointer items-center gap-2 px-4 py-2 bg-white border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-0.5 transition-all font-bold text-sm"
            >
              {currentDepthOption?.icon}
              <span>{currentDepthOption?.label}</span>
              <ChevronDown
                className={`w-4 h-4 transition-transform ${
                  showDepthDropdown ? "rotate-180" : ""
                }`}
              />
            </button>

            {/* Dropdown */}
            {showDepthDropdown && (
              <div className="absolute top-full cursor-pointer left-0 mt-2 w-56 bg-white border-3 border-black shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] z-50">
                {depthOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => handleDepthChange(option.value)}
                    onMouseEnter={() => setHoveredDepth(option.value)}
                    onMouseLeave={() => setHoveredDepth(null)}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-left font-bold text-sm transition-colors ${
                      depth === option.value
                        ? "bg-emerald-400 text-black"
                        : "hover:bg-gray-100"
                    }`}
                  >
                    {option.icon}
                    <div className="flex-1">
                      <span className="block text-black">{option.label}</span>
                      {hoveredDepth === option.value && (
                        <span className="text-xs font-normal text-black/60 mt-1 block">
                          {option.tooltip}
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Auto-Schedule Toggle */}
          <div className="flex flex-col">
            <label className="text-xs font-bold text-black/60 uppercase tracking-wide mb-1">
              Auto-Schedule
            </label>
            <button
              onClick={handleAutoScheduleToggle}
              className={`relative text-black cursor-pointer flex items-center gap-2 px-4 py-2 border-3 border-black font-bold text-sm transition-all ${
                autoSchedule
                  ? "bg-emerald-400 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] translate-x-0.5 translate-y-0.5"
                  : "bg-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-0.5"
              }`}
            >
              <div
                className={`w-4 h-4 border-2 border-black flex items-center justify-center ${
                  autoSchedule ? "bg-black" : "bg-white"
                }`}
              >
                {autoSchedule && (
                  <svg
                    className="w-3 h-3 text-white"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                )}
              </div>
              <span>{autoSchedule ? "Auto-book" : "Review first"}</span>
            </button>
          </div>

          {/* Calendar Status */}
          <div className="flex flex-col">
            <label className="text-xs font-bold text-black/60 uppercase tracking-wide mb-1">
              Calendar
            </label>
            {isCalendarConnected ? (
              <div className="flex items-center gap-2 px-4 py-2 bg-emerald-100 border-3 border-black font-bold text-sm">
                <div className="w-3 h-3 bg-emerald-500 rounded-full border-2 border-black" />
                <CalendarCheck className="w-4 h-4" />
                <span>Connected</span>
              </div>
            ) : (
              <button
                onClick={() => user?.id && handleConnectCalendar(user.id)}
                disabled={!user?.id}
                className="flex items-center text-black gap-2 px-4 py-2 cursor-pointer bg-yellow-300 border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-0.5 transition-all font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Calendar className="w-4 h-4" />
                {profile?.calendar_connected ? '✅Connected' : 'Connect'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlBar;
