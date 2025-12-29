"use client";
import React, { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";

const loadingMessages = [
  "Analyzing requirements...",
  "Breaking down architecture...",
  "Estimating time for tasks...",
  "Mapping dependencies...",
  "Generating milestones...",
  "Optimizing timeline...",
  "Finding empty slots in your schedule...",
];

interface LoadingStateProps {
  message?: string;
}

const LoadingState: React.FC<LoadingStateProps> = ({ message }) => {
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);

  useEffect(() => {
    if (message) return; // If custom message provided, don't cycle

    const interval = setInterval(() => {
      setCurrentMessageIndex((prev) => (prev + 1) % loadingMessages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, [message]);

  const displayMessage = message || loadingMessages[currentMessageIndex];

  return (
    <div className="flex flex-col items-center justify-center py-12">
      <div className="bg-white border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
        {/* Animated blocks */}
        <div className="flex items-center justify-center gap-2 mb-6">
          <div
            className="w-4 h-4 bg-emerald-400 border-2 border-black animate-bounce"
            style={{ animationDelay: "0ms" }}
          />
          <div
            className="w-4 h-4 bg-yellow-300 border-2 border-black animate-bounce"
            style={{ animationDelay: "150ms" }}
          />
          <div
            className="w-4 h-4 bg-pink-400 border-2 border-black animate-bounce"
            style={{ animationDelay: "300ms" }}
          />
          <div
            className="w-4 h-4 bg-emerald-400 border-2 border-black animate-bounce"
            style={{ animationDelay: "450ms" }}
          />
        </div>

        {/* Loading text */}
        <div className="text-center">
          <p className="font-bold text-black text-lg animate-pulse">
            {displayMessage}
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoadingState;
