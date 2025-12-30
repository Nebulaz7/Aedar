"use client";
import React, { useState, useEffect } from "react";
import ChatInput from "./ChatInput";
import RoadmapCard from "./RoadmapCard";
import ZeroState from "./ZeroState";
import ControlBar from "./ControlBar";
import LoadingState from "./LoadingState";
import { useRouter } from "next/navigation";

// Define types for the API response
type Resource = {
  type: string;
  title: string;
  link: string;
};

type Node = {
  id: string | number;
  title: string;
  description: string;
  duration?: string;
  resources: Resource[];
};

type Stage = {
  id: string | number;
  title: string;
  duration?: string;
  nodes: Node[];
};

type RoadmapData = Stage[] | string | null;

type ChatContainerProps = {
  userId: string;
  roadmapId?: string | null;
  userName?: string;
};

const ChatContainer: React.FC<ChatContainerProps> = ({
  userId,
  roadmapId,
  userName = "there",
}) => {
  const [loading, setLoading] = useState(false);
  const [roadmap, setRoadmap] = useState<RoadmapData>(null);
  const [roadmapTitle, setRoadmapTitle] = useState("");
  const [isLoaded, setIsLoaded] = useState(false);
  const [planningDepth, setPlanningDepth] = useState<
    "sprint" | "standard" | "architect"
  >("standard");
  const [autoSchedule, setAutoSchedule] = useState(false);
  const [isCalendarConnected, setIsCalendarConnected] = useState(false);
  const router = useRouter();
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://aedar.onrender.com";

  // Fetch user's calendar connection status from profile
  useEffect(() => {
    const fetchCalendarStatus = async () => {
      if (!userId) return;

      try {
        const res = await fetch(
          `${
            process.env.NEXT_PUBLIC_API_URL || "https://aedar.onrender.com"
          }/user/profile?userId=${userId}`
        );
        if (res.ok) {
          const profile = await res.json();
          setIsCalendarConnected(profile.calendar_connected || false);
        }
      } catch (err) {
        console.error("Error fetching calendar status:", err);
      }
    };

    fetchCalendarStatus();
  }, [userId]);

  // Load existing roadmap if roadmapId is provided
  useEffect(() => {
    const fetchRoadmap = async () => {
      if (!roadmapId) return;

      try {
        const res = await fetch(
          `${
            process.env.NEXT_PUBLIC_API_URL || "https://aedar.onrender.com"
          }/roadmap/${roadmapId}?userId=${userId}`
        );
        if (res.ok) {
          const data = await res.json();
          setRoadmap(data.roadmap);
          setRoadmapTitle(data.title || "Your Roadmap");
        } else {
          console.error("Error loading roadmap:", res.statusText);
          setRoadmap("Could not load roadmap. Please try again.");
        }
      } catch (err) {
        console.error("Error loading roadmap:", err);
        setRoadmap("Could not load roadmap. Please try again.");
      }
    };

    fetchRoadmap();
  }, [roadmapId, userId]);

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  // Handle sending messages to the API
  const handleSendMessage = async (message: string) => {
    setLoading(true);
    setRoadmapTitle(message);

    try {
     const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userMessage: message,
          planningDepth,
          modelType: planningDepth,
        }),
      });

      const data = await res.json();

      if (data.roadmap) {
        setRoadmap(data.roadmap);

        // Save the roadmap
        try {
         const saveRes = await fetch(`${API_BASE}/roadmap`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              userId,
              title: message,
              goal: message,
              roadmap: data.roadmap,
            }),
            }
          );

          if (saveRes.ok) {
            const savedRoadmap = await saveRes.json();
            router.push(`/roadmap/${savedRoadmap.id}`);
          }
        } catch (saveErr) {
          console.error("Error saving roadmap:", saveErr);
        }
      } else if (data.error) {
        setRoadmap(data.error);
      } else {
        setRoadmap("No roadmap received. Try another query.");
      }
    } catch (err) {
      console.error("Error:", err);
      setRoadmap("Failed to fetch roadmap. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  const handleAddToCalendar = async () => {
    // TODO: Implement calendar integration
    console.log("Adding to calendar...");
  };

  const handleModifyPlan = () => {
    // Clear roadmap to show input again
    setRoadmap(null);
  };

  return (
    <div className="min-h-screen w-full bg-[#f5f5f5] flex flex-col">
      {/* Control Bar */}
      <ControlBar
        onDepthChange={setPlanningDepth}
        onAutoScheduleChange={setAutoSchedule}
        isCalendarConnected={isCalendarConnected}
        user={{ id: userId }}
        onCalendarConnected={() => {
          console.log("Setting isCalendarConnected to true");
          setIsCalendarConnected(true);
        }}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center px-4 py-8 overflow-y-auto">
        {/* Decorative Elements */}
        <div className="fixed top-40 left-10 w-16 h-16 bg-emerald-400 border-3 border-black rotate-12 opacity-50 hidden xl:block" />
        <div className="fixed bottom-20 right-10 w-12 h-12 bg-yellow-300 border-3 border-black -rotate-6 opacity-50 hidden xl:block" />
        <div className="fixed top-1/3 right-20 w-8 h-8 bg-pink-400 border-3 border-black rounded-full opacity-50 hidden xl:block" />

        {/* Hero Text */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-black text-black mb-2 uppercase tracking-tight">
            Ready to execute,{" "}
            <span className="text-emerald-500">{userName}</span>?
          </h1>
          <p className="text-lg md:text-xl text-black/70 font-medium max-w-md mx-auto">
            Describe your idea, and Aedar will build the plan.
          </p>
        </div>

        {/* Loading State */}
        {loading && <LoadingState />}

        {/* Input Area - Always visible at top */}
        <div className="w-full max-w-3xl mt-auto pt-8 mb-auto">
          <ChatInput onSendMessage={handleSendMessage} isLoading={loading} />
        </div>

        {/* Roadmap Display */}
        {!loading && roadmap && typeof roadmap !== "string" && (
          <div className="w-full max-w-4xl mb-8">
            <RoadmapCard
              title={roadmapTitle}
              stages={roadmap}
              onAddToCalendar={handleAddToCalendar}
              onModifyPlan={handleModifyPlan}
            />
          </div>
        )}

        {/* Error State */}
        {!loading && typeof roadmap === "string" && (
          <div className="w-full max-w-2xl mb-8">
            <div className="bg-white border-4 border-black p-6 shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
              <p className="text-black font-medium text-center">
                Failed to fetch roadmap. Try again.
                {/* {roadmap} */}
              </p>
            </div>
          </div>
        )}

        {/* Zero State - Show when no roadmap */}
        {!loading && !roadmap && (
          <ZeroState
            userName={userName}
            onSuggestionClick={handleSuggestionClick}
          />
        )}
      </main>
    </div>
  );
};

export default ChatContainer;
