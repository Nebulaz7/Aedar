"use client";
import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Trash2,
  Globe,
  Lock,
  Calendar,
  Loader2,
  AlertCircle,
  Copy,
  Check,
  Home,
} from "lucide-react";
import RoadmapCard from "../../chat/components/RoadmapCard";
import { createBrowserClient } from "@supabase/ssr";

interface Resource {
  type: string;
  title: string;
  link: string;
}

interface RoadmapNode {
  id: string | number;
  title: string;
  description: string;
  duration?: string;
  resources: Resource[];
}

interface RoadmapStage {
  id: string | number;
  title: string;
  duration?: string;
  nodes: RoadmapNode[];
}

interface RoadmapData {
  id: string;
  user_id: string;
  title: string;
  goal: string;
  roadmap: RoadmapStage[];
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

export default function RoadmapPage() {
  const params = useParams();
  const router = useRouter();
  const roadmapId = params.id as string;

  const [roadmap, setRoadmap] = useState<RoadmapData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [userLoaded, setUserLoaded] = useState(false);
  const [isOwner, setIsOwner] = useState(false);
  const [isTogglingPublic, setIsTogglingPublic] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isAddingToCalendar, setIsAddingToCalendar] = useState(false);

  // Get current user
  useEffect(() => {
    const fetchUser = async () => {
      const supabase = createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
      );
      const {
        data: { user },
        error,
      } = await supabase.auth.getUser();

      if (!user || error) {
        console.error("❌ No user found.");
        setUserLoaded(true);
        return;
      }

      setUserId(user.id);
      setUserLoaded(true);
    };
    fetchUser();
  }, []);

  // Fetch roadmap data
  useEffect(() => {
    const fetchRoadmap = async () => {
      if (!roadmapId || !userLoaded) return;

      setLoading(true);
      setError(null);

      try {
        const baseUrl = `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"
        }/roadmap/${roadmapId}`;
        const url = userId ? `${baseUrl}?userId=${userId}` : baseUrl;

        const res = await fetch(url);

        if (!res.ok) {
          if (res.status === 404) {
            setError("Roadmap not found");
          } else if (res.status === 403) {
            setError("You don't have permission to view this roadmap");
          } else {
            setError("Failed to load roadmap");
          }
          return;
        }

        const data = await res.json();
        setRoadmap(data);
        setIsOwner(userId === data.user_id);
      } catch (err) {
        console.error("Error fetching roadmap:", err);
        setError("Failed to load roadmap. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchRoadmap();
  }, [roadmapId, userId, userLoaded]);

  // Update isOwner when userId changes
  useEffect(() => {
    if (roadmap && userId) {
      setIsOwner(userId === roadmap.user_id);
    }
  }, [userId, roadmap]);

  const handleTogglePublic = async () => {
    if (!roadmap || !userId) return;

    setIsTogglingPublic(true);
    try {
      const res = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"
        }/roadmap/${roadmapId}/public`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId,
            isPublic: !roadmap.is_public,
          }),
        }
      );

      if (res.ok) {
        setRoadmap({ ...roadmap, is_public: !roadmap.is_public });
      }
    } catch (err) {
      console.error("Error toggling public:", err);
    } finally {
      setIsTogglingPublic(false);
    }
  };

  const handleDelete = async () => {
    if (!roadmap || !userId) return;

    setIsDeleting(true);
    try {
      const res = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"
        }/roadmap/${roadmapId}?userId=${userId}`,
        { method: "DELETE" }
      );

      if (res.ok) {
        router.push("/chat");
      }
    } catch (err) {
      console.error("Error deleting roadmap:", err);
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handleAddToCalendar = async () => {
    if (!roadmap || !userId) {
      alert("Please sign in to add this roadmap to your calendar");
      return;
    }

    setIsAddingToCalendar(true);
    try {
      const res = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"
        }/mcp/calendar/execute`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId,
            roadmap: roadmap.roadmap,
            startDate: new Date().toISOString().split("T")[0], // Start from today
          }),
        }
      );

      const data = await res.json();

      if (data.success) {
        alert(`🎉 Added ${data.eventCount} events to your Google Calendar!`);
      } else if (data.error?.includes("not connected")) {
        alert("Please connect your Google Calendar first from the chat page.");
      } else {
        alert(data.error || "Failed to add to calendar. Please try again.");
      }
    } catch (err) {
      console.error("Error adding to calendar:", err);
      alert("Failed to add to calendar. Please try again.");
    } finally {
      setIsAddingToCalendar(false);
    }
  };

  const handleModifyPlan = () => {
    router.push(`/chat?roadmapId=${roadmapId}`);
  };

  const handleExport = () => {
    if (!roadmap) return;

    const dataStr = JSON.stringify(roadmap, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${roadmap.title
      .replace(/\s+/g, "-")
      .toLowerCase()}-roadmap.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const calculateTotalDuration = (stages: RoadmapStage[]) => {
    // Simple estimation - just count stages
    return `${stages.length} phases`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#f5f5f5] flex items-center justify-center">
        <div className="text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="w-4 h-4 bg-emerald-400 border-2 border-black animate-bounce" />
            <div
              className="w-4 h-4 bg-yellow-300 border-2 border-black animate-bounce"
              style={{ animationDelay: "0.1s" }}
            />
            <div
              className="w-4 h-4 bg-pink-400 border-2 border-black animate-bounce"
              style={{ animationDelay: "0.2s" }}
            />
          </div>
          <p className="font-bold text-black/60">Loading roadmap...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#f5f5f5] flex items-center justify-center p-6">
        <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-8 max-w-md text-center">
          <div className="w-16 h-16 bg-red-100 border-4 border-black mx-auto mb-4 flex items-center justify-center">
            <AlertCircle className="w-8 h-8 text-red-600" />
          </div>
          <h1 className="text-2xl font-black text-black mb-2">Oops!</h1>
          <p className="text-black/60 font-medium mb-6">{error}</p>
          <Link
            href="/chat"
            className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-400 border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
          >
            <Home className="w-5 h-5" />
            Go to Chat
          </Link>
        </div>
      </div>
    );
  }

  if (!roadmap) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[#f5f5f5]">
      {/* Header */}
      <header className="bg-white border-b-4 border-black sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <Link
                href="/chat"
                className="p-2 bg-white cursor-pointer text-black border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
              >
                <ArrowLeft className="w-5 h-5" />
              </Link>
              <div>
                <h1 className="font-black text-xl text-black truncate max-w-md">
                  {roadmap.title}
                </h1>
                <div className="flex items-center gap-3 text-sm text-black/50">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-4 h-4" />
                    {formatDate(roadmap.created_at)}
                  </span>
                  <span className="flex cursor-pointer items-center gap-1">
                    {roadmap.is_public ? (
                      <>
                        <Globe className="w-4 h-4 text-emerald-600" />
                        <span className="text-emerald-600 font-medium">
                          Public
                        </span>
                      </>
                    ) : (
                      <>
                        <Lock className="w-4 h-4 text-black" />
                        <span className="text-black">Private</span>
                      </>
                    )}
                  </span>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2">
              {/* Copy Link */}
              <button
                onClick={handleCopyLink}
                className="flex items-center cursor-pointer text-black gap-2 px-4 py-2 bg-white border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
                title="Copy link"
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4 text-emerald-600" />
                    <span className="hidden sm:inline">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    <span className="hidden sm:inline">Share</span>
                  </>
                )}
              </button>

              {/* Owner Actions */}
              {isOwner && (
                <>
                  {/* Toggle Public */}
                  <button
                    onClick={handleTogglePublic}
                    disabled={isTogglingPublic}
                    className={`flex items-center gap-2 px-4 py-2 border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all disabled:opacity-50 ${
                      roadmap.is_public ? "bg-emerald-400" : "bg-white"
                    }`}
                    title={roadmap.is_public ? "Make private" : "Make public"}
                  >
                    {isTogglingPublic ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : roadmap.is_public ? (
                      <Globe className="w-4 h-4" />
                    ) : (
                      <Lock className="w-4 h-4" />
                    )}
                    <span className="hidden sm:inline">
                      {roadmap.is_public ? "Public" : "Private"}
                    </span>
                  </button>

                  {/* Delete */}
                  <button
                    onClick={() => setShowDeleteConfirm(true)}
                    className="p-2 cursor-pointer bg-red-100 border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all hover:bg-red-200"
                    title="Delete roadmap"
                  >
                    <Trash2 className="w-4 h-4 text-red-600" />
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-6 py-8">
        {/* Goal Section */}
        {roadmap.goal && (
          <div className="mb-8 bg-emerald-50 border-4 border-black p-6 shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]">
            <h2 className="font-black text-sm text-black/50 uppercase tracking-wide mb-2">
              Project Goal
            </h2>
            <p className="text-lg font-medium text-black">{roadmap.goal}</p>
          </div>
        )}

        {/* Roadmap Card */}
        {Array.isArray(roadmap.roadmap) && roadmap.roadmap.length > 0 ? (
          <RoadmapCard
            title={roadmap.title}
            stages={roadmap.roadmap}
            totalDuration={calculateTotalDuration(roadmap.roadmap)}
            onAddToCalendar={handleAddToCalendar}
            onModifyPlan={handleModifyPlan}
            onExport={handleExport}
            isAddingToCalendar={isAddingToCalendar}
          />
        ) : (
          <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-12 text-center">
            <p className="text-black/60 font-medium">
              No roadmap data available.
            </p>
          </div>
        )}
      </main>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
          <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] w-full max-w-md p-6">
            <div className="w-16 h-16 bg-red-100 border-4 border-black mx-auto mb-4 flex items-center justify-center">
              <Trash2 className="w-8 h-8 text-red-600" />
            </div>
            <h2 className="text-xl font-black text-black text-center mb-2">
              Delete Roadmap?
            </h2>
            <p className="text-black/60 font-medium text-center mb-6">
              This action cannot be undone. Your roadmap will be permanently
              deleted.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="flex-1 px-4 py-3 bg-white border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="flex-1 px-4 py-3 bg-red-500 text-white border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all disabled:opacity-50"
              >
                {isDeleting ? (
                  <Loader2 className="w-5 h-5 animate-spin mx-auto" />
                ) : (
                  "Delete"
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
