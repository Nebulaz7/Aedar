"use client";
import React, { useState, useCallback, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  MessageSquare,
  LogOut,
  User,
  Settings,
  Menu,
  Clock,
} from "lucide-react";
import SignoutModal from "./SignoutModal";
import AccountModal from "./AccountModal";
import SettingsModal from "./SettingsModal";
import Link from "next/link";
import { createBrowserClient } from "@supabase/ssr";

type SidenavProps = {
  // initialChatHistory is reserved for future use when static chat history data is provided
  initialChatHistory?: { id: string; title: string; date: string }[];
  currentRoadmapId?: string;
};

const Sidenav: React.FC<SidenavProps> = ({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  initialChatHistory = [],
  currentRoadmapId,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [showSignoutModal, setShowSignoutModal] = useState(false);
  const [showAccountModal, setShowAccountModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [roadmapHistory, setRoadmapHistory] = useState<
    { id: string; title: string; created_at: string; user_id: string }[]
  >([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchRoadmaps = async () => {
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
        setIsLoading(false);
        return;
      }

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
      if (!apiUrl) {
        console.error("❌ Missing NEXT_PUBLIC_API_URL in .env.local");
        setIsLoading(false);
        return;
      }

      try {
        const res = await fetch(`${apiUrl}/roadmap/user/${user.id}`);
        if (res.ok) {
          const roadmaps = await res.json();
          setRoadmapHistory(roadmaps);
        } else {
          console.error("❌ Failed to fetch roadmaps:", res.statusText);
        }
      } catch (err) {
        console.error("❌ Failed to fetch roadmaps:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchRoadmaps();
  }, []);

  // 🔄 Handle mobile screen resizing
  useEffect(() => {
    const checkMobile = () => {
      if (window.innerWidth >= 1024) {
        setIsMobileOpen(false);
      }
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const handleMouseEnter = useCallback(() => {
    if (window.innerWidth >= 1024) setIsExpanded(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (window.innerWidth >= 1024) setIsExpanded(false);
  }, []);

  const toggleExpanded = () => setIsExpanded(!isExpanded);
  const toggleMobileSidenav = () => {
    setIsMobileOpen(!isMobileOpen);
    if (!isMobileOpen) setIsExpanded(true);
  };

  // Helper function to format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) return "Today";
    if (diffDays === 2) return "Yesterday";
    if (diffDays <= 7) return `${diffDays - 1} days ago`;
    return date.toLocaleDateString();
  };

  // Helper function to truncate title
  const truncateTitle = (title: string, maxLength: number = 30) => {
    if (title.length <= maxLength) return title;
    return title.substring(0, maxLength) + "...";
  };

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={toggleMobileSidenav}
        className="lg:hidden fixed top-4 left-4 z-50 p-3 bg-white border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
        aria-label="Toggle navigation menu"
      >
        <Menu size={20} className="text-black" />
      </button>

      {/* Sidebar */}
      <div
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className={`fixed top-0 left-0 h-full z-40 transition-all duration-300 
          ${isExpanded ? "w-64" : "w-16"} 
          ${
            isMobileOpen
              ? "translate-x-0"
              : "-translate-x-full lg:translate-x-0"
          }
          bg-white border-r-4 border-black
          ${isMobileOpen ? "w-3/4 sm:w-64" : ""}`}
      >
        <div className="flex flex-col h-full">
          {/* Expand/Collapse Button */}
          <button
            onClick={toggleExpanded}
            className="hidden lg:flex self-end p-2 m-2 hover:bg-emerald-100 text-black transition-colors"
          >
            {isExpanded ? (
              <ChevronLeft size={16} />
            ) : (
              <ChevronRight size={16} />
            )}
          </button>

          {/* Close Button on Mobile */}
          {isMobileOpen && (
            <button
              onClick={toggleMobileSidenav}
              className="lg:hidden self-end p-2 m-2 hover:bg-emerald-100 text-black transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
          )}

          {/* New Roadmap Button */}
          <div className="px-3 py-2">
            <Link
              href="/chat"
              className={`flex items-center px-3 py-2 bg-emerald-400 border-3 border-black
                shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]
                hover:translate-x-0.5 hover:translate-y-[2px] transition-all font-bold
                ${isExpanded ? "justify-start" : "justify-center"}`}
            >
              <MessageSquare size={18} className="text-black" />
              {isExpanded && (
                <span className="ml-3 text-sm text-black">New Roadmap</span>
              )}
            </Link>
          </div>

          {/* Roadmap History */}
          <div className="flex-1 overflow-y-auto">
            <div className={`px-3 py-2 ${isExpanded ? "block" : "hidden"}`}>
              <h3 className="text-xs font-bold text-black/50 uppercase tracking-wider">
                Roadmap History
              </h3>
            </div>

            {!isExpanded && (
              <div className="flex justify-center py-2">
                <Clock size={20} className="text-black/40" />
              </div>
            )}

            <div className="space-y-1 px-2">
              {isLoading ? (
                <div
                  className={`text-black/50 text-xs italic px-3 ${
                    isExpanded ? "block" : "hidden"
                  }`}
                >
                  Loading...
                </div>
              ) : roadmapHistory.length > 0 ? (
                roadmapHistory.map((roadmap) => (
                  <Link
                    key={roadmap.id}
                    href={`/roadmap/${roadmap.id}`}
                    className={`flex items-center px-2 py-2 w-full text-left
                      transition-all duration-150 group border-2
                      ${
                        currentRoadmapId === roadmap.id
                          ? "bg-emerald-100 border-black text-black font-bold"
                          : "bg-white border-transparent hover:border-black hover:bg-gray-50 text-black"
                      }
                      ${isExpanded ? "justify-start" : "justify-center"}`}
                  >
                    <MessageSquare
                      size={16}
                      className={`${
                        currentRoadmapId === roadmap.id
                          ? "text-emerald-600"
                          : "text-black/40 group-hover:text-black"
                      }`}
                    />
                    {isExpanded && (
                      <div className="ml-3 flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">
                          {truncateTitle(roadmap.title)}
                        </div>
                        <div className="text-xs text-black/50 mt-0.5">
                          {formatDate(roadmap.created_at)}
                        </div>
                      </div>
                    )}
                  </Link>
                ))
              ) : (
                <div
                  className={`px-2 py-2 text-black/50 text-xs italic ${
                    isExpanded ? "block" : "hidden"
                  }`}
                >
                  No roadmaps yet
                </div>
              )}
            </div>
          </div>

          {/* User Actions */}
          <div className="p-2 border-t-2 border-black/10">
            <button
              onClick={() => setShowAccountModal(true)}
              className={`flex cursor-pointer items-center px-2 py-2 w-full text-left
                text-black hover:bg-emerald-100 transition-colors mb-1
                ${isExpanded ? "justify-start" : "justify-center"}`}
            >
              <User size={18} className="text-black/60" />
              {isExpanded && (
                <span className="ml-3 text-sm font-medium">Account</span>
              )}
            </button>

            <button
              onClick={() => setShowSettingsModal(true)}
              className={`flex cursor-pointer items-center px-2 py-2 w-full text-left
                text-black hover:bg-emerald-100 transition-colors mb-1
                ${isExpanded ? "justify-start" : "justify-center"}`}
            >
              <Settings size={18} className="text-black/60" />
              {isExpanded && (
                <span className="ml-3 text-sm font-medium">Settings</span>
              )}
            </button>

            <button
              onClick={() => setShowSignoutModal(true)}
              className={`flex cursor-pointer items-center px-2 py-2 w-full text-left
                text-red-600 hover:bg-red-100 transition-colors
                ${isExpanded ? "justify-start" : "justify-center"}`}
            >
              <LogOut size={18} className="text-red-500" />
              {isExpanded && (
                <span className="ml-3 text-sm font-medium">Sign out</span>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Modals */}
      {showSignoutModal && (
        <SignoutModal
          isOpen={showSignoutModal}
          onClose={() => setShowSignoutModal(false)}
          onSignout={() => {
            console.log("User signed out");
            setShowSignoutModal(false);
          }}
        />
      )}

      {showAccountModal && (
        <AccountModal
          isOpen={showAccountModal}
          onClose={() => setShowAccountModal(false)}
        />
      )}

      {showSettingsModal && (
        <SettingsModal
          isOpen={showSettingsModal}
          onClose={() => setShowSettingsModal(false)}
        />
      )}

      {/* Mobile backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-30 lg:hidden"
          onClick={toggleMobileSidenav}
          aria-hidden="true"
        />
      )}
    </>
  );
};

export default Sidenav;
