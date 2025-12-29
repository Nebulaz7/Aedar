"use client";
import React, { useEffect, useRef, useState } from "react";
import { X, Mail, Calendar } from "lucide-react";
import { createBrowserClient } from "@supabase/ssr";
import Image from "next/image";

type AccountModalProps = {
  isOpen: boolean;
  onClose: () => void;
};

type UserInfo = {
  name: string;
  email: string;
  joined: string;
  profileImageUrl?: string;
};

const AccountModal: React.FC<AccountModalProps> = ({ isOpen, onClose }) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [avatarError, setAvatarError] = useState(false);

  const generateAvatarUrl = (name: string) => {
    const seed = encodeURIComponent(name.toLowerCase().replace(/\s+/g, ""));
    return `https://api.dicebear.com/7.x/avataaars/svg?seed=${seed}&backgroundColor=0ea5e9,0284c7,0369a1&radius=50`;
  };

  useEffect(() => {
    const fetchUserData = async () => {
      if (!isOpen) return;

      setIsLoading(true);
      setError(null);

      try {
        const supabase = createBrowserClient(
          process.env.NEXT_PUBLIC_SUPABASE_URL!,
          process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
        );
        const {
          data: { user },
          error: authError,
        } = await supabase.auth.getUser();

        if (authError) throw new Error(authError.message);
        if (!user) throw new Error("No user found");

        const userData: UserInfo = {
          name:
            user.user_metadata?.full_name || user.user_metadata?.name || "User",
          email: user.email || "No email provided",
          joined: formatDate(user.created_at),
          profileImageUrl:
            user.user_metadata?.avatar_url || user.user_metadata?.picture,
        };

        setUserInfo(userData);
      } catch (err) {
        console.error("Error fetching user data:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load user data"
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, [isOpen]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        modalRef.current &&
        !modalRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose]);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("keydown", handleEscape);
    }

    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((word) => word[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
      <div
        ref={modalRef}
        className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] w-full max-w-md p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-black text-black uppercase">Account</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 transition-colors"
            aria-label="Close"
          >
            <X size={20} className="text-black" />
          </button>
        </div>

        <div className="w-full h-1 bg-black mb-6"></div>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center py-12">
            <div className="flex gap-2 mb-4">
              <div
                className="w-3 h-3 bg-emerald-400 border-2 border-black animate-bounce"
                style={{ animationDelay: "0ms" }}
              />
              <div
                className="w-3 h-3 bg-yellow-300 border-2 border-black animate-bounce"
                style={{ animationDelay: "150ms" }}
              />
              <div
                className="w-3 h-3 bg-pink-400 border-2 border-black animate-bounce"
                style={{ animationDelay: "300ms" }}
              />
            </div>
            <p className="text-black/60 text-sm font-medium">
              Loading user data...
            </p>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-12">
            <div className="w-16 h-16 bg-red-100 border-3 border-black flex items-center justify-center mb-4">
              <X className="w-8 h-8 text-red-500" />
            </div>
            <p className="text-red-600 text-sm text-center mb-2 font-bold">
              Failed to load user data
            </p>
            <p className="text-black/50 text-xs text-center">{error}</p>
          </div>
        ) : userInfo ? (
          <>
            <div className="flex flex-col items-center mb-6">
              <div className="relative">
                {!avatarError ? (
                  <Image
                    src={
                      userInfo.profileImageUrl
                        ? userInfo.profileImageUrl
                        : generateAvatarUrl(userInfo.name)
                    }
                    alt={userInfo.name}
                    width={96}
                    height={96}
                    className="w-24 h-24 object-cover border-4 border-black"
                    onError={() => setAvatarError(true)}
                    unoptimized
                  />
                ) : (
                  <div className="w-24 h-24 bg-emerald-400 border-4 border-black flex items-center justify-center text-black text-2xl font-black">
                    {getInitials(userInfo.name)}
                  </div>
                )}
              </div>

              <div className="mt-4 text-center">
                <h3 className="text-lg font-black text-black">
                  {userInfo.name}
                </h3>
              </div>
            </div>

            <div className="space-y-3 mb-6">
              <div className="flex items-center gap-3 px-4 py-3 bg-gray-50 border-3 border-black">
                <Mail size={18} className="text-emerald-600" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-black/50 font-bold uppercase">
                    Email
                  </p>
                  <p className="text-sm text-black font-medium truncate">
                    {userInfo.email}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 px-4 py-3 bg-gray-50 border-3 border-black">
                <Calendar size={18} className="text-emerald-600" />
                <div>
                  <p className="text-xs text-black/50 font-bold uppercase">
                    Joined
                  </p>
                  <p className="text-sm text-black font-medium">
                    {userInfo.joined}
                  </p>
                </div>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};

export default AccountModal;
