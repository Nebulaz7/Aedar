"use client";
import React, { useEffect, useRef } from "react";
import { createBrowserClient } from "@supabase/ssr";
import { X } from "lucide-react";

type SignoutModalProps = {
  isOpen: boolean;
  onClose: () => void;
  // Callback for parent component to handle post-signout logic
  onSignout?: () => void;
};

const supabase = createBrowserClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

const SignoutModal: React.FC<SignoutModalProps> = ({
  isOpen,
  onClose,
  onSignout,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);

  const handleSignOut = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      console.error("❌ Sign-out failed:", error.message);
    } else {
      onSignout?.();
      window.location.href = "/";
    }
  };

  // Close modal when clicking outside
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

  // Handle escape key press
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
      <div
        ref={modalRef}
        className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] w-full max-w-md p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-black text-black uppercase">Sign Out</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 transition-colors"
            aria-label="Close"
          >
            <X size={20} className="text-black" />
          </button>
        </div>

        <div className="w-full h-1 bg-black mb-6"></div>

        <div className="mb-6">
          <p className="text-black font-medium">
            Are you sure you want to sign out?
          </p>
        </div>

        <div className="flex space-x-3 justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
          >
            Cancel
          </button>
          <button
            onClick={handleSignOut}
            className="px-4 py-2 bg-red-500 border-3 border-black font-bold text-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
          >
            Sign Out
          </button>
        </div>
      </div>
    </div>
  );
};

export default SignoutModal;
