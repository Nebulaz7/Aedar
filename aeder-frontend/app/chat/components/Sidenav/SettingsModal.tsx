"use client";
import React, { useEffect, useRef, useState } from "react";
import { X, Moon, Sun, Zap, Info, Bell } from "lucide-react";

type SettingsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  initialSettings?: {
    darkMode: boolean;
    notifications: boolean;
    performance: "low" | "medium" | "high";
  };
};

// Default settings for demo purposes
const defaultSettings = {
  darkMode: true,
  notifications: true,
  performance: "medium" as const,
};

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  initialSettings = defaultSettings,
}) => {
  const [settings, setSettings] = useState(initialSettings);
  const modalRef = useRef<HTMLDivElement>(null);

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

  // Reserved for future use when dark mode toggle is implemented
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleDarkModeToggle = () => {
    setSettings((prev) => ({ ...prev, darkMode: !prev.darkMode }));
  };

  const handleNotificationsToggle = () => {
    setSettings((prev) => ({ ...prev, notifications: !prev.notifications }));
  };

  const handlePerformanceChange = (value: "low" | "medium" | "high") => {
    setSettings((prev) => ({ ...prev, performance: value }));
  };

  const saveSettings = () => {
    // Here you would normally save settings to your backend or localStorage
    console.log("Saving settings:", settings);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
      <div
        ref={modalRef}
        className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] w-full max-w-md p-6"
      >
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-black text-black uppercase">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 transition-colors"
            aria-label="Close"
          >
            <X size={20} className="text-black" />
          </button>
        </div>

        <div className="w-full h-1 bg-black mb-6"></div>

        <div className="space-y-6 mb-6">
          {/* Dark Mode Setting */}
          <div className="flex items-center justify-between p-4 bg-gray-50 border-3 border-black">
            <div className="flex items-center gap-3">
              {settings.darkMode ? (
                <Moon size={18} className="text-black" />
              ) : (
                <Sun size={18} className="text-yellow-500" />
              )}
              <div>
                <p className="text-black font-bold">Dark Mode</p>
                <p className="text-xs text-black/50">
                  Toggle between light and dark theme
                </p>
              </div>
            </div>
            <span className="text-xs px-2 py-1 bg-yellow-300 border-2 border-black font-bold">
              Soon
            </span>
          </div>

          {/* Notification Setting */}
          <div className="flex items-center justify-between p-4 bg-gray-50 border-3 border-black">
            <div className="flex items-center gap-3">
              <Bell size={18} className="text-emerald-600" />
              <div>
                <p className="text-black font-bold">Notifications</p>
                <p className="text-xs text-black/50">
                  Receive updates and alerts
                </p>
              </div>
            </div>
            <button
              onClick={handleNotificationsToggle}
              className={`w-12 h-6 border-2 border-black relative transition-colors ${
                settings.notifications ? "bg-emerald-400" : "bg-gray-200"
              }`}
            >
              <div
                className={`absolute top-0.5 w-4 h-4 bg-black transition-all ${
                  settings.notifications ? "left-6" : "left-0.5"
                }`}
              />
            </button>
          </div>

          {/* Performance Setting */}
          <div className="p-4 bg-gray-50 border-3 border-black">
            <div className="flex items-center gap-3 mb-3">
              <Zap size={18} className="text-yellow-500" />
              <div>
                <p className="text-black font-bold">Performance Mode</p>
                <p className="text-xs text-black/50">
                  Adjust animation intensity
                </p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(["low", "medium", "high"] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => handlePerformanceChange(level)}
                  className={`py-2 text-sm font-bold border-2 border-black transition-all ${
                    settings.performance === level
                      ? "bg-emerald-400 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)]"
                      : "bg-white hover:bg-gray-100"
                  }`}
                >
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Info Section */}
          <div className="flex items-start gap-3 p-4 bg-emerald-50 border-3 border-black">
            <Info size={18} className="text-emerald-600 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-black/70 font-medium">
              Higher performance settings may use more system resources. For the
              best experience, adjust based on your device capabilities.
            </p>
          </div>
        </div>

        <div className="flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] transition-all"
          >
            Cancel
          </button>
          <button
            onClick={saveSettings}
            className="px-4 py-2 bg-emerald-400 border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] transition-all"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
