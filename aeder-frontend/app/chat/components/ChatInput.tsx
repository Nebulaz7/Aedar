"use client";

import React, { useState, useRef } from "react";
import { Paperclip, Mic, Send } from "lucide-react";
import SlashCommandMenu from "./SlashCommandMenu";

// Type definitions for SpeechRecognition API
interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

interface ChatInputProps {
  onSendMessage: (message: string) => Promise<void>;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [listening, setListening] = useState(false);
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [slashFilter, setSlashFilter] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInputValue(value);

    // Check for slash commands
    if (value.startsWith("/")) {
      setShowSlashMenu(true);
      setSlashFilter(value);
    } else {
      setShowSlashMenu(false);
      setSlashFilter("");
    }

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  };

  const handleFocus = () => setIsFocused(true);
  const handleBlur = () => {
    setIsFocused(false);
    setTimeout(() => setShowSlashMenu(false), 200);
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    const messageToSend = inputValue.trim();
    setInputValue("");
    setShowSlashMenu(false);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      await onSendMessage(messageToSend);
    } catch (error) {
      console.error("Error sending message:", error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSlashCommand = (command: string) => {
    setInputValue(command + " ");
    setShowSlashMenu(false);
    textareaRef.current?.focus();
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      setInputValue(text);
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
        textareaRef.current.style.height = `${Math.min(
          textareaRef.current.scrollHeight,
          200
        )}px`;
      }
    } catch (err) {
      console.error("Error reading file:", err);
    }
  };

  const startSpeechRecognition = () => {
    if (typeof window === "undefined") return;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech Recognition not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript;
      setInputValue((prev) => (prev ? prev + " " + transcript : transcript));
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      const silentErrors = [
        "aborted",
        "no-speech",
        "not-allowed",
        "service-not-allowed",
      ];
      if (!silentErrors.includes(event.error)) {
        console.error("Speech recognition error:", event.error);
      }
      setListening(false);
    };

    recognition.start();
  };

  return (
    <div className="relative w-full max-w-3xl mx-auto">
      {/* Slash Command Menu */}
      <SlashCommandMenu
        isOpen={showSlashMenu}
        onSelect={handleSlashCommand}
        onClose={() => setShowSlashMenu(false)}
        filterText={slashFilter}
      />

      {/* Input Container */}
      <div
        className={`bg-white border-4 border-black transition-all duration-150 ${
          isFocused
            ? "border-emerald-500 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] translate-x-[-2px] translate-y-[-2px]"
            : "shadow-[6px_6px_0px_0px_rgba(0,0,0,1)]"
        }`}
      >
        {/* Textarea */}
        <div className="p-4">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={handleInputChange}
            onFocus={handleFocus}
            onBlur={handleBlur}
            onKeyPress={handleKeyPress}
            placeholder="Describe your project (e.g., 'Build a fitness app in React')..."
            className="w-full bg-transparent text-black placeholder-black/40 resize-none outline-none text-base leading-relaxed min-h-[60px] max-h-[200px] font-medium"
            rows={2}
          />
        </div>

        {/* Bottom Bar */}
        <div className="flex items-center justify-between px-4 py-3 border-t-2 border-black/10 bg-gray-50">
          {/* Left Actions */}
          <div className="flex items-center gap-2">
            <label
              htmlFor="file-upload"
              className="p-2 bg-white border-2 border-black hover:bg-emerald-100 transition-colors cursor-pointer"
            >
              <Paperclip className="w-4 h-4 text-black" />
            </label>
            <input
              id="file-upload"
              type="file"
              accept=".txt"
              className="hidden"
              onChange={handleFileUpload}
            />

            <button
              onClick={startSpeechRecognition}
              className={`p-2 border-2 cursor-pointer border-black transition-colors ${
                listening
                  ? "bg-red-400 animate-pulse"
                  : "bg-white hover:bg-emerald-100"
              }`}
            >
              <Mic
                className={`w-4 h-4 ${listening ? "text-white" : "text-black"}`}
              />
            </button>

            {listening && (
              <span className="text-xs font-bold text-red-500 animate-pulse">
                Listening...
              </span>
            )}

            {isLoading && (
              <span className="text-xs font-bold text-emerald-600 flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                Generating roadmap...
              </span>
            )}
          </div>

          {/* Hint + Send */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-black/40 font-medium hidden sm:block">
              Enter to send, Shift+Enter for new line
            </span>

            <button
              onClick={handleSend}
              disabled={isLoading || !inputValue.trim()}
              className={`p-3 border-3 border-black font-bold transition-all ${
                inputValue.trim() && !isLoading
                  ? "bg-emerald-400 cursor-pointer shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px]"
                  : "bg-gray-200 cursor-not-allowed"
              }`}
            >
              <Send className="w-5 h-5 text-black" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
