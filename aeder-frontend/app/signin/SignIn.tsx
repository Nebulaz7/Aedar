"use client";
import React from "react";
import { supabase } from "../../lib/supabase";
import { motion } from "framer-motion";
import Link from "next/link";

const SignIn = () => {
  const handleOAuthLogin = async (provider: "google" | "github") => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
        queryParams: {
          access_type: "offline",
          prompt: "consent",
        },
      },
    });

    if (error) {
      console.error("OAuth error:", error.message);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12 bg-[#f5f5f5]">
      {/* Decorative shapes */}
      <div className="absolute top-10 left-10 w-20 h-20 bg-emerald-400 border-3 border-black rotate-12 hidden md:block"></div>
      <div className="absolute bottom-16 right-16 w-16 h-16 bg-yellow-300 border-3 border-black -rotate-6 hidden md:block"></div>
      <div className="absolute top-1/4 right-10 w-12 h-12 bg-pink-400 border-3 border-black rounded-full hidden md:block"></div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-md relative"
      >
        <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-8">
          {/* Logo */}
          <div className="flex justify-center mb-6">
            <h1 className="text-3xl font-black text-black uppercase tracking-tight">
              Welcome To{" "}
              <span className="bg-linear-to-r from-emerald-500 to-black text-transparent bg-clip-text">
                Aedar
              </span>
            </h1>
          </div>

          {/* Divider */}
          <div className="w-full h-1 bg-black mb-6"></div>

          {/* Info text */}
          <p className="text-black font-medium text-center mb-8">
            Log into an existing account or create one using the options below
          </p>

          {/* Sign-in buttons */}
          <div className="space-y-4">
            <button
              onClick={() => handleOAuthLogin("google")}
              className="w-full bg-white cursor-pointer hover:bg-gray-100 text-black font-bold py-4 px-4 border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all duration-150 flex items-center justify-center gap-3"
            >
              <svg
                className="w-5 h-5"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  fill="#4285F4"
                />
                <path
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  fill="#34A853"
                />
                <path
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  fill="#FBBC05"
                />
                <path
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  fill="#EA4335"
                />
              </svg>
              <span>Sign in with Google</span>
            </button>

            <button
              onClick={() => handleOAuthLogin("github")}
              className="w-full cursor-pointer bg-emerald-400 hover:bg-emerald-500 text-black font-bold py-4 px-4 border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all duration-150 flex items-center justify-center gap-3"
            >
              <svg
                className="w-5 h-5"
                viewBox="0 0 24 24"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"
                />
              </svg>
              <span>Sign in with GitHub</span>
            </button>
          </div>

          {/* Divider */}
          <div className="w-full h-1 bg-black mt-8 mb-6"></div>

          {/* Terms and privacy */}
          <p className="text-sm text-center text-black/70 font-medium">
            By signing in, you accept the{" "}
            <Link
              href="#"
              className="text-emerald-600 hover:text-emerald-700 underline underline-offset-2 font-bold"
            >
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link
              href="#"
              className="text-emerald-600 hover:text-emerald-700 underline underline-offset-2 font-bold"
            >
              Privacy Policy
            </Link>
          </p>

          {/* Back to Home */}
          <div className="mt-6 text-center">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-black font-bold hover:text-emerald-600 transition-colors duration-150 border-b-2 border-black hover:border-emerald-600"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z"
                  clipRule="evenodd"
                />
              </svg>
              Back to Home
            </Link>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default SignIn;
