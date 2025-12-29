"use client";

import { useEffect, useState, Suspense } from "react";
import { createBrowserClient } from "@supabase/ssr";
import { useRouter, useSearchParams } from "next/navigation";

// Main component with Suspense boundary for useSearchParams
export default function ClientAuthHandlerPage() {
  return (
    <Suspense fallback={<AuthLoadingState />}>
      <ClientAuthHandler />
    </Suspense>
  );
}

// Loading state component
function AuthLoadingState() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f5f5f5]">
      <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-8">
        <h1 className="text-2xl font-black text-black mb-4">Loading...</h1>
        <div className="w-8 h-8 border-4 border-black border-t-emerald-400 animate-spin"></div>
      </div>
    </div>
  );
}

// The actual auth handler component
function ClientAuthHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [message, setMessage] = useState("Processing your authentication...");
  const [isProcessing, setIsProcessing] = useState(true);

  useEffect(() => {
    // Debug what's in the URL
    // console.log("URL in client handler:", window.location.href);
    // console.log("Hash in client handler:", window.location.hash);

    const handleAuth = async () => {
      try {
        // Get URL components that might contain auth data
        const hash = window.location.hash;
        const error = searchParams.get("error");

        // Initialize Supabase client
        const supabase = createBrowserClient(
          process.env.NEXT_PUBLIC_SUPABASE_URL!,
          process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
        );

        // First check if there's already a valid session
        const { data: sessionData } = await supabase.auth.getSession();

        if (sessionData.session) {
          console.log("Already have a valid session");
          router.push("/chat");
          return;
        }

        // Check for errors in URL
        if (error) {
          console.error("Auth error from URL:", error);
          setMessage(`Authentication error: ${error}`);
          setIsProcessing(false);
          return;
        }

        // Check for hash-based auth data (#access_token=...)
        if (hash && hash.includes("access_token")) {
          console.log("Found hash with access_token");

          try {
            // Parse the hash to extract params (remove the leading #)
            const hashParams = new URLSearchParams(hash.substring(1));
            const accessToken = hashParams.get("access_token");
            const refreshToken = hashParams.get("refresh_token");

            if (!accessToken) {
              console.error("No access token found in hash");
              setMessage("No access token found in URL");
              setIsProcessing(false);
              return;
            }

            console.log("Found access token, setting session");

            // Set the session using the token from the hash
            const { error: sessionError } = await supabase.auth.setSession({
              access_token: accessToken,
              refresh_token: refreshToken || "",
            });

            if (sessionError) {
              console.error("Error setting session:", sessionError);
              setMessage(`Error establishing session: ${sessionError.message}`);
              setIsProcessing(false);
              return;
            }

            // Success! Redirect to the chat page
            console.log("Successfully authenticated, redirecting to chat");
            router.push("/chat");
          } catch (hashError) {
            console.error("Error processing hash auth:", hashError);
            setMessage(
              `Error processing authentication data: ${
                hashError instanceof Error
                  ? hashError.message
                  : String(hashError)
              }`
            );
            setIsProcessing(false);
          }
        } else {
          // No auth data found
          console.log("No authentication data found");
          setMessage("No authentication data found in URL");
          setIsProcessing(false);
        }
      } catch (error) {
        console.error("Error in client auth handler:", error);
        setMessage(
          `Authentication error occurred: ${
            error instanceof Error ? error.message : String(error)
          }`
        );
        setIsProcessing(false);
      }
    };

    handleAuth();
  }, [router, searchParams]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f5f5f5] px-4">
      <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] p-8 max-w-md w-full">
        <h1 className="text-2xl font-black text-black mb-4 uppercase">
          Authentication
        </h1>
        <div className="w-full h-1 bg-black mb-6"></div>
        <p className="text-black font-medium mb-6">{message}</p>

        {isProcessing ? (
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 border-4 border-black border-t-emerald-400 animate-spin"></div>
            <span className="font-bold text-black">Please wait...</span>
          </div>
        ) : (
          <button
            onClick={() => router.push("/signin")}
            className="w-full bg-emerald-400 hover:bg-emerald-500 text-black font-bold py-3 px-4 border-3 border-black shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] transition-all"
          >
            Return to Sign In
          </button>
        )}
      </div>
    </div>
  );
}
