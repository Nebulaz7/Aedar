import Link from "next/link";
import {
  Rocket,
  Calendar,
  Zap,
  Brain,
  ArrowRight,
  CheckCircle,
  GithubIcon,
  Twitter,
  Sparkles,
  Target,
  Clock,
  Users,
  Shield,
  ChevronDown,
} from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#f5f5f5] overflow-x-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b-4 border-black">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* <div className="w-10 h-10 bg-emerald-400 border-3 border-black flex items-center justify-center">
              <Rocket size={20} className="text-black" />
            </div> */}
            <span className="text-2xl font-black bg-linear-to-r from-emerald-500 to-black text-transparent bg-clip-text">
              AEDAR
            </span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <a
              href="#features"
              className="font-bold text-black hover:text-emerald-600 transition-colors"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="font-bold text-black hover:text-emerald-600 transition-colors"
            >
              How It Works
            </a>
            <a
              href="#faqs"
              className="font-bold text-black hover:text-emerald-600 transition-colors"
            >
              FAQs
            </a>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/signin"
              className="px-4 py-2 font-bold text-black hover:text-emerald-600 transition-colors hidden sm:block"
            >
              Sign In
            </Link>
            <Link
              href="/signin"
              className="px-5 py-2 bg-emerald-400 border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-0.5 hover:translate-y-[2px] transition-all"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6 relative">
        {/* Decorative shapes */}
        <div className="absolute top-40 left-10 w-20 h-20 bg-yellow-300 border-4 border-black rotate-12 hidden lg:block" />
        <div className="absolute top-60 right-20 w-16 h-16 bg-pink-400 border-4 border-black rounded-full hidden lg:block" />
        <div className="absolute bottom-20 left-1/4 w-12 h-12 bg-blue-400 border-4 border-black rotate-45 hidden lg:block" />

        <div className="max-w-5xl mx-auto text-center relative">
          {/* <div className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-300 border-3 border-black mb-6">
            <Sparkles size={16} className="text-black" />
            <span className="font-bold text-black text-sm">
              AI-POWERED EXECUTION AGENT
            </span>
          </div> */}

          <h1 className="text-5xl md:text-7xl font-black text-black leading-tight mb-6">
            Turn Ideas Into
            <span className="relative inline-block mx-3">
              <span className="relative z-10">Roadmaps</span>
              <div className="absolute bottom-2 left-0 right-0 h-4 bg-emerald-400 -z-0" />
            </span>
            That Execute
          </h1>

          <p className="text-xl md:text-2xl text-black/70 max-w-3xl mx-auto mb-10 font-medium">
            Transform your project ideas into structured technical roadmaps —
            and watch them automatically appear on your calendar. Planning that
            actually ships.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link
              href="/signin"
              className="px-8 py-4 bg-emerald-400 border-4 border-black font-black text-lg shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[3px] hover:translate-y-[3px] transition-all flex items-center gap-2"
            >
              Start Building Free
              <ArrowRight size={20} />
            </Link>
            <a
              href="#how-it-works"
              className="px-8 py-4 bg-white border-4 text-black border-black font-black text-lg shadow-[6px_6px_0px_0px_rgba(0,0,0,1)] hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[3px] hover:translate-y-[3px] transition-all"
            >
              See How It Works
            </a>
          </div>

          {/* Hero visual */}
          <div className="relative max-w-4xl mx-auto">
            <div className="bg-white border-4 border-black shadow-[12px_12px_0px_0px_rgba(0,0,0,1)] p-6 md:p-8">
              <div className="flex items-center gap-3 mb-4 pb-4 border-b-2 border-black">
                <div className="w-4 h-4 bg-red-500 border-2 border-black rounded-full" />
                <div className="w-4 h-4 bg-yellow-500 border-2 border-black rounded-full" />
                <div className="w-4 h-4 bg-emerald-500 border-2 border-black rounded-full" />
                <span className="ml-4 font-bold text-black/50">Aedar Chat</span>
              </div>

              <div className="space-y-4">
                <div className="flex justify-end">
                  <div className="bg-emerald-100 border-3 border-black p-4 max-w-md">
                    <p className="font-medium text-black">
                      I want to build a SaaS MVP for project management in 2
                      weeks
                    </p>
                  </div>
                </div>

                <div className="flex justify-start">
                  <div className="bg-gray-50 border-3 border-black p-4 max-w-lg">
                    <p className="font-medium text-black mb-3">
                      I&apos;ve created a 2-week roadmap with 4 phases:
                    </p>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm text-black">
                        <CheckCircle size={16} className="text-emerald-600" />
                        <span className="font-bold ">Week 1:</span> Core auth &
                        database setup
                      </div>
                      <div className="flex items-center gap-2 text-sm text-black">
                        <CheckCircle size={16} className="text-emerald-600" />
                        <span className="font-bold ">Week 1:</span> Project CRUD
                        & team features
                      </div>
                      <div className="flex items-center gap-2 text-sm text-black">
                        <CheckCircle size={16} className="text-emerald-600" />
                        <span className="font-bold ">Week 2:</span> Dashboard &
                        notifications
                      </div>
                      <div className="flex items-center gap-2 text-sm text-black">
                        <CheckCircle size={16} className="text-emerald-600" />
                        <span className="font-bold ">Week 2:</span> Testing &
                        deployment
                      </div>
                    </div>
                    <div className="mt-4 flex gap-2">
                      <button className="px-3 py-1.5 bg-emerald-400 border-2 border-black font-bold text-sm flex items-center gap-1">
                        <Calendar size={14} />
                        Add to Calendar
                      </button>
                      <button className="px-3 py-1.5 bg-white border-2 text-black border-black font-bold text-sm">
                        Modify Plan
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Logos Section */}
      <section className="py-16 px-6 bg-white border-y-4 border-black">
        <div className="max-w-5xl mx-auto text-center">
          <p className="font-bold text-black/50 mb-8 uppercase tracking-wide">
            Built for developers who ship
          </p>
          <div className="flex flex-wrap items-center justify-center gap-8 md:gap-16">
            <div className="flex items-center gap-2 text-black/40">
              <div className="w-10 h-10 bg-black/10 border-2 border-black/20" />
              <span className="font-black text-xl">Startups</span>
            </div>
            <div className="flex items-center gap-2 text-black/40">
              <div className="w-10 h-10 bg-black/10 border-2 border-black/20 rounded-full" />
              <span className="font-black text-xl">Agencies</span>
            </div>
            <div className="flex items-center gap-2 text-black/40">
              <div className="w-10 h-10 bg-black/10 border-2 border-black/20 rotate-45" />
              <span className="font-black text-xl">Indie Hackers</span>
            </div>
            <div className="flex items-center gap-2 text-black/40">
              <div className="w-10 h-10 bg-black/10 border-2 border-black/20" />
              <span className="font-black text-xl">Teams</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-pink-400 border-3 border-black mb-4">
              <Zap size={16} className="text-black" />
              <span className="font-bold text-sm">FEATURES</span>
            </div>
            <h2 className="text-4xl md:text-5xl font-black text-black mb-4">
              Everything You Need to Execute
            </h2>
            <p className="text-xl text-black/60 max-w-2xl mx-auto font-medium">
              From brainstorming to calendar events — Aedar handles the entire
              planning workflow.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Feature 1 */}
            <div className="bg-white border-4 border-black p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-1 hover:translate-y-1 transition-all">
              <div className="w-14 h-14 bg-emerald-400 border-3 border-black flex items-center justify-center mb-4">
                <Brain size={28} className="text-black" />
              </div>
              <h3 className="text-xl font-black text-black mb-2">
                AI Chat Assistant
              </h3>
              <p className="text-black/60 font-medium">
                Brainstorm ideas naturally. Our AI understands context and helps
                clarify your goals through conversation.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="bg-white border-4 border-black p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-1 hover:translate-y-1 transition-all">
              <div className="w-14 h-14 bg-yellow-300 border-3 border-black flex items-center justify-center mb-4">
                <Target size={28} className="text-black" />
              </div>
              <h3 className="text-xl font-black text-black mb-2">
                Smart Roadmaps
              </h3>
              <p className="text-black/60 font-medium">
                Convert discussions into structured technical plans with phases,
                milestones, and time estimates.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="bg-white border-4 border-black p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-1 hover:translate-y-1 transition-all">
              <div className="w-14 h-14 bg-pink-400 border-3 border-black flex items-center justify-center mb-4">
                <Calendar size={28} className="text-black" />
              </div>
              <h3 className="text-xl font-black text-black mb-2">
                Auto-Schedule
              </h3>
              <p className="text-black/60 font-medium">
                One click to add your entire roadmap to Google Calendar.
                Milestones become events automatically.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="bg-white border-4 border-black p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-1 hover:translate-y-1 transition-all">
              <div className="w-14 h-14 bg-blue-400 border-3 border-black flex items-center justify-center mb-4">
                <Clock size={28} className="text-black" />
              </div>
              <h3 className="text-xl font-black text-black mb-2">
                Session History
              </h3>
              <p className="text-black/60 font-medium">
                Revisit past roadmaps and evolve your ideas over time. Never
                lose a great planning session.
              </p>
            </div>

            {/* Feature 5 */}
            <div className="bg-white border-4 border-black p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-1 hover:translate-y-1 transition-all">
              <div className="w-14 h-14 bg-orange-400 border-3 border-black flex items-center justify-center mb-4">
                <Users size={28} className="text-black" />
              </div>
              <h3 className="text-xl font-black text-black mb-2">
                Team Collaboration
              </h3>
              <p className="text-black/60 font-medium">
                Share roadmaps with your team. Everyone stays aligned on
                what&apos;s being built and when.
              </p>
            </div>

            {/* Feature 6 */}
            <div className="bg-white border-4 border-black p-6 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)] hover:shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:translate-x-1 hover:translate-y-1 transition-all">
              <div className="w-14 h-14 bg-purple-400 border-3 border-black flex items-center justify-center mb-4">
                <Shield size={28} className="text-black" />
              </div>
              <h3 className="text-xl font-black text-black mb-2">
                Secure & Private
              </h3>
              <p className="text-black/60 font-medium">
                Enterprise-grade security with Supabase. Your ideas and plans
                stay protected.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section
        id="how-it-works"
        className="py-24 px-6 bg-emerald-400 border-y-4 border-black"
      >
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-black text-black mb-4">
              How It Works
            </h2>
            <p className="text-xl text-black/70 max-w-2xl mx-auto font-medium">
              From idea to execution in three simple steps
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="relative">
              <div className="bg-white border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
                <div className="absolute -top-4 -left-4 w-12 h-12 bg-yellow-300 border-4 border-black flex items-center justify-center font-black text-2xl">
                  1
                </div>
                <div className="mt-4">
                  <h3 className="text-2xl font-black text-black mb-3">
                    Describe Your Idea
                  </h3>
                  <p className="text-black/60 font-medium">
                    Chat naturally with Aedar about what you want to build. Be
                    as detailed or vague as you want — the AI will ask
                    clarifying questions.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="relative">
              <div className="bg-white border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
                <div className="absolute -top-4 -left-4 w-12 h-12 bg-pink-400 border-4 border-black flex items-center justify-center font-black text-2xl">
                  2
                </div>
                <div className="mt-4">
                  <h3 className="text-2xl font-black text-black mb-3">
                    Get Your Roadmap
                  </h3>
                  <p className="text-black/60 font-medium">
                    Aedar generates a structured technical roadmap with phases,
                    tasks, and realistic time estimates based on your goals.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="relative">
              <div className="bg-white border-4 border-black p-8 shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
                <div className="absolute -top-4 -left-4 w-12 h-12 bg-blue-400 border-4 border-black flex items-center justify-center font-black text-2xl">
                  3
                </div>
                <div className="mt-4">
                  <h3 className="text-2xl font-black text-black mb-3">
                    Execute & Schedule
                  </h3>
                  <p className="text-black/60 font-medium">
                    One click adds everything to your calendar. Your roadmap
                    becomes scheduled work blocks, ready to execute.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}

      {/* FAQ Section */}
      <section
        id="faqs"
        className="py-24 px-6 bg-yellow-300 border-y-4 border-black"
      >
        <div className="max-w-3xl mx-auto">
          <h2 className="text-4xl md:text-5xl font-black text-black text-center mb-12">
            Questions? Answers.
          </h2>

          <div className="space-y-4">
            {[
              {
                q: "What makes Aeaer different from other planning tools?",
                a: "Aedar doesn't just help you plan — it executes. Your roadmaps become calendar events automatically. No more planning that sits in a document.",
              },
              {
                q: "How does the calendar integration work?",
                a: "Connect your Google Calendar in settings, then click 'Add to Calendar' on any roadmap. Aedar schedules work blocks based on your milestones and availability.",
              },
              {
                q: "Can I modify the generated roadmaps?",
                a: "Absolutely! Every roadmap is fully editable. Adjust phases, change timelines, add or remove tasks — Aedar adapts to your needs.",
              },
              {
                q: "Is my data secure?",
                a: "Yes. We use Supabase for enterprise-grade security. Your ideas and plans are encrypted and never shared with third parties.",
              },
            ].map((faq, i) => (
              <div
                key={i}
                className="bg-white border-4 border-black p-6 shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-lg font-black text-black mb-2">
                      {faq.q}
                    </h3>
                    <p className="text-black/60 font-medium">{faq.a}</p>
                  </div>
                  <ChevronDown
                    size={20}
                    className="text-black flex-shrink-0 mt-1"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-black border-4 border-black p-12 text-center relative overflow-hidden">
            {/* Decorative shapes */}
            <div className="absolute top-4 left-4 w-16 h-16 bg-emerald-400 border-4 border-white/20 rotate-12" />
            <div className="absolute bottom-4 right-4 w-12 h-12 bg-pink-400 border-4 border-white/20 rounded-full" />
            <div className="absolute top-1/2 right-8 w-8 h-8 bg-yellow-300 border-4 border-white/20 rotate-45" />

            <h2 className="text-4xl md:text-5xl font-black text-white mb-4 relative z-10">
              Ready to Ship Faster?
            </h2>
            <p className="text-xl text-white/70 mb-8 max-w-2xl mx-auto font-medium relative z-10">
              Join developers and teams who&apos;ve transformed their planning
              workflow with Aedar.
            </p>
            <Link
              href="/signin"
              className="inline-flex items-center gap-2 px-8 py-4 bg-emerald-400 border-4 border-white font-black text-lg shadow-[6px_6px_0px_0px_rgba(255,255,255,0.3)] hover:shadow-[3px_3px_0px_0px_rgba(255,255,255,0.3)] hover:translate-x-[3px] hover:translate-y-[3px] transition-all relative z-10"
            >
              Get Started — It&apos;s Free
              <ArrowRight size={20} />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 bg-white border-t-4 border-black">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2">
              {/* <div className="w-10 h-10 bg-emerald-400 border-3 border-black flex items-center justify-center">
                <Rocket size={20} className="text-black" />
              </div> */}
              <span className="text-2xl font-black bg-linear-to-r from-emerald-500 to-black text-transparent bg-clip-text">
                AEDAR
              </span>
            </div>

            <div className="flex items-center gap-8">
              <a
                href="#features"
                className="font-bold text-black hover:text-emerald-600 transition-colors"
              >
                Features
              </a>
              <a
                href="#pricing"
                className="font-bold text-black hover:text-emerald-600 transition-colors"
              >
                Pricing
              </a>
              <a
                href="#"
                className="font-bold text-black hover:text-emerald-600 transition-colors"
              >
                Privacy
              </a>
              <a
                href="#"
                className="font-bold text-black hover:text-emerald-600 transition-colors"
              >
                Terms
              </a>
            </div>

            <div className="flex items-center gap-4">
              <a
                href="#"
                className="w-10 h-10 bg-black border-3 border-black flex items-center justify-center hover:bg-emerald-400 transition-colors"
              >
                <Twitter size={18} className="text-white hover:text-black" />
              </a>
              <a
                href="#"
                className="w-10 h-10 bg-black border-3 border-black flex items-center justify-center hover:bg-emerald-400 transition-colors"
              >
                <GithubIcon size={18} className="text-white hover:text-black" />
              </a>
            </div>
          </div>

          <div className="mt-8 pt-8 border-t-2 border-black/10 text-center">
            <p className="text-black/50 font-medium">
              © 2025 Aedar. Built with 💚 for builders who ship.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
