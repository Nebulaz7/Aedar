# Aedar

**Aedar** is an AI-powered execution agent that transforms project ideas into structured technical roadmaps — and automatically schedules them. Built for developers, founders, and teams who want planning that actually executes.

---

## 🧠 Why Aedar?

Building software ideas often starts messy — scattered notes, vague goals, and no clear execution plan. Rave bridges that gap by helping teams turn raw ideas into structured, technical roadmaps with AI-assisted clarity.

**What makes Rave different:** It doesn't just plan — it executes. Calendar events, meetings, and milestones are created automatically when needed.

---

## ✨ Features

### Core Planning
- 🔐 **Authentication with Supabase** – Secure login, OAuth support
- 🤖 **AI Chat Assistant** – Brainstorm ideas and clarify goals interactively
- 🧭 **Smart Roadmap Generation** – Convert discussions into structured technical plans
- 🕘 **Roadmap History** – Revisit and evolve past ideas
- 📁 **Session Management** – Organize multiple projects cleanly

### AI Execution Layer (New)
- 📅 **Calendar MCP Integration** – AI schedules roadmap milestones directly to your Google Calendar
- 🎯 **Intent-Based Tool Invocation** – Calendar features activate only when you need them
- 🔗 **App Connections** – Connect external tools via Settings → "Connect Apps"
- 🛡️ **Safe Tool Execution** – MCP architecture ensures AI actions are auditable and deterministic

### Architecture
- 🧩 **Monorepo Structure** – Cleanly separated API, frontend, and MCP servers
- 🔧 **Protocol-Driven Design** – AI tools are first-class citizens, not afterthoughts

---

## 🛠️ Tech Stack

| Layer        | Tech                                      |
|--------------|-------------------------------------------|
| **Frontend** | Next.js 14, TailwindCSS, React Icons      |
| **Backend**  | NestJS (REST API)                         |
| **Database** | Supabase (PostgreSQL + Auth)              |
| **AI**       | OpenAI API (chat + roadmap generation)    |
| **MCP**      | Custom MCP servers (Calendar, Meet)       |
| **Infra**    | Render (backend), Vercel (frontend)       |
| **Other**    | TypeScript, ESM, Turbo monorepo           |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/cridiv/Aedar.git
cd Rave
```

### 2. Install Dependencies

```bash
npm install
```

Or with pnpm:

```bash
pnpm install
```

### 3. Set Up Environment Variables

Create a `.env` file in the root:

```env
# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_ANON_KEY=your-supabase-anon-key

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Google Calendar (for MCP)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=your-redirect-uri
GOOGLE_REFRESH_TOKEN=your-refresh-token

# Frontend
NEXT_PUBLIC_SUPABASE_URL=your-supabase-url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### 4. Run the App Locally

**Backend (NestJS API)**
```bash
cd aedar-backend
npm run start:dev
```

**Frontend (Next.js)**
```bash
cd aedar-frontend
npm run dev
```

**Calendar MCP Server**
```bash
cd mcp/calendar
npm run dev
```

App will be available at `http://localhost:3000`

---

## 🎯 How It Works

### User Flow

1. **User creates or continues a chat session**
2. **User describes their project idea**
   - "Build a SaaS MVP in 4 weeks and schedule the milestones"
3. **AI Intent Detection**
   - Determines if calendar integration is needed
4. **Roadmap Generation**
   - AI creates structured technical roadmap
5. **Calendar Execution (if requested)**
   - MCP Calendar Server creates events in Google Calendar
   - User sees confirmation with calendar links

### Settings Integration

Users enable calendar features via:

**Settings → Connect Apps → Google Calendar**

- One-click OAuth connection
- Permissions: Read/Write calendar events
- Can disconnect anytime

### AI Tool Invocation

Calendar MCP is triggered when:
- User mentions scheduling, calendar, or timeline
- LLM detects time-bound planning intent
- User explicitly requests calendar sync

**Example prompts that trigger calendar:**
- "Schedule these milestones"
- "Add this to my calendar"
- "Plan a 2-week sprint and create the events"

---

## 🧩 MCP Architecture

### What is MCP?

**Model Context Protocol (MCP)** is a structured way for AI to interact with external tools safely.

Instead of:
```
LLM → Google Calendar API (unsafe, unpredictable)
```

We use:
```
LLM → MCP Router → Calendar MCP Server → Google Calendar API
```

### Benefits

✅ **Deterministic** – Actions are auditable  
✅ **Safe** – No hallucinated API calls  
✅ **Extensible** – Add new tools without touching core AI  
✅ **Protocol-driven** – Tools are typed and validated  

### Calendar MCP Capabilities (V1)

- Create calendar events
- List upcoming events
- Update events (future)

### Tool Contract Example

```typescript
{
  "tool": "calendar.create_event",
  "arguments": {
    "title": "Backend API Kickoff",
    "startTime": "2025-01-20T10:00:00Z",
    "endTime": "2025-01-20T11:00:00Z",
    "attendees": ["team@example.com"]
  }
}
```

---

## 🌐 Deployment

### 📡 Backend (Render)

1. Connect the repo on Render
2. Set root directory to `aedar-backend`
3. Add environment variables
4. Build command:
   ```bash
   npm install && npm run build
   ```
5. Start command:
   ```bash
   npm run start:prod
   ```

### 🧑‍💻 Frontend (Vercel)

1. Connect repo on Vercel
2. Set root directory to `aedar-frontend`
3. Add environment variables
4. Deploy

### 📅 Calendar MCP Server (Render)

1. Deploy as separate service
2. Set root directory to `mcp/calendar`
3. Add Google OAuth credentials
4. Expose internal endpoint to NestJS API

---

## 🎬 Demo Flow (For Hackathon Judges)

**Step 1:** User types  
*"Plan a 4-week MVP and schedule it on my calendar"*

**Step 2:** AI generates roadmap  
- Week 1: Backend setup  
- Week 2: Frontend basics  
- Week 3: Integration  
- Week 4: Testing & launch  

**Step 3:** AI creates calendar events  
- Events appear in Google Calendar  
- Each milestone has a dedicated time block  
- User gets confirmation with links  

**Step 4:** Show live calendar  
→ **Judges see real execution, not just planning** 🎯

---

## 🔮 Roadmap (Future Features)

### Next Phase
- ✅ Google Meet MCP (auto-create meeting links)
- ✅ Slack MCP (post updates to channels)
- ✅ Risk & bottleneck detection
- ✅ AI project memory (remembers past decisions)

### Vision
Transform Rave into a full execution agent:
- Jira integration (create tickets)
- GitHub integration (scaffold repos)
- Notion sync (mirror roadmaps)

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Commit changes: `git commit -m "feat: your update"`
4. Push: `git push origin feat/your-feature`
5. Open a pull request

---

## 👨‍💻 Authors

- **Aderemi Ademola** – Backend Lead, Fullstack & AI Engineer (X: @cridiv)
- **Peters Joshua** – Frontend Lead, Fullstack Developer (X: @joshpet77)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🏆 Built For

**AI-Themed Hackathon Submission**

Rave demonstrates:
- Real-world AI tool orchestration
- Protocol-driven design (MCP architecture)
- Execution over conversation
- Production-ready patterns

---

*Built with ❤️ by developers, for developers*
