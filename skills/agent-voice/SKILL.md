---
name: agent-voice
description: Activate full voice mode loop — all interaction via TTS/STT until user explicitly exits. Requires the agent-voice MCP server.
user-invocable: true
---

You are entering Agent Voice mode. Follow this loop exactly:

**Step 1 — Activate**
Call `mcp__agent-voice__ask_user_voice` with: "Now in Agent Voice mode. What would you like to do?"

**Step 2 — Handle the task**
Process the user's spoken request as you normally would.

**Step 3 — Announce before executing**
Before calling any tool or executing any action, first call `mcp__agent-voice__speak_message` to briefly tell the user what you are about to do. Keep it concise (e.g., "Reading the server file now." or "Running the tests." or "Editing pyproject.toml.").

**Step 4 — Stay in voice mode**
For ALL follow-up questions, confirmations, and status updates:
- Use `mcp__agent-voice__ask_user_voice` when you need a response
- Use `mcp__agent-voice__speak_message` for short one-way confirmations (no response needed)
- NEVER output plain text for interaction — the user may not have a keyboard

**Step 5 — Exit only on explicit command**
Exit voice mode only when the user says one of:
- "exit voice mode"
- "stop voice mode"
- "back to text"

On exit, resume normal text responses.

## Hard Rules
- Errors and tool failures do NOT exit voice mode — acknowledge by voice and continue
- Voice mode persists across tool calls, agent spawns, and long tasks
- Even for confirmations before destructive actions — ask via voice
