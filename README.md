# AI Agents Lab

A hands-on collection of AI agent, automation, and generative AI projects — covering multi-agent systems (LangGraph), retrieval-augmented generation (RAG), workflow automation (n8n), and applied GenAI tools built for real products.

![Architecture Overview](images/architecture_overview.png)

## Why this repo exists

Most of my work sits at the intersection of **agentic AI systems** and **practical automation for SMEs** — from multi-agent business tools to Darija-language voice assistants. This repo is where that work lives: real projects, real architecture decisions, and real trade-offs, documented as I build.

## What's inside

| Project | Description | Stack |
|---|---|---|
| `kryon-cortex` | AI chat interface for a lead-generation SaaS — LangGraph harness agent, RAG over MongoDB, memory-aware conversation | LangGraph, MongoDB, Next.js |
| `hisab-ai` | AI accounting assistant for Moroccan SMEs — WhatsApp/Telegram automation, OCR invoice processing | n8n, Twilio, Claude Vision API, Supabase |
| `atlas-fitness-agent` | Darija-language fitness coaching multi-agent system with local LLM + RAG | n8n, Ollama, Supabase pgvector |
| `agrinova` | Agricultural quality classification agent pipeline (dates, citrus, olives) | YOLOv8, agent orchestration |
| *(more added as built)* | | |

Each project folder has its own README with setup instructions, architecture notes, and demo material.

## Core themes across this repo

- **Multi-agent orchestration** — using LangGraph to coordinate specialized agents (routing, tool use, memory)
- **RAG pipelines** — vector search (MongoDB, Supabase pgvector) combined with LLM reasoning
- **Workflow automation** — n8n-based automation connecting AI reasoning to real business channels (WhatsApp, Telegram)
- **MCP (Model Context Protocol)** — wrapping external APIs and tools for agent use
- **Localization-first design** — Darija/Arabic-first interfaces, built for the Moroccan and MENA market




## Roadmap

- [ ] Add a shared `agents/` utility folder for reusable agent components across projects
- [ ] Document evaluation methodology for agent reliability (not just demos)
- [ ] Add architecture diagrams per project (agent graphs, data flow)
- [ ] Write a comparison post: LangGraph vs. n8n for different automation needs

## About

Built by [Abdelhamid] — AI & Software Engineering student,
