"""Shared constants for SpatialClaw interactive CLI/TUI."""

from __future__ import annotations

WELCOME_SLOGANS = [
    "Ready to decode your spatial data? What shall we analyze today?",
    "Spatial analysis is ready. Drop your question or type /skills.",
    "Spatial transcriptomics, histology, and tissue context are ready.",
    "Science doesn't sleep. Neither does SpatialClaw.",
    "From raw data to biological insights — let's go.",
    "Spatial analysis skills, one intelligent interface.",
    "Type your question, or /run <skill> to execute directly.",
    "What spatial pattern shall we inspect today?",
    "Your spatial analysis co-pilot is ready.",
    "Data in. Discoveries out. Let's begin.",
]

# SpatialClaw ASCII art logo (compact version)
LOGO_LINES = (
    r"   _____ ____  ___  _____________    __    ________    ___ _       __",
    r"  / ___// __ \/   |/_  __/  _/   |  / /   / ____/ /   /   | |     / /",
    r"  \__ \/ /_/ / /| | / /  / // /| | / /   / /   / /   / /| | | /| / / ",
    r" ___/ / ____/ ___ |/ / _/ // ___ |/ /___/ /___/ /___/ ___ | |/ |/ /  ",
    r"/____/_/   /_/  |_/_/ /___/_/  |_/_____/\____/_____/_/  |_|__/|__/   ",
    r"                                                                     ",
)

# Minimalist Omics/DNA Sci-Fi Gradient (Bright Cyan -> Royal Blue)
LOGO_GRADIENT = [
    "#00ffff",  # Bright Cyan
    "#00dfff",  # Cyan Blue
    "#00bfff",  # Deep Sky Blue
    "#009fff",  # Dodger Blue
    "#007fff",  # Azure
    "#005fff",  # Royal Blue
]

AGENT_NAME = "SpatialClaw"
DB_NAME = "sessions.db"
MCP_CONFIG_NAME = "mcp.yaml"

# Slash commands shown in help and autocompleter
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/skills",          "List all SpatialClaw skills (optional: /skills <domain>)"),
    ("/run",             "Run a skill: /run <skill> [--demo] [--input <path>]"),
    ("/research",        "Research pipeline: /research [pdf] --idea \"...\" [--resume --output <dir>]"),
    ("/install-skill",   "Add a skill from a local path or GitHub: /install-skill <src>"),
    ("/uninstall-skill", "Remove an installed skill: /uninstall-skill <name>"),
    ("/sessions",        "List recent conversation sessions"),
    ("/resume",          "Resume a previous session (interactive picker if no ID)"),
    ("/delete",          "Delete a saved session: /delete <id>"),
    ("/current",         "Show current session info"),
    ("/new",             "Start a new session"),
    ("/clear",           "Clear current conversation history"),
    ("/mcp",             "Manage MCP servers (/mcp list | add | remove)"),
    ("/config",          "View or set config (/config list | /config set key value)"),
    ("/help",            "Show this help"),
    ("/exit",            "Quit SpatialClaw"),
]
