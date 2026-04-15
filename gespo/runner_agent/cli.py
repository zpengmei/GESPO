"""Thin CLI for the GESPO runner agent."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import AgentConfig, load_agent_config
from .runtime import Agent, ToolRegistry, ApprovalGate, Sandbox
from .runtime.config import RuntimeConfig
from .runtime.llm import create_client
from .skills import register_default_skills
from .skills.gespo_workflow import GESPOWorkflowProvider
from .prompt import build_system_prompt
from . import terminal as t


def _build_agent(config: AgentConfig) -> Agent:
    """Construct a fully-wired Agent from config."""
    client = create_client(
        provider=config.llm.provider,
        model=config.llm.model,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        api_key_file=config.llm.api_key_file,
        timeout_s=config.llm.timeout_s,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
    sandbox = Sandbox(
        workdir=Path(config.sandbox.workdir),
        write_root=Path(config.sandbox.write_root) if config.sandbox.write_root else None,
    )
    registry = ToolRegistry()
    register_default_skills(
        registry,
        enabled=config.tools.enabled,
        sandbox=sandbox,
        shell_allowlist=config.tools.shell_allowlist,
        timeout_s=config.tools.timeout_s,
        slurm_defaults=config.tools.slurm,
        allow_slurm_submit=config.tools.allow_slurm_submit,
        python_bin=config.tools.python_bin,
        script_root=config.tools.script_root,
    )
    approval = ApprovalGate(mode=config.tools.approval)
    rt_config = RuntimeConfig(
        max_tool_steps=config.max_tool_steps,
        stream=config.llm.stream,
    )
    agent = Agent(client=client, registry=registry, approval=approval, config=rt_config)

    workflow = (
        GESPOWorkflowProvider(config.docs_root, config.max_context_chars)
        if config.docs_root else None
    )
    system_msgs = build_system_prompt(registry, workflow)
    agent.setup(system_msgs)
    return agent


# ── Readline setup ──────────────────────────────────────────────────────

def _setup_readline() -> None:
    """Enable readline history and tab completion if available."""
    try:
        import readline
    except ImportError:
        return
    histfile = Path(os.environ.get("GESPO_AGENT_HISTFILE", "~/.gespo_agent_history"))
    histfile = histfile.expanduser()
    try:
        readline.read_history_file(str(histfile))
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(500)

    import atexit
    atexit.register(_save_readline_history, str(histfile))


def _save_readline_history(path: str) -> None:
    try:
        import readline
        readline.write_history_file(path)
    except Exception:
        pass


# ── Slash commands ──────────────────────────────────────────────────────

_SLASH_HELP = {
    "/help":    "Show this help message.",
    "/tools":   "List available tools with descriptions.",
    "/history": "Show conversation history (last N turns).",
    "/clear":   "Clear conversation history (keep system prompt).",
    "/config":  "Show active configuration summary.",
    "/exit":    "Exit the agent.",
}


def _handle_slash(cmd: str, agent: Agent, config: AgentConfig) -> bool:
    """Handle a slash command. Returns True if handled."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    # arg = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        print(file=sys.stderr)
        print(f"  {t.styled('Commands:', t.BOLD)}", file=sys.stderr)
        for slash, desc in _SLASH_HELP.items():
            print(f"    {t.styled(slash, t.CYAN):20s} {desc}", file=sys.stderr)
        print(file=sys.stderr)
        return True

    if command == "/tools":
        print(file=sys.stderr)
        names = agent.registry.names()
        if not names:
            t.warn("No tools registered.")
        else:
            print(f"  {t.styled('Tools:', t.BOLD)} ({len(names)})", file=sys.stderr)
            for name in names:
                skill = agent.registry.get(name)
                desc = skill.description if skill else ""
                print(
                    f"    {t.tool_name(name):30s} {t.system_msg(desc)}",
                    file=sys.stderr,
                )
        print(file=sys.stderr)
        return True

    if command == "/history":
        msgs = agent.conversation.messages
        # Show last 20 non-system messages
        shown = [m for m in msgs if m.role != "system"][-20:]
        print(file=sys.stderr)
        for m in shown:
            role_color = {
                "user": t.GREEN,
                "assistant": t.MAGENTA,
            }.get(m.role, t.GRAY)
            role_str = t.styled(m.role, t.BOLD, role_color)
            content = m.content
            if len(content) > 200:
                content = content[:197] + "..."
            print(f"  {role_str}: {content}", file=sys.stderr)
        print(file=sys.stderr)
        return True

    if command == "/clear":
        # Keep system messages, clear the rest
        system_msgs = [m for m in agent.conversation.messages if m.role == "system"]
        agent.conversation.messages.clear()
        agent.conversation.messages.extend(system_msgs)
        t.success("Conversation cleared (system prompt retained).")
        return True

    if command == "/config":
        print(file=sys.stderr)
        print(f"  {t.styled('Configuration:', t.BOLD)}", file=sys.stderr)
        print(f"    provider:  {t.styled(config.llm.provider, t.CYAN)}", file=sys.stderr)
        print(f"    model:     {t.styled(config.llm.model, t.CYAN)}", file=sys.stderr)
        print(f"    workdir:   {config.sandbox.workdir}", file=sys.stderr)
        print(f"    stream:    {config.llm.stream}", file=sys.stderr)
        print(f"    approval:  {config.tools.approval}", file=sys.stderr)
        print(f"    tools:     {', '.join(config.tools.enabled)}", file=sys.stderr)
        print(f"    max_steps: {config.max_tool_steps}", file=sys.stderr)
        if config.docs_root:
            print(f"    docs_root: {config.docs_root}", file=sys.stderr)
        print(file=sys.stderr)
        return True

    if command in ("/exit", "/quit"):
        raise SystemExit(0)

    return False


# ── REPL ────────────────────────────────────────────────────────────────

def _repl(agent: Agent, config: AgentConfig) -> None:
    """Interactive REPL loop."""
    _setup_readline()

    # Startup banner
    print(file=sys.stderr)
    print(
        f"  {t.styled('GESPO Agent', t.BOLD, t.MAGENTA)} "
        f"{t.system_msg('— type /help for commands, /exit to quit')}",
        file=sys.stderr,
    )
    tool_names = agent.registry.names()
    if tool_names:
        tools_str = ", ".join(t.tool_name(n) for n in tool_names)
        print(f"  {t.system_msg('tools:')} {tools_str}", file=sys.stderr)
    print(file=sys.stderr)

    while True:
        try:
            prompt_text = input(t.prompt_str()).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {t.system_msg('Bye.')}", file=sys.stderr)
            break

        if not prompt_text:
            continue

        # Slash commands
        if prompt_text.startswith("/"):
            if _handle_slash(prompt_text, agent, config):
                continue
            t.warn(f"Unknown command: {prompt_text.split()[0]}  (try /help)")
            continue

        # Plain exit
        if prompt_text.lower() in ("exit", "quit", "q"):
            print(f"  {t.system_msg('Bye.')}", file=sys.stderr)
            break

        try:
            reply = agent.run_turn(prompt_text)
            # Print final response
            print(file=sys.stderr)
            print(reply)
        except KeyboardInterrupt:
            print(f"\n  {t.system_msg('(interrupted)')}", file=sys.stderr)
        except Exception as e:
            t.error(f"{type(e).__name__}: {e}")


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="gespo-agent",
        description="GESPO runner agent — LLM-powered workflow assistant",
    )
    parser.add_argument("--config", "-c", default="", help="YAML config file")
    parser.add_argument("--provider", default=None, help="LLM provider (deepseek/openai)")
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--workdir", default=None, help="Working directory")
    parser.add_argument("--stream", action="store_true", default=None, help="Stream LLM responses")
    parser.add_argument(
        "--approve", default=None, choices=["prompt", "always", "never"],
        help="Tool approval mode",
    )
    parser.add_argument("--docs-root", default=None, help="Path to workflow docs directory")
    parser.add_argument(
        "--tui", action="store_true", default=False,
        help="Launch full-screen TUI (requires textual; omit for plain REPL)",
    )
    parser.add_argument("prompt", nargs="*", help="Single-shot prompt (omit for REPL)")

    args = parser.parse_args()

    overrides = {}
    for key in ("provider", "model", "workdir", "stream", "approve", "docs_root"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    config = load_agent_config(args.config or None, overrides=overrides if overrides else None)

    # Check API key early with a clear message
    from .runtime.llm.openai_compat import _ENV_KEYS
    provider = config.llm.provider
    if not config.llm.api_key and not config.llm.api_key_file:
        env_var = _ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
        if not os.environ.get(env_var):
            print(
                f"Error: No API key found for provider '{provider}'.\n"
                f"\n"
                f"Set one of:\n"
                f"  export {env_var}=your-key-here\n"
                f"  --config file.yaml  (with llm.api_key or llm.api_key_file)\n"
                f"\n"
                f"Example:\n"
                f"  export {env_var}=sk-...\n"
                f"  python -m gespo.runner_agent --tui --provider {provider}\n",
                file=sys.stderr,
            )
            sys.exit(1)

    agent = _build_agent(config)

    if args.prompt:
        reply = agent.run_turn(" ".join(args.prompt))
        print(reply)
    elif args.tui:
        try:
            from .tui import GESPOApp
        except ImportError:
            print(
                "TUI requires textual and rich: pip install textual rich",
                file=sys.stderr,
            )
            sys.exit(1)
        app = GESPOApp(agent=agent, config=config)
        app.run()
    else:
        _repl(agent, config)


if __name__ == "__main__":
    main()
