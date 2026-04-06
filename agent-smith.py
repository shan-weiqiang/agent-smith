import ast
import inspect
import os
import re
import shutil
import subprocess
import sys
from string import Template
from typing import List, Callable, Tuple, Optional

import click
from dotenv import load_dotenv
import anthropic
import platform


# MiniMax Anthropic-compatible Messages API
MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"

# Assistant identity (injected into the system prompt)
AGENT_NAME = "Smith"


def _terminal_labels_use_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    return sys.stdout.isatty()


# Terminal labels: distinguish user input from agent output (ANSI when supported)
if _terminal_labels_use_color():
    _C_AGENT = "\033[36;1m"  # bright cyan
    _C_USER = "\033[35;1m"  # bright magenta
    _C_RESET = "\033[0m"
    T_AGENT = f"{_C_AGENT}[Agent]{_C_RESET}"
    T_USER = f"{_C_USER}[You]{_C_RESET}"
else:
    T_AGENT = "[Agent]"
    T_USER = "[You]"
# Indent continuation lines under [Agent] blocks (Thought, Question, Message, Feedback, …)
T_INDENT = "  "


def _print_agent_line(text: str) -> None:
    """Single-line agent output: `[Agent] <text>`."""
    print(f"\n\n{T_AGENT} {text}")


def _print_agent_turn_header(step: int) -> None:
    """Open one [Agent] block with the step / wait line (call before the model request)."""
    print(f"\n\n{T_AGENT}")
    print(f"{T_INDENT}Step:")
    print(f"{T_INDENT}{T_INDENT}Step {step} — Requesting model, please wait...")


def _print_agent_turn_sections(sections: List[Tuple[str, str]]) -> None:
    """
    Continue the same [Agent] block: Thought / Action / Message|Question|Feedback / …
    Each subsection is a title line, then body lines indented one level deeper.
    """
    for title, body in sections:
        text = "" if body is None else str(body)
        print(f"{T_INDENT}{title}:")
        if not text.strip():
            continue
        for line in text.rstrip("\n").splitlines():
            print(f"{T_INDENT}{T_INDENT}{line}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool functions — these run in the agent's ReAct loop
# ─────────────────────────────────────────────────────────────────────────────

def read_file(file_path):
    """Read file contents as UTF-8 text."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_to_file(file_path, content):
    """Write content to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "Write succeeded"


def run_terminal_command(command):
    """Run a shell command string."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "Command succeeded" if result.returncode == 0 else result.stderr


def list_messages() -> str:
    """List all protobuf message names defined so far."""
    store = _get_protobuf_store()
    names = store.list_message_names()
    if not names:
        return "No messages defined yet."
    lines = [f"{i+1}. {name}" for i, name in enumerate(names)]
    return "\n".join(lines)


def show_message(name: str) -> str:
    """Return the .proto definition for the message with the given name."""
    store = _get_protobuf_store()
    msg = store.get_message(name)
    if msg is None:
        available = store.list_message_names()
        if not available:
            return f'No message named "{name}". No messages defined yet.'
        return f'No message named "{name}". Available: {", ".join(available)}.'
    return msg.get_proto_string()


def create_message(name: str, fields_str: str) -> str:
    """
    Create a new protobuf message. Fails if a message with this name already exists.

    Args:
        name: the message name (PascalCase).
        fields_str: a newline-separated list of field definitions (same format as update_message).
    """
    store = _get_protobuf_store()
    return store.create_message(name, fields_str)


def update_message(name: str, fields_str: str) -> str:
    """
    Replace the fields of an existing protobuf message. Fails if the message does not exist.

    Args:
        name: the message name (PascalCase).
        fields_str: a newline-separated list of field definitions, e.g.:
            int32 id = 1
            string name = 2
            repeated int32 scores = 3
    """
    store = _get_protobuf_store()
    return store.update_message(name, fields_str)


def delete_message(name: str) -> str:
    """Delete the protobuf message with the given name."""
    store = _get_protobuf_store()
    ok = store.delete_message(name)
    if ok:
        return f'Deleted message "{name}".'
    available = store.list_message_names()
    if not available:
        return f'No message named "{name}". No messages defined yet.'
    return f'No message named "{name}". Available: {", ".join(available)}.'


def get_all_messages_proto() -> str:
    """Return the full .proto file content for all messages."""
    store = _get_protobuf_store()
    return store.get_proto_string()


def ask_user(prompt: str) -> str:
    """
    Read the user's reply after the question was shown in the terminal by ReActAgent.run.
    (Printing is unified with Thought/Action there.)
    """
    try:
        response = input(f"\n{T_USER} ").strip()
    except (EOFError, KeyboardInterrupt):
        response = ""
    return response if response else "(no response)"


TELL_USER_FEEDBACK = (
    "Message shown. End of turn — the user will type their next message at the main prompt. "
    "Your tell_user text should invite them to give their next instruction when appropriate."
)


def tell_user(message: str) -> str:
    """
    Return feedback for the model. The message text is shown by ReActAgent.run in a unified block.
    """
    return TELL_USER_FEEDBACK


def save_proto_file(path: str) -> str:
    """Write the full protobuf definition to the given .proto file path (starts with syntax = "proto3";)."""
    store = _get_protobuf_store()
    content = store.get_proto_string()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved to {path}"


def _compile_proto_messages_protoc(
    proto_path: str,
    output_directory: str,
    out_flag: str,
    success_detail: str,
) -> str:
    """
    Run `protoc --<out_flag>=<dir>` for protobuf message codegen only.

    out_flag: ``python_out`` (``*_pb2.py``) or ``cpp_out`` (``.pb.h`` / ``.pb.cc``).
    Requires ``protoc`` on PATH — project dependency **protobuf-protoc-bin** installs
    it under ``.venv/bin`` (use ``uv run`` so it is found).
    """
    proto_path = os.path.abspath(proto_path)
    output_directory = os.path.abspath(output_directory)
    if not os.path.isfile(proto_path):
        return f"Not a file: {proto_path}"
    if not proto_path.endswith(".proto"):
        return f"Expected a .proto file: {proto_path}"
    os.makedirs(output_directory, exist_ok=True)
    include_dir = os.path.dirname(proto_path)

    if not shutil.which("protoc"):
        return (
            "No `protoc` on PATH. Install dependencies (`uv sync` includes "
            "`protobuf-protoc-bin`) and run the agent with `uv run` so `.venv/bin/protoc` "
            "is used, or install protobuf via Homebrew/apt."
        )
    try:
        result = subprocess.run(
            ["protoc", f"-I{include_dir}", f"--{out_flag}={output_directory}", proto_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "protoc timed out after 120s."
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return f"protoc failed (--{out_flag}, exit {result.returncode}): {err}"
    return f"compiled ({success_detail}): {proto_path} → {output_directory}"


def compile_proto_to_python(proto_path: str, output_directory: str) -> str:
    """
    Compile one .proto file to **protobuf message** Python only (`*_pb2.py`, `--python_out`).
    Uses the same `protoc` on PATH as C++ (see **protobuf-protoc-bin** in the project).
    Paths must be absolute; use save_proto_file first for in-memory definitions.
    """
    return _compile_proto_messages_protoc(
        proto_path,
        output_directory,
        "python_out",
        "Python protobuf messages *_pb2.py",
    )


def compile_proto_to_cpp(proto_path: str, output_directory: str) -> str:
    """
    Compile one .proto file to **protobuf message** C++ (``*.pb.h``, ``*.pb.cc``, ``--cpp_out``).
    Same `protoc` executable as :func:`compile_proto_to_python`.
    """
    return _compile_proto_messages_protoc(
        proto_path,
        output_directory,
        "cpp_out",
        "C++ protobuf messages .pb.h/.pb.cc",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ProtobufStore — in-memory protobuf message definitions
# ─────────────────────────────────────────────────────────────────────────────

class ProtobufField:
    def __init__(self, line: str):
        line = line.strip().strip(";")
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid field: {line}")

        # Handle leading [] for repeated
        if parts[0] == "repeated":
            self.repeated = True
            parts.pop(0)
        else:
            self.repeated = False

        if len(parts) < 2:
            raise ValueError(f"Invalid field: {line}")

        self.proto_type = parts[0]
        self.name = parts[1]
        self.tag = int(parts[-1]) if parts[-1].isdigit() else parts[-1]

    def get_proto_string(self) -> str:
        repeated = "repeated " if self.repeated else ""
        return f"  {repeated}{self.proto_type} {self.name} = {self.tag};"


class ProtobufMessage:
    def __init__(self, name: str):
        self.name = name
        self.fields: List[ProtobufField] = []

    def add_field(self, line: str):
        self.fields.append(ProtobufField(line))

    def get_proto_string(self) -> str:
        field_lines = "\n".join(f.get_proto_string() for f in self.fields)
        return f"message {self.name} {{\n{field_lines}\n}}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fields": [{"type": f.proto_type, "name": f.name, "tag": f.tag,
                        "repeated": f.repeated} for f in self.fields],
        }


# Prepended to full-store .proto output (save_proto_file, get_all_messages_proto)
PROTO_SYNTAX_LINE = 'syntax = "proto3";'


class ProtobufStore:
    def __init__(self):
        self._messages: dict[str, ProtobufMessage] = {}

    def _apply_fields(self, msg: ProtobufMessage, fields_str: str) -> Optional[str]:
        """Populate msg.fields from fields_str. Returns an error string, or None on success."""
        msg.fields.clear()
        for line in fields_str.strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    msg.add_field(line)
                except ValueError as e:
                    return f"Invalid field '{line}': {e}"
        return None

    def create_message(self, name: str, fields_str: str) -> str:
        if name in self._messages:
            return f'Message "{name}" already exists. Use update_message("{name}", ...) to change it.'
        msg = ProtobufMessage(name)
        err = self._apply_fields(msg, fields_str)
        if err:
            return err
        self._messages[name] = msg
        return f"Message '{name}' created."

    def update_message(self, name: str, fields_str: str) -> str:
        if name not in self._messages:
            available = self.list_message_names()
            if not available:
                return f'No message named "{name}". No messages defined yet. Use create_message first.'
            return f'No message named "{name}". Available: {", ".join(available)}. Use create_message to add it.'
        msg = self._messages[name]
        err = self._apply_fields(msg, fields_str)
        if err:
            return err
        return f"Message '{name}' updated."

    def get_message(self, name: str) -> Optional[ProtobufMessage]:
        return self._messages.get(name)

    def delete_message(self, name: str) -> bool:
        if name in self._messages:
            del self._messages[name]
            return True
        return False

    def list_message_names(self) -> List[str]:
        return list(self._messages.keys())

    def get_proto_string(self) -> str:
        if not self._messages:
            return f"{PROTO_SYNTAX_LINE}\n\n// No messages defined yet."
        body = "\n\n".join(m.get_proto_string() for m in self._messages.values())
        return f"{PROTO_SYNTAX_LINE}\n\n{body}"

    def clear(self):
        self._messages.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Global store accessor (module-level singleton for the agent session)
# ─────────────────────────────────────────────────────────────────────────────

_store: Optional[ProtobufStore] = None


def _get_protobuf_store() -> ProtobufStore:
    global _store
    if _store is None:
        _store = ProtobufStore()
    return _store


# ─────────────────────────────────────────────────────────────────────────────
# ReAct Agent
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [
    read_file,
    write_to_file,
    run_terminal_command,
    list_messages,
    show_message,
    create_message,
    update_message,
    delete_message,
    get_all_messages_proto,
    save_proto_file,
    compile_proto_to_python,
    compile_proto_to_cpp,
    ask_user,
    tell_user,
]


def _build_tool_signature_reference(tools: List[Callable]) -> str:
    """One line per tool: name(signature) — matches what the agent parses and executes."""
    return "\n".join(f"  {fn.__name__}{inspect.signature(fn)}" for fn in tools)


# Shown in the system prompt so the model's <action> matches parse_action() + tool dispatch.
ACTION_FORMAT_GUIDE = """
### How to write <action> (must match the agent parser)
The agent takes **exactly one** Python-style call between `<action>` and `</action>`:

  <action>tool_name(arg1, arg2, ...)</action>

Rules:
- **One call only** — no semicolons, no multiple statements, no extra text after the closing `)`.
- **Name** — must be one of the tool function names below (snake_case), e.g. `list_messages`, not `ListMessages`.
- **Parentheses** — the parser finds the matching `)` for the `(` after `tool_name`. Parentheses **inside** quoted strings are literal and do not end the call.
- **Commas** — separate top-level arguments only. Commas inside `"..."` or `'...'` belong to the string.
- **Strings** — use `"double"` or `'single'` quotes. Inside strings, escape backslash and quotes: `\\\\` `\\"` `\\'`, and newlines as `\\n`, tabs as `\\t`.
- **No raw newlines inside the call** — keep the whole call on one line. For multi-line content (e.g. protobuf fields), put `\\n` inside the string.
- **Paths** — use absolute paths as string arguments, e.g. `read_file("/Users/me/proj/x.proto")`.
- **No-argument tools** — `list_messages()` and `get_all_messages_proto()` use empty `()`.
- **`protoc`** — `compile_proto_to_python` / `compile_proto_to_cpp` need `protoc` on PATH (from **protobuf-protoc-bin** after `uv sync`; run with `uv run`).

**Examples (valid shape):**
  list_messages()
  get_all_messages_proto()
  show_message("Robot")
  delete_message("Robot")
  create_message("Robot", "int32 id = 1\\nstring name = 2")
  update_message("Robot", "int32 id = 1\\nstring name = 2")
  save_proto_file("/Users/you/project/out.proto")
  read_file("/Users/you/project/in.proto")
  write_to_file("/Users/you/out.txt", "a\\nb")
  run_terminal_command("ls -la")
  ask_user("Which message should I open?")
  tell_user("Done. What would you like to do next?")
  compile_proto_to_python("/Users/you/project/messages.proto", "/Users/you/project/gen/py")
  compile_proto_to_cpp("/Users/you/project/messages.proto", "/Users/you/project/gen/cpp")

**Exact signatures (argument names and order):**
${tool_signature_reference}
"""


SYSTEM_PROMPT_TEMPLATE = """
You are **${agent_name}**, a protobuf definition assistant. Your name is ${agent_name}; when asked who you are or what your name is, answer ${agent_name}.
The user describes protobuf messages they want to define, and you maintain an in-memory store of message definitions throughout the conversation.

${action_format_guide}

User requests and their tool calls:

1. "list messages" → list_messages()
   Lists all defined message names, one per line.

2. "show all messages" / "show everything" → get_all_messages_proto()
   Returns the full .proto text for all messages.

3. "show message <Name>" → show_message("<Name>")
   Shows the .proto definition for the named message.
   Example: show_message("Robot")

4. "create message <Name> with fields: <fields>" → create_message("<Name>", "<fields>")
   "define message <Name>" / "add message <Name>" / "new message <Name>" (only when the name is new)
   Fails if the message already exists — then use update_message.

5. "update message <Name> with fields: <fields>" / "modify <Name>" → update_message("<Name>", "<fields>")
   Replaces the field list of an existing message. Fails if the message does not exist — then use create_message.

   For both create_message and update_message, fields_str is a newline-separated list of field definitions.
   Field format: "<proto_type> <field_name> = <tag_number>"
   For repeated fields, prepend "repeated ": "repeated int32 ids = 1"
   Example fields_str: "int32 id = 1\\nstring name = 2\\nrepeated double scores = 3"

6. "delete message <Name>" → delete_message("<Name>")
   Removes the named message from the store.

7. "save to <path>" → save_proto_file("<absolute_path>")
   Writes all current message definitions as a .proto file.
   Example: save_proto_file("/Users/me/my_project/robot.proto")

8. Compile .proto (protobuf **messages** only — no gRPC stubs):
   compile_proto_to_python("...", "...") → Python `*_pb2.py` via `protoc --python_out` (same `protoc` as below).
   compile_proto_to_cpp("...", "...") → C++ `*.pb.h` / `*.pb.cc` via `protoc --cpp_out`.
   Both require **`protoc` on PATH** — the project depends on **protobuf-protoc-bin**; use `uv run` so `.venv/bin/protoc` is found (or use a system-installed protoc).
   One .proto file per call. To compile everything currently in memory, call save_proto_file first, then compile that path.
   To compile several on-disk .proto files, call the compile tool once per file (or use run_terminal_command with a shell loop).

9. Generic file operations (if needed):
   read_file("<absolute_path>")
   write_to_file("<absolute_path>", "content with \\n for newlines")
   run_terminal_command("shell command")

10. Ask the user a question when intent is unclear:
   ask_user("<question>")
   The user's reply is returned as <feedback> — use it to decide the next action.

11. Reply to the user when you are done with their current request:
   tell_user("<your message to the user>")
   Include summaries, explanations, or data from prior <feedback> in this string.
   **Always end handling of each user message with tell_user(...)** (unless you are still calling other tools first).
   **Default:** your tell_user message should invite the user to give their next instruction (e.g. ask what they want to do next), unless the user already stated a clear follow-up.

Rules:
- Follow the **How to write <action>** section above: your tool call must be parseable exactly as described (one `tool_name(...)`, quoting, `\\n`, absolute paths).
- Every reply must include <thought> first, then exactly one <action>...</action>.
- Take **only one action** per model response, then stop; you will receive <feedback>...</feedback> with the tool result (or error) and may act again.
- After each tool action, continue in the same user turn with another <thought> + <action> until you call tell_user.
- **When the user asks to write/save to a .proto file:** call get_all_messages_proto() as an <action>, then in a later step call tell_user with the .proto text (or a summary plus path).
- File paths in tool arguments must be absolute, not relative
- The store persists across all turns; you do not need to re-create messages each turn

Available tools (names and parameters — same as in **Exact signatures** above):
${tool_list}

---
Environment: ${operating_system}, working directory: ${project_directory}
Files in this directory: ${file_list}
"""


class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Create a .env file with your API key.")
        base_url = os.getenv("ANTHROPIC_BASE_URL", MINIMAX_ANTHROPIC_BASE_URL).strip().rstrip("/")
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def run(self, user_input: str, max_steps: int = 64) -> None:
        system_prompt = self.render_system_prompt(SYSTEM_PROMPT_TEMPLATE)
        messages = [{"role": "user", "content": [{"type": "text", "text": f"<question>{user_input}</question>"}]}]

        for step in range(1, max_steps + 1):
            _print_agent_turn_header(step)
            content = self.call_model(system_prompt, messages)

            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            thought_text = thought_match.group(1).strip() if thought_match else ""

            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                reason = "missing or invalid <action>"
                recovery_msg = (
                    "I could not run a tool from that reply "
                    f"({reason}). Please say what you want next, or repeat your request."
                )
                q = "What would you like me to do next?"
                sections: List[Tuple[str, str]] = []
                if thought_text:
                    sections.append(("Thought", thought_text))
                sections.append(("Message", recovery_msg))
                sections.append(("Question", q))
                _print_agent_turn_sections(sections)
                try:
                    user_reply = input(f"\n{T_USER} ").strip()
                except (EOFError, KeyboardInterrupt):
                    user_reply = ""
                recovery_feedback = (
                    f"<feedback>Recovery (no valid <action> in model reply; {reason}). "
                    f"The user was prompted and replied: {user_reply!r}. "
                    f"Continue with <thought> and <action>. "
                    f"If their reply is a new request, handle it; otherwise retry the last step.</feedback>"
                )
                messages.append({"role": "user", "content": [{"type": "text", "text": recovery_feedback}]})
                continue

            action = action_match.group(1).strip()
            try:
                tool_name, args = self.parse_action(action)
            except ValueError as e:
                feedback_text = (
                    f"<feedback>Could not parse <action> as tool_name(...). Reason: {e}. "
                    f"Use one line, e.g. tell_user(\"hello\") or list_messages(). "
                    f"Strings must be quoted; use \\n inside strings for newlines.</feedback>"
                )
                sections_err: List[Tuple[str, str]] = []
                if thought_text:
                    sections_err.append(("Thought", thought_text))
                sections_err.append(("Parse error", str(e)))
                _print_agent_turn_sections(sections_err)
                messages.append({"role": "user", "content": [{"type": "text", "text": feedback_text}]})
                continue

            if tool_name not in self.tools:
                available = ", ".join(sorted(self.tools.keys()))
                feedback_text = (
                    f"<feedback>No tool named \"{tool_name}\". "
                    f"Reason: that action does not exist. "
                    f"Available tools: {available}.</feedback>"
                )
                sections_ut: List[Tuple[str, str]] = []
                if thought_text:
                    sections_ut.append(("Thought", thought_text))
                sections_ut.append(("Action", tool_name))
                sections_ut.append(("Error", f'No tool named "{tool_name}". Available: {available}.'))
                _print_agent_turn_sections(sections_ut)
                messages.append({"role": "user", "content": [{"type": "text", "text": feedback_text}]})
                continue

            if tool_name == "ask_user":
                prompt = args[0]
                action_display = "ask_user"
                sections_ask: List[Tuple[str, str]] = []
                if thought_text:
                    sections_ask.append(("Thought", thought_text))
                sections_ask.append(("Action", action_display))
                sections_ask.append(("Question", prompt))
                _print_agent_turn_sections(sections_ask)
                tool_result = self.tools["ask_user"](prompt)
            elif tool_name == "tell_user":
                msg = args[0]
                action_display = "tell_user"
                sections_tell: List[Tuple[str, str]] = []
                if thought_text:
                    sections_tell.append(("Thought", thought_text))
                sections_tell.append(("Action", action_display))
                sections_tell.append(("Message", msg))
                _print_agent_turn_sections(sections_tell)
                tool_result = self.tools["tell_user"](msg)
            else:
                args_display = ", ".join(repr(a) for a in args)
                action_display = f"{tool_name}({args_display})"
                tool_fn = self.tools[tool_name]
                try:
                    tool_result = tool_fn(*args)
                except TypeError as e:
                    sig = inspect.signature(tool_fn)
                    tool_result = (
                        f"Wrong parameters for {tool_name}{sig}. Reason: {e}. "
                        f"Match the argument count and types to this signature."
                    )
                except Exception as e:
                    tool_result = f"{tool_name} raised {type(e).__name__}: {e}"
                feedback_str = tool_result if isinstance(tool_result, str) else str(tool_result)
                sections_fb: List[Tuple[str, str]] = []
                if thought_text:
                    sections_fb.append(("Thought", thought_text))
                sections_fb.append(("Action", action_display))
                sections_fb.append(("Feedback", feedback_str))
                _print_agent_turn_sections(sections_fb)

            feedback_str = tool_result if isinstance(tool_result, str) else str(tool_result)
            messages.append({"role": "user", "content": [{"type": "text", "text": f"<feedback>{feedback_str}</feedback>"}]})

            if tool_name == "tell_user":
                return

        _print_agent_line(f"⚠️ Stopped: exceeded {max_steps} agent steps for this message.")

    def get_tool_list(self) -> str:
        tool_descriptions = []
        for func in self.tools.values():
            sig = str(inspect.signature(func))
            doc = inspect.getdoc(func) or "(no description)"
            tool_descriptions.append(f"- {func.__name__}{sig}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, template: str) -> str:
        tool_list = self.get_tool_list()
        try:
            files = ", ".join(os.path.abspath(os.path.join(self.project_directory, f))
                             for f in os.listdir(self.project_directory))
        except OSError:
            files = "(unable to list directory)"
        return Template(template).substitute(
            agent_name=AGENT_NAME,
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=files,
            project_directory=os.path.abspath(self.project_directory),
            action_format_guide=Template(ACTION_FORMAT_GUIDE).substitute(
                tool_signature_reference=_build_tool_signature_reference(list(self.tools.values())),
            ),
        )

    def get_operating_system_name(self) -> str:
        return {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(platform.system(), "Unknown")

    def _extract_text_from_message(self, message) -> str:
        return "".join(block.text for block in message.content if block.type == "text")

    def call_model(self, system_prompt: str, messages: list):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=messages,
        )
        content = self._extract_text_from_message(message)
        messages.append({"role": "assistant", "content": message.content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        """
        Parse a single tool call: name(arg1, arg2, ...). The closing ) matches the ( after
        the name (greedy regex on (.*) would break if the last ) appears inside a string).
        """
        code_str = code_str.strip()
        m = re.match(r"^(\w+)\s*\(", code_str)
        if not m:
            raise ValueError("Expected tool_name(...) — identifier and '('")

        func_name = m.group(1)
        i = m.end()
        depth = 1
        in_string = False
        string_char = None

        while i < len(code_str) and depth > 0:
            c = code_str[i]
            if not in_string:
                if c in ('"', "'"):
                    in_string = True
                    string_char = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            else:
                if c == "\\" and i + 1 < len(code_str):
                    i += 2
                    continue
                if c == string_char:
                    in_string = False
                    string_char = None
            i += 1

        if depth != 0:
            raise ValueError("Unclosed ')' in action (check balanced parentheses and string quotes)")

        close_idx = i - 1
        args_str = code_str[m.end() : close_idx]
        tail = code_str[i:].strip()
        if tail:
            raise ValueError(f"Unexpected text after tool call: {tail!r}")

        raw_args = self._split_action_arguments(args_str)
        return func_name, [self._parse_single_arg(a) for a in raw_args]

    def _split_action_arguments(self, args_str: str) -> List[str]:
        """Split top-level comma-separated arguments (respects quotes and nested parens)."""
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        paren_depth = 0
        i = 0
        while i < len(args_str):
            char = args_str[i]
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == "(":
                    paren_depth += 1
                    current_arg += char
                elif char == ")":
                    paren_depth -= 1
                    current_arg += char
                elif char == "," and paren_depth == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i - 1] != "\\"):
                    in_string = False
                    string_char = None
            i += 1
        if current_arg.strip():
            args.append(current_arg.strip())
        return args

    def _parse_single_arg(self, arg_str: str):
        arg_str = arg_str.strip()
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            inner = arg_str[1:-1]
            inner = inner.replace('\\"', '"').replace("\\'", "'")
            inner = inner.replace("\\n", "\n").replace("\\t", "\t")
            inner = inner.replace("\\r", "\r").replace("\\\\", "\\")
            return inner
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            return arg_str


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("project_directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(project_directory: str):
    load_dotenv()
    project_dir = os.path.abspath(project_directory)

    print("=" * 60)
    print(f"  Protobuf Message Agent — {AGENT_NAME}")
    print("  Type 'list messages' to see all messages")
    print("  Type 'show message <Name>' to see one message")
    print("  Type 'save to <path>' to write a .proto file")
    print("  Type Ctrl+C to exit")
    print("=" * 60)

    agent = ReActAgent(tools=TOOLS, model="MiniMax-M2.7", project_directory=project_dir)

    while True:
        try:
            user_input = input(f"\n{T_USER} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        try:
            agent.run(user_input)
        except Exception as e:
            _print_agent_line(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
