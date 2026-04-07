# Agent demo

**Smith** is a small terminal agent that helps you **design and maintain Protocol Buffers message definitions** in a project folder. You chat in natural language; it follows a ReAct-style loop (`<thought>` + one tool `<action>` per step) to list, create, update, or delete messages, **save** them to `.proto` files, and **compile** them to Python (`*_pb2.py`) or C++ (`.pb.h` / `.pb.cc`) using `protoc` from this repo’s dependencies.

## Demo
**内容：json数据-->抽象出protobuf消息定义-->接收人工指令修改-->编译成库，全程自然语言交互**
**仅用于展示Agent**
![Agent demo](output.gif)

## Model

The app uses the **Anthropic-compatible Messages API** with the chat model **`MiniMax-M2.7`**, pointed at MiniMax’s endpoint by default (`ANTHROPIC_BASE_URL` in `.env` can override). Set **`ANTHROPIC_API_KEY`** in a `.env` file (or your environment) — the same variable name the Anthropic client expects.

## Run

From the repo root (with `uv`):

```bash
uv sync
uv run python agent-smith.py /path/to/your_project
```

Use an existing directory; Smith uses it as the working directory context when listing files and resolving paths.
