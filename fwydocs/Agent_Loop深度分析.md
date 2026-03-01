# Nanobot Agent Loop 深度分析

## 概述

Agent Loop 是 nanobot 的**核心处理引擎**，负责协调整个 AI 助手的运行流程。

### 核心问题解答

#### 1. 是否用了 LangGraph 这样的框架？

**答案：NO。完全自实现，没有依赖任何 agent 框架。**

nanobot 的 Agent Loop 是纯自实现的，没有依赖 LangGraph、AutoGPT、CrewAI 或其他 agent 框架。这正是 nanobot "极简" 的体现——用 ~200 行代码实现了传统 agent 框架千行代码的功能。

#### 2. Agent Loop 可以理解为 ReAct 架构吗？

**答案：YES，但更准确的说法是 ReAct 的"循环迭代"形式。**

ReAct (Reasoning + Acting) 的核心是：
```
思考 (Thought) → 行动 (Action) → 观察 (Observation) → 再思考 → ...
```

nanobot Agent Loop 正是这个流程：

```
LLM Call (Thought + Action)
    ↓
Parse Tool Calls (Action)
    ↓
Execute Tools (Observation)
    ↓
Feed Back to LLM (for next Thought)
    ↓
Loop until no more tool calls
```

所以可以说：**nanobot Agent Loop ≈ ReAct 循环实现**

---

## 源码阅读路线图

### 第一步：理解整体架构（15 分钟）

**文件**：`nanobot/agent/loop.py`

**关键类**：`AgentLoop`

**应该看的部分**：
1. 类定义和 `__init__` (行 35-126)
2. 核心方法列表

```python
class AgentLoop:
    """主类"""

    def __init__(self, ...):
        """初始化，注册所有工具"""
        # 行 49-107

    async def run(self):
        """主循环，监听 Message Bus"""
        # 行 250-267

    async def _dispatch(self, msg: InboundMessage):
        """分发消息到处理函数"""
        # 行 285-305

    async def _process_message(self, msg: InboundMessage):
        """核心消息处理"""
        # 行 321-444

    async def _run_agent_loop(self, initial_messages):
        """ReAct 循环的核心实现"""
        # 行 174-248
```

**理解路线**：
```
run()
  ↓
_dispatch()
  ↓
_process_message()
  ↓
_run_agent_loop() ← ⭐ 最核心
```

---

### 第二步：深入 ReAct 循环实现（30 分钟）

**文件**：`nanobot/agent/loop.py` 行 174-248

**方法**：`_run_agent_loop()`

这是 **最关键的方法**，体现了 ReAct 的完整流程：

```python
async def _run_agent_loop(
    self,
    initial_messages: list[dict],
    on_progress: Callable[..., Awaitable[None]] | None = None,
) -> tuple[str | None, list[str], list[dict]]:
    """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""

    messages = initial_messages
    iteration = 0
    final_content = None
    tools_used: list[str] = []

    while iteration < self.max_iterations:  # ⭐ ReAct 循环开始
        iteration += 1

        # 第一步：调用 LLM (Thought + possible Actions)
        response = await self.provider.chat(
            messages=messages,
            tools=self.tools.get_definitions(),  # 传递所有可用工具 schema
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 检查 LLM 是否返回了 tool calls (Actions)
        if response.has_tool_calls:
            # 第二步：LLM 要求执行工具 (Acting)

            # 构建 tool call 消息格式
            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in response.tool_calls
            ]

            # 添加 assistant 消息到历史 (LLM 的思考和决定)
            messages = self.context.add_assistant_message(
                messages, response.content, tool_call_dicts,
                reasoning_content=response.reasoning_content,
            )

            # 第三步：执行每个工具 (Observation)
            for tool_call in response.tool_calls:
                tools_used.append(tool_call.name)

                # 执行工具
                result = await self.tools.execute(tool_call.name, tool_call.arguments)

                # 添加工具结果到消息 (Observation 反馈给 LLM)
                messages = self.context.add_tool_result(
                    messages, tool_call.id, tool_call.name, result
                )

            # ⭐ 循环：下次迭代会用这些消息调用 LLM，LLM 会看到工具结果

        else:
            # LLM 没有调用工具，直接返回最终回复
            clean = self._strip_think(response.content)
            if response.finish_reason == "error":
                final_content = clean or "Sorry, I encountered an error..."
                break
            messages = self.context.add_assistant_message(
                messages, clean, reasoning_content=response.reasoning_content,
            )
            final_content = clean
            break  # 循环结束

    # 如果达到最大迭代次数
    if final_content is None and iteration >= self.max_iterations:
        logger.warning("Max iterations ({}) reached", self.max_iterations)
        final_content = f"I reached the maximum number of tool call iterations..."

    return final_content, tools_used, messages
```

**关键理解点**：

1. **ReAct 循环构造**：
   ```
   while iteration < max_iterations:
       LLM Call (see all tools and history)
       if has_tool_calls:
           for each tool_call:
               Execute & capture result
               Add result to messages
       else:
           No more tools needed → Done
   ```

2. **消息流向**：
   ```
   messages = [
       {"role": "system", "content": system_prompt},
       {"role": "user", "content": user_input},
       {"role": "assistant", "content": "I'll search...", "tool_calls": [...]},
       {"role": "tool", "tool_call_id": "...", "name": "web_search", "content": "results..."},
       {"role": "assistant", "content": "Based on the search...", "tool_calls": [...]},
       {"role": "tool", ...},
       ... (多轮迭代)
       {"role": "assistant", "content": "Final answer"}
   ]
   ```

3. **为什么这样设计**：
   - **无状态迭代**：每次都把完整的历史传给 LLM，LLM 可以看到全上下文
   - **简洁明了**：不需要复杂的状态机，只用消息列表表示对话历史
   - **兼容所有 LLM**：任何支持 tool_calls 的 LLM 都能用

---

### 第三步：理解上下文构建（20 分钟）

**文件**：`nanobot/agent/context.py`

**关键方法**：

1. **`build_messages()`** (行 105-120)
   ```python
   def build_messages(
       self,
       history: list[dict[str, Any]],          # 过去的对话
       current_message: str,                   # 用户新消息
       skill_names: list[str] | None = None,
       media: list[str] | None = None,
       channel: str | None = None,
       chat_id: str | None = None,
   ) -> list[dict[str, Any]]:
       """Build the complete message list for an LLM call."""
       return [
           {"role": "system", "content": self.build_system_prompt(skill_names)},
           *history,  # ⭐ 过去的对话
           {"role": "user", "content": self._build_runtime_context(channel, chat_id)},
           {"role": "user", "content": self._build_user_content(current_message, media)},
       ]
   ```

2. **`build_system_prompt()`** (行 26-53)

   系统提示词包括：
   - nanobot 身份介绍
   - 运行时信息 (OS, Python 版本, 工作目录)
   - 记忆内容 (`~/.nanobot/memory/MEMORY.md`)
   - 激活的技能
   - 可用技能列表

3. **`add_assistant_message()`** (行 148-161)
   - 添加 LLM 的响应消息
   - 如果有 tool_calls，也一起添加

4. **`add_tool_result()`** (行 140-146)
   - 添加工具执行结果
   - 格式：`{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}`

**数据流**：
```
用户消息
    ↓
load_history (最近 N 条)
    ↓
build_system_prompt() ← 注册表、记忆、技能
    ↓
build_messages() ← 组装最终消息列表
    ↓
LLM Call
    ↓
... (迭代)
```

---

### 第四步：理解工具系统（20 分钟）

**文件**：
- `nanobot/agent/tools/registry.py` - 工具注册表
- `nanobot/agent/tools/base.py` - 工具基类
- `nanobot/agent/tools/shell.py` - 具体工具示例

#### 4.1 工具注册表

**文件**：`nanobot/agent/tools/registry.py`

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool

    def get_definitions(self) -> list[dict[str, Any]]:
        """获取所有工具的 OpenAI schema 格式

        这会被传给 LLM 的 tools 参数，告诉 LLM 有哪些工具可用
        """
        return [tool.to_schema() for tool in self._tools.values()]

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """执行指定的工具

        1. 查找工具
        2. 验证参数
        3. 执行工具
        4. 捕获异常和结果
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found..."

        try:
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters..."

            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + "\n\n[Analyze the error above and try a different approach.]"

            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
```

#### 4.2 工具基类

**文件**：`nanobot/agent/tools/base.py`

```python
class Tool(ABC):
    """所有工具的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，LLM 会用这个来调用"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，告诉 LLM 这个工具做什么"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema 格式的参数定义"""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具，返回字符串结果"""
        pass

    def to_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

#### 4.3 具体工具示例：Shell 工具

**文件**：`nanobot/agent/tools/shell.py` (行 1-60)

```python
class ExecTool(Tool):
    """执行 shell 命令的工具"""

    def __init__(self, timeout: int = 60, working_dir: str | None = None, ...):
        self.timeout = timeout
        self.working_dir = working_dir
        # 安全检查：拒绝危险命令
        self.deny_patterns = [
            r"\brm\s+-[rf]{1,2}\b",      # rm -r, rm -rf
            r"\bdel\s+/[fq]\b",          # del /f, del /q
            # ... 更多危险命令
        ]

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory"
                }
            },
            "required": ["command"]
        }

    async def execute(self, command: str, working_dir: str | None = None) -> str:
        """执行命令"""
        # 安全检查
        if self._is_dangerous(command):
            return f"Error: Command denied for security reasons"

        # 执行
        try:
            result = await self._run_subprocess(command, working_dir or self.working_dir)
            return result[:500]  # 限制输出大小
        except Exception as e:
            return f"Error: {str(e)}"
```

#### 4.4 工具注册（在 Agent Loop 初始化）

**文件**：`nanobot/agent/loop.py` 行 109-125

```python
def _register_default_tools(self) -> None:
    """Register the default set of tools."""

    # 文件系统工具
    allowed_dir = self.workspace if self.restrict_to_workspace else None
    for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
        self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))

    # Shell 工具
    self.tools.register(ExecTool(
        working_dir=str(self.workspace),
        timeout=self.exec_config.timeout,
        restrict_to_workspace=self.restrict_to_workspace,
    ))

    # 网络工具
    self.tools.register(WebSearchTool(api_key=self.brave_api_key))
    self.tools.register(WebFetchTool())

    # 消息工具（用于发送信息到特定渠道）
    self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))

    # 生成子任务工具
    self.tools.register(SpawnTool(manager=self.subagents))

    # 计划任务工具
    if self.cron_service:
        self.tools.register(CronTool(self.cron_service))
```

**工具流程总结**：
```
Tool Schema → LLM → Tool Call Request → Tool Registry.execute()
→ Tool.execute() → Result String → back to LLM
```

---

### 第五步：理解 LLM 提供商层（20 分钟）

**文件**：`nanobot/providers/`

#### 5.1 基类和响应格式

**文件**：`nanobot/providers/base.py`

```python
@dataclass
class ToolCallRequest:
    """LLM 的工具调用请求"""
    id: str          # 工具调用 ID，用来关联结果
    name: str        # 工具名称
    arguments: dict  # 工具参数

@dataclass
class LLMResponse:
    """LLM 的响应"""
    content: str | None           # 文本内容
    tool_calls: list[ToolCallRequest] = ...  # 如果有工具调用
    finish_reason: str = "stop"   # 完成原因
    reasoning_content: str | None = None  # 思考内容（某些模型有）

    @property
    def has_tool_calls(self) -> bool:
        """是否有工具调用"""
        return len(self.tool_calls) > 0
```

#### 5.2 LiteLLM 提供商（支持 20+ 提供商）

**文件**：`nanobot/providers/litellm_provider.py`

```python
class LiteLLMProvider(LLMProvider):
    """
    基于 LiteLLM 的通用提供商。

    LiteLLM 是一个统一接口库，支持：
    - OpenAI (GPT)
    - Anthropic (Claude)
    - OpenRouter (所有模型)
    - DeepSeek
    - Gemini
    - ... 等 20+ 提供商
    """

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],  # OpenAI function schema 格式
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        调用 LLM

        1. 规范化消息格式
        2. 调用 LiteLLM acompletion()
        3. 解析工具调用
        4. 返回 LLMResponse
        """

        # 清理消息（某些提供商不支持空内容）
        messages = self._sanitize_empty_content(messages)

        # 调用 LLM (通过 LiteLLM)
        response = await acompletion(
            model=self.model_to_provider(model),  # 自动前缀
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            api_base=self.api_base,
            extra_headers=self.extra_headers,
        )

        # 解析工具调用
        tool_calls = []
        if response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LLMResponse(
            content=response.choices[0].message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
        )
```

#### 5.3 提供商注册表

**文件**：`nanobot/providers/registry.py`

```python
PROVIDERS = (
    ProviderSpec(
        name="openrouter",
        keywords=("openrouter", "claude", "gpt"),  # 自动匹配关键词
        litellm_prefix="openrouter",  # 自动添加前缀
        env_key="OPENROUTER_API_KEY",
    ),
    ProviderSpec(
        name="anthropic",
        keywords=("claude", "anthropic"),
        env_key="ANTHROPIC_API_KEY",
    ),
    # ... 更多提供商
)

def find_by_model(model: str) -> ProviderSpec | None:
    """根据模型名称找到提供商"""
    model_lower = model.lower()
    for spec in PROVIDERS:
        if any(kw in model_lower for kw in spec.keywords):
            return spec
    return None
```

**LLM 调用流程**：
```
Agent Loop
    ↓
await self.provider.chat(messages, tools, model, ...)
    ↓
LiteLLMProvider.chat()
    ↓
find_by_model(model) → ProviderSpec
    ↓
自动前缀、设置 env vars
    ↓
acompletion() via LiteLLM
    ↓
解析响应，返回 LLMResponse
    ↓
检查 has_tool_calls
```

---

### 第六步：消息处理主流程（20 分钟）

**文件**：`nanobot/agent/loop.py`

**核心方法链**：

1. **`run()`** (行 250-267) - 主循环

```python
async def run(self) -> None:
    """Run the agent loop, dispatching messages as tasks."""
    self._running = True
    await self._connect_mcp()  # MCP 连接
    logger.info("Agent loop started")

    while self._running:
        try:
            # 从 Message Bus 读取入站消息
            msg = await asyncio.wait_for(
                self.bus.consume_inbound(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            continue

        # 特殊命令处理
        if msg.content.strip().lower() == "/stop":
            await self._handle_stop(msg)
        else:
            # 创建异步任务处理消息（不阻塞）
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
```

2. **`_dispatch(msg)`** (行 285-305) - 消息分发

```python
async def _dispatch(self, msg: InboundMessage) -> None:
    """Process a message under the global lock."""
    async with self._processing_lock:  # 全局锁，防止并发问题
        try:
            response = await self._process_message(msg)  # ⭐ 处理消息
            if response is not None:
                await self.bus.publish_outbound(response)  # 发送回复
        except Exception:
            logger.exception("Error processing message")
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="Sorry, I encountered an error.",
            ))
```

3. **`_process_message(msg)`** (行 321-444) - 处理消息

**简化版**：
```python
async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
    """Process a single inbound message and return the response."""

    # 1. 获取或创建会话
    key = msg.session_key
    session = self.sessions.get_or_create(key)

    # 2. 处理特殊命令
    cmd = msg.content.strip().lower()
    if cmd == "/new":
        # 清空会话
        session.clear()
        self.sessions.save(session)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                              content="New session started.")

    # 3. 自动记忆整理（当会话过长时）
    if len(session.messages) - session.last_consolidated >= self.memory_window:
        # 后台启动记忆整理任务
        ...

    # 4. 构建消息列表
    history = session.get_history(max_messages=self.memory_window)
    initial_messages = self.context.build_messages(
        history=history,
        current_message=msg.content,
        media=msg.media,
        channel=msg.channel,
        chat_id=msg.chat_id,
    )

    # 5. 运行 ReAct 循环（最核心）
    final_content, tools_used, all_msgs = await self._run_agent_loop(
        initial_messages, on_progress=_bus_progress
    )

    # 6. 保存到会话历史
    self._save_turn(session, all_msgs, 1 + len(history))
    self.sessions.save(session)

    # 7. 返回回复
    return OutboundMessage(
        channel=msg.channel,
        chat_id=msg.chat_id,
        content=final_content,
    )
```

---

## 完整的请求-响应生命周期

```
用户在 Telegram 发送消息
         │
         ▼
    [TelegramChannel] 捕获消息
         │
         ▼
    InboundMessage(channel="telegram", content="Search for...", user_id="123")
         │
         ▼
    [Message Bus] 入站队列
         │
         ▼
    [Agent Loop] run()
    监听 bus.consume_inbound()
         │
         ▼
    接到消息，创建 asyncio.Task
         │
         ▼
    _dispatch(msg)
         │
    全局锁（防止并发竞态）
         │
         ▼
    _process_message(msg)
         │
    ├─→ [Session Manager] 加载会话历史
    │   (最近 100 条消息)
    │
    ├─→ [Context Builder] 构建消息列表
    │   - System prompt (身份 + 记忆 + 技能)
    │   - Past history (过去的对话)
    │   - Current message (用户新消息)
    │   - Runtime context (时间、渠道信息)
    │
    ├─→ [Agent Loop] _run_agent_loop() ← ⭐ 关键
    │   ├─→ while iteration < max_iterations:
    │   │   ├─→ [LLM Provider] await provider.chat(
    │   │   │       messages, tools, model
    │   │   │   )
    │   │   │   通过 LiteLLM 路由到具体提供商
    │   │   │   (OpenRouter/Anthropic/OpenAI/...)
    │   │   │
    │   │   ├─→ 接收 LLMResponse
    │   │   │   (content, tool_calls, finish_reason)
    │   │   │
    │   │   ├─→ if has_tool_calls:
    │   │   │   ├─→ for each tool_call:
    │   │   │   │   ├─→ [Tool Registry] execute(tool_name, args)
    │   │   │   │   ├─→ 执行具体工具（Shell/File/Web/...）
    │   │   │   │   ├─→ 捕获结果（限制 500 字符）
    │   │   │   │   └─→ 添加到 messages
    │   │   │   └─→ 循环下一次迭代
    │   │   │
    │   │   └─→ else (no tool calls):
    │   │       ├─→ LLM 完成回复
    │   │       └─→ 跳出循环
    │   │
    │   └─→ return (final_content, tools_used, messages)
    │
    ├─→ [Session Manager] 保存消息到会话
    │   (_save_turn() 截断大的工具结果)
    │
    └─→ 返回 OutboundMessage(content=final_content)
         │
         ▼
    [Message Bus] 出站队列
         │
         ▼
    [Channel Manager] 路由消息
         │
         ▼
    [TelegramChannel] 发送消息
         │
         ▼
    用户在 Telegram 收到回复 ✓
```

---

## 关键设计细节

### 1. 为什么用消息列表而不是状态机？

```python
# ❌ 传统状态机方式（复杂）
class AgentState:
    status: str  # "thinking", "executing_tool", "done"
    thoughts: str
    tool_call: ToolCall | None
    tool_result: str | None
    ...

# ✓ Nanobot 方式（简洁）
messages = [
    {"role": "system", ...},
    {"role": "user", "content": "user input"},
    {"role": "assistant", "content": "thinking...", "tool_calls": [...]},
    {"role": "tool", "content": "tool result"},
    ...
]
```

**优点**：
- LLM 可以看到完整历史
- 消息格式是 OpenAI 标准，兼容所有 LLM
- 无需维护复杂的状态转移
- 易于持久化（直接保存为 JSON）

### 2. 为什么限制工具结果 500 字符？

```python
# nanobot/agent/loop.py 行 47
_TOOL_RESULT_MAX_CHARS = 500

# 在 _run_agent_loop 中
if role == "tool" and len(content) > self._TOOL_RESULT_MAX_CHARS:
    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
```

**原因**：
- 防止消息列表无限增长
- 节省 token 和 API 费用
- LLM 的上下文窗口有限
- 工具结果通常只需要关键部分

### 3. 并发和锁

```python
# 全局处理锁
self._processing_lock = asyncio.Lock()

# 每个会话的记忆整理锁
self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock]

# 活跃任务追踪
self._active_tasks: dict[str, list[asyncio.Task]]
```

**为什么**：
- 防止多个消息同时修改同一个会话
- 防止记忆整理竞态条件
- 支持取消任务 (`/stop`)

### 4. MCP 集成

```python
# MCP (Model Context Protocol) 是外部工具服务器
async def _connect_mcp(self) -> None:
    """Connect to configured MCP servers (one-time, lazy)."""
    if self._mcp_connected or not self._mcp_servers:
        return

    # 第一次调用时连接，之后复用
    self._mcp_stack = AsyncExitStack()
    await connect_mcp_servers(self._mcp_servers, self.tools, ...)
    self._mcp_connected = True
```

MCP 工具会被自动注册到 Tool Registry，和内置工具一样使用。

---

## 源码阅读顺序总结

| 步骤 | 文件 | 行号 | 时间 | 关键函数 |
|------|------|------|------|---------|
| 1 | loop.py | 35-126 | 15 分钟 | `__init__`, 类结构 |
| 2 | loop.py | 174-248 | 30 分钟 | `_run_agent_loop()` ⭐⭐⭐ |
| 3 | context.py | 26-162 | 20 分钟 | `build_messages()`, `build_system_prompt()` |
| 4 | tools/*.py | 全部 | 20 分钟 | `ToolRegistry`, `Tool`, `ExecTool` |
| 5 | providers/*.py | 全部 | 20 分钟 | `LLMProvider`, `LiteLLMProvider` |
| 6 | loop.py | 250-444 | 20 分钟 | `run()`, `_dispatch()`, `_process_message()` |

**总时间**：约 2-3 小时深入理解

---

## 代码复杂度分析

| 组件 | 行数 | 复杂度 | 说明 |
|------|------|--------|------|
| Agent Loop 核心 (`_run_agent_loop`) | 75 | 低 | 清晰的 while 循环，无复杂逻辑 |
| Context Builder | 162 | 低 | 模板组装，无复杂算法 |
| Tool Registry | 67 | 低 | 简单的字典操作 |
| LLM Provider | 150+ | 中 | 多提供商支持，但大部分由 LiteLLM 处理 |
| 整体架构 | ~4000 | 低 | 模块化、无复杂交互 |

---

## 常见问题

### Q: Agent Loop 有没有使用 Async 流式处理？

**A**: 是的，但不是在核心循环中。

```python
# 核心循环是顺序的（await）
response = await self.provider.chat(...)  # 等待 LLM 响应
result = await self.tools.execute(...)     # 等待工具执行

# 但消息处理是异步的（多个用户可并发）
task = asyncio.create_task(self._dispatch(msg))  # 非阻塞
self._active_tasks.setdefault(msg.session_key, []).append(task)
```

### Q: 怎样修改 Agent Loop 的行为？

**A**: 主要有三种方式：

1. **修改 system prompt**：编辑 `AGENTS.md` 或 `SOUL.md`
2. **添加工具**：继承 `Tool` 基类，注册到 ToolRegistry
3. **修改 LLM 选择**：改 `config.json` 的 model 或 provider

### Q: Agent Loop 支持流式响应吗？

**A**: 不直接支持，但：
- 可以通过 `on_progress` 回调得到进度信息
- Message Tool 可以在循环中发送中间结果

---

## 总结

**Agent Loop 就是 ReAct 的循环实现**：

```
┌────────────────────────────────────────┐
│  while iteration < max_iterations:     │
│    ├─ Call LLM (see all tools & hist)  │ ← Thought
│    ├─ Parse tool calls                 │ ← Action
│    ├─ Execute tools                    │ ← Observation
│    ├─ Feed back to messages            │
│    └─ Loop                             │
└────────────────────────────────────────┘
```

关键特点：
- **无框架依赖**：纯自实现，~200 行核心代码
- **消息驱动**：完整历史一次次发给 LLM，无复杂状态
- **可扩展**：通过 Registry 模式添加工具、提供商、渠道
- **异步并发**：支持多个用户同时交互
- **标准兼容**：OpenAI function calling 格式，任何 LLM 都能用
