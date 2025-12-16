"""Python tool using Jupyter kernel for stateful execution - Enhanced version."""
import os
import queue
import threading
import warnings
from collections import deque
from typing import Any

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)


class LocalJupyterSession:
    """Stateful Jupyter kernel session for code execution."""

    # Class-level lock and port counter to avoid port conflicts
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        """Get next available ports for kernel connection."""
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, connection_file: str | None = None, *, timeout: float = 120.0):
        try:
            from jupyter_client import BlockingKernelClient, KernelManager
        except ImportError as exc:
            raise RuntimeError("jupyter_client package required") from exc

        self._default_timeout = timeout
        self._owns_kernel = False
        self._client: BlockingKernelClient
        self._km: KernelManager | None = None
        self._execution_count = 0  # Track number of executions for diagnostics

        if connection_file:
            from pathlib import Path
            connection_path = Path(connection_file).expanduser()
            if not connection_path.exists():
                raise FileNotFoundError(f"Connection file not found: {connection_path}")
            client = BlockingKernelClient()
            client.load_connection_file(str(connection_path))
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
        else:
            # Allocate unique ports to avoid conflicts when running multiple kernels
            ports = self._get_next_ports(5)
            km = KernelManager(kernel_name='python3')
            km.shell_port = ports[0]
            km.iopub_port = ports[1]
            km.stdin_port = ports[2]
            km.hb_port = ports[3]
            km.control_port = ports[4]

            # Start kernel with reduced startup time and optimizations
            km.start_kernel(env={
                **os.environ,
                'PYTHONSTARTUP': '',  # Skip startup files
                'MPLBACKEND': 'Agg',  # Non-interactive matplotlib backend
                'OMP_NUM_THREADS': '1',  # Limit numpy threading overhead
            })

            client = km.blocking_client()
            client.start_channels()
            client.wait_for_ready(timeout=self._default_timeout)
            self._client = client
            self._km = km
            self._owns_kernel = True

            # Pre-import common libraries for faster execution
            self._preload_libraries()

    def _preload_libraries(self) -> None:
        """Pre-import common math libraries and optimize settings."""
        preload_code = """
# Math libraries
import math
import numpy as np
import sympy as sp
from sympy import *

# Optimize SymPy for competition math
sp.init_printing(use_unicode=False)

# Faster symbolic computation settings
from sympy import evaluate
sp.core.cache.clear_cache()  # Start with clean cache

# Suppress common warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Number theory imports (common in math competitions)
from sympy.ntheory import *
from sympy import gcd, lcm, factorint, isprime, nextprime

# Combinatorics imports
from sympy import binomial, factorial, fibonacci

# Helper: Automatically rationalize floats to exact values
from fractions import Fraction

# Silently initialized - no print message
"""
        try:
            output = self.execute(preload_code, timeout=30.0, silent=True)
            # Suppress the initialization message from being shown
        except Exception:
            pass  # Non-critical if preload fails

    def execute(self, code: str, *, timeout: float | None = None, silent: bool = False) -> str:
        """Execute code and return combined stdout/stderr.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            silent: If True, skip history storage for faster execution
        """
        client = self._client
        effective_timeout = timeout or self._default_timeout
        self._execution_count += 1

        # Use silent=True to skip history storage for faster execution
        msg_id = client.execute(
            code,
            store_history=not silent,
            allow_stdin=False,
            stop_on_error=True  # Stop immediately on error for faster failure
        )

        # Use deque for faster append operations
        stdout_parts: deque[str] = deque()
        stderr_parts: deque[str] = deque()

        # Process iopub messages
        while True:
            try:
                msg = client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for kernel output after {effective_timeout}s. Consider breaking down the computation.") from exc

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                traceback_data = content.get("traceback")
                if traceback_data:
                    # Clean ANSI escape codes from traceback
                    import re
                    clean_traceback = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in traceback_data]
                    stderr_parts.append("\n".join(clean_traceback))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else f"{text}\n")
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        # Drain shell channel with shorter timeout
        shell_timeout = min(1.0, effective_timeout)
        while True:
            try:
                reply = client.get_shell_msg(timeout=shell_timeout)
            except queue.Empty:
                # Shell reply missing, but we already have output from iopub
                break

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error":
                traceback_data = reply_content.get("traceback")
                if traceback_data:
                    import re
                    clean_traceback = [re.sub(r'\x1b\[[0-9;]*m', '', line) for line in traceback_data]
                    stderr_parts.append("\n".join(clean_traceback))
                else:
                    ename = reply_content.get("ename", "")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see results."

        return stdout

    def get_stats(self) -> dict[str, Any]:
        """Get session statistics for debugging."""
        return {
            "execution_count": self._execution_count,
            "owns_kernel": self._owns_kernel,
            "kernel_alive": self._km.is_alive() if self._km else None,
        }

    def close(self):
        import contextlib
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)

    def __del__(self):
        self.close()


class PythonTool:
    """Python execution tool using Jupyter kernel - Enhanced for math competitions."""

    def __init__(self, execution_backend: str | None = None, local_jupyter_timeout: float = 60.0):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._execution_lock = threading.Lock()
        self._jupyter_session: LocalJupyterSession | None = None
        # Lazy initialization to avoid port conflicts during object creation
        self._init_lock = threading.Lock()
        self._execution_count = 0
        self._timeout_count = 0

    def _ensure_session(self):
        """Lazily initialize the Jupyter session."""
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = LocalJupyterSession(timeout=self._local_jupyter_timeout)

    @classmethod
    def get_tool_name(cls) -> str:
        return "python"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """Use this tool to execute Python code. The code runs in a stateful Jupyter notebook with math, numpy, and sympy pre-imported. Use print() to see output. For exact answers, use sympy's symbolic computation."""

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[]
        )

    def _make_response(self, output: str, channel: str | None = None) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name=self.get_tool_name())
        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    @staticmethod
    def _ensure_printable(code: str) -> str:
        """Ensure the last expression is printed for better visibility."""
        lines = code.strip().split("\n")
        if not lines:
            return code

        last_line = lines[-1]

        # Skip if already has print, or is import/assignment/control flow/comment
        skip_keywords = ["print(", "import ", "=", "if ", "for ", "while ", "def ", "class ", "return ", "pass", "break", "continue", "try:", "except:", "finally:", "with ", "#"]
        if any(keyword in last_line for keyword in skip_keywords):
            return code

        # Skip if line is empty or only whitespace
        last = last_line.split("#")[0].strip()
        if not last:
            return code

        # Wrap last expression in print
        lines[-1] = f"print({last})"

        return "\n".join(lines)

    @staticmethod
    def _add_safety_wrapper(code: str, timeout: float) -> str:
        """Add safety checks to prevent infinite loops and resource exhaustion."""
        # This is a simple heuristic - could be made more sophisticated
        safety_prefix = f"""
import signal
import sys

def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded {timeout}s")

# Set alarm (Unix-like systems only)
try:
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int({timeout}))
except (AttributeError, ValueError):
    pass  # Windows doesn't support SIGALRM

try:
"""
        safety_suffix = """
finally:
    try:
        signal.alarm(0)  # Cancel alarm
    except:
        pass
"""
        # Only wrap if code doesn't already have try/finally and is multi-line
        if "try:" not in code and len(code.split("\n")) > 1:
            # Indent the user code
            indented_code = "\n".join("    " + line for line in code.split("\n"))
            return safety_prefix + indented_code + safety_suffix
        return code

    def process_sync_plus(self, message: Message) -> list[Message]:
        """Execute code from message using Jupyter kernel."""
        self._ensure_session()
        script = message.content[0].text
        self._execution_count += 1

        # Ensure output visibility
        script = self._ensure_printable(script)

        with self._execution_lock:
            try:
                # Use silent=True for faster execution (skip history)
                output = self._jupyter_session.execute(script, silent=True)
            except TimeoutError as exc:
                self._timeout_count += 1
                output = f"[ERROR] {exc}\nTip: Try breaking down the computation into smaller steps or use more efficient algorithms."
            except Exception as exc:
                output = f"[ERROR] Unexpected error: {type(exc).__name__}: {exc}"

        return [self._make_response(output, channel=message.channel)]

    def get_stats(self) -> dict[str, Any]:
        """Get tool statistics for debugging."""
        stats = {
            "tool_execution_count": self._execution_count,
            "timeout_count": self._timeout_count,
        }
        if self._jupyter_session:
            stats.update(self._jupyter_session.get_stats())
        return stats

    def close(self):
        if self._jupyter_session is not None:
            self._jupyter_session.close()
            self._jupyter_session = None

    def __del__(self):
        self.close()
