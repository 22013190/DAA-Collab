"""
A2A-Compliant Analysis Agent Service using Official A2A SDK
===========================================================

This creates a proper A2A-compliant service using the official a2a-sdk.
"""

"""
CODING STANDARDS FOR THIS FILE:
================================
1. NEVER USE UNICODE EMOJIS OR SPECIAL CHARACTERS
   - Windows console (cp1252 codec) cannot display Unicode emojis
   - This causes UnicodeEncodeError and server crashes
   - All emojis have been completely removed from this file

2. STICK TO ASCII TEXT ONLY:
   - Use plain text descriptions instead of emojis
   - Use status indicators like "OK", "ERROR", "WARNING"
   - Use ASCII symbols if needed: *, +, -, =, |, etc.

3. ALWAYS TEST ON WINDOWS CONSOLE BEFORE DEPLOYMENT
================================
"""





import os
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Official A2A SDK imports
import httpx
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.push_notification_sender import PushNotificationSender
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    InvalidParamsError,
    Message,
    Part,
    SendMessageRequest,
    Task,
    TaskState,
    TextPart,
    DataPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError

from src.agents.analysis.pipeline.streaming_utils import (
    StatusAccumulator,
    StreamThrottle,
    ascii_sanitize,
    strip_think_blocks,
)

from structures.interfaces.a2a_schema import (
    A2AUpdaterArtifactAdapter,
    A2AUpdaterStatusAdapter,
    SchemaValidator,
)

# Our analysis agent imports
# The modular pipeline is the preferred implementation in this extracted repo.
# Keep the legacy react agent optional so the service can import/run even when
# legacy dependencies (or helper APIs) are not present.
from src.agents.analysis.pipeline import create_analysis_pipeline_agent

# Add current directory to Python path
# current_dir = Path(__file__).parent
# sys.path.insert(0, str(current_dir))


class SimplePushNotificationSender(PushNotificationSender):
    """Simple in-memory push notification sender for A2A compliance"""

    def __init__(self, httpx_client=None):
        self.httpx_client = httpx_client
        print("SimplePushNotificationSender initialized (no-op implementation)")

    async def send_notification(self, task: Task) -> None:
        """Send push notification for task updates (simple logging implementation)"""
        print(f"Push notification: Task {task.id} status: {task.status}")


class A2AAnalysisAgentExecutor(AgentExecutor):
    """AgentExecutor for the A2A Analysis Agent following the tutorial pattern."""

    def __init__(self):
        """Initialize the AgentExecutor."""
        self.analysis_agent = None
        self.pipeline_agent = None
        self.session_id = None
        self.pipeline_mode = os.getenv("ANALYSIS_PIPELINE_MODE", "loop").lower()

    async def initialize_analysis_agent(self):
        """Initialize the LangGraph analysis agent"""
        self.pipeline_mode = os.getenv("ANALYSIS_PIPELINE_MODE", "loop").lower()
        use_pipeline = os.getenv("ANALYSIS_PIPELINE_ENABLED", "true").lower() in ("1", "true", "yes")
        if use_pipeline:
            if not self.pipeline_agent:
                self.pipeline_agent = await create_analysis_pipeline_agent()
                print("Modular analysis pipeline initialized")
        else:
            if not self.analysis_agent:
                try:
                    from src.agents.analysis.langgraph_react_analysis_agent import create_analysis_agent
                except Exception as e:
                    raise RuntimeError(
                        "Legacy react analysis agent is not available. "
                        "Set ANALYSIS_PIPELINE_ENABLED=true to use the modular pipeline, "
                        "or include the legacy agent module and its compatible dependencies. "
                        f"Original import error: {e}"
                    ) from e

                self.analysis_agent = await create_analysis_agent()
                self.session_id = await self.analysis_agent.start_session()
                print(f"Analysis agent initialized with session: {self.session_id}")

    async def _execute_pipeline_streaming(
        self,
        *,
        updater: TaskUpdater,
        context: RequestContext,
        send_request: SendMessageRequest,
    ) -> dict:
        """Stream pipeline execution and forward updates via TaskUpdater.

        Returns a dict describing whether execution completed or interrupted.

        Phase 1 scope:
        - Stream Contract-A status snapshots (accumulated text)
        - Emit a small set of artifacts when detected
        - No HITL interrupts yet
        """
        if not self.pipeline_agent:
            return {"interrupted": False, "result_text": "Analysis pipeline not available."}

        def _extract_interrupt_payload(obj):
            """Best-effort extraction of LangGraph interrupt payload.

            LangGraph surfaces interrupts under the '__interrupt__' key, typically
            as a list containing an Interrupt object with a '.value'.
            """
            try:
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    if "__interrupt__" in obj:
                        return obj.get("__interrupt__")
                    for v in obj.values():
                        found = _extract_interrupt_payload(v)
                        if found is not None:
                            return found
                if isinstance(obj, (list, tuple)):
                    for v in obj:
                        found = _extract_interrupt_payload(v)
                        if found is not None:
                            return found
            except Exception:
                return None
            return None

        def _normalize_interrupt_value(interrupt_obj):
            try:
                if interrupt_obj is None:
                    return None
                if isinstance(interrupt_obj, list) and interrupt_obj:
                    first = interrupt_obj[0]
                    if hasattr(first, "value"):
                        return getattr(first, "value")
                    if isinstance(first, dict) and "value" in first:
                        return first.get("value")
                    return first
                if hasattr(interrupt_obj, "value"):
                    return getattr(interrupt_obj, "value")
                if isinstance(interrupt_obj, dict) and "value" in interrupt_obj:
                    return interrupt_obj.get("value")
            except Exception:
                return None
            return interrupt_obj

        def _iso_ts() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _action_for_node(node_name: str) -> str:
            n = (node_name or "").strip().lower()
            if not n:
                return "Working..."
            if "ingest" in n:
                return "Reading request and preparing data"
            if "planner" in n or "plan" in n:
                return "Planning analysis steps"
            if "execute" in n or "tool" in n:
                return "Executing analysis steps"
            if "interpret" in n:
                return "Interpreting intermediate results"
            if "reflect" in n:
                return "Reviewing results and refining"
            if "synthesis" in n or "synth" in n:
                return "Synthesizing final report"
            if "persist" in n or "cleanup" in n:
                return "Finalizing outputs"
            if n in ("end", "finish", "done"):
                return "Finishing"
            return f"Working in {node_name}"

        # Streaming controls (env-tunable)
        try:
            max_status_chars = int(os.getenv("ANALYSIS_A2A_MAX_STATUS_CHARS", "80000"))
        except Exception:
            max_status_chars = 80000
        try:
            min_interval_ms = int(os.getenv("ANALYSIS_A2A_STREAM_MIN_INTERVAL_MS", "1000"))
        except Exception:
            min_interval_ms = 1000
        try:
            min_chars_delta = int(os.getenv("ANALYSIS_A2A_STREAM_MIN_CHARS_DELTA", "120"))
        except Exception:
            min_chars_delta = 120

        accumulator = StatusAccumulator(max_chars=max_status_chars)
        throttle = StreamThrottle(
            min_interval_s=max(0.05, float(min_interval_ms) / 1000.0),
            min_chars_delta=max(1, int(min_chars_delta)),
        )

        final_report_text: str = ""
        seen_artifact_paths = set()

        async for mode, chunk in self.pipeline_agent.astream_a2a_message(
            send_request,
            context_id=context.context_id,
            stream_mode=["values", "updates", "messages", "custom"],
        ):
            if not chunk:
                continue

            # Phase 4: HITL interrupt detection (works across values/updates).
            try:
                interrupt_obj = _extract_interrupt_payload(chunk)
                if interrupt_obj is not None:
                    interrupt_value = _normalize_interrupt_value(interrupt_obj)
                    payload = interrupt_value
                    # Emit a structured artifact containing the plan review payload.
                    artifact_data = {
                        "agent_task_id": str(context.task_id),
                        "content": [payload] if isinstance(payload, dict) else [str(payload)],
                        "tool_calls": [],
                        "current_action": "Plan review required",
                        "current_node": "planner",
                    }
                    try:
                        if await SchemaValidator(A2AUpdaterArtifactAdapter, artifact_data):
                            await updater.add_artifact(
                                metadata={
                                    "node": "planner",
                                    "ts": _iso_ts(),
                                },
                                parts=[Part(root=DataPart(data=artifact_data, kind="data"))],
                                name="plan_review",
                            )
                    except Exception:
                        pass

                    # Move task to input-required with an instruction message.
                    hint = (
                        "Plan is ready for review. Reply with one of:\n"
                        "- APPROVE\n"
                        "- REVISE: <your feedback>\n"
                        "- ABORT\n\n"
                        "Programmatic option: send a DataPart like {\"hitl_resume\": {\"decision\": \"approve|revise|abort\", \"feedback\": \"...\"}}"
                    )
                    response_data = {
                        "agent_task_id": str(context.task_id),
                        "content": (accumulator.get() + "\n\n" + hint).strip(),
                        "tool_call_chunks": [],
                        "current_action": "Input required: plan review",
                        "current_node": "planner",
                    }
                    parts = []
                    if await SchemaValidator(A2AUpdaterStatusAdapter, response_data):
                        parts.append(Part(root=DataPart(data=response_data, kind="data")))
                    parts.append(Part(root=TextPart(text=hint)))
                    try:
                        await updater.requires_input(
                            message=updater.new_agent_message(parts=parts),
                            final=False,
                        )
                    except Exception:
                        await updater.requires_input(final=False)

                    return {
                        "interrupted": True,
                        "interrupt_type": "plan_review",
                        "result_text": accumulator.get().strip(),
                    }
            except Exception:
                # Never let interrupt parsing crash the stream.
                pass

            if mode == "messages":
                try:
                    msg = chunk[0]
                    meta = chunk[1] if isinstance(chunk, (list, tuple)) and len(chunk) > 1 else {}
                    current_node = ""
                    try:
                        current_node = meta.get("langgraph_node") or meta.get("node") or ""
                    except Exception:
                        current_node = ""

                    delta = getattr(msg, "content", "")
                    if isinstance(delta, list):
                        delta_text = " ".join(str(x) for x in delta)
                    else:
                        delta_text = str(delta)

                    delta_text = strip_think_blocks(delta_text)
                    delta_text = ascii_sanitize(delta_text)
                    if delta_text:
                        accumulator.append(delta_text)

                    now_s = time.monotonic()
                    if throttle.should_emit(now_s=now_s, current_len=len(accumulator)):
                        response_data = {
                            "agent_task_id": str(context.task_id),
                            "content": accumulator.get(),
                            "tool_call_chunks": list(getattr(msg, "tool_call_chunks", []) or []),
                            "current_action": _action_for_node(current_node or "analysis_pipeline"),
                            "current_node": current_node or "analysis_pipeline",
                        }
                        if await SchemaValidator(A2AUpdaterStatusAdapter, response_data):
                            status_part = Part(root=DataPart(data=response_data, kind="data"))
                            await updater.update_status(
                                TaskState.working,
                                updater.new_agent_message(parts=[status_part]),
                            )
                except Exception:
                    # Never let a malformed chunk crash the server; continue streaming.
                    continue

            elif mode == "updates":
                # Best-effort artifact detection / final output capture.
                if isinstance(chunk, dict):
                    for node_name, node_val in chunk.items():
                        # Capture final report text if present
                        try:
                            if hasattr(node_val, "report_text"):
                                candidate = getattr(node_val, "report_text")
                                if isinstance(candidate, str) and candidate.strip():
                                    final_report_text = candidate.strip()
                            elif isinstance(node_val, dict) and isinstance(node_val.get("report_text"), str):
                                final_report_text = node_val.get("report_text", "").strip()
                        except Exception:
                            pass

                        # Emit artifact paths if they appear in updates
                        try:
                            artifact_log = None
                            if hasattr(node_val, "artifact_log"):
                                artifact_log = getattr(node_val, "artifact_log")
                            elif isinstance(node_val, dict):
                                artifact_log = node_val.get("artifact_log")
                            if isinstance(artifact_log, list):
                                for p in artifact_log:
                                    sp = str(p)
                                    if not sp or sp in seen_artifact_paths:
                                        continue
                                    seen_artifact_paths.add(sp)
                                    artifact_data = {
                                        "agent_task_id": str(context.task_id),
                                        "content": sp,
                                        "tool_calls": [],
                                        "current_action": "Artifact generated",
                                        "current_node": str(node_name),
                                    }
                                    if await SchemaValidator(A2AUpdaterArtifactAdapter, artifact_data):
                                        await updater.add_artifact(
                                            metadata={
                                                "node": str(node_name),
                                                "ts": _iso_ts(),
                                            },
                                            parts=[Part(root=DataPart(data=artifact_data, kind="data"))],
                                            name="ongoing_artifact",
                                        )
                        except Exception:
                            continue

            elif mode == "custom":
                # Phase 3: custom progress events emitted by pipeline nodes/tools.
                # These are used to keep progress visible even if LLM token streaming is quiet.
                try:
                    node_name = "analysis_pipeline"
                    event = "custom"
                    tool_name = None
                    status = None

                    if isinstance(chunk, dict):
                        node_name = str(chunk.get("node") or chunk.get("langgraph_node") or node_name)
                        event = str(chunk.get("event") or chunk.get("type") or event)
                        tool_name = chunk.get("tool")
                        status = chunk.get("status")
                    else:
                        # String or other payload
                        event = str(chunk)

                    # Build a short, human-readable current_action label.
                    action = "Working..."
                    try:
                        if event == "node_start":
                            action = _action_for_node(node_name)
                        elif event == "tool_start":
                            action = f"Starting tool {str(tool_name)}" if tool_name else "Starting tool"
                        elif event == "tool_end":
                            if tool_name and status:
                                action = f"Finished tool {str(tool_name)} ({str(status)})"
                            elif tool_name:
                                action = f"Finished tool {str(tool_name)}"
                            else:
                                action = "Finished tool"
                        else:
                            # Fallback: keep action short
                            action = str(event)[:120]
                    except Exception:
                        action = "Working..."

                    action = ascii_sanitize(strip_think_blocks(action), keep_newlines=False)
                    # If we have no accumulated text yet, seed it with a minimal progress line.
                    if len(accumulator) == 0 and action:
                        accumulator.append(action + "\n")

                    now_s = time.monotonic()
                    if throttle.should_emit(now_s=now_s, current_len=len(accumulator), force=True):
                        response_data = {
                            "agent_task_id": str(context.task_id),
                            "content": accumulator.get(),
                            "tool_call_chunks": [],
                            "current_action": action or _action_for_node(node_name),
                            "current_node": node_name or "analysis_pipeline",
                        }
                        if await SchemaValidator(A2AUpdaterStatusAdapter, response_data):
                            status_part = Part(root=DataPart(data=response_data, kind="data"))
                            await updater.update_status(
                                TaskState.working,
                                updater.new_agent_message(parts=[status_part]),
                            )
                except Exception:
                    continue

            else:
                # Ignore values/debug/custom in Phase 1.
                continue

        # If the graph didn't surface a report_text update, fall back to the accumulated stream.
        if not final_report_text:
            final_report_text = accumulator.get().strip() or "Analysis completed successfully."

        return {"interrupted": False, "result_text": final_report_text}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the analysis agent following the A2A tutorial pattern."""

        # Import A2A types at method level to avoid scope issues
        from a2a.types import (
            MessageSendParams,
            Role,
        )

        # Validation from tutorial pattern
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        # Create TaskUpdater as shown in tutorials
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        # Request validation
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        try:
            # Initialize our analysis agent
            await self.initialize_analysis_agent()

            # Ensure these exist for both streaming and non-streaming paths.
            response = {}
            result_payload = {}

            # Get user input from context (tutorial pattern)
            query = context.get_user_input()
            print(f"Analysis Agent received A2A request: {query}")

            # Determine if this is a HITL resume message.
            try:
                current_state = None
                if getattr(context, "current_task", None) is not None:
                    try:
                        current_state = getattr(getattr(context.current_task, "status", None), "state", None)
                    except Exception:
                        current_state = None
                state_val = getattr(current_state, "value", current_state)
                is_hitl_resume = (str(state_val) == str(TaskState.input_required.value))
            except Exception:
                is_hitl_resume = False

            # Use session_id from context for continuity
            session_id = context.context_id

            # Execute the actual LangGraph analysis agent
            use_pipeline = os.getenv("ANALYSIS_PIPELINE_ENABLED", "true").lower() in ("1", "true", "yes")
            if use_pipeline:
                print("Executing Modular Pipeline...")
            else:
                print("Executing LangGraph React Analysis Agent...")

            # Parse the query to extract dataset and instructions
            import re
            import json

            # Try to extract dataset path and instructions from the query
            dataset_match = re.search(r"Dataset:\s*(.+)", query)
            # CRITICAL FIX: Use Session ID as delimiter instead of "Please"
            # This prevents the regex dependency that required duplicated content
            instructions_match = re.search(
                r"Instructions:\s*(.+?)(?=\nSession ID:|$)", query, re.DOTALL

            )

            dataset_path = (
                dataset_match.group(1).strip()
                if dataset_match
                else None  # No default fallback - let the analysis agent handle detection

            )
            instructions = (
                instructions_match.group(1).strip() if instructions_match else query
            )

            # ENHANCED: Check if instructions contain JSON with dataset_path
            # This handles the a2a_client case where JSON metadata is sent
            try:
                instructions_json = json.loads(instructions)
                if isinstance(instructions_json, dict) and 'dataset_path' in instructions_json:
                    # Use the dataset_path from the JSON metadata
                    dataset_path = instructions_json['dataset_path']
                    # Use the analysis_instruction as the actual instructions
                    if 'analysis_instruction' in instructions_json:
                        instructions = instructions_json['analysis_instruction']
                    elif 'task' in instructions_json:
                        instructions = instructions_json['task']
                    print(f"A2A Service: Extracted dataset_path from JSON metadata: {dataset_path}")
                    print(f"A2A Service: Using analysis instructions from JSON metadata")
            except (json.JSONDecodeError, TypeError):
                # Not JSON or invalid JSON, use as-is
                pass

            # FIXED: Remove duplicated content, Session ID delimiter makes this safe

            # Preserve incoming parts from the original request (this keeps any
            # DataPart or FilePart intact). Then append a human-readable TextPart
            # with the summary so the analysis agent has both typed parts and
            # a readable context. If a DataPart already provides dataset/insructions
            # prefer synthesizing the TextPart from that DataPart to keep them consistent.
            try:
                parts = list(context.message.parts) if getattr(context.message, "parts", None) else []
            except Exception:
                parts = []

            # Check for existing DataPart content to synthesize human-readable summary
            existing_datapart = None
            for p in parts:
                try:
                    if getattr(p, "root", None) is not None and hasattr(p.root, "data"):
                        # prefer the first DataPart we find
                        existing_datapart = p.root.data
                        break
                except Exception:
                    continue

            if existing_datapart:
                # Normalize common keys
                ds = existing_datapart.get("dataset") or existing_datapart.get("dataset_path") or existing_datapart.get("path")
                instr = existing_datapart.get("instructions") or existing_datapart.get("analysis_instruction") or existing_datapart.get("task")
                # Build message content from DataPart so TextPart and DataPart are consistent
                message_content = f"""
Dataset: {ds if ds else (dataset_path if dataset_path else "No explicit dataset provided")}
Instructions: {instr if instr else instructions}
Session ID: {session_id}
"""
            else:
                message_content = f"""
Dataset: {dataset_path if dataset_path else "No explicit dataset provided"}
Instructions: {instructions}
Session ID: {session_id}
"""

            # Append a TextPart summary for human readability / fallback parsing
            # Only append if there is no existing text part or if existing text part
            # does not already contain the Dataset/Instructions summary. This
            # prevents sending a redundant TextPart when the client already
            # supplied a human-readable message (common in our helper clients).
            existing_text_part = None
            for p in parts:
                try:
                    if getattr(p, "root", None) is not None and hasattr(p.root, "text") and p.root.text and p.root.text.strip():
                        existing_text_part = p.root.text
                        break
                except Exception:
                    continue

            if existing_text_part:
                # If the existing text part already includes Dataset: and Instructions:
                # prefer keeping it and avoid appending another TextPart to reduce duplication.
                if ("Dataset:" in existing_text_part and "Instructions:" in existing_text_part):
                    print("A2A Service: Incoming message already contains a human-readable TextPart; skipping append.")
                else:
                    # Append only when existing text is not formatted as expected
                    parts.append(Part(root=TextPart(text=message_content)))
            else:
                parts.append(Part(root=TextPart(text=message_content)))

            # If this task is resuming from HITL, attach a hitl_resume DataPart if not provided.
            try:
                has_hitl_resume = False
                for p in parts:
                    try:
                        if getattr(p, "root", None) is not None and hasattr(p.root, "data"):
                            d = p.root.data
                            if isinstance(d, dict) and ("hitl_resume" in d or "hitl_decision" in d or "decision" in d):
                                has_hitl_resume = True
                                break
                    except Exception:
                        continue

                if is_hitl_resume and not has_hitl_resume:
                    q = str(query or "").strip()
                    q_low = q.lower()
                    decision = None
                    feedback = None
                    if q_low in ("approve", "approved", "yes", "y", "ok", "continue"):
                        decision = "approve"
                    elif q_low.startswith("revise") or q_low.startswith("edit") or q_low.startswith("change") or q_low.startswith("replan"):
                        decision = "revise"
                        feedback = q
                    elif ("revise" in q_low) or ("edit" in q_low) or ("change" in q_low):
                        decision = "revise"
                        feedback = q
                    elif q_low in ("abort", "cancel", "stop") or ("abort" in q_low) or ("cancel" in q_low) or ("stop" in q_low):
                        decision = "abort"
                    else:
                        # Heuristic: long free-text is likely feedback -> revise.
                        if len(q) >= 12:
                            decision = "revise"
                            feedback = q
                        else:
                            decision = "approve"
                    payload = {"decision": decision}
                    if feedback:
                        payload["feedback"] = feedback
                    parts.append(Part(root=DataPart(data={"hitl_resume": payload}, kind="data")))
            except Exception:
                pass

            # If we parsed a dataset_path and there's no DataPart already, include metadata
            has_datapart = existing_datapart is not None or any(getattr(p, "root", None) is not None and hasattr(p.root, "data") for p in parts)
            if dataset_path and not existing_datapart:
                try:
                    metadata = {
                        "dataset_path": dataset_path,
                        "analysis_instruction": instructions,
                        "session_id": session_id,
                    }
                    parts.append(Part(root=DataPart(data=metadata)))
                    print("A2A Service: Added DataPart with dataset metadata to message parts")
                except Exception as e:
                    print(f"Warning: could not build DataPart metadata: {e}")

            send_request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(
                    message=Message(
                        message_id=str(uuid.uuid4()),
                        role=Role.user,
                        parts=parts,
                        context_id=session_id,
                    )
                ),
            )

            # Call the appropriate analysis path
            enable_streaming = os.getenv("ANALYSIS_A2A_STREAMING", "true").lower() in (
                "1",
                "true",
                "yes",
            )

            if use_pipeline and self.pipeline_agent and enable_streaming:
                print("Executing Modular Pipeline (streaming enabled)...")
                stream_result = await self._execute_pipeline_streaming(
                    updater=updater,
                    context=context,
                    send_request=send_request,
                )
                if isinstance(stream_result, dict) and stream_result.get("interrupted"):
                    print("Analysis paused for HITL input (streaming path)")
                    return
                result = stream_result.get("result_text") if isinstance(stream_result, dict) else stream_result
                print("Analysis completed successfully (streaming path)!")
            else:
                if use_pipeline and self.pipeline_agent:
                    response = await self.pipeline_agent.receive_a2a_message(send_request)
                else:
                    response = await self.analysis_agent.receive_a2a_message(send_request)

                print("Analysis completed successfully!")

                # Extract result from response
                result_payload = response.get("result", {}) if isinstance(response.get("result"), dict) else {}

                if result_payload.get("status", {}).get("state") == "completed":
                    result = result_payload.get("analysis_result")
                    if not result:
                        # Check for alternate fields used by some agent paths
                        for candidate_key in ("report", "summary", "output", "final_report"):
                            candidate_value = result_payload.get(candidate_key)
                            if candidate_value:
                                result = candidate_value
                                break
                    if not result:
                        result = "Analysis completed successfully"
                elif response.get("error"):
                    error_msg = response.get("error", {}).get(
                        "message", "Unknown error occurred"
                    )
                    result = f"Analysis failed: {error_msg}"
                else:
                    result = "Analysis completed but status unclear"

            # Log a concise preview of the analysis output so local runs show the result
            try:
                preview_text = ""
                if isinstance(result, str):
                    preview_text = result.strip()
                elif isinstance(result, (dict, list)):
                    import json

                    preview_text = json.dumps(result, indent=2, ensure_ascii=True)
                elif result is not None:
                    preview_text = str(result).strip()

                if preview_text:
                    if len(preview_text) > 1200:
                        preview_text = preview_text[:1200] + "... [truncated]"
                    print("\n=== ANALYSIS RESULT PREVIEW ===")
                    print(preview_text)
                    artifacts = result_payload.get("artifacts") if isinstance(result_payload, dict) else None
                    if artifacts:
                        print("Artifacts:", artifacts)
                    print("==============================\n")
                else:
                    print("Analysis result payload was empty. Full response keys:", list(response.keys()) if isinstance(response, dict) else [])
                    if result_payload:
                        print("Result payload keys:", list(result_payload.keys()))
            except Exception as preview_error:
                print(f"Warning: Unable to display analysis preview ({preview_error})")

        except Exception as e:
            print(f"Error executing analysis agent: {e}")
            import traceback

            traceback.print_exc()
            # Provide a more informative error message
            result = f"Analysis Agent encountered an error: {str(e)}\n\nThe agent is available but encountered an issue during execution."

        # Final artifact: schema-stable DataPart plus a readable TextPart.
        try:
            if isinstance(result, str):
                result_text = result
            elif isinstance(result, (dict, list)):
                import json

                result_text = json.dumps(result, indent=2, ensure_ascii=True)
            elif result is None:
                result_text = ""
            else:
                result_text = str(result)
        except Exception:
            result_text = str(result)

        completion_data = {
            "agent_task_id": str(context.task_id),
            "content": result_text,
            "tool_calls": [],
            "current_action": "Completed",
            "current_node": "analysis_pipeline",
        }

        completion_parts: list[Part] = []
        if await SchemaValidator(A2AUpdaterArtifactAdapter, completion_data):
            completion_parts.append(
                Part(root=DataPart(data=completion_data, kind="data"))
            )
        completion_parts.append(Part(root=TextPart(text=result_text or "Analysis completed.")))

        try:
            await updater.add_artifact(
                metadata={"node": "analysis_pipeline"},
                parts=completion_parts,
                name="completed_artifact",
                last_chunk=True,
            )
        except TypeError:
            await updater.add_artifact(completion_parts)

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation following tutorial pattern."""
        print("Analysis Agent: Task cancellation requested")
        # Following the tutorial pattern for unsupported operations
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate the request context."""
        return False  # No validation errors for now


def create_a2a_server(host="127.0.0.1", port=10002):

    """Create and configure A2A server following the tutorial pattern"""

    # Create A2A Agent Capabilities following tutorial examples
    capabilities = AgentCapabilities(
        streaming=True,  # Enable streaming like the tutorial examples
        push_notifications=True,
        # Note: A2A SDK may have limited predefined capabilities
        # Actual capabilities are advertised through skills below
    )

    # Define skills following tutorial pattern
    skills = [
        AgentSkill(
            id="statistical_analysis",
            name="Statistical Analysis",
            description="EDA, descriptive statistics, correlation analysis, ANOVA, t-tests, hypothesis testing, automated statistical test selection",
            tags=["statistics", "eda", "correlation", "anova", "t_test", "automated"],
            examples=[
                "Comprehensive statistical analysis",
                "Automated statistical testing",
                "Correlation and hypothesis testing",
            ],
        ),
        AgentSkill(
            id="temporal_analysis",
            name="Temporal Analysis",
            description="Dedicated temporal pattern analysis, hourly/daily/weekly trends, incident timing, seasonal patterns, time-series analysis",
            tags=["temporal", "time", "patterns", "hourly", "seasonal", "time_series"],
            examples=[
                "Analyze temporal patterns in data",
                "Hourly and weekly trend analysis",
                "Time-based incident analysis",
            ],
        ),
        AgentSkill(
            id="pattern_detection",
            name="Pattern Detection & Data Quality",
            description="Pattern recognition, anomaly detection, outlier analysis, data quality assessment, missing value analysis",
            tags=["patterns", "anomaly", "outliers", "quality", "assessment"],
            examples=[
                "Detect patterns and anomalies",
                "Data quality assessment",
                "Outlier detection and analysis",
            ],
        ),
        AgentSkill(
            id="data_visualization",
            name="Data Visualization", 
            description="Advanced plotting: histogram, scatter, box, bar charts, temporal plots. Auto-detects optimal visualization types",
            tags=["visualization", "plots", "charts", "temporal_plots"],
            examples=[
                "Generate pairplots for numeric relationships",
                "Create temporal distribution plots", 
                "Auto-select optimal visualizations",
            ],
        ),
    ]

    # Create Agent Card following tutorial pattern
    agent_card = AgentCard(
        name="Data Analysis Agent",
        description="""Select this agent as long as analysis is needed. This is a comprehensive data analysis agent with statistical analysis, visualization.

[IMPORTANT: Please send original user query/instructions without any generalization or modification]
CAPABILITIES:
• Statistical Analysis: EDA, correlation, ANOVA, t-tests, automated statistical test selection
• Temporal Analysis: Dedicated temporal patterns, hourly/daily/weekly trends, seasonal analysis
• Pattern Detection: Anomaly detection, outlier analysis, data quality assessment
• Data Visualization: Advanced plots with auto-detection, temporal plots, correlation heatmaps
• Note: Correlation heatmaps are provided via dedicated tools; general plotting supports pairplots for numeric relationships.

MESSAGE FORMAT:
Dataset: [file_path_or_data]
Instructions: [analysis_requirements]
Session ID: [unique_id]

HOW TO REQUEST ANALYSIS TYPES:

[IMPORTANT: Please send original user query/instructions without any generalization or modification]

STATISTICAL ANALYSIS - Use:
• "statistical analysis", "EDA", "automated statistical testing"
• "correlation analysis", "ANOVA testing", "t-test"
• "hypothesis testing", "statistical significance"

TEMPORAL ANALYSIS - Use:
• "temporal patterns", "analyze temporal patterns", "time-based analysis"
• "hourly trends", "weekly patterns", "seasonal analysis"
• "incident timing", "time-series analysis"

PATTERN DETECTION - Use:
• "pattern detection", "anomaly detection", "data quality assessment"
• "outlier analysis", "assess data quality", "detect patterns"

VISUALIZATION - Use:
• "generate plots", "pairplot", "auto-categorical"
• "temporal visualizations", "advanced plotting"

EXAMPLES:
Dataset: [path to dataset]
Instructions: generate comprehensive insights.
Session ID: stats_001

Dataset: [path to dataset]
Instructions: Analyze temporal heatmap of incidents by month
Session ID: temporal_002

Session ID: viz_004""",
        url=f"http://{host}:{port}/",
        version="2.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=capabilities,
        skills=skills,
    )

    # Create HTTP client for push notifications (tutorial pattern)
    httpx_client = httpx.AsyncClient()

    # Create request handler following tutorial pattern
    request_handler = DefaultRequestHandler(
        agent_executor=A2AAnalysisAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_sender=SimplePushNotificationSender(httpx_client),
    )

    # Create A2A Starlette application following tutorial pattern
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    return server, agent_card


def main():
    """Main function to start the A2A-compliant service following tutorial pattern"""

    print("=" * 80)
    print("A2A-COMPLIANT ANALYSIS AGENT SERVICE")
    print("=" * 80)
    print("Using Official A2A SDK v0.3.0")
    print("LangGraph React Analysis Agent with FastMCP Integration")
    print("Agent-to-Agent Protocol Compliant Service")
    print("Following Tutorial Best Practices")
    print("=" * 80)
    print("")

    # ---------------------------------------------------------------------
    # End-to-end defaults (only set when missing)
    # ---------------------------------------------------------------------
    def _set_default_env(key: str, value: str) -> None:
        try:
            cur = os.getenv(key)
            if cur is None or str(cur).strip() == "":
                os.environ[key] = value
        except Exception:
            return

    # Ensure the modular pipeline is used.
    _set_default_env("ANALYSIS_PIPELINE_ENABLED", "true")

    # Enable delegation by default for end-to-end validation.
    _set_default_env("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION", "1")
    _set_default_env("ANALYSIS_MAX_CI_DELEGATIONS", "1")
    _set_default_env("ANALYSIS_FORCE_DELEGATION_FOR_TOOL_REQUEST", "1")
    # Do not stop early just because the model claims it answered; allow refinement/delegation.
    _set_default_env("ANALYSIS_END_ON_ANSWER", "0")

    # Child graph defaults: prefer subprocess sandbox unless overridden.
    _set_default_env("TOOLGEN_SANDBOX_MODE", "subprocess")

    # Try to auto-locate the sibling codeGen repo and its local venv.
    try:
        service_file = Path(__file__).resolve()
        daa_root = service_file.parents[3]  # .../DAA/src/agents/analysis/a2a_compliant_service.py
        child_root = daa_root.parent / "codeGen" / "MCP_Tool_Code_Interpreter_Generator"
        if child_root.exists():
            _set_default_env("ANALYSIS_CODE_INTERPRETER_CHILD_ROOT", str(child_root))
            child_py = child_root / ".venv" / "Scripts" / "python.exe"
            if child_py.exists():
                _set_default_env("ANALYSIS_CODE_INTERPRETER_PYTHON", str(child_py))
    except Exception:
        pass

    # Configuration
    host = os.getenv("AGENT_DOMAIN", "127.0.0.1")
    port = int(os.getenv("A2A_ANALYSIS_AGENT_PORT", "10002"))

    # Preflight: fail fast with a readable message if the port is already bound.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as e:
        # Windows: WinError 10048 is "only one usage of each socket address"
        if getattr(e, "winerror", None) == 10048:
            print("\nSTARTUP ERROR: Port is already in use")
            print(f"Port {port} is already bound on {host}.")
            print("Fix options:")
            print("  - Stop the other process using that port")
            print("  - Or pick a new port by setting A2A_ANALYSIS_AGENT_PORT")
            print("Example (PowerShell):")
            print("  $env:A2A_ANALYSIS_AGENT_PORT=10003; uv run src\\agents\\analysis\\a2a_compliant_service.py")
            import sys

            sys.exit(1)
        raise

    # Create A2A server following tutorial pattern
    server, agent_card = create_a2a_server(host=host, port=port)

    print("A2A Service Configuration:")
    print(f"   Agent Name: {agent_card.name}")
    print(f"   Skills: {len(agent_card.skills)} capabilities")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   URL: {agent_card.url}")
    print("")
    print("A2A Protocol Features:")
    print("    Standard Agent Discovery")
    print("    Message Protocol Compliance")
    print("    Skill-based Capability Negotiation")
    print("    Agent Card Serving")
    print("    Task Management")
    print("    Streaming Support")

    print("")
    print("Agent Skills Available:")
    for skill in agent_card.skills:
        print(f"   • {skill.name}: {skill.description}")
    print("")
    print("Starting A2A-compliant service...")
    print("   Press Ctrl+C to stop")
    print("=" * 80)

    try:
        # Import uvicorn for serving
        import uvicorn

        # Start the A2A Starlette application using tutorial pattern
        print(f" Starting server on http://{host}:{port}")
        print(f" Agent Card: http://{host}:{port}/.well-known/agent-card.json")
        
        # Configure uvicorn with shorter timeout for faster shutdown
        config = uvicorn.Config(
            server.build(),
            host=host,
            port=port,
            log_level="info",
            timeout_graceful_shutdown=5,  # 5 second timeout instead of default 120
        )
        server_instance = uvicorn.Server(config)
        server_instance.run()

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("SHUTDOWN REQUESTED")
        print("=" * 80)
        print("A2A Analysis Agent service stopped gracefully")
        print("Goodbye!")
        print("=" * 80)
    except Exception as e:
        print(f"\nSTARTUP ERROR: {e}")
        print("Check your configuration and A2A SDK installation")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    main()