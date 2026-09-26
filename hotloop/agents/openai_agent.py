"""Reference agent: a minimal tool-calling loop for any OpenAI-compatible endpoint.

    api="responses" - OpenAI Responses API (OpenAI models)
    api="chat"      - Chat Completions (vLLM / SGLang / other servers for open-weight models)

Imports only hotloop.interface: it knows nothing about scoring or backends.
"""

import json
import os
import time

from hotloop.interface import AgentResult, Budget, BudgetExceeded, Environment

SYSTEM = """You are an expert GPU kernel engineer working autonomously on a remote Linux machine with an NVIDIA GPU \
and no internet access. The task statement follows (it is also in TASK.md in your working directory). Write code, \
benchmark, profile and iterate. Call `submit` when you are done; the solution file is scored as it is at that moment."""

MAX_OUTPUT_CHARS = 12000

_BASH = {"name": "bash",
         "description": "Run a shell command in the working directory. Returns combined stdout/stderr "
                        f"(last {MAX_OUTPUT_CHARS} characters) and the exit code.",
         "parameters": {"type": "object", "properties": {
             "command": {"type": "string"},
             "timeout": {"type": "integer", "description": "seconds (default 600)"}},
             "required": ["command"]}}
_WRITE = {"name": "write_file", "description": "Write text to a file (relative paths are under the working directory).",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"], "additionalProperties": False}}
_SUBMIT = {"name": "submit", "description": "Finish; the solution file is scored.",
           "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}
TOOL_DEFS = [_BASH, _WRITE, _SUBMIT]


class OpenAIAgent:
    def __init__(self, model: str, effort: str | None = "medium", api: str = "responses",
                 base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY", max_turns: int = 150,
                 name: str | None = None, temperature: float | None = None, top_p: float | None = None,
                 top_k: int | None = None, repetition_penalty: float | None = None,
                 max_tokens: int | None = None, provider: str | None = None, quantization: str | None = None):
        self.model = model
        # Output cap per request. Some providers (e.g. OpenRouter) reserve credit for the
        # model's maximum output on every request when this is unset.
        self.max_tokens = None if max_tokens in (None, "", "none") else int(max_tokens)
        # Sampling (e.g. a model's recommended settings); None = server default.
        self.sampling = {k: float(v) for k, v in (("temperature", temperature), ("top_p", top_p)) if v is not None}
        self.extra_body = {k: v for k, v in (("top_k", None if top_k is None else int(top_k)),
                                             ("repetition_penalty", None if repetition_penalty is None
                                              else float(repetition_penalty))) if v is not None}
        # Hosted open models (OpenRouter): pin who serves the model and at what precision,
        # with no silent fallback, so API results are reproducible and comparable.
        if provider or quantization:
            pin = {"allow_fallbacks": False}
            if provider:
                pin["order"] = [p.strip() for p in provider.split(",")]
            if quantization:
                pin["quantizations"] = [q.strip() for q in quantization.split(",")]
            self.extra_body["provider"] = pin
        self.effort = None if effort in (None, "", "none", "None") else effort
        self.api = api
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_turns = int(max_turns)
        self.name = (name or model).replace("/", "_")

    def preflight(self) -> str:
        """One minimal request, so a bad key or empty balance fails before any GPU is used."""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=os.environ.get(self.api_key_env), max_retries=0)
        if self.api == "responses":
            client.responses.create(model=self.model, input="ok", max_output_tokens=16)
        else:
            # Same output cap as real requests, so credit/limit problems show up here, not mid-run.
            client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": "ok"}],
                                           max_tokens=self.max_tokens or 1,
                                           tools=[{"type": "function", "function": t} for t in TOOL_DEFS])
        return "ok"

    # --- tools -------------------------------------------------------------------
    def _tool(self, env: Environment, name: str, args: dict) -> str:
        if name == "bash":
            res = env.exec(args.get("command", ""), timeout=int(args.get("timeout") or 600))
            out = res.output[-MAX_OUTPUT_CHARS:]
            return out + f"\n[exit code {res.exit_code}{', timed out' if res.timed_out else ''}]"
        if name == "write_file":
            path = args["path"] if args["path"].startswith("/") else os.path.join(env.workdir, args["path"])
            env.write_text(path, args["content"])
            return f"Wrote {len(args['content'])} bytes to {path}."
        return f"Unknown tool {name}."

    def _status(self, env: Environment, turns: int) -> str:
        return f"\n[{env.seconds_left() / 60:.0f} min and {self.max_turns - turns} tool calls left]"

    # --- loop --------------------------------------------------------------------
    def run(self, env: Environment, task: str, budget: Budget) -> AgentResult:
        from openai import OpenAI

        # Generous retries: many episodes run in parallel and rate limits must not end one early.
        client = OpenAI(base_url=self.base_url, api_key=os.environ.get(self.api_key_env), max_retries=8)
        step = self._responses_step if self.api == "responses" else self._chat_step
        state = {"prev": None, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}],
                 "pending": [{"role": "user", "content": task}]}
        transcript = budget.events  # recorded live, so it survives a deadline overrun
        transcript.append({"role": "user", "content": task})
        usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0,
                 "api_seconds": 0.0, "api_calls": 0}
        turns, nudges, truncations = 0, 0, 0
        stop = "budget"
        while turns < self.max_turns and env.seconds_left() > 0:
            t_call = time.time()
            try:
                texts, calls = step(client, state, usage)
            except BudgetExceeded:
                break
            except Exception as e:
                transcript.append({"role": "error", "content": repr(e)})
                # Running out of context is the agent's own limit, not a provider failure.
                if "context length" in str(e).lower() or "maximum context" in str(e).lower():
                    stop = "context_exhausted"
                else:
                    stop = f"error: api: {e!r}"[:300]
                break
            finally:
                usage["api_seconds"] += time.time() - t_call
                usage["api_calls"] += 1
            transcript += [{"role": "assistant", "content": t} for t in texts if t]
            if not calls:
                if state.get("truncated"):
                    truncations += 1
                    note = ("Your last response was cut off at the output-token limit before it finished. "
                            "Think more briefly, and write large files in smaller pieces.")
                    transcript.append({"role": "harness", "content": f"response truncated at output limit ({truncations})"})
                    if truncations > 8:
                        stop = "output_limit"
                        break
                else:
                    nudges += 1
                    note = "Keep working with the tools, or call `submit` if you are done."
                    transcript.append({"role": "harness", "content": f"no tool call; nudged ({nudges})"})
                    if nudges > 3:
                        stop = "no_tool_calls"
                        break
                self._add_user(state, note)
                continue
            outputs, submitted, out_of_budget = [], False, False
            for call_id, name, raw in calls:
                turns += 1
                try:
                    args = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    args = {}
                transcript.append({"role": "tool_call", "name": name, "args": args, "t": round(time.time())})
                if name == "submit":
                    out, submitted = "Submitted.", True
                elif out_of_budget:
                    out = "Budget exhausted."
                else:
                    try:
                        out = self._tool(env, name, args)
                    except BudgetExceeded:
                        out, out_of_budget = "Budget exhausted.", True
                    except Exception as e:
                        out = f"[tool error: {e!r}]"
                out += self._status(env, turns)
                transcript.append({"role": "tool_output", "name": name, "content": out})
                outputs.append((call_id, name, out))
            self._add_outputs(state, outputs)
            if submitted or out_of_budget:
                stop = "submitted" if submitted else "budget"
                break
        if usage["api_seconds"]:
            usage["output_tokens_per_api_second"] = round(usage["output_tokens"] / usage["api_seconds"], 1)
        return AgentResult(stop_reason=stop, transcript=transcript, usage=usage,
                           metadata={"model": self.model, "effort": self.effort, "api": self.api,
                                     "base_url": self.base_url, "turns": turns, "sampling": self.sampling,
                                     "extra_body": self.extra_body})

    # --- Responses API -----------------------------------------------------------
    def _responses_step(self, client, state, usage):
        kw = {"reasoning": {"effort": self.effort}} if self.effort else {}
        state["truncated"] = False
        resp = client.responses.create(model=self.model, instructions=SYSTEM, input=state["pending"],
                                       tools=[{"type": "function", "strict": t["name"] != "bash", **t} for t in TOOL_DEFS],
                                       previous_response_id=state["prev"], **kw)
        state["prev"] = resp.id
        state["truncated"] = getattr(resp, "status", "") == "incomplete"
        if resp.usage:
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens
            usage["cached_tokens"] += getattr(resp.usage.input_tokens_details, "cached_tokens", 0) or 0
            usage["reasoning_tokens"] += getattr(resp.usage.output_tokens_details, "reasoning_tokens", 0) or 0
        texts, calls = [], []
        for o in resp.output:
            if o.type == "message":
                texts.append("".join(c.text for c in o.content if getattr(c, "type", "") == "output_text"))
            elif o.type == "function_call":
                calls.append((o.call_id, o.name, o.arguments))
        state["pending"] = []
        return texts, calls

    # --- Chat Completions API ----------------------------------------------------
    def _chat_step(self, client, state, usage):
        kw = {"reasoning_effort": self.effort} if self.effort else {}
        kw.update(self.sampling)
        if self.max_tokens:
            kw["max_tokens"] = self.max_tokens
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        resp = client.chat.completions.create(model=self.model, messages=state["messages"],
                                              tools=[{"type": "function", "function": t} for t in TOOL_DEFS], **kw)
        if resp.usage:
            usage["input_tokens"] += resp.usage.prompt_tokens
            usage["output_tokens"] += resp.usage.completion_tokens
            details = getattr(resp.usage, "prompt_tokens_details", None)
            usage["cached_tokens"] += (getattr(details, "cached_tokens", 0) or 0) if details else 0
        served_by = (resp.model_extra or {}).get("provider")  # OpenRouter reports the serving provider
        if served_by:
            usage.setdefault("served_by", {})
            usage["served_by"][served_by] = usage["served_by"].get(served_by, 0) + 1
        msg = resp.choices[0].message
        state["truncated"] = resp.choices[0].finish_reason == "length"
        state["messages"].append(msg.model_dump(exclude_none=True))
        calls = [(c.id, c.function.name, c.function.arguments) for c in (msg.tool_calls or [])]
        return [msg.content or ""], calls

    # --- history helpers -----------------------------------------------------------
    def _add_user(self, state, text):
        state["pending"].append({"role": "user", "content": text})
        state["messages"].append({"role": "user", "content": text})

    def _add_outputs(self, state, outputs):
        for call_id, _, out in outputs:
            state["pending"].append({"type": "function_call_output", "call_id": call_id, "output": out})
            state["messages"].append({"role": "tool", "tool_call_id": call_id, "content": out})
