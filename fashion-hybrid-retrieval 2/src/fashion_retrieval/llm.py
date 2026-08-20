"""LLM client abstraction (Section 2.3).

Every backend here is free of charge and free of subscription. No paid API is
used anywhere in this project.

``ollama`` (recommended for the reported experiments)
    A local, open-weight model served by Ollama (https://ollama.com), which is
    free and open source. The model runs on the student's own machine; no
    account, no key, no billing. Recommended models, in ascending order of
    cost in hardware: ``qwen2.5:3b-instruct``, ``llama3.1:8b``,
    ``qwen2.5:7b-instruct``.

``transformers``
    A local open-weight model loaded directly through Hugging Face
    ``transformers``. Slower to set up than Ollama and heavier on RAM, but it
    needs no separate server, which helps on machines where installing Ollama
    is not possible (some lab computers).

``openai-compatible``
    Any endpoint exposing the OpenAI chat-completions shape. Included because
    several free, self-hosted servers (llama.cpp's ``llama-server``, vLLM,
    LM Studio) speak this protocol. Point it at a local server and it costs
    nothing. Do NOT point it at a paid vendor for this work.

``mock``
    A deterministic, rule-based stand-in that returns well-formed JSON. It
    lets the pipeline, the tests and the evaluation harness run with no model
    at all. Its extraction quality is deliberately crude; it is a plumbing
    device, not a system under study, and its numbers must never be reported.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class LLMClient(Protocol):
    """Minimal interface the rest of the codebase depends on."""

    name: str

    def complete(self, system: str, user: str) -> str:  # pragma: no cover
        ...


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in a model response.

    Small open-weight models wrap JSON in prose or in Markdown fences more
    often than large ones do, so the parser is tolerant by design. A failure
    to parse is treated as an empty extraction rather than as a crash, and is
    counted in the evaluation as a parse failure, which is itself a result:
    parse reliability is part of what a small local model costs.
    """
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, depth = None, 0
    for i, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    log.warning("Could not parse JSON from model response")
    return {}


def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    """POST a JSON body and decode the JSON response (stdlib only)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaLLM:
    """Local open-weight model served by Ollama. Free, no account, no key.

    Setup:
        1. install Ollama from https://ollama.com (free, open source)
        2. ``ollama pull qwen2.5:3b-instruct``
        3. ``ollama serve`` (usually already running after install)
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str | None = None,
        temperature: float = 0.0,
        num_predict: int = 512,
        timeout: int = 300,
    ):
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def _check(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5):
                return
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` "
                f"and pull a model with `ollama pull {self.model}`. "
                "Ollama is free and runs locally; see the README."
            ) from exc

    def complete(self, system: str, user: str) -> str:
        self._check()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Ollama can constrain decoding to valid JSON, which matters a lot
            # for small models asked to emit a structured object.
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        try:
            response = _post_json(f"{self.host}/api/chat", payload, self.timeout)
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}. Is the model '{self.model}' pulled? "
                f"Try `ollama pull {self.model}`."
            ) from exc
        return response.get("message", {}).get("content", "")


class TransformersLLM:
    """Local open-weight model loaded directly through Hugging Face."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers and torch are not installed. Install them with "
                "`pip install -r requirements-full.txt`, or use the 'ollama' "
                "backend, which is lighter to set up."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map="auto"
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.name = f"transformers:{model}"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - heavy
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature or None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = generated[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True)


class OpenAICompatibleLLM:
    """Client for any local server speaking the OpenAI chat-completions shape.

    Intended for free, self-hosted servers such as llama.cpp's ``llama-server``,
    vLLM or LM Studio. The base URL must point at a local process.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL") or "http://localhost:8080/v1"
        ).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.name = f"openai-compatible:{model}"

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - network
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = _post_json(f"{self.base_url}/chat/completions", payload, self.timeout)
        choices = response.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")


class MockLLM:
    """Offline, rule-based stand-in producing the same JSON shape."""

    name = "mock"

    def complete(self, system: str, user: str) -> str:
        if "SYNTHETIC QUERY GENERATION" in system:
            return self._mock_query(user)
        return self._mock_parse(system, user)

    # -- query parsing ----------------------------------------------------
    def _mock_parse(self, system: str, user: str) -> str:
        """Match vocabulary values from the prompt against the query text."""
        query = user.lower()
        vocabulary = extract_json(_between(system, "<vocabulary>", "</vocabulary>"))
        hard: dict[str, str] = {}
        confidence: dict[str, float] = {}

        for field, values in vocabulary.items():
            best = None
            for value in values:
                token = str(value).lower()
                if re.search(rf"\b{re.escape(token)}\b", query):
                    if best is None or len(token) > len(best.lower()):
                        best = str(value)
            if best is not None:
                hard[field] = best
                confidence[field] = 0.9

        soft = query
        for value in hard.values():
            soft = re.sub(rf"\b{re.escape(value.lower())}\b", " ", soft)
        soft = re.sub(r"\s+", " ", soft).strip()

        return json.dumps(
            {"hard_filters": hard, "soft_intent": soft or query, "confidence": confidence}
        )

    # -- synthetic query generation ---------------------------------------
    def _mock_query(self, user: str) -> str:
        text = re.sub(r"\s+", " ", user).strip()
        words = _TOKEN_RE.findall(text.lower())
        stop = {"the", "a", "an", "with", "and", "in", "of", "for", "product", "description"}
        keep = [w for w in words if w not in stop][:12]
        return json.dumps({"query": " ".join(keep)})


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError:
        return "{}"


def build_llm(cfg) -> LLMClient:
    """Instantiate the LLM client selected in the configuration."""
    backend = cfg.llm_backend
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        return OllamaLLM(
            model=cfg.llm_model,
            host=cfg.llm_host,
            temperature=cfg.llm_temperature,
            num_predict=cfg.llm_max_tokens,
        )
    if backend == "transformers":
        return TransformersLLM(
            model=cfg.llm_model,
            max_new_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
        )
    if backend == "openai-compatible":
        return OpenAICompatibleLLM(
            model=cfg.llm_model,
            base_url=cfg.llm_host,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )
    raise ValueError(
        f"Unknown LLM backend: {backend}. "
        "Valid: mock, ollama, transformers, openai-compatible."
    )
