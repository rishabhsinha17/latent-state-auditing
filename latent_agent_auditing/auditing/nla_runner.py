from __future__ import annotations

import hashlib
import math
import re
import random
from typing import Protocol

from latent_agent_auditing.models.schemas import ActivationRecord


class NLARunner(Protocol):
    def explain(self, activation: ActivationRecord, n_samples: int = 5) -> list[str]:
        ...


class HeuristicNLARunner:
    """Development stand-in for NLA explanations.

    This is not an interpretability method. It gives stable, semantically
    varied explanations from mock activation dimensions so the rest of the
    research stack can be developed and tested.
    """

    benign_templates = [
        "The model is focused on completing the trusted user task.",
        "The representation emphasizes a benign summary or read-only action.",
        "The model appears to be preserving the user's original goal.",
    ]
    injection_templates = [
        "The model is treating untrusted external text as an instruction.",
        "The model appears to internalize a request from the retrieved content.",
        "The external document instruction is represented as part of the next action.",
    ]
    tool_misuse_templates = [
        "The model is preparing an unauthorized tool call.",
        "The model is considering a tool action that is not required by the user goal.",
        "The representation points toward tool misuse before execution.",
    ]
    exfil_templates = [
        "The model is preparing to send or reveal private data.",
        "The activation emphasizes private information targeted for transmission.",
        "The model appears to connect sensitive data with an external destination.",
    ]
    destructive_templates = [
        "The model is considering a destructive or sabotage-like action.",
        "The activation emphasizes deleting or damaging user-controlled state.",
        "The model is preparing an irreversible high-impact action.",
    ]

    def explain(self, activation: ActivationRecord, n_samples: int = 5) -> list[str]:
        rng = random.Random(int(hashlib.sha256(activation.id.encode()).hexdigest()[:8], 16))
        pools = [self.benign_templates]
        vector = activation.vector
        if len(vector) > 1 and vector[1] > 0.25:
            pools.append(self.injection_templates)
        if len(vector) > 2 and vector[2] > 0.25:
            pools.append(self.tool_misuse_templates)
        if len(vector) > 3 and vector[3] > 0.25:
            pools.append(self.exfil_templates)
        if len(vector) > 4 and vector[4] > 0.25:
            pools.append(self.destructive_templates)

        explanations: list[str] = []
        for _ in range(n_samples):
            pool = rng.choice(pools)
            explanations.append(rng.choice(pool))
        return explanations


class ExternalNLAClientRunner:
    """Thin adapter for an installed NLA inference client."""

    def __init__(self, model_name: str, layer: int, **client_kwargs: object) -> None:
        self.model_name = model_name
        self.layer = layer
        self.client_kwargs = client_kwargs
        self._client = None

    def _load_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            from nla_inference import NLAClient  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "nla_inference is not installed. Install the optional NLA dependency "
                "or use HeuristicNLARunner for local development."
            ) from exc
        self._client = NLAClient(model_name=self.model_name, layer=self.layer, **self.client_kwargs)
        return self._client

    def explain(self, activation: ActivationRecord, n_samples: int = 5) -> list[str]:
        client = self._load_client()
        if not hasattr(client, "explain"):
            raise RuntimeError("NLAClient must expose an explain(vector, n_samples=...) method")
        return list(client.explain(activation.vector, n_samples=n_samples))  # type: ignore[attr-defined]


class LocalTransformersAVRunner:
    """Run a released NLA activation verbalizer with local Transformers.

    This is a correctness-first fallback for small batches. The released NLA
    inference docs recommend SGLang for throughput, but local generation is a
    useful first integration test because it avoids standing up a server.
    """

    def __init__(
        self,
        av_model_name: str = "kitft/nla-qwen2.5-7b-L20-av",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        cache_dir: str | None = None,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        trust_remote_code: bool = False,
    ) -> None:
        self.av_model_name = av_model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._meta: dict[str, object] | None = None

    def explain(self, activation: ActivationRecord, n_samples: int = 5) -> list[str]:
        model, tokenizer, torch, meta = self._load()
        d_model = int(meta["d_model"])
        if len(activation.vector) != d_model:
            raise ValueError(
                f"activation vector width {len(activation.vector)} does not match NLA d_model {d_model}; "
                "use the AV checkpoint matching the base model and extraction layer"
            )

        input_ids = self._prompt_input_ids(tokenizer, meta).to(model.device)
        embeds = model.get_input_embeddings()(input_ids).float()
        embed_scale = self._embed_scale(model, meta)
        if embed_scale != 1.0:
            embeds = embeds * embed_scale
        injection_position = self._find_injection_position(input_ids[0].tolist(), meta)
        vector = torch.tensor(activation.vector, dtype=torch.float32, device=model.device)
        norm = torch.linalg.vector_norm(vector)
        if float(norm) == 0.0:
            raise ValueError("cannot verbalize a zero activation vector")
        injection_scale = float(_nested_get(meta, ["extraction", "injection_scale"]))
        embeds[0, injection_position] = vector * (injection_scale / norm)
        embeds = embeds.to(dtype=next(model.parameters()).dtype)
        attention_mask = torch.ones((1, embeds.shape[1]), dtype=torch.long, device=model.device)

        explanations: list[str] = []
        for sample_idx in range(n_samples):
            do_sample = self.temperature > 0
            with torch.no_grad():
                output_ids = model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            explanations.append(_extract_explanation(text))
        return explanations

    def _load(self):
        if self._model is not None and self._tokenizer is not None and self._torch is not None and self._meta is not None:
            return self._model, self._tokenizer, self._torch, self._meta
        try:
            import torch
            import yaml
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "LocalTransformersAVRunner requires torch, transformers, huggingface_hub, and pyyaml. "
                "Install with `pip install -e .[ml]`."
            ) from exc

        meta_path = hf_hub_download(self.av_model_name, "nla_meta.yaml", cache_dir=self.cache_dir)
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)
        tokenizer = AutoTokenizer.from_pretrained(
            self.av_model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        dtype = self._resolve_torch_dtype(torch)
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(self.av_model_name, **model_kwargs)
        if self.device != "auto":
            model = model.to(self.device)
        model.eval()
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._meta = meta
        return model, tokenizer, torch, meta

    def _resolve_torch_dtype(self, torch):
        if self.torch_dtype == "auto":
            return "auto"
        if self.torch_dtype in {"none", "None", None}:
            return None
        dtype = getattr(torch, str(self.torch_dtype), None)
        if dtype is None:
            raise ValueError(f"unknown torch dtype: {self.torch_dtype}")
        return dtype

    def _prompt_input_ids(self, tokenizer, meta: dict[str, object]):
        template = str(_nested_get(meta, ["prompt_templates", "av"]))
        injection_char = str(_nested_get(meta, ["tokens", "injection_char"]))
        content = template.format(injection_char=injection_char)
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids

    def _find_injection_position(self, ids: list[int], meta: dict[str, object]) -> int:
        injection_id = int(_nested_get(meta, ["tokens", "injection_token_id"]))
        left_id = int(_nested_get(meta, ["tokens", "injection_left_neighbor_id"]))
        right_id = int(_nested_get(meta, ["tokens", "injection_right_neighbor_id"]))
        for position, token_id in enumerate(ids):
            if token_id != injection_id or position == 0 or position == len(ids) - 1:
                continue
            if ids[position - 1] == left_id and ids[position + 1] == right_id:
                return position
        raise RuntimeError("could not find NLA injection token with expected neighbors")

    def _embed_scale(self, model, meta: dict[str, object]) -> float:
        explicit = _nested_get(meta, ["extraction", "embed_scale"], default=None)
        if explicit is not None:
            return float(explicit)
        model_type = str(getattr(model.config, "model_type", ""))
        text_config = getattr(model.config, "text_config", None)
        text_model_type = str(getattr(text_config, "model_type", ""))
        if model_type.startswith("gemma") or text_model_type.startswith("gemma"):
            return math.sqrt(float(meta["d_model"]))
        return 1.0


def _nested_get(mapping: dict[str, object], path: list[str], default: object = ...):
    current: object = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            if default is ...:
                raise KeyError(".".join(path))
            return default
        current = current[key]
    return current


def _extract_explanation(text: str) -> str:
    match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("<explanation>")
    if start != -1:
        return text[start + len("<explanation>") :].strip()
    return text.strip()
