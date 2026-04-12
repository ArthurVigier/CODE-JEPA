from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .contracts import ActivationSnapshot


class ResidualHookCollector:
    def __init__(self, layer_ids: list[int]):
        self.layer_ids = sorted(layer_ids)
        self._residuals: dict[int, np.ndarray] = {}
        self._handles = []

    def attach(self, model) -> None:
        for layer_id in self.layer_ids:
            handle = model.model.layers[layer_id].register_forward_hook(self._make_hook(layer_id))
            self._handles.append(handle)

    def _make_hook(self, layer_id: int) -> Callable:
        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._residuals[layer_id] = hidden.detach().cpu().float().squeeze(0).numpy()

        return hook

    def snapshot(self, prompt_id: str, source: str, category: str, token_step: int, model_name: str) -> ActivationSnapshot:
        if set(self.layer_ids) != set(self._residuals):
            missing = sorted(set(self.layer_ids) - set(self._residuals))
            raise RuntimeError(f"Missing captured residuals for layers {missing}")
        sample = self._residuals[self.layer_ids[0]]
        snapshot = ActivationSnapshot(
            residuals_by_layer={layer: self._residuals[layer].copy() for layer in self.layer_ids},
            prompt_id=prompt_id,
            source=source,
            category=category,
            token_step=token_step,
            model_name=model_name,
            layers=self.layer_ids,
            seq_len=int(sample.shape[0]),
        )
        snapshot.validate()
        return snapshot

    def clear(self) -> None:
        self._residuals.clear()

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@dataclass(slots=True)
class TransformerActivationExtractor:
    model_name: str
    layer_ids: list[int]
    torch_dtype: str = "bfloat16"
    tokenizer: object = field(init=False, repr=False)
    model: object = field(init=False, repr=False)
    collector: ResidualHookCollector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.collector = ResidualHookCollector(self.layer_ids)
        self.collector.attach(self.model)

    def _input_device(self):
        try:
            return self.model.model.embed_tokens.weight.device
        except Exception:
            return next(self.model.parameters()).device

    def extract_snapshot(self, prompt_id: str, source: str, category: str, text: str, token_step: int = 0) -> ActivationSnapshot:
        input_device = self._input_device()
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        import torch

        self.collector.clear()
        with torch.no_grad():
            self.model(**inputs)
        return self.collector.snapshot(prompt_id, source, category, token_step, self.model_name)

    def close(self) -> None:
        self.collector.detach()
