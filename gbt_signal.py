"""GBT integration adapter for Trading Bot V2.

The adapter is deliberately safe-by-default: if a trained checkpoint and the
GBT dependency tree are unavailable, it reports that GBT is unavailable rather
than silently generating fake predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


class GBTSignal:
    def __init__(self, checkpoint: Optional[str] = None, seq_len: int = 32,
                 pred_len: int = 1, threshold: float = 0.001):
        self.checkpoint = checkpoint
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.threshold = float(threshold)
        self.model = None
        self.available = False
        self.error = None
        if checkpoint:
            self._load(checkpoint)

    def _load(self, checkpoint: str) -> None:
        try:
            import torch
            from GBT.GBT import GBT
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(f"GBT checkpoint not found: {path}")
            # Model architecture is checkpoint-specific. A serialized module is
            # supported directly; state_dict-only checkpoints need matching config.
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if hasattr(obj, "eval") and callable(obj.eval):
                self.model = obj.eval()
            else:
                raise ValueError("Checkpoint contains a state_dict/config, not a serialized GBT model")
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.available = False

    def predict(self, prices: pd.Series) -> Optional[float]:
        if not self.available or self.model is None:
            return None
        import torch
        values = pd.Series(prices).astype(float).dropna().to_numpy()
        if len(values) < self.seq_len:
            return None
        x = values[-self.seq_len:]
        scale = max(abs(x.mean()), 1e-8)
        x = (x / scale).astype(np.float32)
        tensor = torch.from_numpy(x).view(1, self.seq_len, 1)
        marks = torch.zeros((1, self.seq_len, 1), dtype=torch.float32)
        dec = tensor[:, -1:, :]
        with torch.no_grad():
            out = self.model(tensor, marks, dec, marks, flag="first stage")
        if isinstance(out, tuple):
            out = out[0]
        return float(out.reshape(-1)[-1].cpu().item() * scale)

    def signal(self, prices: pd.Series) -> int:
        """Return +1 buy, -1 sell, 0 neutral based on predicted next price."""
        prediction = self.predict(prices)
        if prediction is None:
            return 0
        current = float(pd.Series(prices).iloc[-1])
        change = (prediction - current) / max(abs(current), 1e-8)
        if change > self.threshold:
            return 1
        if change < -self.threshold:
            return -1
        return 0
