"""Export compact inference-only weights from a Phase 5 PPO checkpoint.

Usage:
    python training/tools/export_weights.py --checkpoint submission/weights/emlrp_best.pt --output submission/weights/agent.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env.limits import MAX_REPO_BYTES  # noqa: E402
from submission.models import EMLRPConfig, EMLRPPolicy  # noqa: E402


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def export_inference_weights(
    checkpoint_path: str | Path,
    output_path: str | Path,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Strip optimizer/RND training state and write compact weights_only-safe `.pt` file."""
    src = Path(checkpoint_path)
    dst = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing 'model_state_dict': {src}")

    raw_sd = ckpt["model_state_dict"]
    clean_sd = {k: v.detach().cpu().to(torch.float32).clone() for k, v in raw_sd.items()}
    model_config = _to_builtin(ckpt.get("model_config", {}))
    preprocess_config = _to_builtin(ckpt.get("preprocess_config", {}))

    payload: dict[str, Any] = {
        "model_state_dict": clean_sd,
        "model_config": model_config,
        "preprocess_config": preprocess_config,
        "inference_temperature": float(temperature),
        "source_update": int(ckpt.get("update_idx", 0)),
        "source_step": int(ckpt.get("global_step", 0)),
        "curriculum_stage": int(ckpt.get("curriculum_stage", 0)),
        "best_val_score": float(ckpt.get("best_val_score", 0.0)),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    # Verify strict weights_only=True reload + architecture parity
    reloaded = torch.load(dst, map_location="cpu", weights_only=True)
    policy = EMLRPPolicy(EMLRPConfig(**reloaded["model_config"])).eval()
    policy.load_state_dict(reloaded["model_state_dict"])

    out_bytes = dst.stat().st_size
    if out_bytes > MAX_REPO_BYTES:
        raise ValueError(f"Exported weights exceed MAX_REPO_BYTES: {out_bytes} > {MAX_REPO_BYTES}")

    return {
        "source": str(src),
        "output": str(dst),
        "source_bytes": src.stat().st_size,
        "output_bytes": out_bytes,
        "source_update": payload["source_update"],
        "source_step": payload["source_step"],
        "best_val_score": payload["best_val_score"],
        "inference_temperature": payload["inference_temperature"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact inference-only weights for submission/agent.py")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/emlrp_best.pt")
    parser.add_argument("--output", type=str, default="submission/weights/agent.pt")
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = deterministic (logits > 0); >0 = Bernoulli temperature")
    args = parser.parse_args()

    info = export_inference_weights(args.checkpoint, args.output, temperature=args.temperature)
    src_mb = info["source_bytes"] / (1024 * 1024)
    dst_mb = info["output_bytes"] / (1024 * 1024)
    print(
        f"Exported {info['output']} ({dst_mb:.2f} MB, down from {src_mb:.2f} MB) "
        f"[upd={info['source_update']}, step={info['source_step']}, best_val={info['best_val_score']:.4f}, "
        f"temp={info['inference_temperature']}]"
    )


if __name__ == "__main__":
    main()
