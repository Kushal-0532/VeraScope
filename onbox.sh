#!/usr/bin/env bash
# One-time setup on the g4dn box. Idempotent enough to re-run.
# Usage:  bash onbox.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv --system-site-packages .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q --upgrade pip

# whisperx pulls faster-whisper + ctranslate2 + pyannote.audio; the DLAMI
# already ships a CUDA torch, so let pip keep it.
pip install -q whisperx python-dotenv streamlit groq tavily-python "huggingface_hub>=1.5" "transformers>=5"  # groq+tavily: Stage 2. hf_hub>=1.5: HF zero-shot response shape changed, 0.36 raises TypeError in verify_claims; transformers 4.57 refuses hf_hub 1.x, 5.16 works with whisperx 3.8.6 + pyannote 4.0.7 (GPU-verified 2026-09-09). pip check still flags whisperx metadata pin hf_hub<1; harmless.

# ponytail: install the cuDNN 9 runtime unconditionally rather than waiting for
# the ctranslate2 mismatch to blow up at the first transcribe() (STAGE3-PLAN
# risk 1). It is a wheel, it is small, and it is the documented fix.
pip install -q nvidia-cudnn-cu12 || true

echo "--- versions"
python -c "
import importlib.metadata as md
for p in ('torch','whisperx','faster-whisper','ctranslate2','pyannote.audio',
          'transformers','numpy','speechbrain'):
    try: print(f'{p}=={md.version(p)}')
    except Exception as e: print(f'{p}: {e}')
"
echo "--- gpu"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "--- stage 2 still imports"
python -c "import claim_detect, verify_claims, retrieve_evidence, verify_pipeline; print('stage 2 ok')"
echo
echo "setup done. next:  source .venv/bin/activate && python onbox_check.py"
