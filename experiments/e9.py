"""E9 — Gemma 4 E2B audio diarization: zero-shot vs QLoRA fine-tuned, with
pyannote 3.1 on the identical windows as the reference system.

  python -m experiments.e9 prep    windows + pyannote per window (base env)
  python -m experiments.e9 gemma   zero-shot eval, Unsloth QLoRA train, fine-tuned eval
  python -m experiments.e9 report  summary, CIs, error cases (runs even if gemma failed)

Windows are WINDOW_S long (Gemma 4 audio accepts <= 30 s). Targets use the
text format in experiments.common (serialize/parse): local labels S1.. by
first appearance, window-relative times at 0.1 s; overlapping turns allowed.
Same-speaker gaps <= 0.3 s are merged in TRAINING TARGETS only; scoring
always uses the raw reference RTTM cropped to the window.

Splits by recording: train = AMI train meetings + VoxConverse dev files,
validation = AMI dev meetings, test = AMI test meetings + the VoxConverse test
files E1 uses. Speaker disjointness across AMI splits is not verified here
(split is by meeting per the official full-corpus partition).
"""

import json
import random
import sys
import time

from experiments.common import COLLARS_PM, DATA, RESULTS, SEED, SMOKE, Exp, bootstrap, crop, der, \
    log, merge_turns, parse, pooled_der, serialize, write_json
from experiments import speech as sp

WINDOW_S = 25.0
MIN_SPEECH = 0.3                     # fraction of window with reference speech
MIN_SPK_S = 0.2                      # a speaker counts in a window with >= this much speech
N_TRAIN = 40 if SMOKE else 3000
N_VAL = 8 if SMOKE else 100
N_TEST_AMI, N_TEST_VOX = (5, 5) if SMOKE else (128, 120)
AMI_TRAIN_MEETINGS = 2 if SMOKE else 40
VOX_DEV_FILES = 2 if SMOKE else 80
MODEL_ID = "unsloth/gemma-4-E2B-it"
MAX_NEW_TOKENS = 256
TRAIN = dict(epochs=1, lr=5e-5, batch=2, grad_accum=4, r=8, alpha=16, max_steps=10 if SMOKE else -1,
             save_steps=5 if SMOKE else 50)
EXAMPLES = ["S1 0.0-3.4; S2 3.4-7.9; S1 7.9-9.2",
            "S1 0.0-6.1; S2 5.8-9.0; S3 9.5-14.2; S1 14.0-20.3"]
SYSTEM = "You are a speaker diarization system."
WDIR = DATA / "e9"


def prompt():
    return (f"Segment this {WINDOW_S:.0f}-second audio clip by speaker. Output only the speaker "
            "turns as 'S<n> <start>-<end>' separated by '; ', times in seconds from the start of "
            "the clip with one decimal, speakers numbered S1, S2, ... in order of first "
            "appearance. If two people talk at once, list both turns. Examples of the format: "
            + " | ".join(EXAMPLES))


# ---------------------------------------------------------------- windows
def windows_of(uri, turns, dur, rng):
    """Non-overlapping windows from a random offset -> list of window dicts."""
    out, t = [], rng.uniform(0, WINDOW_S)
    while t + WINDOW_S <= dur:
        w = crop(turns, t, t + WINDOW_S)
        spk = {}
        for s, e, k in w:
            spk[k] = spk.get(k, 0) + e - s
        spk = {k: v for k, v in spk.items() if v >= MIN_SPK_S}
        tl = sp.overlap_timeline(w)
        speech = _union(w)
        if 1 <= len(spk) <= 4 and speech >= MIN_SPEECH * WINDOW_S:
            out.append({"id": f"{uri}@{t:.1f}", "uri": uri, "start": round(t, 3), "turns": w,
                        "n_spk": len(spk), "speech_s": speech,
                        "overlap_ratio": sum(b - a for a, b in tl) / speech})
        t += WINDOW_S
    return out


def _union(turns):
    tot, end = 0.0, -1.0
    for s, e, _ in sorted(turns):
        if e > end:
            tot += e - max(s, end)
            end = e
    return tot


def build_split(name, sources, n, rng):
    """sources: [(corpus, uri, path, turns)] -> manifest of n windows (wavs on disk)."""
    man = WDIR / f"{name}.jsonl"
    if man.exists():
        return [json.loads(l) for l in man.read_text().splitlines()]
    per_uri = {}
    for corpus, uri, path, turns in sources:
        try:
            dur = sp.duration(path)
            ws = windows_of(uri, turns, dur, rng)
        except Exception as e:
            log(f"E9 windows {uri}: {e!r}")
            continue
        for w in ws:
            w.update(corpus=corpus, path=str(path))
        per_uri[uri] = ws
    # round-robin across recordings so no single file dominates
    for ws in per_uri.values():
        rng.shuffle(ws)
    pool, i = [], 0
    while len(pool) < n and any(i < len(ws) for ws in per_uri.values()):
        pool += [ws[i] for ws in per_uri.values() if i < len(ws)]
        i += 1
    pool = pool[:n]
    for w in pool:
        wav = WDIR / name / f"{w['id'].replace('@', '_')}.wav"
        if not wav.exists():
            sp.write_wav(wav, sp.load16k(w["path"], start=w["start"], dur=WINDOW_S))
        w["wav"] = str(wav)
        w["target"] = serialize(merge_turns(w["turns"]))
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text("".join(json.dumps(w) + "\n" for w in pool))
    log(f"E9 {name}: {len(pool)} windows from {len(per_uri)} recordings")
    return pool


def ami_sources(split, meetings):
    out = []
    for m in meetings:
        try:
            out.append(("ami", m, sp.ami_audio(m), sp.ami_rttm(m, split)))
        except Exception as e:
            log(f"E9 skip AMI {m}: {e!r}")
    return out


def splits():
    rng = random.Random(SEED)
    train_m = sp.ami_meetings("train")
    rng.shuffle(train_m)
    from experiments.e1 import N_VOX, VOX_MIN_S
    vox_dev = [("vox", u, p, t) for u, p, t, _ in sp.vox_files("dev", VOX_DEV_FILES)]
    vox_test = [("vox", u, p, t) for u, p, t, _ in sp.vox_files("test", N_VOX, VOX_MIN_S)]
    train = build_split("train", ami_sources("train", train_m[:AMI_TRAIN_MEETINGS]) + vox_dev,
                        N_TRAIN, rng)
    val = build_split("val", ami_sources("dev", sp.ami_meetings("dev")[:3]), N_VAL, rng)
    test_m = sp.ami_meetings("test")[:2 if SMOKE else None]
    test = (build_split("test_ami", ami_sources("test", test_m), N_TEST_AMI, rng)
            + build_split("test_vox", vox_test, N_TEST_VOX, rng))
    tr, te = {w["uri"] for w in train}, {w["uri"] for w in test}
    assert not tr & te, f"recording leakage train/test: {tr & te}"
    assert not tr & {w["uri"] for w in val}, "recording leakage train/val"
    return train, val, test


# ---------------------------------------------------------------- scoring
def score(w, hyp, info, seconds, system):
    ref = w["turns"]
    rec = {"system": system, "window": w["id"], "uri": w["uri"], "corpus": w["corpus"],
           "n_spk": w["n_spk"], "overlap_ratio": w["overlap_ratio"], "seconds": seconds,
           "hyp": hyp, "n_hyp_spk": len({k for *_, k in hyp}), **info}
    for c in COLLARS_PM:
        rec[f"der_c{c}"] = der(ref, hyp, c, uem=(0.0, WINDOW_S))
    return rec


# ---------------------------------------------------------------- prep: pyannote
def prep():
    exp = Exp("E9", config())
    train, val, test = splits()
    import torch
    import diarize_demo as dd
    pipe = dd._diarization_pipeline("cuda" if torch.cuda.is_available() else "cpu")
    done = exp.done()
    for w in test:
        key = f"pyannote|{w['id']}"
        if key in done:
            continue
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            df = pipe(sp.load16k(w["wav"]), min_speakers=None, max_speakers=None)
            rows = df.to_dict("records") if hasattr(df, "to_dict") else df
            hyp = sorted((float(r["start"]), float(r["end"]), str(r["speaker"])) for r in rows)
            rec = score(w, hyp, {"parse_failure": False}, time.perf_counter() - t, "pyannote")
            rec["peak_torch_mem_mb"] = (torch.cuda.max_memory_allocated() / 2**20
                                        if torch.cuda.is_available() else None)
            exp.add(key, rec)
        except Exception as e:
            exp.fail(key, repr(e))
    log(f"E9 prep: {len(train)} train / {len(val)} val / {len(test)} test windows")


def config():
    return {"window_s": WINDOW_S, "min_speech_frac": MIN_SPEECH, "min_spk_s": MIN_SPK_S,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": [N_TEST_AMI, N_TEST_VOX],
            "ami_train_meetings": AMI_TRAIN_MEETINGS, "vox_dev_files": VOX_DEV_FILES,
            "model": MODEL_ID, "train": TRAIN, "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy (do_sample=False)", "prompt": prompt(), "system": SYSTEM,
            "target_merge_gap_s": 0.3, "der": "per-window optimal mapping, uem = window"}


# ---------------------------------------------------------------- gemma
def messages(w, audio, with_answer=False):
    m = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
         {"role": "user", "content": [{"type": "audio", "audio": audio},
                                      {"type": "text", "text": prompt()}]}]
    if with_answer:
        m.append({"role": "assistant", "content": [{"type": "text", "text": w["target"]}]})
    return m


def fix_fp16_audio_mask(model):
    """Gemma4AudioAttention masks with config.attention_invalid_logits_value
    (-1e9), which overflows fp16 on a T4 (Unsloth Gemma 4 docs). Clamp it to a
    finite fp16 value; a no-op for models without that attribute."""
    n = 0
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and getattr(cfg, "attention_invalid_logits_value", 0) < -6e4:
            cfg.attention_invalid_logits_value = -1e4
            n += 1
    return n


def load_model(adapter=None):
    """Unsloth FastModel (as in the official Gemma4 E2B Audio notebook), 4-bit.
    Falls back to transformers + bitsandbytes if Unsloth fails; reason recorded."""
    import torch
    try:
        from unsloth import FastModel
        model, proc = FastModel.from_pretrained(model_name=adapter or MODEL_ID, dtype=None,
                                                max_seq_length=2048, load_in_4bit=True,
                                                full_finetuning=False)
        backend = "unsloth"
    except Exception as e:
        reason = repr(e)
        log(f"E9 Unsloth load failed, falling back to transformers+peft: {reason}")
        from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
        base = MODEL_ID.replace("unsloth/", "google/")
        model = AutoModelForMultimodalLM.from_pretrained(base, quantization_config=q,
                                                         torch_dtype=torch.float16, device_map="auto")
        proc = AutoProcessor.from_pretrained(base)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        backend = f"transformers-fallback ({reason[:300]})"
    fix_fp16_audio_mask(model)
    return model, proc, backend


def generate(model, proc, w):
    import torch
    audio = sp.load16k(w["wav"])
    inputs = proc.apply_chat_template(messages(w, audio), add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to(model.device)
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    secs = time.perf_counter() - t
    text = getattr(proc, "tokenizer", proc).decode(out[0][inputs["input_ids"].shape[1]:],
                                                   skip_special_tokens=True)
    return text.strip(), secs, int(out.shape[1] - inputs["input_ids"].shape[1])


def eval_gemma(exp, model, proc, test, system, done):
    import torch
    torch.cuda.reset_peak_memory_stats()
    for i, w in enumerate(test):
        key = f"{system}|{w['id']}"
        if key in done:
            continue
        try:
            text, secs, ntok = generate(model, proc, w)
            hyp, info = parse(text, WINDOW_S)
            rec = score(w, hyp, info, secs, system)
            rec.update(raw_output=text, new_tokens=ntok,
                       peak_torch_mem_mb=torch.cuda.max_memory_allocated() / 2**20)
            exp.add(key, rec)
        except Exception as e:
            exp.fail(key, repr(e))
        if i % 20 == 0:
            log(f"E9 {system} {i}/{len(test)}")


class WinDS:
    """Lazy dataset: audio read from disk per item (3000 windows would not fit
    in Colab RAM as float arrays)."""

    def __init__(self, ws):
        self.ws = ws

    def __len__(self):
        return len(self.ws)

    def __getitem__(self, i):
        w = self.ws[i]
        return {"messages": messages(w, sp.load16k(w["wav"]), with_answer=True)}


def train(exp, model, proc, backend, train_ws, val_ws):
    import torch
    from trl import SFTConfig, SFTTrainer
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
               "post", "linear_start", "linear_end", "embedding_projection",
               "ffw_layer_1", "ffw_layer_2", "output_proj"]      # Gemma4 E2B Audio notebook
    if backend == "unsloth":
        from unsloth import FastModel
        from unsloth.trainer import UnslothVisionDataCollator
        model = FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True, r=TRAIN["r"],
            lora_alpha=TRAIN["alpha"], lora_dropout=0, bias="none", random_state=SEED,
            use_rslora=False, loftq_config=None, target_modules=targets,
            use_gradient_checkpointing="unsloth")
        FastModel.for_training(model)          # zero-shot eval left it in inference mode
        collator = UnslothVisionDataCollator(model, proc)
    else:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(r=TRAIN["r"], lora_alpha=TRAIN["alpha"],
                                                 lora_dropout=0, target_modules=targets,
                                                 task_type="CAUSAL_LM"))

        def collator(batch):
            enc = proc.apply_chat_template([b["messages"] for b in batch], tokenize=True,
                                           return_dict=True, return_tensors="pt", padding=True)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            enc["labels"] = labels
            return enc
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ckpt = RESULTS / "E9" / "ckpt"
    args = SFTConfig(
        per_device_train_batch_size=TRAIN["batch"], gradient_accumulation_steps=TRAIN["grad_accum"],
        warmup_ratio=0.03, num_train_epochs=TRAIN["epochs"], max_steps=TRAIN["max_steps"],
        learning_rate=TRAIN["lr"], logging_steps=5, optim="adamw_8bit", weight_decay=0.001,
        lr_scheduler_type="cosine", seed=SEED, output_dir=str(ckpt), report_to="none",
        save_strategy="steps", save_steps=TRAIN["save_steps"], save_total_limit=2,
        eval_strategy="steps", eval_steps=TRAIN["save_steps"], per_device_eval_batch_size=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False, dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=2048)
    trainer = SFTTrainer(model=model, train_dataset=WinDS(train_ws), eval_dataset=WinDS(val_ws),
                         processing_class=proc.tokenizer, data_collator=collator, args=args)
    torch.cuda.reset_peak_memory_stats()
    resume = ckpt.exists() and any(ckpt.glob("checkpoint-*"))
    t = time.perf_counter()
    stats = trainer.train(resume_from_checkpoint=resume or None)
    wall = time.perf_counter() - t
    adapter = RESULTS / "E9" / "adapter"
    model.save_pretrained(str(adapter))
    proc.save_pretrained(str(adapter))
    hist = trainer.state.log_history
    exp.add("train|meta", {
        "backend": backend, "trainable_params": trainable, "total_params": total,
        "train_windows": len(train_ws), "val_windows": len(val_ws),
        "audio_seconds_seen": trainer.state.global_step * TRAIN["batch"] * TRAIN["grad_accum"] * WINDOW_S,
        "steps": trainer.state.global_step, "wall_s": wall, "resumed": resume,
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        "best_eval_loss": trainer.state.best_metric, "log_history": hist})
    return model


def set_mode(model, backend, training):
    if backend == "unsloth":
        from unsloth import FastModel
        (FastModel.for_training if training else FastModel.for_inference)(model)
    else:
        model.train(training)


def gemma():
    exp = Exp("E9", config())
    train_ws, val_ws, test = splits()
    done = exp.done()
    adapter = RESULTS / "E9" / "adapter"
    have_adapter = (adapter / "adapter_config.json").exists()
    zs_left = any(f"gemma_zeroshot|{w['id']}" not in done for w in test)
    if zs_left or not have_adapter:
        model, proc, backend = load_model()
        exp.config["backend"] = backend
        if zs_left:
            set_mode(model, backend, False)
            eval_gemma(exp, model, proc, test, "gemma_zeroshot", done)
        if not have_adapter:
            model = train(exp, model, proc, backend, train_ws, val_ws)
        else:
            del model
            model, proc, backend = load_model(str(adapter))
    else:
        model, proc, backend = load_model(str(adapter))
    set_mode(model, backend, False)
    eval_gemma(exp, model, proc, test, "gemma_finetuned", exp.done())


# ---------------------------------------------------------------- report
def report():
    exp = Exp("E9", config())
    recs = [r for r in exp.done().values() if r["key"] != "train|meta"]
    meta = exp.done().get("train|meta")
    systems = ["pyannote", "gemma_zeroshot", "gemma_finetuned"]
    for name, n in (("train", N_TRAIN), ("val", N_VAL)):
        p = WDIR / f"{name}.jsonl"
        if p.exists():
            ws = [json.loads(l) for l in p.read_text().splitlines()]
            exp.dataset("AMI + VoxConverse windows", f"{WINDOW_S:.0f} s windows", name, len(ws),
                        "CC BY 4.0", None, f"{len({w['uri'] for w in ws})} recordings")
    ids = {s: {r["window"] for r in recs if r["system"] == s} for s in systems}
    common = set.intersection(*[v for v in ids.values() if v]) if any(ids.values()) else set()
    summary = {"n_common_windows": len(common), "train": {k: v for k, v in (meta or {}).items()
                                                          if k != "log_history"}}

    def by_uri(rs, c):
        g = {}
        for r in rs:
            g.setdefault(r["uri"], []).append(r[f"der_c{c}"])
        return list(g.values())

    def cluster_der(groups):
        return pooled_der([d for g in groups for d in g])

    for s in systems:
        rs = [r for r in recs if r["system"] == s and r["window"] in common]
        if not rs:
            continue
        m = {"n": len(rs), "n_recordings": len({r["uri"] for r in rs})}
        for c in COLLARS_PM:
            m[f"der_c{c}"] = bootstrap(by_uri(rs, c), cluster_der)
            tot = sum(r[f"der_c{c}"]["total"] for r in rs) or 1
            m[f"components_c{c}"] = {k: sum(r[f"der_c{c}"][k] for r in rs) / tot
                                     for k in ("fa", "miss", "conf")}
        m["by_n_spk"] = {k: {"n": len(g), "der_c0.25": pooled_der([r["der_c0.25"] for r in g])}
                         for k in (1, 2, 3, 4) if (g := [r for r in rs if r["n_spk"] == k])}
        bins = [(0, 0), (0, 0.1), (0.1, 0.25), (0.25, 1.01)]
        m["by_overlap"] = {f"{a}-{b}": {"n": len(g), "der_c0.25": pooled_der([r["der_c0.25"] for r in g])}
                           for a, b in bins
                           if (g := [r for r in rs if (r["overlap_ratio"] == 0 if b == 0
                                                       else a < r["overlap_ratio"] <= b)])}
        m["spk_count_acc"] = sum(r["n_hyp_spk"] == r["n_spk"] for r in rs) / len(rs)
        m["parse_failure_rate"] = sum(r.get("parse_failure", False) for r in rs) / len(rs)
        if s != "pyannote":
            m["strict_format_rate"] = sum(r.get("strict", False) for r in rs) / len(rs)
            for k in ("out_of_window", "inverted", "self_overlap"):
                m[f"windows_with_{k}"] = sum(r.get(k, 0) > 0 for r in rs) / len(rs)
            m["windows_max_concurrent_gt2"] = sum(r.get("max_concurrent", 0) > 2 for r in rs) / len(rs)
            m["new_tokens_mean"] = sum(r.get("new_tokens", 0) for r in rs) / len(rs)
        m["seconds_per_window_mean"] = sum(r["seconds"] for r in rs) / len(rs)
        m["peak_torch_mem_mb"] = max((r.get("peak_torch_mem_mb") or 0) for r in rs)
        summary[s] = m
    ft = {r["window"]: r for r in recs if r["system"] == "gemma_finetuned"}
    py = {r["window"]: r for r in recs if r["system"] == "pyannote"}
    errs = [{"window": k, "n_spk": r["n_spk"], "overlap_ratio": r["overlap_ratio"],
             "gemma_finetuned_output": r.get("raw_output"), "gemma_der": r["der_c0.25"],
             "pyannote_der": py[k]["der_c0.25"] if k in py else None,
             "reference": serialize(next(w for w in _test_manifest() if w["id"] == k)["turns"]),
             "parse": {x: r.get(x) for x in ("strict", "parse_failure", "out_of_window",
                                             "inverted", "self_overlap")}}
            for k, r in sorted(ft.items(), key=lambda kv: -kv[1]["der_c0.25"]["der"])[:10]]
    exp.finish(summary, errs, records=recs + ([meta] if meta else []))


def _test_manifest():
    out = []
    for n in ("test_ami", "test_vox"):
        p = WDIR / f"{n}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in p.read_text().splitlines()]
    return out


def _selfcheck():
    rng = random.Random(0)
    turns = [(0, 12, "A"), (11, 20, "B"), (30, 40, "A"), (40, 60, "C"), (60, 80, "D"), (80, 100, "E")]
    ws = windows_of("m", turns, 100.0, rng)
    assert ws and all(1 <= w["n_spk"] <= 4 and w["speech_s"] >= MIN_SPEECH * WINDOW_S for w in ws)
    assert all(0 <= s < e <= WINDOW_S + 1e-9 for w in ws for s, e, _ in w["turns"])
    assert abs(_union([(0, 5, "A"), (3, 8, "B"), (10, 11, "A")]) - 9) < 1e-9
    w = windows_of("m", [(0, 12, "A"), (11, 20, "B")], 26.0, random.Random(1))
    if w:
        assert w[0]["overlap_ratio"] > 0
    print("e9 selfcheck ok")


if __name__ == "__main__":
    {"prep": prep, "gemma": gemma, "report": report, "--selfcheck": _selfcheck}[sys.argv[1]]()
