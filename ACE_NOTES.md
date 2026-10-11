# ACE-Step 1.5 XL worker: stage 1 recon notes (Oct 10 2026)

Branch `ace-xl`. Written for stage 2 (the `ace_handler.py` author) and for reviewers.
Every upstream claim cites `path:line` in the pinned checkout `F:/KadeAI-work/ace/upstream`
(commit `ca1e85fe9430179831e6bc6be790c332190a3866`, a shallow clone with one commit; line numbers are for that commit).
Nothing here was run on a GPU. Where a statement is an estimate or inference it says so.

How facts were obtained:
- READ = read in the pinned upstream source or the worker repo (primary source).
- HF-LISTING = public Hugging Face file listing and `config.json` files fetched read-only on Oct 10 2026
  (`huggingface.co/api/models/<repo>/tree/main...`). The fetch tool summarised the pages with a small model, so
  names and sizes are reliable but a copied 40-character sha could hold a typo: re-check before pinning (see 3.5).
- INFERENCE = reasoned from code or docs, not observed.

Contents: 1 Python API (a) | 2 dependencies and install recipe (b) | 3 cache layout vs upstream layout and the three
HF repos (c) | 4 things that will bite (d) | 5 yue_handler.py conventions to reuse (e) | 6 suggested shape of stage 2 | 7 not verified |
8 stage 3: the image, CI, endpoint settings, model-cache resolution, tests, smoke-test watch list (Dockerfile.ace, ace.yml) |
9 stage 6: trained styles (a LoRA adapter folded into the xl-sft decoder, restored and verified; why upstream's unload is not used; the style smoke-test watch list).
Stage 3 supersedes parts of section 2.3 (pins) and 3.5 (revisions): see 8.2. Stage 6 supersedes 1.7.

TL;DR for stage 2 (details and citations below):
1. Entry point: `acestep.inference.generate_music(dit, llm, GenerationParams, GenerationConfig, save_dir=None)`; two handlers are created by hand
   (`AceStepHandler().initialize_service(...)`, `LLMHandler().initialize(...)`), both return `(status, ok)` and never raise. Returns CPU float32 tensors at 48 kHz, stereo; the worker writes its own files.
2. Model selection is by checkpoint FOLDER NAME under one flat checkpoints dir (`acestep-v15-xl-turbo`, `acestep-v15-xl-sft`, plus `vae`, `Qwen3-Embedding-0.6B`, `acestep-5Hz-lm-1.7B`,
   and a weighted `acestep-v15-turbo` that upstream insists on). Build that dir at boot from symlinks into the RunPod snapshots, but give each DiT a REAL directory with the pinned
   upstream `*.py` files COPIED in, otherwise upstream's code sync writes through symlinks into the shared cache (1.2, 3.3).
3. Fully offline works: every `from_pretrained` on the generation path takes a local folder (1.2).
4. LM backend: `pt` (no compiler, no flash-attn; nano-vllm would run eager without flash-attn on 3.12). Planner time under `pt` is probably as large as the xl-turbo render (1.3, estimate).
5. Gotchas that change results: `use_random_seed=False` plus an explicit `seeds` list, seed the global RNG for the LM, `thinking=False` needs the three `use_cot_*` flags off too, the LM rewrites caption
   and language unless told not to, xl-sft gets `dcw_enabled=False`, shift 3.0 for both, steps 8 vs 50 (1.4, 1.6).
6. Install: torch 2.10.0 + torchaudio 2.10.0 from the cu128 index, transformers 4.57.6, diffusers 0.37.1 and the short list in 2.3; clone upstream at the pinned commit onto `PYTHONPATH`, do not pip-install it.
   Arch assertion on `sm_86` and `sm_120`, not `sm_89` (2.3).
7. Silence loguru's INFO sink: it prints captions and lyrics into RunPod logs (4.4).

---------------------------------------------------------------------------------------------------------------------

## 1. The Python inference API (a)

### 1.1 Entry points and call chain

Official in-process usage (docs/en/INFERENCE.md:26-73, run_generate_test.py:31-110, profile_inference.py:874-898):

```
dit  = acestep.handler.AceStepHandler()                       # handler.py:62
llm  = acestep.llm_inference.LLMHandler()                     # llm_inference.py:48
status, ok = dit.initialize_service(project_root=..., config_path="acestep-v15-xl-turbo", device="cuda", ...)
status, ok = llm.initialize(checkpoint_dir=..., lm_model_path="acestep-5Hz-lm-1.7B", backend="pt", device="cuda", ...)
result = acestep.inference.generate_music(dit, llm, params, config, save_dir=None)   # inference.py:514
```

- Both `initialize_service` (init_service_orchestrator.py:48-220) and `LLMHandler.initialize` (llm_inference.py:499-777)
  return `(status_string, ok_bool)` and never raise (they catch everything). Stage 2 must check `ok`.
- `generate_music` (inference.py:514-1115) also never raises: it returns `GenerationResult(success=False, error=str(e))`
  (inference.py:1107-1115). `error` can be a long string; map it to one plain sentence, log only the exception type.
- Everything is synchronous and not thread-safe (single handler instances); RunPod runs one job at a time per worker, fine.

### 1.2 DiT config selection, checkpoint names and layout

How a DiT "config" is selected: there is no config object. The DiT is chosen by the NAME of a folder under the checkpoints dir,
passed as `config_path` ("acestep-v15-xl-turbo" or "acestep-v15-xl-sft"). Three things hang off that name:

1. `model_path = os.path.join(checkpoint_dir, config_path)` and the weights are loaded from there with
   `AutoModel.from_pretrained(path, trust_remote_code=True, attn_implementation=..., dtype=self.dtype)`
   (init_service_orchestrator.py:155-162; init_service_loader.py:175-180). The model class comes from the repo's own
   `config.json` `auto_map` (HF-LISTING: turbo -> `modeling_acestep_v15_xl_turbo.AceStepConditionGenerationModel`,
   sft -> `modeling_acestep_v15_xl_base.AceStepConditionGenerationModel`; AutoConfig -> `configuration_acestep_v15.AceStepConfig`).
2. Behaviour switches come from the repo's `config.json`, not from the name: `is_turbo` (HF-LISTING: xl-turbo `true`; xl-sft `false`,
   `model_version` "turbo" vs "base"). `is_turbo_model()` = `getattr(config, "is_turbo", False)` (init_service_catalog.py:75-79).
   It decides: guidance forced to 1.0 (handler generate_music.py:312-318), steps clamped to max 8 (service_generate_request.py:52-57),
   and the DCW default (generate_music.py:181-193 and service_generate_execute.py:64-70).
3. The NAME is used for (a) the code-sync map `_CHECKPOINT_TO_VARIANT` (model_downloader.py:25-43: `acestep-v15-xl-turbo`->`xl_turbo`,
   `acestep-v15-xl-sft`->`xl_sft`) and (b) VRAM heuristics `get_dit_type_from_path` (gpu_config.py:193-217; token match, so an
   absolute path still works for this one).

CODE SYNC (important): before loading, `_sync_model_code_if_needed` (init_service_downloads.py:64-76) hashes every `*.py` in
`acestep/models/<variant>/` (xl_turbo: `configuration_acestep_v15.py`, `modeling_acestep_v15_xl_turbo.py`; xl_sft:
`configuration_acestep_v15.py`, `apg_guidance.py`, `modeling_acestep_v15_xl_base.py`) against the same names inside the checkpoint
folder and, on any difference, OVERWRITES the checkpoint folder's copies with `shutil.copy2` (model_downloader.py:60-133).
Blob sizes in the pinned commit vs the HF repos (HF-LISTING): `modeling_acestep_v15_xl_turbo.py` 109,260 vs 103,821;
`modeling_acestep_v15_xl_base.py` 108,816 vs 105,981; `configuration_acestep_v15.py` is a 212-byte re-export of
`acestep.models.common` upstream vs 13,225 bytes on the Hub; `apg_guidance.py` 602 vs 8,340. They all differ, so the sync
WILL fire on an unprepared snapshot. On a RunPod cache snapshot those files are symlinks into `blobs/`; `copy2` onto a symlink
writes through to the shared blob (or fails if read-only). So never point `config_path` at a snapshot folder directly. The
pinned upstream code expects its own modeling files: the pinned `service_generate_execute.py:103-141` passes `dcw_*`, `repaint_*`,
`retake_*` kwargs that the older HF copy may not accept (INFERENCE). See 3.3 for the layout recipe that avoids all of this.

`config_path` could be an absolute path (os.path.join and Path / both discard the left part for an absolute right operand:
init_service_orchestrator.py:155, model_downloader.py:424), but then `_CHECKPOINT_TO_VARIANT.get()` finds nothing, no code sync
happens, and the HF modeling copy is used. Not recommended.

DiT dtype/device handling (init_service_orchestrator.py:85-101): CUDA with compute capability major >= 8 -> `torch.bfloat16`,
older CUDA -> float16 (ROCm -> float32 unless `ACESTEP_ROCM_DTYPE`); `gpu_config.cuda_supports_bfloat16` is gpu_config.py:111-121.
A40/A6000 (sm_86), L40/6000 Ada (sm_89) and 5090 (sm_120) all get bf16. Attention: `initialize_service(use_flash_attention=False)`
is the default -> `sdpa` (init_service_loader.py:144-162), falling back to `eager` if sdpa fails to load
(init_service_loader.py:164-185; check the status string "Attention: sdpa", an `eager` fallback would be slow and memory hungry).
`flash_attn` is not installed in the recommended image (2.3), so `use_flash_attention=True` would just warn and use SDPA.
The model is moved with `.to(device).to(dtype)` (init_service_loader.py:197-204); `offload_to_cpu=False` keeps everything on the GPU.
`compile_model=False` is the default (orchestrator line 54); leave it off (Triton compile adds minutes and sm_120 risk, INFERENCE).
`quantization=None` is the default; leave it (contract: no INT8). Weights stay fp32 on disk and are cast to bf16 on load (see 4).

Signature (init_service_orchestrator.py:48-61):
`initialize_service(project_root: str, config_path: str, device: str="auto", use_flash_attention: bool=False, compile_model: bool=False,
offload_to_cpu: bool=False, offload_dit_to_cpu: bool=False, quantization: Optional[str]=None, prefer_source: Optional[str]=None,
use_mlx_dit: bool=True, vae_checkpoint: Optional[str]=None) -> Tuple[str, bool]`.

Checkpoints directory resolution (orchestrator lines 118-129; model_downloader.py:344-359): env `ACESTEP_CHECKPOINTS_DIR` if set
(then `.expanduser().resolve()`), else `<project_root>/checkpoints`. `project_root` defaults to `ACESTEP_PROJECT_ROOT` or cwd
(progress.py:20-33); the handler also writes `.cache/acestep/progress_estimates.json` under it (handler.py:143-149, errors
swallowed in progress.py:47-54). Point both at a writable folder under /tmp.

What `initialize_service` demands to exist (all by NAME, all under the checkpoints dir):
- `check_main_model_exists` (model_downloader.py:386-402) requires ALL FOUR main components to hold weights:
  `acestep-v15-turbo` (the 2B DiT, never used by us), `vae`, `Qwen3-Embedding-0.6B`, `acestep-5Hz-lm-1.7B`
  (MAIN_MODEL_COMPONENTS, model_downloader.py:307-312). "Holds weights" = one of `model.safetensors`,
  `model.safetensors.index.json`, `pytorch_model.bin`, ..., `diffusion_pytorch_model.safetensors` exists (model_downloader.py:362-383).
  If any is missing it calls the downloader (init_service_downloads.py:31-36), which first opens a socket to www.google.com
  to choose HF vs ModelScope (model_downloader.py:140-159, 247) and then tries to download ~10 GB. Under `HF_HUB_OFFLINE=1`
  that fails and `initialize_service` returns `(error, False)`. So the dir must be complete BEFORE the call.
- `check_model_exists(config_path)` (init_service_downloads.py:43) for the XL folder; if missing it downloads from
  `SUBMODEL_REGISTRY` (model_downloader.py:290-304: `acestep-v15-xl-turbo` -> `ACE-Step/acestep-v15-xl-turbo`,
  `acestep-v15-xl-sft` -> `ACE-Step/acestep-v15-xl-sft`) into `<ckpt>/<name>/`.
- VAE at `<ckpt>/vae` (`resolve_vae_path`, model_downloader.py:710-738; default variant "official"; `vae_checkpoint` may also be an
  ABSOLUTE path, init_service_loader_components.py:19-66). Loaded with diffusers `AutoencoderOobleck.from_pretrained`
  (loader_components:53), dtype bf16 on Ampere+ (memory_utils.py:199-216).
- Text encoder at `<ckpt>/Qwen3-Embedding-0.6B` (loader_components:90), `AutoTokenizer` + `AutoModel.from_pretrained`, cast to the DiT dtype.
- `<ckpt>/<config_path>/silence_latent.pt` (init_service_loader.py:211-216).
Every `from_pretrained` on the generation path takes a LOCAL directory (audit: init_service_loader.py:175,
loader_components:53/94/95, llm_inference.py:404/612). The only hub calls are in `model_downloader.py:178/204`.
=> With a prepared checkpoints dir the whole path runs under `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; nothing resolves a hub id.
(`trust_remote_code=True` on a local folder copies the .py files into `HF_MODULES_CACHE`, default `~/.cache/huggingface/modules`;
that dir must be writable and `acestep` must be importable because the modeling file does
`from acestep.models.common.dcw_correction import DCWCorrector`, modeling_acestep_v15_xl_turbo.py:53.)

Each `AceStepHandler` also loads its own VAE and text encoder (init_service_orchestrator.py:163-172), so two DiT handlers
duplicate ~1.5 GB. Upstream's own way to switch DiTs is to call `initialize_service(config_path=<other>)` again on the SAME handler
(api/job_model_selection.py:73-107 `_swap_primary_model`): `_load_main_model_from_checkpoint` first `del self.model` and
`torch.cuda.empty_cache()` (init_service_loader.py:130-135). There is no separate unload method for the DiT handler (grep:
only `LLMHandler.unload` exists, llm_inference.py:125). A manual release that mirrors upstream's failure path is
`h.model = h.vae = h.text_encoder = h.text_tokenizer = h.config = h.silence_latent = None; gc.collect(); torch.cuda.empty_cache()`
(orchestrator lines 211-218).

### 1.3 The 1.7B planner LM: loading and backends

`LLMHandler.initialize(checkpoint_dir: str, lm_model_path: str, backend: str="vllm", device: str="auto", offload_to_cpu: bool=False,
dtype: Optional[torch.dtype]=None) -> Tuple[str, bool]` (llm_inference.py:499-507).
- `full_lm_model_path = os.path.join(checkpoint_dir, lm_model_path)` (line 590): `lm_model_path` may be a relative folder name
  ("acestep-5Hz-lm-1.7B") or an absolute path (os.path.join). It must exist or you get `(msg, False)` (line 591-592).
  The LM size is read from the NAME ("1.7B" substring) for memory heuristics (gpu_config.py:894-912), so keep "1.7B" in the folder name.
- dtype default: bf16 on cuda/xpu, float32 otherwise (lines 566-577). Device "cuda" or "auto".
- Backends: `"vllm"` (nano-vllm), `"pt"` (HF transformers, custom decode loop), `"mlx"` (Apple only). Resolution:
  `backend == "pt"` goes straight to `_load_pytorch_model` (lines 767-772; 401-416: `AutoModelForCausalLM.from_pretrained(path,
  trust_remote_code=True)` then `.to(device).to(dtype)`). `backend == "vllm"` silently falls back to pt if CUDA is absent, if
  free VRAM < 2 GB (VRAM_SAFE_FREE_GB, line 33/735) or if nano-vllm fails to import/init (lines 721-766). The status string says which
  backend actually ran: log it (`llm.llm_backend` is "pt" or "vllm" after init).
- The global GPU config picks the recommended backend per VRAM tier (gpu_config.py:316-470; >= 24 GB = tier "unlimited", recommended LM
  4B, backend vllm, max duration with LM 600 s, max batch 8, no offload, no quantization). `resolve_lm_backend` (gpu_config.py:285-309).
  Cards of the target class: A40/A6000 48 GB, L40/L40S/6000 Ada, 5090 32 GB are all "unlimited". The global config is built lazily
  from the detected VRAM (gpu_config.py:1569-1582), initialising CUDA the first time anything asks (so also at `AceStepHandler()`
  construction: handler.py:175).
- vLLM path details (INFERENCE where unverified): nano-vllm is pure Python (setuptools), allocates a KV cache pool up front sized
  from FREE VRAM (`get_lm_gpu_memory_ratio`, gpu_config.py:976-1075) and uses `enforce_eager=True` when `flash_attn` or `triton` is missing
  (llm_inference.py:652-675), with a Python-loop SDPA decode fallback (nano-vllm attention.py:233-285, 327-330).
  See 2.3 for why the recommendation is `pt`.
- pt path details: single item = custom decode loop with HF KV cache; with `lm_cfg_scale > 1` (default 2.0) it runs cond+uncond as a
  batch of 2 (llm_inference.py:1042-1087, 2618-2790). Constrained decoding (FSM for the metadata block, audio-code-only vocabulary) runs in
  Python each step. Codes are generated at 5 per second of audio (llm_inference.py:278-282): 180 s = 900 codes, 360 s = 1800 codes.
  ESTIMATE (not measured): ~25-40 decoded tokens/s on an A40, so planner time is roughly 40-70 s for a 3-minute song and 70-110 s for 6 minutes,
  on top of ~7 s for the metadata (CoT) phase. That is longer than the xl-turbo render itself (~25 s for 2:30 on a 5090 per the research file).
  Report it separately as `timing.plan_s`. `thinking=False` removes it entirely.
- `max_model_len` is 4096 on the handler (line 62) and caps `max_new_tokens` at 4032 (lines 299-301): plenty for 600 s (3000 codes).

### 1.4 Generation entry point: names, types, defaults

`acestep.inference.generate_music(dit_handler, llm_handler, params: GenerationParams, config: GenerationConfig, save_dir: Optional[str]=None,
progress=None) -> GenerationResult` (inference.py:514-521). Decorator `_get_spaces_gpu_decorator` is a no-op outside HF Spaces.

`GenerationParams` (inference.py:55-233, dataclass). Fields the worker sets (worker field -> param, type, default, notes):

| worker input | GenerationParams field | type / upstream default | notes |
|---|---|---|---|
| caption | `caption` | str, "" | docs say < 512 chars. Text encoder hard-truncates the whole prompt (instruction + caption + metas) at 256 tokens (conditioning_text.py:127-133). |
| lyrics | `lyrics` | str, "" | Instrumental = the literal string `"[Instrumental]"`. `GenerationParams.instrumental` is NOT consumed by text2music (grep: only echoed in params and used by create_sample), so set the lyrics string. Lyric tokens truncate silently at 2048 (conditioning_text.py:138-144); 4096 chars of normal English is ~1,100 tokens. Lyrics are wrapped as `# Languages\n<lang>\n\n# Lyric\n<lyrics><|endoftext|>` (prompt_utils.py:27-29). |
| duration | `duration` | float, -1.0 (auto) | seconds, 10-600 (constants.py:58-59). The LM clamps its target to `[10, min(gpu tier max, 600)]` (llm_inference.py:266-278). |
| bpm | `bpm` | Optional[int], None | 30-300 (constants.py:52-53). |
| keyscale | `keyscale` | str, "" | LM and constrained decoder use exactly `"<A-G><accidental> <major|minor>"` with accidental none, `#`, `b`, or the Unicode sharp/flat signs, lowercase mode, e.g. "G major", "F# minor" (constants.py:27-42). Normalise "A Minor"->"A minor"; drop an unparsable value with a worker note. |
| timesignature | `timesignature` | str, "" | "2","3","4","6" (constants.py:66; docs INFERENCE.md:348), passed as a string. |
| vocal_language | `vocal_language` | str, "unknown" | ISO-ish code from `VALID_LANGUAGES` (constants.py:13-20: 'en','es','fr',...,'yue','zh','unknown'). Pass "en" explicitly. |
| seed | `GenerationConfig.seeds` (see 1.5) | int | see seed handling below |
| thinking | `thinking` | bool, True | True = LM writes the plan AND the 5 Hz audio codes the DiT then follows. |
| steps | `inference_steps` | int, 8 | turbo clamps to <= 8 in the handler (service_generate_request.py:52-57), the turbo sampler itself allows 1-20 (modeling_acestep_v15_xl_turbo.py:1982-1991). |
| guidance_scale | `guidance_scale` | float, 7.0 | turbo forces 1.0 (generate_music.py:312-318). |
| shift | `shift` | float, 1.0 | `__post_init__` turns shift <= 0 into 1.0 (inference.py:223-229). Upstream HTTP API and Gradio use 3.0 (api/http/release_task_request_builder.py:83; model_config.py:121,138). |
| (DCW) | `dcw_enabled` Optional[bool] None, `dcw_mode` "double", `dcw_scaler` 0.05, `dcw_high_scaler` 0.02, `dcw_wavelet` "haar" | | None = on for turbo, off otherwise (docs/en/DCW.md:81; generate_music.py:181-193). |
| lm_temperature | `lm_temperature` | float, 0.85 | optional; sent only when the caller sets it and the planner runs; reaches only the planner (8.11). |
| batch_size | `GenerationConfig.batch_size` | int, 2 | default is 2, not 1: always set. |

Other fields worth knowing: `use_cot_metas/use_cot_caption/use_cot_language` (default True), `lm_temperature` 0.85, `lm_cfg_scale` 2.0, `lm_top_k` 0,
`lm_top_p` 0.9, `use_constrained_decoding` True, `enable_normalization` True with `normalization_db=-1.0` (peak-normalises each take to -1 dBFS,
inference.py:134-138, 998-1013), `fade_in_duration`/`fade_out_duration` 0, `infer_method` "ode", `sampler_mode` "euler", `use_adg` False,
`cfg_interval_start/end` 0.0/1.0, `task_type` "text2music". (`timesteps`, repaint/cover/flow_edit fields are unused for text2music.)

`GenerationConfig` (inference.py:236-267): `batch_size=2`, `allow_lm_batch=False` (not read by generate_music), `use_random_seed=True`,
`seeds: Optional[List[int]]=None`, `lm_batch_chunk_size=8`, `constrained_decoding_debug=False`, `audio_format="flac"`,
`mp3_bitrate="128k"`, `mp3_sample_rate=48000`.

Behaviours that are easy to get wrong (all READ):
1. SEEDS. `use_random_seed=True` (the default) makes `prepare_seeds` ignore every seed you pass and draw random ones
   (task_utils.py:26-28). So set `use_random_seed=False` AND `seeds=[s, s+1, ...]` (list; inference.py:597-600). A single int seed with
   batch > 1 gives only take 0 that seed; the others are random (task_utils.py:48-52). Seeds are parsed as `int(float(str))`
   (task_utils.py:37), which loses precision above 2^53, and upstream itself draws seeds in [0, 2^32-1]; keep worker seeds <= 2,147,483,647
   (the booth/YuE limit) and compute `seed + i` modulo that. The seed that was used for take i is `audios[i]["params"]["seed"]`
   (inference.py:969) and `extra_outputs["seed_value"]` is the joined string. The DiT noise uses `torch.Generator(device).manual_seed(int(s))`
   per take (modeling_acestep_v15_xl_turbo.py:1811-1827), so the DiT part is deterministic per seed.
   The LM is NOT seeded for batch 1: the `seeds` argument to the LM is ignored in single mode ("TODO: not used yet", llm_inference.py:1318-1319)
   and `_run_pt` only calls `torch.manual_seed` in batch mode (llm_inference.py:1186-1191). To make `seed` reproduce the plan, call
   `torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed)` once before `generate_music` (INFERENCE: cheap, bitwise
   reproducibility across GPU models is not promised anyway).
2. THINKING OFF does not skip the LM by itself. `use_lm = (thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized`
   with `need_lm_for_cot = use_cot_caption or use_cot_language or use_cot_metas` (inference.py:653-655). With thinking False and the cot flags
   at their default True the LM still runs a metadata-only pass. To honour "thinking false -> LM skipped", pass `use_cot_metas=False,
   use_cot_caption=False, use_cot_language=False` (or pass `llm_handler=None`). Do NOT set `use_cot_metas=False` while thinking is True
   and no user metadata exists (generate_with_stop_condition would call `.items()` on None, llm_inference.py:1429); duration is always set so
   user_metadata is never empty in the worker, still guard it.
3. THE LM OVERWRITES THE CAPTION AND THE LANGUAGE by default. With `use_cot_caption=True` the DiT's caption becomes the LM's rewritten caption
   (inference.py:825-826); with `use_cot_language=True` the DiT language becomes the LM's detection (827-828; user language is not part of
   the metadata sent to the LM, 672-700). Recommended: `use_cot_language = False` whenever the caller gave `vocal_language` (the worker never turns it on, and an absent
   vocal_language is sent as "unknown", upstream's own default, because the booth has no Language setting and the family writes in other languages); `use_cot_caption` is a product decision (upstream default True expands a short direction into a rich one;
   False keeps the caller's exact words). Record the choice in `worker_notes`.
4. If bpm + keyscale + timesignature + duration are ALL supplied, phase 1 (metadata CoT) is skipped (llm_inference.py:1245-1251, 1367, 1423-1429)
   and the caption is not rewritten, but phase 2 (audio codes) still runs when thinking is True. With only some supplied, the constrained decoder
   injects the given values and the LM fills the rest (user_metadata, inference.py:672-700).
5. LM-decided values are copied back into the (mutated) `params` object: `params.cot_bpm`, `cot_keyscale`, `cot_timesignature`, `cot_duration`,
   `cot_vocal_language`, `cot_caption` (inference.py:809-822). Raw dict: `result.extra_outputs["lm_metadata"]` (keys bpm, caption, duration,
   keyscale, language, timesignature; line 1071). Use these for the contract's `plan`; do not copy `audios[i]["params"]` wholesale: it carries the
   full lyrics, the caption and the entire `audio_codes` string (lines 977-983).
6. BATCH + LM: with batch > 1, phase 1 is shared and phase 2 is generated per item; under pt that is SEQUENTIAL per take with `torch.manual_seed(seeds[i])`
   (llm_inference.py:1180-1218), so planner time scales with batch size.
7. `_vram_guard_reduce_batch` can silently lower the batch size when free VRAM is tight (memory_utils.py:138-197; XL x1.7, sft/base x2) and a
   pre-flight can refuse the request (generate_music.py:106-179; skip with `ACESTEP_SKIP_VRAM_PREFLIGHT=1`). The number of returned `audios`
   can be smaller than `batch_size`; build `takes` from what came back and add a worker note.
8. Latents with NaN/Inf or all zeros raise inside the handler and come back as `success=False` (generate_music_decode.py:67-86).

### 1.5 What it returns

`GenerationResult(audios: List[dict], status_message: str, extra_outputs: dict, success: bool, error: Optional[str])` (inference.py:278-302).
Each `audios[i]` (inference.py:1059-1065): `{"path": str, "tensor": torch.Tensor [channels, samples] float32 on CPU, "key": uuid-from-params, "sample_rate": 48000, "params": dict}`.
- Sample rate 48,000 Hz (handler.py:121), stereo (VAE config: 2 channels, 48000 Hz, downsample product 1920 = 25 Hz latents; HF-LISTING vae/config.json).
- Tensors are returned in memory. Files are written only if `save_dir` is not None (inference.py:1042). The worker should pass `save_dir=None` and write
  its own files, because upstream's saver goes through `torchaudio.save(..., backend='soundfile')` (audio_utils.py:303-309), and torchaudio 2.9+ moved I/O onto
  torchcodec (INFERENCE, unverified); on failure it falls back to plain soundfile (lines 322-343) but why depend on it.
- Output audio is clamped (divide by peak if > 1.0, generate_music_decode.py:199-203) and then, by default, peak-normalised to -1 dBFS in `generate_music`.
- WAV 24-bit 48 kHz: `soundfile.write(path, tensor.T.numpy(), 48000, subtype="PCM_24")` (tensor.T because upstream is channels-first; audio_utils.py:292
  does the same transpose), or float WAV -> `ffmpeg -c:a pcm_s24le` exactly like yue_handler.py:1373. MP3: `ffmpeg -i master.wav -c:a libmp3lame -b:a 320k`
  (yue_handler.py:1374). 24-bit stereo 48 kHz is 288,000 bytes per second: 3 min = 52 MB, 6 min = 104 MB, 10 min = 173 MB.
- Timing: `extra_outputs["time_costs"]` (inference.py:1073-1094) holds `lm_phase1_time`, `lm_phase2_time`, `lm_total_time` (only when the LM ran),
  `dit_encoder_time_cost`, `dit_diffusion_time_cost`, `dit_diffusion_per_step_time_cost`, `dit_vae_decode_time_cost`, `dit_total_time_cost`,
  `dit_offload_time_cost`, `pipeline_total_time`. Map: `plan_s = lm_total_time`, `render_s = dit_total_time_cost`.
- Memory: `ACESTEP_SAVE_MEMORY=1` makes the payload skip the big intermediate tensors (generate_music_payload.py:52-64, gpu_config.py:1575-1581; said to save
  4-8 GB of host RAM per generation). Set it.
- Progress: pass `progress=` any callable accepting `(ratio, desc=...)` AND `(ratio, "text")` (llm_inference.py:1366 passes the text positionally; the DiT handler,
  core/generation/handler/generate_music.py:331, passes `desc=`). Define `def progress(*a, **k)` and forward a plain phrase to `runpod.serverless.progress_update`.

### 1.6 Recommended settings per checkpoint (from upstream docs and UI defaults)

| | xl-turbo | xl-sft |
|---|---|---|
| steps | 8 (Tutorial.md:261; model_config.py:116; allowed 1-20, clamped to 8 by the handler) | 50 (Tutorial.md:262; model_config.py:131) |
| guidance | ignored (forced 1.0) | 7.0 default, typical 5-9 (INFERENCE.md:367) |
| shift | 3.0 ("the only turbo parameter to set manually", INFERENCE.md:354-362; Gradio model_config.py:121) | 3.0 (Gradio model_config.py:138; HTTP default 3.0) |
| DCW | model default = on, "double" 0.05/0.02 (DCW.md:81, 148-157). Gradio uses 0.02/0.06 when Think is on (DCW.md:144-146); the HTTP route and `GenerationParams` carry only the 0.05/0.02 pair. (Stage 1 said "start with the defaults"; stage 4 changed it: the worker now passes 0.02/0.06 when the planner is used and 0.05/0.02 when it is not, see 8.10.) | pass `dcw_enabled=False` explicitly (contract; upstream default is already off because `is_turbo` is false in its config.json, but do not rely on that) |
| sampler | ode / euler, `use_adg=False`, cfg interval 0..1 | same |
| think | thinking True gives the LM's 5 Hz codes (the "plan") | same |

With 8 steps and shift 3.0 the turbo sampler builds exactly its distilled schedule `[1.0, .9545, .9, .8333, .75, .6429, .5, .3]`
(modeling_acestep_v15_xl_turbo.py:1938-1942, 1982-1991; checked by hand). If the caller passes a different `steps`, the turbo class computes a linspace +
shift schedule instead; allowed 1-20.

### 1.7 LoRA seam (reserved)

`AceStepHandler` already has `add_lora(path, adapter_name)`, `load_lora(path)`, `set_lora_scale(name, scale)`, `set_use_lora(bool)`, `remove_lora(name)`,
`unload_lora()` (core/generation/handler/lora_manager.py:17-36; lifecycle.py:191-325). They need `peft` (not installed in the recommended image), return
status STRINGS starting with a check mark / cross instead of raising, refuse quantized models (lifecycle.py:200-205, which is why INT8 stays off), and back up the
whole decoder state_dict to CPU on first use (lifecycle.py:246-260). Per-request loading would go in `apply_adapter(pipe, key)`: fetch the adapter from B2 to /tmp,
`h.add_lora(dir, name)`, `h.set_lora_scale(name, scale)`, generate, then `h.unload_lora()` in a `finally`. **Superseded in stage 6 (section 9):** `apply_adapter` is now a context manager that folds the
adapter into the decoder weights and puts them back from a pristine copy, because running upstream's real `unload_lora()` against real PEFT showed that it leaves the adapter in the decoder (9.2).

---------------------------------------------------------------------------------------------------------------------

## 2. Dependencies and install recipe (b)

### 2.1 What upstream pins

- Python: `requires-python >=3.11,<3.13` (pyproject.toml:6; uv.lock:3). 3.12 is allowed.
- Linux x86_64 torch stack: `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`, `torchaudio==2.10.0+cu128` (pyproject.toml:14-16; requirements.txt:13-15).
  Index `https://download.pytorch.org/whl/cu128` (pyproject.toml:64-67).
- uv.lock resolved versions (the set upstream tests): transformers 4.57.6 (pyproject says `>=4.51,<4.58`), diffusers 0.37.1, huggingface-hub 0.36.0,
  tokenizers 0.22.2, safetensors 0.7.0, accelerate 1.12.0, numpy 2.3.5, scipy 1.17.0, soundfile 0.13.1, loguru 0.7.3, einops 0.8.2,
  vector-quantize-pytorch 1.27.20, pytorch-wavelets 1.3.0, pywavelets 1.9.0, torchcodec 0.10.0, torchao 0.16.0 (pyproject pins `>=0.16,<0.17`), triton 3.6.0,
  peft 0.18.1, lycoris-lora 3.4.0, numba 0.63.1, xxhash 3.6.0, nano-vllm 0.2.0, flash-attn 2.8.3+cu128torch2.10 (cp311 only).
- `transformers>=4.56` is required for the `dtype=` keyword used in init_service_loader.py:179 (older versions took `torch_dtype`; INFERENCE from the
  transformers 4.56 rename). 4.57.6 satisfies it. Keep the pin.
- Things that need compilation: nothing on the recommended path. `flash-attn` is a prebuilt cp311 wheel from a third-party GitHub release
  (uv.lock:609-621, acestep/third_parts/nano-vllm/pyproject.toml:20); on Python 3.12 the markers (`python_version == '3.11'`, lines 16-20) do not match, so it is
  simply not installed (there is no cp312 wheel in the lock; building flash-attn from source on a slim image needs nvcc plus 30+ minutes). `nano-vllm` itself is pure Python.
  `triton` arrives with torch on Linux (torch 2.10 cu128 depends on triton 3.6.0, INFERENCE) but nano-vllm's own triton requirement is also restricted to Python 3.11 (pyproject.toml:17).
  `numba`, `vector-quantize-pytorch`, `pytorch-wavelets`, `pywavelets`, `soundfile` are wheels.

### 2.2 What the inference path actually imports (measured with F:/KadeAI-work/ace/scratch/import_graph.py, a static AST walk)

Module level, from `acestep.inference`, `acestep.handler`, `acestep.llm_inference`, `acestep.gpu_config`, `acestep.model_downloader`:
`torch, torchaudio, transformers, numpy, soundfile, loguru, tqdm, yaml`. Inside functions: `diffusers` (VAE), `safetensors`, `huggingface_hub`
(downloader only), `modelscope` (downloader only), `torchao` (quantization only), `flash_attn`, `triton`, `nanovllm`, `mlx*` (Apple), `torchcodec` (audio loading in the
training/dataset modules only), `spaces`. The remote modeling code adds `einops`, `vector_quantize_pytorch` (module level, modeling file lines 23 and 42) and
`pytorch_wavelets` (lazy, only when DCW is active; missing = warning and DCW silently skipped, models/common/dcw_loader.py:32-47).
NOT needed for text2music: gradio, fastapi, uvicorn, scipy, numba, matplotlib, lightning, tensorboard, peft, lycoris, modelscope, torchvision, torchao, torchcodec.

### 2.3 Recommended install recipe for `python:3.12-slim-bookworm`

```
FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN python -c "import torch; f = torch._C._cuda_getArchFlags(); print('ACE env CUDA archs:', f); assert 'sm_86' in f and 'sm_120' in f, f"
RUN git clone https://github.com/ace-step/ACE-Step-1.5.git /opt/ace-step && cd /opt/ace-step && git checkout ca1e85fe9430179831e6bc6be790c332190a3866
RUN pip install --no-cache-dir transformers==4.57.6 huggingface_hub==0.36.0 tokenizers==0.22.2 safetensors==0.7.0 accelerate==1.12.0 diffusers==0.37.1 \
    numpy==2.3.5 soundfile==0.13.1 loguru==0.7.3 einops==0.8.2 vector-quantize-pytorch==1.27.20 pytorch-wavelets==1.3.0 pywavelets==1.9.0 \
    pyyaml tqdm 'runpod>=1.7,<2' 'boto3>=1.35,<2' 'requests>=2.32,<3'
ENV PYTHONPATH=/opt/ace-step ACE_STEP_COMMIT=ca1e85fe9430179831e6bc6be790c332190a3866
```
STAGE 3 CORRECTIONS to this recipe (the shipped file is `Dockerfile.ace`, details in 8.2): `setuptools==71.1.0` and `six==1.17.0` were added (pytorch-wavelets needs `pkg_resources`), `requests` is 2.34.2 not 2.32.x
(runpod 1.12.0 needs it), runpod, boto3 and botocore are exact pins, and the lock-file version of every other package is used as listed here.
Notes on the recipe:
- Mirrors `Dockerfile.yue` lines 10-15 (same base, same cu128 index, same git clone + checkout style). pip resolves `torch==2.10.0` to the `+cu128` build from that
  index exactly as Dockerfile.yue:13 does; later `pip install` calls will not replace it because `==2.10.0` is satisfied by `2.10.0+cu128`.
- Do NOT `pip install` upstream itself (`pip install .` would drag in gradio, fastapi, uvicorn, numba, lightning, tensorboard, torchao...). Put the clone on
  `PYTHONPATH`; `acestep` is a plain package (acestep/__init__.py is a docstring only) and its code-sync source dir is `Path(__file__).parent/"models"`
  (model_downloader.py:46-48). The repo is ~21 MB; no pruning needed.
- The arch assertion must NOT look for `sm_89`: PyTorch wheels list `sm_80/sm_86/sm_90/sm_100/sm_120`-style cubins and run on sm_89 (L40, 6000 Ada, 4090)
  through CUDA binary compatibility within the 8.x family (INFERENCE: standard CUDA rule, and `Dockerfile.yue:6,31` assert only sm_86 and sm_120).
  The contract text says sm_89; assert `sm_86` and `sm_120`, print the full flag list so CI shows it, and say so in the Dockerfile comment.
- LM backend CHOICE: `pt`. Reasons (READ/INFERENCE): (1) nano-vllm installs fine on 3.12 (pure Python, no compiler) but without `flash_attn` it runs `enforce_eager=True`
  with a Python-loop SDPA decode (attention.py:233-285, llm_inference.py:675), losing most of vLLM's speed advantage, and no cp312 flash-attn wheel exists in the lock;
  (2) it pre-allocates a KV pool from free VRAM (gpu_config.py:976-1075) which competes with the lazily loaded xl-sft and with the 32 GB 5090 budget;
  (3) its Triton KV kernel and shared-memory multiprocess runner are extra moving parts on sm_120 (nano-vllm utils/compat.py:3-6 itself warns about Blackwell Triton);
  (4) upstream falls back to pt automatically on any vLLM failure, so `pt` is the behaviour we would end up with anyway. Cost: planner time (1.3). Make the backend an env knob
  (`ACE_LM_BACKEND`, default `pt`) so the smoke test can try `vllm` after `pip install --no-deps /opt/ace-step/acestep/third_parts/nano-vllm xxhash`.
- torchao is deliberately NOT installed (contract: no INT8). Upstream warns that an unpinned torchao 0.18.x breaks model load (requirements.txt:37-41); if a build-time import
  check ever demands it, pin `torchao==0.16.0` (pyproject.toml:42), never unpinned. `peft==0.18.1` and `lycoris-lora==3.4.0` come with the LoRA phase later.
- Size estimate (INFERENCE, not measured): torch 2.10 cu128 + nvidia libs about 7-8 GB on disk, the rest about 1.5 GB, base 0.15 GB, ffmpeg 0.1 GB: roughly 9-10 GB, no weights.
  Dockerfile.yue is the existing example of an image of this kind that fits the 15 GB limit.
- Build-time checks to put in the Dockerfile: the arch assertion above; `python -c "import acestep.inference, acestep.handler, acestep.llm_inference, acestep.model_downloader;
  from acestep.inference import GenerationParams, GenerationConfig, generate_music"` (imports only; `AceStepHandler()` would also work with no GPU but is not needed);
  `python -c "import ace_handler"`; `python -m unittest test_ace_handler.py`; and a check that the code-sync source files exist:
  `ls /opt/ace-step/acestep/models/xl_turbo/modeling_acestep_v15_xl_turbo.py /opt/ace-step/acestep/models/xl_sft/modeling_acestep_v15_xl_base.py`.
- Env for the runtime (all read by upstream): `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub TOKENIZERS_PARALLELISM=false
  ACESTEP_PROJECT_ROOT=/tmp/ace-root ACESTEP_CHECKPOINTS_DIR=/tmp/ace-checkpoints ACESTEP_DISABLE_TQDM=1 ACESTEP_SAVE_MEMORY=1`.

---------------------------------------------------------------------------------------------------------------------

## 3. RunPod cache layout vs upstream layout (c)

### 3.1 What upstream expects (a flat `checkpoints/` folder)

```
checkpoints/
  acestep-v15-turbo/            2B DiT from the main repo: only needs to hold weights (check_main_model_exists); never loaded by us
  acestep-v15-xl-turbo/         config.json + modeling/config .py + shard files + silence_latent.pt   (the DiT we load)
  acestep-v15-xl-sft/           same shape, loaded lazily
  vae/                          config.json + diffusion_pytorch_model.safetensors
  Qwen3-Embedding-0.6B/         tokenizer + model.safetensors
  acestep-5Hz-lm-1.7B/          tokenizer + model.safetensors
```
Names are exact and case sensitive (model_downloader.py:287-312, init_service_loader_components.py:90, llm_inference.py:590).

### 3.2 What RunPod's cache gives us

`/runpod-volume/huggingface-cache/hub/models--<org>--<name>/snapshots/<sha>/...` (standard HF cache; files are symlinks to `../../blobs/<hash>`; this is what
`Dockerfile.yue:17-18,28-30` already relies on). For this engine: `models--ACE-Step--acestep-v15-xl-turbo`, `models--ACE-Step--acestep-v15-xl-sft`,
`models--ACE-Step--Ace-Step1.5`. A snapshot folder of the DiT repos is flat (the DiT files sit directly in `snapshots/<sha>/`); the main repo's snapshot holds the four sub-folders.

### 3.3 Build the checkpoints dir at boot (recipe)

Do this once per process, in a writable scratch dir (default `/tmp/ace-checkpoints`), after the three snapshot folders are resolved:

1. `vae`, `Qwen3-Embedding-0.6B`, `acestep-5Hz-lm-1.7B` and `acestep-v15-turbo`: symlink each to `<main snapshot>/<same name>` (upstream only reads them).
   If `acestep-v15-turbo` is absent from the snapshot, create a real folder with an empty `model.safetensors`; `_contains_model_weights` only tests `.exists()`
   (model_downloader.py:381-383) and the 2B model is never loaded.
2. For each DiT (`acestep-v15-xl-turbo`, `acestep-v15-xl-sft`): create a REAL directory, symlink every non-`.py` entry of the snapshot into it (`config.json`,
   `model-0000N-of-00004.safetensors`, `model.safetensors.index.json`, `silence_latent.pt`), and COPY (not link) the code files from
   `/opt/ace-step/acestep/models/<xl_turbo|xl_sft>/*.py` except `__init__.py`. Then `_check_code_mismatch` finds no difference, the sync never writes, and the shared cache
   is never touched. (Files in a snapshot are symlinks to blobs; `shutil.copy2` onto them would write through, see 1.2.)
3. Export `ACESTEP_CHECKPOINTS_DIR` to that dir (so `initialize_service` and `get_checkpoints_dir()` agree) and pass the same path as `checkpoint_dir` to `LLMHandler.initialize`.
4. Validate before calling upstream: check the specific files (`vae/diffusion_pytorch_model.safetensors`, `Qwen3-Embedding-0.6B/model.safetensors`,
   `acestep-5Hz-lm-1.7B/model.safetensors`, `acestep-v15-xl-*/model.safetensors.index.json` and its 4 shards, `silence_latent.pt`). `yue_handler._filled()` (any file in the folder, lines 49-53)
   is too weak here: a half-filled main snapshot would pass it and fail later inside upstream with a long message.
5. If a repo is missing from the cache, download its pinned revision once with `huggingface_hub.snapshot_download(repo, revision=sha, cache_dir="/tmp/hf-hub")` as `yue_handler.fetch_model`
   does (lines 76-87), then build from that snapshot. Never call upstream's `model_downloader` (it opens a socket to google.com and falls back to ModelScope).

The HF repos' own `*.py` files are NOT used (the pinned upstream copies replace them), only `config.json` (auto_map, `is_turbo`), weights, the index and `silence_latent.pt`.

### 3.4 Contents of the three HF repos (HF-LISTING, Oct 10 2026; sizes in bytes, summed by hand)

`ACE-Step/acestep-v15-xl-turbo` (not gated, not private): `.gitattributes`, `README.md`, `config.json` (2,407), `configuration_acestep_v15.py` (13,225),
`model-00001-of-00004.safetensors` 4,986,971,456; `-00002-` 4,986,942,776; `-00003-` 4,986,942,808; `-00004-` 4,988,483,464 (total 19,949,340,504 = 19.95 GB),
`model.safetensors.index.json` (71,471), `modeling_acestep_v15_xl_turbo.py` (103,821), `silence_latent.pt` (3,841,215).
config.json: `auto_map` as in 1.2, `"dtype": "bfloat16"`, `"is_turbo": true`, `"model_version": "turbo"`, hidden_size 2560, 32 layers, `transformers_version` 4.57.0.
The shards are fp32-sized (about 5.0 B parameters at 4 bytes); the `dtype` field says bf16 but the files are not. Loading with `dtype=bf16` halves it to about 10 GB in VRAM
(upstream's own estimate is 9.0 GB, gpu_config.py:151).

`ACE-Step/acestep-v15-xl-sft`: same four shards (identical sizes), `model.safetensors.index.json`, `silence_latent.pt`, `config.json` (2,525; `"is_turbo": false`, `"model_version": "base"`,
auto_map -> `modeling_acestep_v15_xl_base.AceStepConditionGenerationModel`), `configuration_acestep_v15.py` (13,225), `apg_guidance.py` (8,340), `modeling_acestep_v15_xl_base.py` (105,981), README.

`ACE-Step/Ace-Step1.5` (28 files): `acestep-v15-turbo/{config.json, configuration_acestep_v15.py, model.safetensors (4,787,825,604), modeling_acestep_v15_turbo.py, silence_latent.pt}`,
`acestep-5Hz-lm-1.7B/{added_tokens.json, chat_template.jinja, config.json, merges.txt, model.safetensors (3,708,521,528), special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.json}`,
`Qwen3-Embedding-0.6B/{added_tokens.json, chat_template.jinja, config.json, merges.txt, model.safetensors (1,191,586,416), special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.json}`,
`vae/{config.json, diffusion_pytorch_model.safetensors (337,431,388)}`, plus root `config.json`, `README.md`, `.gitattributes`. Total about 10.1 GB.
LM config: model_type qwen3, 28 layers, hidden 2048, vocab 217,204 (summarised; matches the 1.7B naming). VAE config: `AutoencoderOobleck`, 2 channels, 48 kHz.
Cache footprint of the whole engine: 19.95 + 19.95 + 10.1 = about 50 GB; VRAM resident after boot (turbo + LM + VAE + text encoder, bf16): about 10 + 3.7 + 0.3 + 1.2 = 15 GB;
with xl-sft loaded as a second handler add about 11.5 GB (INFERENCE from file sizes).

### 3.5 Revision pins as of today (HF-LISTING; re-verify before pinning, the summariser may have mistyped a character)

| repo | sha | lastModified |
|---|---|---|
| ACE-Step/acestep-v15-xl-turbo | d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee | 2026-04-07 |
| ACE-Step/acestep-v15-xl-sft | d06de46b4622f781cf07f4a013a67d591ca52819 | 2026-04-07 |
| ACE-Step/Ace-Step1.5 | 19671f406d603126926c1b7e2adc169acbcade22 | 2026-02-03 |

All three have been stable for months, so pinning the Dockerfile `ENV ACE_TURBO_DIR=/runpod-volume/huggingface-cache/hub/models--ACE-Step--acestep-v15-xl-turbo/snapshots/<sha>` style variables (like `YUE_MODEL_DIR`)
is low risk. The orchestrator can confirm each sha with `git ls-remote` on the Hub URL (a metadata call, not made here). Whether RunPod's cache can hold three repos on one endpoint (`MODEL_NAME` vs
a list) is not answerable from upstream; the research file says caching handles one repo per endpoint, in which case two repos arrive through the download fallback on first boot (about 30-40 GB).

---------------------------------------------------------------------------------------------------------------------

## 4. Things that will bite on a real worker (d)

1. COLD START IS BIG. 50 GB cached, about 25 GB read from the network volume at boot (turbo shards 19.95 GB + LM 3.7 + TE 1.2 + VAE 0.3), fp32 shards cast to bf16 on load, plus the LM tokenizer
   (upstream's own comment says the tokenizer load "may take 80~90s", llm_inference.py:609-611). Do all of it in `warm()` so FlashBoot snapshots a ready worker, and log per-stage seconds.
2. THE CODE-SYNC WRITE. Described in 1.2: an unprepared snapshot makes upstream overwrite files inside the shared cache (or fail with a permission error and report only "Error initializing model").
   Use the real-directory-plus-copied-code recipe in 3.3. After building, assert `_check_code_mismatch("acestep-v15-xl-turbo", ckpt) == []` as a boot self-check.
3. `check_main_model_exists` WANTS THE 2B TURBO. A checkpoints dir without a weighted `acestep-v15-turbo` makes upstream start a download (socket to google.com, then Hub, then ModelScope) even
   though we never use that model. In an offline container this shows up as a confusing failure (1.2, 3.3 step 1).
4. LOG PRIVACY AND VOLUME. loguru's default sink prints at INFO the full DiT text input including the caption and lyrics (handler `conditioning_text.py:118-125`) and the LM prompt with caption and lyrics
   (`llm_inference.py:1377,1490`). RunPod keeps worker logs. Remove the default sink at import and add one at WARNING with `backtrace=False, diagnose=False`
   (`from loguru import logger; logger.remove(); logger.add(sys.stderr, level="WARNING", backtrace=False, diagnose=False)`). The two extra arguments are NOT optional: loguru's defaults are both True, and with `diagnose`
   every `logger.exception(...)` that upstream calls around a render (`inference.py:1108` "Music generation failed", `generate_music.py:522`, `prompt_utils.py:91,132`) prints the VALUES of the variables on each traceback line,
   and those frames hold the whole caption and lyrics (stage 4 found this; the first version of this note and of the worker had only the level). Never return upstream's
   `error`/`status_message` text (it embeds tracebacks). `yue_handler` logs only exception types (lines 1428-1429); the worker does the same for request-time failures. The two load-failure lines (model, planner) print upstream's first
   status line with web links and Hugging Face tokens removed (`scrub()`), because that line is the only clue when a first boot fails on the cache layout, and no request text is involved at load time.
5. LM REWRITES CAPTION AND LANGUAGE BY DEFAULT (1.4 item 3). Left at defaults, the booth's "Music direction" text is replaced by the LM's own caption and the caller's language is replaced by the LM's guess.
6. THINKING=FALSE STILL RUNS THE LM unless the three `use_cot_*` flags are also False (1.4 item 2).
7. SEEDS (1.4 item 1): without `use_random_seed=False` the reported seed would not reproduce anything; the LM is unseeded for batch 1; `seed + i` must stay in range.
8. PT PLANNER TIME may exceed the render time (1.3). Surface `timing.plan_s`; cost accounting by execution seconds will see it. A 6-minute song with thinking on is the worst case (about 1,800 codes).
9. TURBO CLAMPS AND IGNORES. Steps above 8 are clamped by the handler (service_generate_request.py:52-57) and guidance is overridden (generate_music.py:312-318): report the values USED in `worker_notes`,
   not the ones requested. The booth's xl-turbo path must not promise guidance control.
10. SFT HAS TWO FORWARD PASSES PER STEP (CFG) and 50 steps: with the VRAM pre-flight (`xl_base` 1.0 GB per batch item per minute of audio, generate_music.py:139-154) a 4-take 6-minute sft request needs
    about 24 GB of activation headroom by upstream's own estimate; the VRAM guard may cut the batch silently (1.4 item 7). Contract caps the booth at batch 1 for v1, keep the capability but test 1 only.
11. TWO DiTs ON ONE CARD. Each handler duplicates VAE + text encoder; a second 10 GB DiT plus first on a 32 GB card is tight. Contract: release turbo before loading sft below 40 GB. The release is manual (1.2);
    the swap re-reads 20 GB, so expect about a minute of `model_load_s` on the first sft request after turbo and again when returning to turbo. Do not thrash: keep whichever model was used last.
12. `HF_MODULES_CACHE` / `trust_remote_code`: transformers copies the modeling files to `~/.cache/huggingface/modules/transformers_modules/<folder name>`; the folder names `acestep-v15-xl-turbo` and `acestep-v15-xl-sft`
    must stay distinct (they are). The cache dir must be writable; `acestep` must be importable (PYTHONPATH) at that moment or the load fails on `from acestep.models.common...`.
13. SDPA vs EAGER. If the sdpa load fails, upstream retries with eager silently (init_service_loader.py:164-185). Grep the init status string for "Attention: sdpa" at boot and fail loudly if not (INFERENCE: eager would also use more memory for the sliding-window layers).
14. DEVICE SELECTION. Everything defaults to `cuda:0`; the GPU config is global and computed from device 0's total memory. RunPod gives one GPU, fine. `MAX_CUDA_VRAM` must not be set in the endpoint env (it caps the allocator, gpu_config.py:24).
15. AUDIO SAVING PATH. Do not use `save_dir` (1.5). The tensor is already peak-normalised to -1 dBFS; keep that (it is upstream's default and keeps takes comparable); do not normalise twice.
16. SILENT TRUNCATIONS: tokens, not characters: the whole text prompt is cut at 256 tokens, lyrics at 2048 (conditioning_text.py:127-144). The booth trims the caption at 512 characters; a caption of 512 characters of dense terms can still exceed 256 tokens
    after the instruction and metas are added. Add a worker note if `len(tokenizer(...)) >= 256` is cheap to check (it is: the DiT handler's tokenizer is `h.text_tokenizer`).
17. `batch_size` default is 2 in `GenerationConfig` (inference.py:254) and `prepare_seeds`, the VRAM guard and the pre-flight all read it. Always pass it.
18. `fetch_model` in yue_handler leaves `huggingface_hub.constants.HF_HUB_OFFLINE = False` after a download (it resets only the environment variable, yue_handler.py:86-87). If you copy it, restore both.
    And import the constants module explicitly (`import huggingface_hub.constants as constants`): huggingface_hub 0.36.0 builds its package lazily (`_attach(..., submodules=[])`), so `huggingface_hub.constants` is not an attribute until
    someone has imported that submodule, and on a cold boot the download fallback runs before upstream is imported (a stage 4 review reproduced the AttributeError in the first version of ace_handler.py; whether yue_handler.py is exposed to the same order was not checked).
19. TORCHAUDIO/TORCHCODEC. `import torchaudio` happens at module level (audio_utils.py:21), so torchaudio must be installed; torchcodec is only needed for loading audio files (reference audio, cover) and is not installed in the
    recommended image. Do not add reference-audio features without adding torchcodec and checking that it finds the system FFmpeg libraries (INFERENCE).
20. sm_89 STRING in the arch assertion (2.3): asserting it would fail the build for a harmless reason.
21. Possible vLLM-only footgun if someone flips `ACE_LM_BACKEND=vllm`: `LLMHandler.initialize` returns success on the pt fallback too, so log `llm.llm_backend`; the vLLM engine takes its memory before the lazily loaded sft arrives.
22. Output WAV sizes (288 kB/s, 1.5 above) matter downstream: the fork contract's cover-source 20 MB check will reject an ACE WAV longer than about 70 s if it reads `wav_url`; `url` (the 320 kbps MP3, 40 kB/s, 20 MB = 8 minutes)
    is the safer cover source. This is the fork stage's problem; it is noted here because the worker decides the file sizes.

---------------------------------------------------------------------------------------------------------------------

## 5. yue_handler.py conventions to reuse (e)

All line numbers are `F:/KadeAI-work/ace/worker/yue_handler.py` unless stated. Files in the working tree are CRLF on this PC (`core.autocrlf=true`); the git index stores LF (`git ls-files --eol`: `i/lf w/crlf`),
so commit whatever endings your editor wrote, they are normalised to LF in the repo. New files may be LF.

Copy (small, standard library or lazy imports, no yue dependencies):
- `_filled(folder)` (49-53), `model_folder(pinned, repo, roots=None)` (56-73): pinned snapshot, else newest non-empty cached snapshot of `models--<org>--<name>` under `[pinned.parents[2], $HF_HUB_CACHE, /runpod-volume/huggingface-cache/hub]`,
  returns `(Path|None, how)`. Add a `required=` list of relative paths so a half-filled snapshot is not accepted (3.3 step 4).
- `fetch_model(repo, revision)` (76-87): `huggingface_hub.snapshot_download(repo, revision=..., cache_dir="/tmp/hf-hub")` with offline mode switched off for the call. Fix the constants reset (item 18).
- `resolve_models(download=True)` (90-106) and the `MODELS` tuple (44-45), `RESOLVED` flag (46): per model, pinned env var -> folder, writes the resolved path back into `os.environ[env]`, prints `"<name> <how>"` per model
  (boot log line `YuE2 models: ...`). ACE equivalent: three entries (`ACE_TURBO_DIR`/`ACE-Step/acestep-v15-xl-turbo`, `ACE_SFT_DIR`/`ACE-Step/acestep-v15-xl-sft`, `ACE_MAIN_DIR`/`ACE-Step/Ace-Step1.5`); sft is optional at boot (resolve it
  without downloading at boot, download lazily on first use). Probe mode calls `resolve_models()` and reports the three `how` values.
- `storage()` (830-834): `boto3.client('s3', endpoint_url=os.environ['AWS_ENDPOINT_URL'], region_name=os.environ.get('AWS_REGION', 'us-east-005'))` and `os.environ['AWS_BUCKET_NAME']`. The access key and secret come from
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` through boto3's own environment lookup (not read in code). Imports boto3 lazily inside the function: keep it so the unit tests run without boto3 (the CI step in `yue.yml:16` runs the
  tests on a bare `ubuntu-latest` Python with no torch).
- `gpu_name()` (801-806) and `gpu_peak(reset=False)` (809-820): lazy `import torch`, return `None` when unavailable; the booth prices by the `gpu` string, so it is `torch.cuda.get_device_name()`.
- `warm()` (1264-1273): `try: engine(); print(f'... ready at boot in {t:.1f}s') except Exception as error: print('... boot load skipped:', type(error).__name__)`; a failure falls back to loading on first use.
- `engine()` (770-779): module global `PIPE`, built once; ACE equivalent holds the turbo handler, the optional sft handler and the one `LLMHandler`.
- `__main__` (1432-1435): `import runpod; warm(); runpod.serverless.start({'handler': handler})`.
- Handler skeleton (1276-1302, 1424-1429): `start = time.monotonic()`, one `try`, validation helpers that raise `ValueError('<plain sentence>')`, `except ValueError as error: return {'error': str(error)}`,
  `except Exception as error: print('X failed:', type(error).__name__, ...); return {'error': 'X could not finish (<Type>). Your writing is saved.'}` (the YuE wording; the contract wants one plain sentence and no URLs or secrets).
  Validation style to imitate: `request_input` (185-208) and `sound_controls` (156-164; the nested `number()` helper rejects bools, NaN, inf and out-of-range values: `type(value) not in (int, float)` already excludes `bool`).
- `runpod.serverless.progress_update(job, 'text')` calls (1306, 1323) imported lazily inside the handler: `import runpod` is at 1298, not at module level (so tests do not need it).
- Output assembly (1383-1414): `prefix = f'<engine>/{uuid.uuid4()}'`; `client.upload_file(str(path), bucket, prefix + '/' + name, ExtraArgs={'ContentType': mime})`; presigned GET with
  `client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=604800)`; result keys `engine, key, wav_key, url, wav_url, duration_s (round(...,2) from sf.info(wav).duration), bytes (mp3.stat().st_size),
  processing_ms (int((time.monotonic()-start)*1000)), seed, truncated, timing {...}, gpu, memory {'render_peak_gib': ...}, features, worker_notes`. Take keys for ACE are `ace/<uuid4>/master.mp3` and `master.wav`.
  The whole body runs inside `with tempfile.TemporaryDirectory() as td:` (1301) so nothing is left on disk.
- `TAKE_KEY` regex (108) is the pattern the other YuE modes use to accept only well-formed keys: if ACE ever accepts a key from the caller (it does not in v1), copy that style.
- `FEATURES` tuple (113-115) reported in outputs; ACE: `("ace-xl-turbo", "ace-xl-sft", "batch", "instrumental", "probe")` per contract.
- ffmpeg: `from auk_handler import ffmpeg` (1300) is the shared helper: `subprocess.run(["ffmpeg","-nostdin","-y","-v","error",*map(str,args)], check=True, timeout=180, capture_output=True)` (auk_handler.py:61-63).
  Importing it drags in `auk_contract` (auk_handler.py:15). `Dockerfile.ace` is meant to be self-contained, so COPY the four-line `ffmpeg()` into `ace_handler.py` instead of importing `auk_handler`.
  Exact calls to repeat: `ffmpeg('-i', work/'audio.flac', '-c:a', 'pcm_s24le', wav)` then `ffmpeg('-i', wav, '-c:a', 'libmp3lame', '-b:a', '320k', mp3)` (1373-1374).
  ACE variant: write the 24-bit WAV directly with `soundfile` from the tensor, or write a float WAV and run the first command; then the second command unchanged.
- Test conventions (`test_yue_controls.py:1-30`, CI `yue.yml:14-16`): plain `unittest`, `import yue_handler` at the top (so the module must import with nothing but the standard library),
  `types.SimpleNamespace`/small fake classes for the pipeline, `unittest.mock.patch.object` for module globals, no GPU, no network. `test_yue_styles.py` is the one test that imports torch (CPU tensors), which is why the
  Docker build runs it inside the image and the workflow does not. Keep ACE tests to the first style (fake pipeline, fake S3) so the CI step works without torch.

Do NOT copy: the YuE-specific cover, score, ABC, sync and style code (lines 107-770 and 837-1262), `STYLE_KEY`, `apply_style` (the YuE adapter folding; ACE's `apply_adapter` is a separate seam, 1.7).

---------------------------------------------------------------------------------------------------------------------

## 6. Suggested shape for stage 2 (not binding)

```
ace_handler.py
  constants: MODELS (3 entries), FEATURES, VERSIONS, DEFAULTS per checkpoint (steps, guidance, shift, dcw)  [1.6]
  pure helpers: trim_caption(text, 512) -> (text, note); normalise_keyscale; request_input(raw) -> dict (ValueError sentences);
                take_seeds(seed, n); settings_for(model, overrides)
  resolve_models / model_folder / fetch_model / storage / gpu_name / gpu_peak            [copied, 5]
  build_checkpoints(root, snaps, upstream_dir) -> Path                                    [3.3]
  engine(model) -> handler; load_lm(); release_turbo()                                    [1.2, 1.3]
  render(job_input) -> takes via generate_music(save_dir=None) + soundfile/ffmpeg + upload
  apply_adapter(pipe, key): raise NotImplementedError                                     [1.7]
  warm(); handler(job); __main__
test_ace_handler.py   fake pipeline (callable returning a fake GenerationResult), fake S3 client, no torch
Dockerfile.ace        recipe 2.3 + COPY ace_handler.py test_ace_handler.py + build-time checks
.github/workflows/ace.yml
```
Probe mode (contract): `resolve_models()` + (optional, cheap, symlinks only) `build_checkpoints()` + `torch.cuda.is_available()` / device name; report
`models` as the three `how` strings plus the resolved snapshot folder NAMES (sha only, no paths with job data), `versions = {ace_step_commit: os.environ['ACE_STEP_COMMIT'],
torch: torch.__version__ (None if import fails), worker: os.environ.get('ACE_WORKER_VERSION')}`. No weights are read, so it stays cheap.
Boot self-checks worth logging (each is one line): `torch.cuda.is_available()` (a CPU-only boot would silently pick GPU tier 1, 240 s max, gpu_config.py:317-334, 753-757),
`_check_code_mismatch(...) == []` after building, the `initialize_service` status text (Attention / Dtype lines, init_service_setup.py:166-178) and `llm.llm_backend`.

Open decisions for stage 2 / the orchestrator (each is a product call, not a technical one): whether the LM may rewrite the caption (1.4 item 3); whether to expose `ACE_LM_BACKEND`; whether sft is resolved at boot
(read-only check) or only on first use; whether to cap `batch_size` at 1 for sft until the VRAM guard behaviour is observed. The smoke test should print, per model: init status string (attention, dtype), LM backend actually used, and the
`time_costs` keys, because those are the unknowns above.

## 7. Not verified here (honest list)

- Nothing was executed on a GPU or in Linux: all timings, tokens/s, image size and VRAM numbers above are estimates from file sizes and docs.
- HF-LISTING facts (file names, sizes, config.json content, shas) came through a summarising fetch tool; names are consistent with upstream's downloader code, but treat sizes and shas as "re-verify before pinning".
- Upstream issue #1299 (DCW and xl-sft) and issue numbers quoted by the research file could not be read (GitHub is off limits); the DCW handling above is from the pinned source and docs.
- Whether `torchaudio` 2.10 import needs torchcodec at import time is not established; the build-time `import acestep.inference` check will show it. If it does, add `torchcodec==0.10.0` (it needs the apt FFmpeg libraries).
- Whether `pytorch-wavelets` pulls `scipy`/`six` transitively is left to pip; the build-time import check covers it. (Settled in stage 3, see 8.2: it declares numpy, six and torch, imports `pkg_resources` and `pywt`, and imports no scipy.)

---------------------------------------------------------------------------------------------------------------------

## 8. Stage 3: the worker image, CI, endpoint settings, tests and the smoke-test watch list (Oct 10 2026)

What was and was not run: nothing in this section ran on Docker, Linux, a GPU or GitHub (none exist on this PC, and the rules forbid GitHub, RunPod and paid APIs).
What WAS done: read-only metadata reads of PyPI (JSON pages), the PyTorch wheel index page, the Hugging Face model API (JSON, shas and file lists) and one RunPod documentation page
(summarised by the fetch tool); a static validation of both new files (parse, quoting, COPY sources, pins, commit and sha agreement, 190 unit tests on Python 3.12 and 3.14); and local emulations
of parts of the build-time checks (8.3). Scratch scripts live in `F:/KadeAI-work/ace/scratch/` (`check_pins.py`, `closure.py`, `lockdiff.py`, `validate_ace_files.py`); they are not part of the repo.

### 8.1 What stage 3 added

- `Dockerfile.ace` (single stage, `python:3.12-slim-bookworm`, no weights, expected about 9 GB), `.github/workflows/ace.yml`, and five tests in `test_ace_handler.py` (class `TestLayoutPassesUpstreamsOwnChecks`):
  they build the checkpoints folder with the REAL pinned code folders and judge it with upstream's own `check_main_model_exists`, `check_vae_exists`, `check_model_exists`, `_check_code_mismatch` and
  `_sync_model_code_files` (loaded from the pinned `model_downloader.py` by file path), and test `upstream_models_dir()`. On this Windows PC the links are copies (no symlink rights); in the image build they are real symlinks.
  Mutation check: removing the code copy, the 2B placeholder file or the clone-folder lookup each makes these tests fail.
- No change to `ace_handler.py` and none to any YuE2 or AuK file.

### 8.2 The image recipe, the pins and how they were checked

Order of the Dockerfile (cheap-to-change last): apt (git, ffmpeg, libsndfile1) -> torch and torchaudio from the cu128 index -> arch assertion -> `git clone` of the ACE-Step repo, `git checkout ca1e85f...`, verify `HEAD`,
delete `.git` -> one `pip install` of the pinned set -> `pip check` -> `ENV PYTHONPATH=/opt/ace-step` -> `COPY ace_handler.py test_ace_handler.py /app/` -> build-time checks (8.3) -> runtime `ENV` -> `CMD python -u /app/ace_handler.py`.
`ACE_STEP_COMMIT` is an `ENV` set before the clone so the same value is used by the build and reported by the worker (`versions.ace_step_commit`); check 3 asserts it equals `ace_handler.UPSTREAM_COMMIT`.

| what | pin | how it was checked |
|---|---|---|
| torch, torchaudio | 2.10.0 / 2.10.0 from `https://download.pytorch.org/whl/cu128` | index page lists `torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl` and the torchaudio one; same torch as Dockerfile.yue's main env |
| transformers, huggingface_hub, tokenizers, safetensors, accelerate, diffusers, numpy, soundfile, loguru, einops, vector-quantize-pytorch, pytorch-wavelets, pywavelets, six, pyyaml, tqdm | 4.57.6, 0.36.0, 0.22.2, 0.7.0, 1.12.0, 0.37.1, 2.3.5, 0.13.1, 0.7.3, 0.8.2, 1.27.20, 1.3.0, 1.9.0, 1.17.0, 6.0.3, 4.67.1 | upstream `uv.lock` values. Each version exists on PyPI, is not yanked, requires a Python that includes 3.12, and ships a wheel installable on CPython 3.12 / glibc linux x86_64 (no compiler in the slim image). Every declared requirement of each package agrees with the other pins (script `check_pins.py`, 0 conflicts) |
| setuptools | 71.1.0 | uv.lock value and upstream's `setuptools<72`. WHY: `pytorch_wavelets/dtcwt/coeffs.py` does `from pkg_resources import resource_stream` at module level (read from the 1.3.0 wheel, in memory). The latest setuptools (84.0.0) wheel has NO `pkg_resources`; 71.1.0 and 80.9.0 do. Without it `from pytorch_wavelets import DWT1DForward` fails, and upstream's lazy loader (`dcw_loader.py:34-46`) only logs a warning and skips DCW. DCW is on by default for xl-turbo, so that would silently change the sound. Build check 2 proves the wavelets work |
| requests | 2.34.2 | runpod 1.12.0 requires `requests>=2.34.2`, `urllib3>=2.7.0`, `boto3>=1.43.66`, `cryptography>=50`, `fastapi[all]>=0.141.1` (PyPI metadata). Upstream's lock has requests 2.32.5, which would conflict with runpod; the range `requests>=2.32,<3` in the recon recipe would have resolved fine, but an exact pin has to follow runpod |
| runpod, boto3, botocore | 1.12.0, 1.43.100, 1.43.100 | newest runpod 1.x (2026-08-10); boto3 from 2026-09-22 (about 3 weeks old, still the 1.43 series); boto3 needs `botocore>=1.43.100,<1.44` and they release in lockstep. The YuE2 image installs `runpod>=1.7,<2` and `boto3>=1.35,<2` unpinned, so it very probably runs the same series (assumption: its build date is not recorded here) |
| einx, filelock, fsspec, hf-xet, regex, pillow, packaging, importlib-metadata | 0.3.0, 3.20.3, 2026.1.0, 1.2.0, 2026.1.15, 12.1.0, 26.0, 8.7.1 | transitive packages the model code leans on, pinned to the lock versions because pip would otherwise take much newer releases that upstream never ran with (einx 0.4.3 feeds the ResidualFSQ quantizer inside the DiT at run time, which no build-time import check exercises; filelock 4.1.0 is a new major). `lockdiff.py` showed that each of these lock versions satisfies every requirement of the rest of the set, and `check_pins.py` and `closure.py` re-ran with them: 0 conflicts |
| everything else (urllib3, certifi, fastapi, aiohttp, pydantic, cryptography, paramiko, cffi, ...) | floats to the newest that pip's resolver accepts | mostly runpod's web stack, which needs the newer releases anyway (lock versions of aiohttp, click, fastapi, psutil, pydantic-core, tomlkit, typing-extensions, urllib3, yarl conflict with runpod 1.12.0). The closure of the whole set (83 packages, torch and triton excluded) was walked on PyPI metadata (`closure.py`): every package in it has a CPython 3.12 linux x86_64 wheel. A real pip resolve was NOT run |

Not installed on purpose: upstream itself (gradio, fastapi UI, numba, lightning, tensorboard...), torchvision, torchcodec, torchao (contract: no INT8), peft, lycoris, scipy (no import of it on this path), flash-attn, nano-vllm, triton as a direct pin (torch brings its own).
`triton` arrives with torch on Linux, as 2.1 said (the lock lists it as a dependency of torch 2.10.0+cu128, and the cu128 index has a triton page).

Hub revisions: `GET https://huggingface.co/api/models/<repo>` on Oct 10 2026 returned exactly the shas that stage 1 copied from the summarising fetch tool: `ACE-Step/acestep-v15-xl-turbo` d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee,
`ACE-Step/acestep-v15-xl-sft` d06de46b4622f781cf07f4a013a67d591ca52819, `ACE-Step/Ace-Step1.5` 19671f406d603126926c1b7e2adc169acbcade22 (all public, not gated, last modified 2026-04-07, 2026-04-07, 2026-02-03). The file lists match 3.4 exactly
(11, 12 and 28 files). These are the `ACE_TURBO_DIR`, `ACE_SFT_DIR`, `ACE_MAIN_DIR` snapshot folder names in the Dockerfile; the folder name doubles as the download revision.

The arch assertion: `torch._C._cuda_getArchFlags()` must contain `sm_86` and `sm_120` and torch must be `2.10.0` with CUDA `12.8`. `sm_89` is deliberately not asserted by name: PyTorch wheels carry `sm_80/sm_86`-style cubins and an Ada card
(L40, L40S, RTX 6000 Ada, 4090) runs the `sm_86` one by CUDA binary compatibility inside the 8.x family, so a literal `sm_89` assertion would fail a correct wheel. The contract text names sm_89; this is the one place the file departs from it,
and the reason is in the Dockerfile comment. Not verified here: the actual flag list of the 2.10.0+cu128 wheel (the build log prints it). Dockerfile.yue runs the same torch 2.10.0 cu128 in production on A40-class cards and prints, but does not assert, its flags.

Image size (estimate, not measured): torch + CUDA libraries about 7 to 8 GB, ffmpeg and git about 0.4 GB, other Python packages about 0.8 GB, upstream source 21 MB, base 0.15 GB, so about 9 GB. Build check 6 prints `du -sh` of site-packages, `/opt/ace-step` and `/app` in the log.

### 8.3 Build-time checks in `Dockerfile.ace` (all CPU, no weights, no network except the build's own installs)

The numbers match the numbered comments in the Dockerfile. Before them: the arch assertion (right after torch is installed) and `pip check` (after the pinned set).
1. Upstream imports: `torch, torchaudio, acestep.inference, acestep.handler, acestep.llm_inference, acestep.model_downloader`, the names `GenerationParams, GenerationConfig, generate_music, AceStepHandler, LLMHandler`, AND both DiT modeling files
   (`acestep.models.xl_turbo.modeling_acestep_v15_xl_turbo`, `acestep.models.xl_sft.modeling_acestep_v15_xl_base`), which import transformers internals, einops and vector_quantize_pytorch the way the real load will. `torch.cuda.is_available()` is printed (False in the build).
2. DCW: `WAVELET_CACHE.get(cpu, float32, 'haar')` is not None and `dcw_double(...)` on random CPU latents changes the tensor (the correction really runs).
3. Worker module: `import ace_handler`; `ACE_STEP_COMMIT` equals `UPSTREAM_COMMIT`; a `mode: probe` job returns `engine: ace` and torch `2.10.0...`; a job with a malformed `lora_key` (`k`) returns the refusal sentence (`ADAPTER_SENTENCE`, since stage 6 `That style is not available right now.`; the check compares with the constant, so it is unchanged and a test pins it).
4. Real encoding: the ffmpeg encoder list must name `libmp3lame` and `pcm_s24le`; then `encode_take` on one second of noise (real soundfile, libsndfile, ffmpeg) gives a 24-bit, 48 kHz, 2-channel WAV and a non-trivial MP3.
5. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest test_ace_handler.py` (the model-folder variables are not set yet, so no test can find or fetch a model). In the image the upstream clone exists, so `TestPinnedUpstream` and the five layout tests run too
   (and `test_weights_are_links_into_the_snapshot_and_code_is_copied`, the real-symlink test that is skipped on this PC).
6. No `*.safetensors`, `*.ckpt`, `*.gguf` or `pytorch_model*.bin` anywhere on the root filesystem, then the size report.
Local emulation done: check 3's exact code ran against `ace_handler` with a stub torch (passed); the WAV half of check 4 ran with real numpy and soundfile (FLOAT, 48000, 2 channels, 1.0 s); the unit tests passed with a stub torch and loguru on the path,
with `HF_HUB_OFFLINE=1`, and on Python 3.12 where numpy, soundfile and huggingface_hub are really installed. The arch assertion, `pip check`, checks 1 and 2, the ffmpeg half of 4 and check 6 could not run here.
If a check fails in CI, its line in the log says which assumption broke; none of them needs a GPU, so a fix can be iterated without RunPod spend.

### 8.4 The workflow `.github/workflows/ace.yml` (name `build-ace`)

- Triggers: `push` to branch `ace-xl` only, and only when `Dockerfile.ace`, `ace_handler.py`, `test_ace_handler.py` or `.github/workflows/ace.yml` changed (`ACE_NOTES.md` is deliberately not listed); `workflow_dispatch`.
  GitHub documents that `workflow_dispatch` only fires for workflow files that exist on the default branch (not tested here): on `ace-xl` alone, rely on the push trigger (a push that touches a listed file). It cannot trigger on `main`, `kade` or any YuE branch.
- Steps, same actions and versions as `yue.yml`: checkout, `python -m unittest test_ace_handler.py` (380 tests after stage 6; about 61 are skipped on the runner: 34 need the upstream clone, which only the image build has, 24 need torch, safetensors or PEFT, one needs `huggingface_hub` and two need `loguru`; stage 3 to 5 had 234 tests and 29 skipped), free disk (the identical `rm -rf` line), buildx, ghcr login, `docker/build-push-action@v6`
  with `file: Dockerfile.ace`, `tags: ghcr.io/kademurdock/scenema-audio-runpod:ace-${{ github.sha }}`, `cache-to: type=inline`. No `cache-from` (the only existing cache candidates are YuE tags, which would feed a different image's layers into this one).
  Consequence: every build is cold, expect roughly 15 to 30 minutes (estimate). `permissions: contents read, packages write`, as in yue.yml; the package already exists with `yue2-` tags, so the push adds `ace-` tags to the same package (not tested).
- The RunPod template should point at the full `ace-<40-character sha>` tag, never at a moving tag.

### 8.5 What the RunPod endpoint needs

Names only; never put values in a file. Copy them from the YuE2 template's environment.

| variable | needed | note |
|---|---|---|
| `AWS_ENDPOINT_URL`, `AWS_BUCKET_NAME` | yes | B2's S3 endpoint and the bucket the booth reads (read by `storage()` exactly like `yue_handler.storage()`) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | yes | read by boto3 itself; the key must be allowed to write under `ace/` and create presigned reads (7 days = 604800 s) |
| `AWS_REGION` | optional | default `us-east-005`, same as YuE2 |
| `MODEL_NAME` / `MODEL_NAMES` | see below | RunPod model cache: exactly the three repos `ACE-Step/acestep-v15-xl-turbo`, `ACE-Step/acestep-v15-xl-sft`, `ACE-Step/Ace-Step1.5` |
| `ACE_TURBO_DIR`, `ACE_SFT_DIR`, `ACE_MAIN_DIR` | no | baked into the image (8.2); set them only to point at other snapshot folders |
| `ACE_LM_BACKEND` | no | `pt` (default) or `vllm` (an untested knob, see 4.21) |
| `ACE_PLANNER_REWRITES_CAPTION` | no | `1` lets the planner rewrite the caption; default keeps the booth's words |
| `ACE_CHECKPOINTS_DIR`, `ACE_STEP_DIR`, `ACE_WORKER_VERSION` | no | `/tmp/ace-checkpoints`, `/opt/ace-step`, `ace-worker-1` by default |
| `ACE_LORA_DIR` | no | where trained styles are cached on the worker, `/tmp/ace-lora` by default (a style is 335 MB; section 9). The endpoint's B2 key must also be allowed to READ the `ace-loras/` prefix |
| `ACESTEP_*`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_MODULES_CACHE`, `TOKENIZERS_PARALLELISM` | no | the handler sets defaults before upstream is imported (`configure_environment`); the image sets the two offline flags |
| `MAX_CUDA_VRAM` | MUST NOT be set | it caps the allocator inside upstream (gpu_config.py:24) |
| `HF_TOKEN` | not needed | all three repos are public and ungated |

`MODEL_NAME` / `MODEL_NAMES`: the contract uses these names. The RunPod documentation page on cached models (fetched Oct 10 2026, through a summarising tool) describes ONE cached model per endpoint, set in the endpoint's Model field, mentions `MODEL_NAME`
only for official worker images, and documents no list form, no separator and no way to pick a revision. Yet the YuE2 endpoint evidently gets four repos from the cache (yue_handler.py:39-43 and Dockerfile.yue:17-30). So: configure the ACE endpoint
the same way YuE2's was configured (the orchestrator can look; this stage did not call RunPod) and list exactly the three repos above. The worker does not depend on it: it works with 3, 2, 1 or 0 repos cached.
What changes is the cold start (8.6). If only one repo can be cached, give the slot to `ACE-Step/acestep-v15-xl-turbo` (about 20 GB, needed at boot); the main repo then downloads at first boot (about 5.3 GB with the unused 2B model skipped) and xl-sft at its first use (about 20 GB).
Other endpoint settings, all copies of the YuE2 choices unless stated (research file and NEXT_SESSION_PLAN): FlashBoot on, min workers 0, max 1 during the test spend, explicit GPU list A40, RTX A6000, RTX 6000 Ada, L40, L40S, RTX 5090, no network volume.
Different from YuE2: (a) set the execution timeout explicitly and high (RunPod's default was not checked); the worker raises upstream's own 600 s render limit to 3600 s (`ACESTEP_GENERATION_TIMEOUT`), so the endpoint limit is the real one: use 1800 s or more for xl-sft;
(b) container disk: with the cache complete, a few GB of `/tmp` is enough; if any repo may have to be downloaded, allow about 50 GB (20 + 5.3 + 20 + outputs) because the fallback writes to `/tmp/hf-hub` (RunPod's disk accounting not verified);
(c) a first boot reads about 25 GB (8.8), so give the first job a long wait and do not judge latency from it.

### 8.6 How the model-cache folders are resolved (`resolve_models`, `model_folder`, `fetch_model` in ace_handler.py)

Per repo key (`turbo`, `sft`, `main`), in this order, using a per-repo completeness test (a DiT snapshot needs `config.json`, `silence_latent.pt` and every shard named in `model.safetensors.index.json`; the main snapshot needs `vae/{config.json, diffusion_pytorch_model.safetensors}`,
`Qwen3-Embedding-0.6B/{config.json, model.safetensors}` and `acestep-5Hz-lm-1.7B/{config.json, model.safetensors}`; a half-filled snapshot is never accepted, unlike yue's any-file test):
1. the pinned folder from the environment variable (`pinned`);
2. else the newest complete snapshot of `models--<org>--<name>` by modification time, searched in this order: the pinned path's cache root (`parents[2]`), `$HF_HUB_CACHE` (the image sets `/runpod-volume/huggingface-cache/hub`), `/runpod-volume/huggingface-cache/hub`, `/tmp/hf-hub` (`cached <first 12 of the sha>`);
3. else, only when downloads are allowed, `huggingface_hub.snapshot_download(repo, revision=<the pinned folder's name = its sha>, cache_dir=/tmp/hf-hub, ignore_patterns=...)` with offline mode switched off for that call only and restored (flag and library constant) after it (`downloaded`); the DiT repos skip `*.py`, `*.md`, `.gitattributes`, the main repo skips the 2B folder; otherwise `missing`.
The winner is written back to its environment variable and one line `ACE-Step models: <repo> <how>; ...` is printed. At boot: turbo and main resolve with downloads allowed, sft is only looked for (no download); sft, and main if needed, resolve again with downloads allowed on the first xl-sft request.
`mode: probe` resolves without downloading and reports, per key, `{repo, source (pinned | cached <sha12> | downloaded | missing), revision (snapshot folder name)}`, plus `loaded` flags for turbo, sft and the planner, `versions` and `gpu`. It is the cheap boot check.
Then `ensure_checkpoints` builds `/tmp/ace-checkpoints` once per process (recipe 3.3 above): symlinks for the VAE, text encoder, planner and DiT weights/json, real folders for the DiT code copies and the 2B placeholder. Nothing is written into the cache.
What a stale pin means: if a repo's `main` moves, RunPod's cache holds a new sha, the pinned folder is missing and step 2 picks the newest complete snapshot. The code comes from the pinned upstream either way, but weights or `config.json` (`is_turbo`) could differ, so look at `source` and `revision` in the probe.

### 8.7 How to run the tests

- This PC (Windows, no torch, no network): from `F:/KadeAI-work/ace/worker`, `python -I -B -m unittest discover -s . -p "test_ace_handler.py"` (Python 3.14 or the 3.12 install) -> after stage 6 `Ran 380 tests ... OK (skipped=28)` on 3.14 (the real-symlink test, the real-`huggingface_hub` test, the two real-`loguru` tests and the 24 tests that need torch, safetensors or PEFT); on the 3.12 scratch venv that has the pinned `huggingface_hub` 0.36.0, `loguru` 0.7.3, CPU torch 2.10 and safetensors (`F:/KadeAI-work/ace/scratch/review/venv312`) it is `OK (skipped=2)` (the symlink test and the PEFT test); on `F:/KadeAI-work/ace/scratch/lora_pod/venv` (the same plus PEFT 0.18.1) `OK (skipped=1)`. Stages 3 and 4 had 207 tests, stage 5 234. `python -I -m unittest test_ace_handler.py` does NOT work (isolated mode cannot find the module); without `-I` the plain form works.
  Plain `python -m unittest` over the whole folder also runs the YuE tests, which have 6 problems on this PC that predate this work (torch missing; hash checks against CRLF files).
- CI runner (Linux, no clone, no torch): `python -m unittest test_ace_handler.py` from the repo root -> 380 tests, about 61 skipped (34 need the upstream source, 24 need torch, safetensors or PEFT, 1 needs `huggingface_hub`, 2 need `loguru`), 0 failures expected. Not run on Linux: the number is the Windows 3.14 run of a copy with the clone and the adapter path pointed nowhere (62 skipped) minus the symlink test, which does run on Linux; treat 61 as an estimate (stage 3 to 5 had 234 tests and 29 skipped).
- Image build (Linux, torch, clone at `/opt/ace-step`): check 5 above -> 380 tests, only the tests that need PEFT (one) and the trained adapter file (one) skipped; the rest, including the real-torch fold tests, run (the image has the clone, torch, safetensors, `huggingface_hub`, `loguru` and real symlinks). On a pod: `cd /app && python -m unittest test_ace_handler.py`.
- Against another clone of the pinned source: `ACE_UPSTREAM_DIR=<folder>` (the tests look there first, then `/opt/ace-step`, then the F: clone).
- By hand on a pod, no GPU needed for the first line: `python -c "import ace_handler as a; print(a.handler({'input': {'mode': 'probe'}}))"`.

### 8.8 Things to watch in the smoke test

Capture for every job the whole answer (`timing`, `memory`, `gpu`, `worker_notes`, `plan`, `versions`) and the container log. Boot lines the worker prints: `ACE-Step models: ...`, `ACE-Step xl-turbo loaded in <s>`, `ACE-Step xl-turbo: Attention: sdpa; Dtype: ...`,
`ACE-Step planner loaded in <s>, backend pt`, `ACE-Step ready at boot in <s>` (or `boot load skipped: <ExceptionType>`). Estimates below are from file sizes and upstream's own constants, not measurements.

1. Memory on 32 GB cards (RTX 5090). Resident after boot is about 15 GB (DiT 9 to 10, planner 3.4 to 3.7, text encoder 1.2, VAE 0.33; upstream's table gpu_config.py:148-165). `BOTH_MODELS_GIB = 40`: a 5090 reports about 31 to 32 GiB, so turbo is released before xl-sft loads (and the planner stays);
   an A40/L40S (about 45 GiB) keeps both (about 26 GB resident). Upstream's own activation estimate is only a pre-flight: 0.5 GB per take per minute for xl-turbo and 1.0 for xl-sft plus 0.5 GB margin (generate_music.py:106-179, gpu_config.py:173-181; it assumes flash attention and may be low for the sdpa path). Read `memory.render_peak_gib`
   for a 30 s, 180 s and 360 s song on both models on the 5090; a "free VRAM" refusal comes back as the "too long for the memory" sentence. The cutoff to find is the longest xl-sft song the 5090 renders. `ACESTEP_SKIP_VRAM_PREFLIGHT=1` exists but hides a real shortage. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a possible fix for fragmentation with changing song lengths; it is not set, because it is untested.
2. Lazy xl-sft load. First xl-sft request = download if uncached, read 20 GB of fp32 shards, cast to bf16, `model_load_s` probably 1 to 3 minutes; the booth gets "Loading the music model" through progress. On a small card turbo is released first, so the next xl-turbo request reloads 20 GB: send turbo, sft, turbo in a row and read `model_load_s` each time.
   Alternating requests on a 32 GB card would thrash; if that happens the fork side may need to route by card or batch by model.
3. Symlinked cache layout. After boot, `ls -laL /tmp/ace-checkpoints/acestep-v15-xl-turbo` should show real `.py` files and linked weights; the log line `upstream code differs from the checkpoint copy` must NOT appear; the shared snapshot's `.py` files must be untouched.
   Slow first reads from a network-backed `/runpod-volume` (mmap of 5 GB shards) are normal; a load that dies with "no such file" means a dangling link (the cache was evicted or the sha moved: run the probe). `HF_MODULES_CACHE` (`/tmp/hf-modules`) must contain `transformers_modules/acestep_hyphen_v15_hyphen_xl_hyphen_turbo/...` after the load (transformers rewrites `-` to `_hyphen_` in the folder name: `_sanitize_module_name`, dynamic_module_utils.py:51-68; the sft one is `acestep_hyphen_v15_hyphen_xl_hyphen_sft`), so grep for that spelling, not for `acestep-v15-xl-turbo`.
4. Offline flags. With `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` nothing should resolve a hub id: watch for retries, timeouts or "couldn't connect to huggingface.co" lines, and for upstream's google.com probe (it must not run: `check_main_model_exists` has to be true, which the layout tests prove for fake files).
   The only legitimate network use is the download fallback (`downloaded` in the probe). `fetch_model` restores both the variable and the library constant afterwards; after a fallback download, run a second job and check nothing else goes online.
5. First-request latency. Boot reads about 25 GB (turbo 19.95, planner 3.7, text encoder 1.2, VAE 0.34) and the planner tokenizer alone "may take 80~90s" per upstream's comment (llm_inference.py:609-611). Record `ready at boot in` for a cold worker, for a cached restart, and whether FlashBoot keeps the loaded models
   (YuE2's first job on a new image waited 360 s). Compare job 1 with job 2 on the same worker: the first render also pays CUDA/cuBLAS/SDPA initialisation. No warm-up render runs at boot; if job 1 is more than a few seconds slower than job 2 at equal length, a 10 s xl-turbo warm-up inside `warm()` is the cheap fix.
6. Concurrency. `runpod.serverless.start({'handler': handler})` runs one job per worker at a time and the worker keeps no concurrency modifier; the handler, `STATE` and upstream's handlers are not thread-safe, so do not add one. Two queued jobs on one worker run back to back; `workersMax` is the only parallelism, and every extra worker pays the cold start.
7. torch.compile and warm-up. `compile_model=False` and nothing is compiled or autotuned; Triton is present (torch brings it) but unused on this path, so no compile minutes and no sm_120 Triton risk are expected. Confirm the log has no "compiling" lines. Optional experiments after the first good run: `ACE_LM_BACKEND=vllm` (needs `pip install --no-deps /opt/ace-step/acestep/third_parts/nano-vllm xxhash` on the pod; no flash-attn on 3.12, so it would run eager; it reserves VRAM from free memory) and a boot warm-up render.
8. fp32 checkpoints loaded as bf16. The shards are about 20 GB because they are fp32; the load casts to bf16 (about 10 GB on the card) and `dtype` in config.json says bf16 as well. Watch host RAM during the load (the cast happens on the CPU side: plausibly 10 to 20 GB peak; a container killed at boot with no Python error is the signature of the host-RAM limit, not VRAM),
   the `Dtype:` line (expect `torch.bfloat16`), and listen to xl-turbo vs xl-sft at the same seed for obvious quality differences that bf16 could cause; upstream runs both in bf16, so a difference from their published samples is not expected.
   The sm_86 cards (A40, A6000) and the 5090 all take the bf16 branch (compute capability 8 or higher, ACE_NOTES 1.2).
9. Long songs. The booth sends 30 to 360 s. Planner time grows with length (5 codes per second: 1,800 codes at 6 minutes; under the `pt` backend an estimated 25 to 40 tokens per second, so roughly 45 to 75 s plus metadata, reported as `timing.plan_s`), xl-sft adds 50 steps with two passes each, and the 24-bit WAV is 288,000 bytes per second (104 MB at 6 minutes, over the fork's 20 MB cover-source limit past 70 s: the MP3 `url` is the safe cover source).
   Try 360 s on xl-turbo and xl-sft with thinking on and off and note total execution seconds (the booth prices by `gpu` and `executionTime`), the endpoint timeout, `/tmp` use for a batch of takes, and the upload time. Planner output was seeded for batch 1 via the global generators; check that the same seed gives the same plan twice.
10. Also look at: the `Attention:` line (eager fallback means a slower and larger load), `worker_notes` (steps used, planner used or skipped, trimmed caption), whether the planner rewrote the key/tempo the caller did not set (`plan`), and that no caption or lyrics text appears in the container log (loguru is cut to WARNING with `backtrace=False, diagnose=False`; the worker prints only exception types, plus the scrubbed first line of a failed model or planner load).
11. DCW strengths. The default xl-turbo job (planner on) now sends `dcw_scaler=0.02, dcw_high_scaler=0.06`, upstream's Gradio Think pair; with the planner off or unavailable it sends 0.05/0.02. Nobody has listened to either on this worker, and there is no runtime switch for the pair. If the first xl-turbo samples sound phasey or harsh in the highs, swap the pair in `DCW_STRENGTHS` (one line, plus the tests that pin it to upstream's values) and rebuild before blaming the model.

### 8.9 Not verified in stage 3 (honest list)

- No Docker build, no Linux, no GPU: every build-time check except the ones emulated in 8.3 is unexercised, and a CI failure on first run is possible. The most likely suspects, in order: the arch-flag string format or the sm_120 token (the arch assertion), an import of a transformers internal in the XL modeling files (check 1), `pip check` complaints from floating transitive packages, the ffmpeg encoders (check 4), a unit test that behaves differently with real torch and loguru on the path (emulated with stubs only).
- The pin set was checked from package metadata, not by a real pip resolve. Floating transitive packages can change between today and the first build; if `pip check` or an import fails after such a change, freeze the closure listed by `closure.py`.
- The RunPod model-cache semantics (several repos on one endpoint, revision choice) come from one summarised documentation page plus the YuE2 evidence, not from the endpoint.
- GitHub behaviour (path filters on the first push of a new branch, `workflow_dispatch` needing the file on the default branch, `GITHUB_TOKEN` write access to the existing package) is from general knowledge, not tested.
- Image size and all timings and memory figures are estimates.

### 8.10 Stage 4: fixes after the two reviews

Two independent reviews returned nine findings (one of them reported by both); each was checked against the code before anything was changed. Changed files: `ace_handler.py`, `test_ace_handler.py`, this file. No Dockerfile or workflow change.

| finding | verdict | what was done |
|---|---|---|
| `fetch_model` raised AttributeError on a cold boot, so the "download once" fallback never worked when it was the first code to touch `huggingface_hub` | real; reproduced on the pinned hub 0.36.0 (`import huggingface_hub; huggingface_hub.constants` -> AttributeError) | explicit `import huggingface_hub.constants as constants`. Tests: the stand-in package in `TestFetchModel` now behaves like the real lazy one (constants not an attribute), plus a subprocess test against the REAL installed `huggingface_hub` (only `snapshot_download` replaced), skipped where the library is absent. Also re-ran the real boot order by hand (runpod and torch imported first, hub not yet): `resolve_models(download=True)` -> `downloaded`, env and library constant restored |
| loguru's `backtrace`/`diagnose` defaults print caption and lyrics on any upstream `logger.exception` | real; reproduced by the review and again here: the reviewer's script with the real upstream `generate_music` gave 4 hits for each marker before the change and 0 after | `logger.add(sys.stderr, level='WARNING', backtrace=False, diagnose=False)`. Tests: the call arguments; a subprocess test with the REAL loguru that fails a render and checks the secrets are absent, and a control run with loguru's plain WARNING sink that must show them (so the test can see a leak); a static test that no module under the pinned `acestep/` calls `logger.add/remove/configure` (only `cli.py` does, and the worker never imports it) |
| DCW strengths left at the dataclass pair 0.05/0.02 while the planner is on | judgement call, changed: upstream is inconsistent (the Gradio UI uses 0.02/0.06 with Think, `GenerationParams` and the HTTP route keep one pair, and its own comment says the single pair "sits near" the Think optimum) | `DCW_STRENGTHS = {True: (0.02, 0.06), False: (0.05, 0.02)}`, chosen by whether the planner is actually used; sent only when DCW is on (xl-turbo). Tests pin the choice and compare the constants with upstream's `dcw_defaults.py`. NOT listened to: smoke-test item 8.8.11 |
| module-cache folder name in the smoke-test list | real (checked in transformers 4.57.6: `_sanitize_module_name`) | the note now says `acestep_hyphen_v15_hyphen_xl_hyphen_turbo` |
| a list or dict `model` raised TypeError instead of the plain sentence | real | `isinstance(model, str)` first. Test: every request field fed 20 odd JSON-like values must give `RequestError` or a valid request, never another exception (only `model` failed before) |
| no test pinned which link goes with which file | real (mutant survived) | the tests now compare each link with the key it was signed for, for take 0, every take, and the set of uploaded keys |
| peak reset, `model_load_s`, summed `encode_s`/`upload_s`, `require_gpu` in `engine` were unpinned | real (mutants survived) | clock-driven tests for the first three and the reset order (load, planner, reset, render, read); `engine` must refuse before resolving or loading anything when there is no card |
| load-failure lines print upstream's status text, not just a type | a literal deviation from the contract; kept on purpose in a safer form | `scrub()` now also removes web links and Hugging Face tokens (and takes a `limit`); the model and planner load-failure lines use it. Reason for keeping the text: it is the only clue when a first boot fails on the cache layout, and no caption or lyrics exist at load time |

Mutation check (`F:/KadeAI-work/ace/scratch/stage4/mutate4.py`, run on the 3.12 venv so the real-library tests take part): 22 breaks of the new behaviour (old `fetch_model`, old loguru call with and without `diagnose`, no or wrong DCW pair, DCW pair sent for sft, no `isinstance`, three kinds of swapped links, peak never reset or reset before the load, zeroed or unsummed timings, missing `require_gpu`, raw status printing for the model and the planner, links or tokens kept by `scrub`, ignored `limit`): all 22 made at least one test fail.
Not verified: nothing in this stage ran on Linux, in Docker or on a GPU. The download fallback was exercised with a replaced `snapshot_download`, not against Hugging Face. Residual log risk: a few upstream WARNING/ERROR lines can carry text derived from the planner's own output (`constrained_logits_processor.py` warnings are debug-gated; `llm_inference.py:2501` prints the exception message), which could repeat user words if the planner fails on them; the review found no such line on the render path.

### 8.11 Stage 5: the Creative variation (weirdness) and Prompt guidance dials

The booth gets two dials like YuE2's. This file is the worker side; the mapping from dial to number is the booth's. Weirdness 0-100 becomes `lm_temperature` 0.40 at 0, 0.85 at 50 (upstream's default) and 1.30 at 100. Prompt guidance 1-3 becomes `guidance_scale = 7 + 3 x (value - 1)` for xl-sft only, so 7, 10 and 13. At the defaults the booth sends neither field.

**What the worker takes and answers.** New optional request field `lm_temperature`: a number from 0 to 2 (not a bool, not NaN or infinity). Anything else returns `Creative variation must be a number from 0 to 2.` The value goes to `GenerationParams.lm_temperature` only when it was set AND the planner really runs (`planner_temperature()`); otherwise the field is left out and upstream keeps its own 0.85. A request without the field is unchanged in everything it sends and returns (a test compares the two requests field by field). The answer reports what was used: `controls.lm_temperature` (a float) exists only when the planner ran with the caller's value, and `worker_notes` says `The planner ran at temperature 1.3.` When the field was sent but the planner did not run (thinking false, or the planner could not load) it is not sent, `controls` has no `lm_temperature`, and the note is `The creative variation setting was not used because the planner did not run.` An older worker image ignores unknown fields, so the booth can also tell "not applied" from the missing key. `guidance_scale` is unchanged: number 1 to 15 for both models, so 13 is accepted. Turbo still ignores it (controls shows 1.0, note `The fast model ignores the guidance setting.`); xl-sft uses it whole (7, 10 and 13 reach `GenerationParams.guidance_scale` as 7.0, 10.0, 13.0, with DCW still off).

**What `lm_temperature` does upstream (READ in the pinned source, ca1e85f).**
1. It is the field `lm_temperature: float = 0.85` of `GenerationParams` (inference.py:204; docstring "0.0-2.0" at line 101; the Gradio slider is 0.0 to 2.0, generation_advanced_primary_controls.py:75-84). Upstream does not validate the range; the worker does.
2. One consumer: `generate_music` hands it to `llm_handler.generate_with_stop_condition(temperature=params.lm_temperature, ...)` (inference.py:726-745). The DiT never sees it. Every other grep hit is the Gradio UI, the HTTP API or a test.
3. The planner runs when `use_lm = (thinking or need_lm_for_cot) and llm_handler is not None and llm_handler.llm_initialized and not skip_lm` (inference.py:653-655). The worker passes the planner with `thinking=True` and `use_cot_metas=True`, and with thinking off passes `llm=None` and all three cot flags off, so the planner does not run at all and the temperature would do nothing. That is why it is not sent then.
4. With the planner on it affects BOTH passes, so it changes the 5 Hz audio codes and not only the metadata (the blueprint). Pass 1 (metadata CoT: tempo, key, time signature, duration; llm_inference.py:1367-1404) and pass 2 (the audio codes the DiT follows; 1604-1626) both get `"temperature": temperature`. On the pt backend both end in `_sample_tokens` (llm_inference.py:438-450: logits divided by T, softmax, one random draw; T of 0 or less would be argmax): pass 1 has CFG scale 1.0 and goes through `_generate_with_constrained_decoding` (sampling at 2585), pass 2 has `lm_cfg_scale` 2.0 and goes through `_generate_with_cfg_custom` (2753). The vllm backend passes the same number to `SamplingParams` (939-948). Upstream's separate per-phase temperatures (`metadata_temperature`, `codes_temperature`) stay None because `generate_music` never sets them (llm_inference.py:344-350, constrained_logits_processor.py:1666-1695), so they cannot override it; a test pins that the worker never sets them.
5. Pass 1 is skipped only when bpm, key, time signature AND duration are all supplied (`has_all_metas`, llm_inference.py:1245-1251). The booth always sends a length and a tempo only when the direction text has one; as ACE_CONTRACT describes it, it has no key or time-signature setting, so pass 1 normally runs. Pass 2 always runs while thinking is on (no codes are supplied), so the dial always reaches the audio codes.
6. It is milder than the numbers suggest: top-p 0.9 and constrained decoding are applied to the logits BEFORE the temperature divides them (llm_inference.py:2581-2585 and 2742-2753; `lm_top_p` 0.9 is passed on, inference.py:669-670). A hot planner only re-weights tokens already inside the 90 percent nucleus, and the format rules still hold, so 1.30 is a moderate step toward variety and 0.40 is a firmer, safer plan, not chaos or a fixed answer. Upstream's own guidance is 0.7 to 0.85 for consistency and 0.9 to 1.1 for diversity (INFERENCE.md:1027-1035).
7. Seeds: the worker seeds Python and torch from take 0's seed before the render, so the same seed, temperature and card repeat the plan (not promised across cards). The same seed with another temperature can give another plan, which is how the dial is meant to be used. Planner time depends on the length (5 codes per second), not on the temperature.

**Guidance, for the record.** Upstream's slider is 1.0 to 15.0 (generation_advanced_dit_controls.py:37-41); the xl-sft sampler has no upper cap and turns CFG on for any value above 1.0 (modeling_acestep_v15_xl_base.py:2010). So 13 is valid, and costs the same as 7 (a doubled batch per step either way; only a value of 1.0 would skip it, and the booth's lowest is 7). Upstream's typical range is 5 to 9 (INFERENCE.md:367), so 13 should sound more literal and less natural; that matches the booth's hint, but nobody has listened to it. xl-turbo: upstream forces 1.0 (generate_music.py:312-318), and the worker sets 1.0 before it gets there.

**Also fixed.** `_real()` called `math.isfinite()` on whole numbers, which raises OverflowError for an integer above about 1e308, so a 400-digit `duration`, `guidance_scale` or `shift` came back as "could not finish (OverflowError)". Whole numbers now pass through and fail the range check with the usual sentence. The new field uses the same check.

**Tests.** 207 -> 234 (27 new): request validation (absent, null, 0 and 2 edges, 22 refusals, 13 on both models, huge numbers), the pass-through with a fake pipeline (set and absent, both models, zero is not absent, planner off or missing, nothing else in the request changes, controls and notes, both dials together), guidance 13 on sft and the turbo override end to end, and three checks against the pinned source (the field and its 0.85 default, where the planner reads it, the guidance range and turbo override). A mutation check on a copy (`F:/KadeAI-work/ace/scratch/stage5/mutate5.py`, 18 breaks: zero treated as absent, field not sent, sent when absent, sent without the planner, either edge excluded, range too wide, bool or NaN accepted, overflow back, controls or notes missing or wrong, not a float, guidance cap lowered, turbo guidance honoured, request key dropped) killed all 18.
Not verified: nothing ran on a GPU or in Docker, and no one has listened to the dial. How far 0.40 and 1.30 move the music is untested; they are the booth's choice of end points inside the 0 to 2 range.

---------------------------------------------------------------------------------------------------------------------

## 9. Stage 6: trained styles (a LoRA adapter folded into the xl-sft decoder)

The Soul adapter's samples were listened to and approved, with the word to turn the style on. This is the worker side (`ace_handler.py`, `test_ace_handler.py`); the contract is `F:/KadeAI-work/ace/ACE_STYLE_CONTRACT.md`. The seam reserved in 1.7 (`apply_adapter`) is now real. What did and did not run: everything below ran on CPU only (Python 3.14, the 3.12 scratch venv, and a 3.12 venv that has PEFT 0.18.1); nothing ran on a GPU, in Docker or against B2 or RunPod. Scratch scripts are in `F:/KadeAI-work/ace/scratch/stage6/` (not part of the repo): `upstream_unload_probe.py`, `bf16_fold_probe.py`, `fold_vs_peft_probe.py`, `inplace_probe.py`, `mutate6.py`.

### 9.1 What a styled request does

Two new optional request fields. `lora_key` is a B2 prefix such as `ace-loras/soul-real-v1`: it must match `^ace-loras/[A-Za-z0-9._-]{1,60}$` (checked with `fullmatch`, so a trailing newline fails), may not contain `..` and the name may not start with a dot; null and the empty string mean "no style". `lora_scale` is a number from 0 to 1.5 (default 1.0; not a bool, NaN or infinity; checked even when there is no key, and then ignored). Every refusal is one plain sentence and happens before any work:

| case | answer |
|---|---|
| a malformed key, or any non-string | `That style is not available right now.` (`ADAPTER_SENTENCE`, the same sentence as an unknown, missing or damaged adapter; Dockerfile.ace build check 3, which sends the key `k`, still passes unchanged) |
| a valid key with `xl-turbo` (or no `model`) | `A style needs the best-quality model.` (the adapter was trained on xl-sft; transfer to turbo is unproven) |
| a bad `lora_scale` | `Style strength must be a number from 0 to 1.5.` |
| `lora_scale` 0 with a valid key and xl-sft | not an error: a plain song, nothing fetched or folded, no trigger word (the evaluation harness also renders scale 0 as the plain model), and the note `The style strength was 0, so the song was made without the style.` |

A styled song: `vocal_language` is `en` when the caller sent none (the adapter was trained with `en` in the lyric header; another language is kept, with the note `The style was taught on English songs, so another language may sound less like it.`). The adapter is fetched and checked first (before the model loads, so a bad style costs no model load), then `engine('xl-sft')` and the planner as usual, then `generate_music` runs with the adapter folded in. The planner is shown the caller's PLAIN caption; the DiT gets `"<tag>, <caption>"` (tag from serve.json), with the caption cut at a word boundary to fit 512 characters in total and the tag never cut (note `The music description was shortened to make room for the style.`). `ACE_PLANNER_REWRITES_CAPTION=1` cannot drop the trigger word: a styled request never lets the planner rewrite the caption. The answer adds `style: {key, name, scale, tag, sha256}`, `controls.style / style_scale / style_tag`, the note `Made in the <name> style at strength <s>, with its trigger word "<tag>" ahead of the music description.`, `timing.style_fetch_s / style_apply_s / style_restore_s`, and `lora` in `features` (every answer and the probe). A plain request is unchanged except for that `features` entry.

**Files** under the prefix (B2, same client and `AWS_*` variables as the uploads): `serve.json`, `adapter_config.json`, `adapter_model.safetensors`. They are cached on this worker in `$ACE_LORA_DIR/<name>/` (default `/tmp/ace-lora/<name>/`), each written through a `.part` file and renamed only after its checks, so a half download or a file that failed its checksum is never at the final name. A later request on the same worker reuses the cached adapter (checked by file size only); if the file vanished it is fetched again. The weights are never re-hashed after the first fetch. Immutable by convention: an updated adapter goes under a new prefix.

**What the worker assumes about the files (serve.json was not on this PC, only its two field names are in the contract; the orchestrator should check them):** serve.json is a JSON object with `adapter_model_sha256` (64 hex characters; compared with the SHA-256 of the safetensors, lower case) and the trigger word as `trigger_tag` (else `trigger_word`, else `custom_tag`; 1 to 40 printable characters). An optional `name` (printable, 40 characters at most) is the display name; without it the name is the last part of the key (`soul-real-v1`). adapter_config.json is a PEFT `LORA` config (`peft_type`, `r`, `lora_alpha`, `target_modules` within q/k/v/o_proj) with none of: rsLoRA, DoRA, qaLoRA, `lora_bias`, `fan_in_fan_out`, a rank or alpha pattern, saved extra modules, a bias other than `none` (anything else is refused rather than folded wrongly: the fold is only exact for PEFT's plain `scale = alpha / r` rule). The safetensors file holds only keys of the form `base_model.model.layers.N.(self_attn|cross_attn).(q|k|v|o)_proj.lora_(A|B).weight` (decoder-relative, no adapter name; checked against the real adapter: 512 tensors, 256 pairs, all F32), every module with both halves and the rank of the config. SHA-256 of the local copy of the trained adapter, for comparison with serve.json: `8c40775a0a5e368d8bdd64a2ce3e7d5edf22be13004c8fd6cc39d0fe94134983`. If serve.json uses other field names, every styled request is refused and the log says `ACE-Step style refused: serve.json has no usable checksum` or `... no usable trigger word`.

### 9.2 Why not upstream's add_lora / unload_lora: it does not restore the plain model

Question from the contract: does `unload_lora()` (lifecycle.py:437-510) restore correctly, given the "Missing keys / Unexpected keys when restoring decoder" lines? **No.** Proved by running upstream's real `AceStepHandler.add_lora / set_lora_scale / unload_lora` (the pinned source, imported from the clone) with real PEFT 0.18.1 on a toy decoder that has the real module names (`scratch/stage6/upstream_unload_probe.py`):

- add_lora backs up `decoder.state_dict()` to the CPU (keys like `layers.0.self_attn.q_proj.weight`), then wraps the decoder with `PeftModel.from_pretrained`, which REPLACES each target `nn.Linear` with a PEFT `lora.Linear` (the original weight moves to `...q_proj.base_layer.weight`, the adapter to `...lora_A.<name>.weight`).
- unload_lora does `self.model.decoder = self.model.decoder.get_base_model()` and `load_state_dict(self._base_decoder, strict=False)`. `get_base_model()` returns the underlying module WITH the injected PEFT layers still in it (removing them takes `unload()` / `merge_and_unload()`, which upstream never calls). The backup keys do not exist in that module (every backup key `q_proj.weight` is "unexpected", every live `q_proj.base_layer.weight` is "missing"), `strict=False` swallows it, and the call reports success and clears its flags.
- Result of the probe: after "✅ LoRA unloaded, using base model", `q_proj` is still a `peft.tuners.lora.layer.Linear`, `lora_loaded` is False, and the decoder's output still differs from the plain model's by exactly the adapter's amount (max difference 2.88, the same as with the adapter loaded). A second add_lora on that module is accepted (the probe's second call reported success), so the wrapping is not even undone by loading again. The warnings that look like noise are the symptom of a restore that restores nothing. A plain song after a styled one through this path would stay styled.

The evaluation harness (`lora/pod/onpod/sample.py`) never hits this: its scale-0 "no adapter" renders are made once before any adapter is loaded. Two more upstream facts the fold relies on, both pinned by tests: upstream's `set_lora_scale` clamps to 0..1 (so the contract's 1.5 ceiling is this worker's own extension; the booth stays at 0..1) and applies a scale by multiplying PEFT's base scaling (`alpha / r`) by it, which is exactly the fold factor.

### 9.3 The fold

`apply_adapter(pipe, adapter, scale)` (a context manager; the reserved seam's name) folds, for every pair, `W += (lora_alpha / r) * scale * (B @ A)` with the product and the sum in fp32 and the result written back in the weight's own dtype (round to nearest even), into `pipe.model.decoder`: the `AceStepDiTModel` that holds the DiT weights (`handler.model.decoder`; upstream keeps no other copy of them, `_base_decoder` is only filled by add_lora, which is never called). The lyric and timbre encoders have `q_proj` and the rest too, but under `model.encoder...`; the fold only looks up names under the decoder, so they cannot be hit (tested with a sibling `encoder.layers.0.self_attn.q_proj`). Shapes in the real adapter (checked against its file): q 4096x2560, k and v 1024x2560, o 2560x4096, rank 64, `lora_alpha` 128, so PEFT scaling 2.0; 32 layers x self/cross x 4 = 256 weights, 1,677,721,600 values.

- **Dtype.** The decoder is bf16 on every target card (init_service_loader.py: `.to(device).to(self.dtype)`, bf16 for compute capability 8+), the adapter is fp32. The sum is done in fp32 and rounded once to bf16. Measured against the real adapter (`bf16_fold_probe.py`; the real xl-sft weights are not on this PC, so the base weights were synthetic Gaussians with sigma 0.01, 0.02 and 0.04, random inputs): the adapter's delta has an rms of 5.6e-4 to 6.7e-4 per element at scaling 2.0, about ten times bf16's half step (6e-5) for weights of size 0.02, so it is not rounded away; the folded weights give the adapter's output with norm ratio 1.00 to 1.01 and cosine 0.99 to 1.00, relative error 3% (sigma 0.01) to 13% (sigma 0.04) of the adapter's own contribution (about 5% at 0.02), against 0.3% for the unfused bf16 branch. Stochastic rounding was tried and is worse (4% to 19%), so plain round-to-nearest stays. Meaning: the strength of the style survives the fold and the rounding noise is small next to the weights' own bf16 noise, but the audio is NOT bit-identical to the approved samples, which came from upstream's unfused PEFT branch (add_lora casts the adapter to bf16 and adds a separate low-rank path). Listen for it in the smoke test (9.6 item 2).
- **Equality with PEFT.** `TestFoldOnTorch` compares the folded weights with `W + (alpha/r) * scale * B @ A` computed in float64 (bf16: within one rounding step everywhere and equal in 99%+; fp32: allclose 1e-5; scales 1.0, 0.6, 1.5). Where PEFT is installed (the 3.12 venv `scratch/lora_pod/venv` has 0.18.1, the version that trained the adapter) `TestFoldAgainstPeft` builds the same adapter in a PEFT model, scales it the way upstream's set_lora_scale does, calls `merge_and_unload()` and compares with the fold (the test allows one rounding step; `fold_vs_peft_probe.py` printed bitwise equality in bf16 and a largest difference of 1.2e-7 in fp32 for scales 1.0, 0.6 and 1.5).
- **In-place writes on parameters.** Upstream never freezes the decoder (no `requires_grad_(False)`; `model.eval()` only), so its weights are ordinary Parameters with `requires_grad=True`, and a plain `param.copy_()` raises "a leaf Variable that requires grad is being used in an in-place operation" (reproduced with torch 2.10). `torch.no_grad()` fixes that, but a parameter created inside `inference_mode` (a different way to load) can only be written inside inference mode ("Inplace update to inference tensor outside InferenceMode", reproduced). `torch.inference_mode()` accepts both, so every fold, snapshot, restore and checksum runs inside it (`inplace_probe.py`; `TestFoldOnTorch` covers requires-grad, frozen and inference-tensor parameters). Parameter objects keep their identity, dtype and flags.
- **Memory.** The pristine copy of the 256 touched weights is 3,355,443,200 bytes = 3.125 GiB of host RAM (bf16), held only for the length of a styled request and released after the restore (not kept between requests: nothing stale can outlive a reload, a FlashBoot snapshot never contains it, and a styled song pays about a second to copy it again). The adapter is read one pair at a time (under 2 MB of fp32 per pair), the fp32 product and working copy of the largest weight are 42 MB each on the card plus a checksum temporary of about 250 MB; `empty_cache()` runs after the fold and after the restore. Peak host RAM of a styled request is therefore about 3.4 GB above an idle worker. ACE_NOTES 8.8 item 8 expects 10 to 20 GB of host RAM at boot for the fp32 to bf16 cast, so any container that survives boot has the room; the adapter takes 335 MB of container disk in `/tmp`. Estimates, not measured: copy about 1 s, fold under 1 s, restore plus check about 2 s, first fetch of 335 MB a few seconds.
- **Speed and VRAM during the render.** Nothing extra: the adapter is part of the weights, so no per-step branch and no extra activation memory (unlike PEFT's unfused path, which adds four low-rank products per attention per layer). `render_s` and `render_peak_gib` of a styled and a plain song of the same length and model should match.

### 9.4 Order of events and what each failure leaves behind

`StyleFold` does, in this order: (1) resolve every target by name under `model.decoder`, checking shape, number format (bf16, fp16 or fp32 only, so a quantised decoder is refused), that no two pairs share one weight, that the decoder is not already a PEFT model and the handler has no upstream adapter loaded (`lora_loaded`), and that the model is not marked dirty; (2) copy all 256 weights to the CPU and take two integer checksums of each (the sum of the bit patterns and a position-weighted sum, taken on the card, so a flipped bit, a swap and a sign all show); nothing has been written yet; (3) mark the model dirty (`STATE['dirty']`); (4) write the folded weights; (5) on the way out, in a `finally`: copy the pristine weights back, recompute the checksums, compare, retry the ones that differ once, and only then clear the dirty mark. The restore looks each weight up by name again in the decoder as it is then (not the objects seen at fold time), so a render that replaced a Parameter or the whole decoder is still put right, or reported as a failed restore if the weights are not there.

| what goes wrong | what happens | state afterwards |
|---|---|---|
| bad key, scale, model | plain sentence before any work | nothing touched |
| style missing, damaged (checksum), unreadable, serve.json or config unusable, disk or network problem | `WorkerError(ADAPTER_SENTENCE)` before the model loads; only a fixed reason or the exception type is logged | nothing touched, nothing cached |
| decoder does not fit (missing weight, wrong shape, quantised, shared, already PEFT) | refused in step 1 | weights untouched, not dirty |
| the fold fails half way (CUDA error, unreadable adapter file) | the `finally` restores everything that was snapshotted | weights verified equal to the original, error returned |
| the render fails, raises, runs out of memory, or the planner dies inside it | same restore, then the usual sentence | clean |
| cancellation: KeyboardInterrupt, SystemExit, CancelledError | same restore, the exception goes on | clean |
| encoding or upload fails after the render | the weights were already put back when `generate_music` returned | clean |
| the restore itself fails or the check finds a difference twice | the model stays marked dirty, the log says `RESTORE FAILED for N weights`, the answer (if there is one) carries `The music model will be reloaded before the next song.` | the next `engine(model)` call, plain or styled, releases that handler and loads the model again from the checkpoint (`model_load_s` of one to three minutes), then clears the mark; if that load fails the mark stays and the next request tries again |
| the process is killed (OOM kill, SIGKILL, preemption) | nothing in memory survives | the checkpoint on disk was never written, so the next start is clean |
| a second job at the same time | `RENDER_LOCK` makes renders run one at a time (see 9.6 item 6) | clean |

### 9.5 The planner and the caption

`generate_music` hands one `params.caption` to the planner and to the DiT. The approved samples were made with the planner on the plain caption (one plan per brief, replayed for every adapter render, whose caption carried the trigger word). The worker reproduces that in one `generate_music` call with `PlainCaptionPlanner`, a thin wrapper that forwards everything to the real planner but replaces the `caption` keyword of `generate_with_stop_condition` with the plain one. Pinned in the source (`TestPinnedUpstream`): `generate_music` calls the planner exactly once, by keyword, with `caption=params.caption or ""`, reads only `llm_initialized` and that method from the planner object, and hands the DiT `dit_input_caption`, which only the planner's own caption rewrite or a direct-conditioning task changes (a styled request turns the rewrite off). The plan the booth sees (`plan`, `timing.plan_s`) is the same as for a plain song. With `thinking` false, or when the planner cannot load, the DiT still gets the tagged caption and the song is made without a plan. Seeds, DCW off for xl-sft, steps 50, guidance 7.0 and shift 3.0 are the unchanged xl-sft defaults, and a test pins that the only generation fields that differ from the same song without a style are `caption` and `vocal_language`.

### 9.6 Things to watch in the style smoke test

Send, on one worker, a plain xl-sft song, a styled one with the same seed and length, then a plain one again; read the whole answer and the container log for each.

1. **Host RAM.** A styled request holds 3.4 GB more than an idle worker (9.3). Signature of trouble: the container killed during a styled request with no Python error. Weights are then lost with the process (the next start is clean).
2. **bf16 weights, fp32 adapter.** The fold is not the numerical path the samples came from (9.3). Render one of the evaluation briefs with its seed and plan and compare with the sample MP3 from the pod: expect the same character at the same strength, not the same bytes. If the style sounds weaker, flatter or noisy against the samples, the fallback is forward hooks that add `scale * B(A(x))` to the four projections (PEFT's own numerics, about 4% more attention-projection FLOPs, no weight writes and no pristine copy at all); that is a different implementation of `apply_adapter`, not a setting.
3. **The module that holds the weights.** The log line after the fold must say `folded into 256 weights`. Any other number, or `ACE-Step style refused: ...` with a reason, means the decoder is not laid out as the adapter expects (the DiT is `handler.model.decoder`).
4. **Writes on parameters.** A `RuntimeError` mentioning "leaf Variable" or "inference tensor" in a styled answer (`could not finish (RuntimeError)`) would mean the model was loaded in a way this was not tested for; the fold runs in `inference_mode`, which handles both on torch 2.10 CPU. It is the same API on CUDA but nobody has run it there.
5. **FlashBoot.** The fold exists only between the fold and the restore inside one handler call; the idle state a snapshot can capture is always the plain model, and `STATE['dirty']` and the adapter cache live in process memory, so they come back together. If a restored worker has memory but not `/tmp`, the cache check (file size) notices the missing file and fetches again (`style_fetch_s` shows it).
6. **Concurrency.** runpod 1.12.0 (the pinned SDK) calls a plain handler directly on the event loop (`rp_job.run_job` does `handler_return = handler(job)` with no thread), and the default concurrency is 1, so one job runs at a time. `RENDER_LOCK` is belt and braces for a concurrency setting added later: a second render waits for the first. Probe jobs and validation do not take it.
7. **The song after a failed styled one.** Try a styled song that fails after the fold (for example a length the card cannot do) and then a plain one: the plain one must show `render_peak_gib` and a result like the first plain song, and the log must NOT contain `weights reloaded after a style could not be removed cleanly`, `restore error` or `RESTORE FAILED`. If it does, copy the lines before it: the restore failed and the model reloaded (slow) as designed.
8. **Cancellation.** In this SDK `task.cancel()` of a running job is only delivered after the sync handler has returned (the event loop is blocked while it runs), so a cancel cannot land in the middle of a fold; SIGINT calls `sys.exit(0)` inside the handler, which raises SystemExit through the fold's `finally` (tested), and SIGTERM stops gracefully after the job.
9. **A sticky CUDA error** (illegal address, device assert) poisons the whole process; a failed restore after one cannot be repaired in process and neither can the reload. This is the same limit plain songs already have. The SDK offers `{'refresh_worker': True}` in the handler's answer (popped into `stopPod`, verified in runpod 1.12.0's `rp_job.py`) to recycle the worker; the handler does not use it.
10. **The plan really comes from the plain caption.** Send the same seed, length and lyrics plain and styled with `thinking` true: `plan` (tempo, key, time signature) and `timing.plan_s` should be the same on the same card, because the planner sees the same words and the same seed.
11. **Speed and VRAM.** `render_s` and `memory.render_peak_gib` of the styled song should match the plain song of the same length; styled adds `style_apply_s` and `style_restore_s` (expected a few seconds in total) and, on a worker's first styled song, `style_fetch_s`.
12. **B2 access.** The key the endpoint runs with must be allowed to read `ace-loras/` as well as write `ace/`. A refusal with `ACE-Step style could not be fetched: ClientError` (or another exception type) in the log is the signature; the three objects are read by exact key.
13. **serve.json and adapter_config.json** are assumed (9.1). The refusal reasons in the log are fixed phrases that say which check failed.
14. **Strength above 1.** Upstream clamps its own scale to 1; the worker allows up to 1.5 because the fold has no ceiling, but nothing has been heard above 1. The booth stays at 0..1.
15. **Language.** A styled song with another language than English gets the note and is untested.
16. **Progress.** A cold worker announces `Loading the style` once while it fetches; later songs on the same worker do not.

### 9.7 Tests and the mutation pass

`test_ace_handler.py` went from 234 to 380 tests. New groups: the request (keys, model, scale, language, 20 odd values in every field with and without a style), the tagged caption (the tag is never cut, over a grid of tag and caption lengths), the planner wrapper, the fold's bookkeeping on a plain-Python stand-in for the tensor operations (order of events, restore after success, exception and cancellation, a fold that fails half way, a restore that fails or writes wrong values, the dirty mark, refusals before any write, replaced weights and decoders), fetching and checking the files with a fake B2 client (checksum, every missing object, a half download, every serve.json and adapter_config.json variant, the cache), whole styled songs through `handler()` (planner on the plain caption, DiT on the tagged one, the weights folded during the render and equal afterwards, the answer, the order of events, the lock, the song after each kind of failure, a plain request unchanged), `engine()` reloading a dirty model, and the real arithmetic on torch tensors (reference formula, parameters that require grad, are frozen or are inference tensors, one flipped bit or swap in the checksum, PEFT's own merge, the real adapter file against a meta-device decoder with the XL widths), plus tests of the pinned upstream source (9.2, 9.5, no CUDA graph or stale cache in the decoder path). Tests that need torch, safetensors or PEFT are skipped where those are missing (Python 3.14 on this PC has none of them), so the arithmetic tests run in the 3.12 venvs and in the image build, not on the CI runner.

How to run (each from `F:/KadeAI-work/ace/worker`): `python -I -B -m unittest discover -s . -p test_ace_handler.py` on Python 3.14: `Ran 380 tests ... OK (skipped=28)`; with `F:/KadeAI-work/ace/scratch/review/venv312/Scripts/python.exe` (torch 2.10 CPU, safetensors, loguru, huggingface_hub; no PEFT): `OK (skipped=2)`; with `F:/KadeAI-work/ace/scratch/lora_pod/venv/Scripts/python.exe` (the same plus PEFT 0.18.1): `OK (skipped=1)`. The one skip left there is the real-symlink test (no symlink rights on Windows).

Mutation pass (`scratch/stage6/mutate6.py`, run on the PEFT venv so every test takes part, four processes in parallel): 95 breaks of the new behaviour, each applied once to a copy of the worker and required to make at least one test fail. Groups: the key rules (a `..`, a leading dot, `match` for `fullmatch`, 61 characters, a second slash), the model gate, the scale range, its edges and default, scale 0, the empty key, the language default and its note, the order of the checks; the tag room (comma and space, the tag itself), the tag order and comma, the shortening note; the checksum (not compared, case, regex), the cache (never checked, not stored), the progress phrase, tag and name rules, every refused adapter_config setting (rsLoRA, DoRA, bias, rank pattern, other targets, boolean rank, zero alpha), a part file left behind (two ways), the exception types caught while fetching, rank, half pair, number format and name checks of the safetensors header; the scaling (`alpha` for `alpha / r`), the strength ignored or not passed on, restore only on success, the dirty mark never set, set too late or never cleared, restore never verified or tried once, `restored` claimed regardless, unreported, shape, number format and shared-weight checks, an adapter already on the decoder, a dirty model folded again, the cache emptied before the render, the pristine copy kept, restore into the objects seen at fold time instead of the live ones, the decoder remembered from fold time; no_grad or no mode instead of inference_mode, the delta added in bf16 instead of fp32, subtracted, not scaled, the product not in fp32, a snapshot that is only an alias, a put-back that does nothing, a checksum without positions or with half the bits, quantised weights accepted; a dirty model reused or never cleared by a reload; the planner shown the tagged caption, the DiT the plain one, the planner allowed to rewrite, the fold never applied or with the wrong strength, no render lock, no style in the answer, controls, timing or notes, the peak read after the restore, the log scrub missing the tagged caption, the fetch after the model load, `lora` missing from the features. First run: 93 killed and two survivors, both real gaps in the tests and not in the code: a serve.json checksum with a trailing newline was refused anyway by the later checksum comparison (so the format check was never what said no), and an `encoder.` module name was refused anyway as half a pair. The tests now assert the reason in the log and use complete pairs; the whole pass was repeated on the final tests: 95 of 95 killed, no setup problems.

### 9.8 Not verified in stage 6 (honest list)

- Nothing ran on a GPU, in Docker, against B2 or RunPod. `download_file(bucket, key, filename)` is called positionally and matches boto3 1.43.100's signature (`Bucket, Key, Filename`); the fake client in the tests has the same one.
- The real xl-sft weights are not on this PC, so the bf16 error figures in 9.3 use synthetic base weights; the sound of the fold against the PEFT-branch samples is unheard.
- serve.json's field names beyond the two in the contract are guessed (9.1); a real file might need a one-line change in `read_serve`.
- The bucket permissions of the endpoint's key for `ace-loras/` are unchecked (credentials were not read).
- FlashBoot's exact snapshot semantics and what happens to `/tmp` across one are not known; the design only needs the idle state to be the plain model, which it always is.
- Host RAM figures are arithmetic from the shapes (and upstream's own table), not measured.
