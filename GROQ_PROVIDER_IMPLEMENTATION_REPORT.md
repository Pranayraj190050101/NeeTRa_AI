# Groq Provider — Implementation Report

**Task:** Make Groq a first-class provider and turn the provider architecture
into something real and production-ready — without touching Hybrid Search or
fine-tuning.

**Status:** Complete. All eight requirements implemented and verified.
**Test result:** 211 passed (183 prior baseline + 28 provider tests), 0 failed.

---

## What changed

| Layer | File | Change |
|-------|------|--------|
| Config | `config.py` | `PROVIDERS`, `_PROVIDER_ALIASES`, `ProviderConfigError`, `mask_secret`; typed `llm_provider` + `groq_*` fields (prefix-free `validation_alias`); `resolved_llm_provider` / `active_llm_model` / `groq_configured`; `validate_llm_provider()` / `llm_provider_diagnostic()`; `summary()` gains `llm_provider` + `llm_model_active`. |
| Provider | `llm/groq_client.py` | **New** `GroqChatClient(LLMClient)` on the official SDK — lazy import, streaming, secret-safe `describe()`. |
| Factory | `llm/factory.py` | Resolves via `resolved_llm_provider`, validates, then returns Mock / Groq / HuggingFace. |
| Exports | `llm/__init__.py` | `GroqChatClient` exported. |
| Health | `agents/tutor.py` | `health()` (LLM-not-loaded branch) reports resolved provider + model + masked key. |
| `--check` | `app.py` | Provider-aware `_dependency_report` + `GROQ_MODULES`; `ProviderConfigError` captured first; `settings_panel` shows Provider / masked key / Active model. |
| Deps | `requirements.txt` | `groq>=0.11.0,<1.0` (imported lazily). |
| Tests | `tests/test_provider_architecture.py` | **New** — 25 tests. `tests/conftest.py` pins the provider fixture against ambient env. |
| Docs | `PROVIDER_ARCHITECTURE.md` | **New** — class diagram, config flow, selection rules, migration guide. |

---

## Requirements traceability

1. **Config layer** — typed `LLM_PROVIDER` / `GROQ_API_KEY` / `GROQ_MODEL`
   (+ temperature / top_p / max_tokens); supports `groq` / `huggingface` /
   `mock`; `NEET_LLM_BACKEND=hf` still works; priority is `LLM_PROVIDER` then
   `NEET_LLM_BACKEND` (`resolved_llm_provider`). ✔
2. **Provider architecture** — `LLMClient` with `GroqChatClient`,
   `HuggingFaceChatClient`, `MockLLM`; everything above the factory is
   provider-agnostic. ✔
3. **Groq implementation** — official `groq` SDK, key from environment,
   configurable model / temperature / max tokens. ✔
4. **Startup validation** — `groq` without a key raises a clear,
   secret-free `ProviderConfigError`; a blank `GROQ_MODEL` / `NEET_LLM_MODEL`
   for a real provider is also rejected; keys masked everywhere. ✔
5. **Health check** — `--check` shows provider + model, never the key. ✔
6. **Tests** — resolution, environment loading, Groq validation, fallback
   compatibility; all existing tests still pass. ✔
7. **Documentation** — `PROVIDER_ARCHITECTURE.md`. ✔
8. **Verification** — `pytest`, `offline_check`, `compileall`. ✔

Out of scope and untouched, as instructed: Hybrid Search, fine-tuning.

---

## Verification evidence

### `pytest` — 211 passed

```text
tests/test_provider_architecture.py  ............................  28 passed
full suite: 211 passed, 1 warning in 8.60s
```

The only warning is a pre-existing `langchain-community` sunset notice, unrelated
to this work. The `test_graph` assertion `health["llm"]["backend"] == "mock"`
still holds (an injected `MockLLM.describe()` returns `backend="mock"`).

The 28 new tests cover:

- **Provider resolution** — `groq`; `hf`/`HuggingFace` → `huggingface`; `mock`;
  supported set; `active_llm_model` per provider.
- **Environment loading** — `LLM_PROVIDER` / `GROQ_API_KEY` / `GROQ_MODEL` read
  from env; `LLM_PROVIDER` wins over `NEET_LLM_BACKEND`.
- **Groq validation** — missing key raises; error never leaks the secret;
  unknown provider raises; blank model rejected (`GROQ_MODEL`/`NEET_LLM_MODEL`);
  mock needs neither key nor model; `mask_secret` behaviour; diagnostic shape.
- **Fallback compatibility** — legacy `NEET_LLM_BACKEND=hf`/`mock` still resolve
  when `LLM_PROVIDER` is absent.
- **Factory wiring** — returns `MockLLM` / `GroqChatClient` /
  `HuggingFaceChatClient`; fails fast for `groq` without a key.
- **`GroqChatClient` behaviour** (injected SDK stub, fully offline) — blocking
  params, streaming callbacks + full-text return, empty-message guard, masked
  `describe()`, client-level missing-key guard, SDK errors wrapped in `LLMError`.

### `offline_check` — all passed

```text
All offline checks passed.
```

### `compileall` — clean

```text
python -m compileall -q .   → exit 0   (no syntax errors)
```

### Live `--check` runs

- **Groq + key:** provider `groq`, model `llama-3.3-70b-versatile`,
  `api_key: "...1239"`, `configured: true` — the full secret `gsk_secretVALUE1239`
  never appears in output.
- **Groq without key:**
  `[FAILED] LLM_PROVIDER=groq requires GROQ_API_KEY. ...` — a clear, actionable,
  secret-free message.
- **Legacy `NEET_LLM_BACKEND=hf`** (no `LLM_PROVIDER`): resolves to
  `huggingface` — backward compatibility confirmed.

---

## Design decisions

- **`ProviderConfigError` lives in `config.py`,** not `llm`, so the factory
  (which imports config) has no dependency cycle and `--check` can report a
  misconfiguration without importing provider SDKs.
- **Prefix-free aliases** (`validation_alias="LLM_PROVIDER"`, `"GROQ_*"`) with
  `populate_by_name=True` make the `.env` read naturally and let tests build
  `Settings(llm_provider=...)` by field name, while `extra="ignore"` keeps every
  new key harmless for existing deployments.
- **Lazy SDK import + injectable client** in `GroqChatClient` keeps the offline
  suite and `--check` runnable on machines without `groq` installed (verified:
  the dependency is reported `MISSING` yet everything still passes).
- **Unknown providers pass through** the resolver and are rejected in one place
  (`validate_llm_provider`), so logging/summary never throws.
- **Secret masking is centralised** in `mask_secret()` and reused by the client,
  the diagnostic and the UI panels.

---

## Backward compatibility

Nothing that worked before stops working. Deployments that only set
`NEET_LLM_BACKEND` keep resolving exactly as before; `summary()` retained its
old `llm_backend` / `llm_model` keys and only added two. `.env.example` was
already provider-oriented, so no template change was required.
