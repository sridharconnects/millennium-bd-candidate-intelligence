# Deployment checklist

## 1 · Populate the LLM cache (once, ~$0.10)

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY
sed -i '' 's/^DEMO_MODE=1/DEMO_MODE=0/' .env
python scripts/run_pipeline.py
sed -i '' 's/^DEMO_MODE=0/DEMO_MODE=1/' .env
```

This writes `data/llm_cache/` and `data/exports/`. **Commit both.** From here on the
app, the notebook and the tests run offline with no key and no cost.

## 2 · Regenerate the measured artefacts

```bash
python scripts/run_eval.py                    # accuracy, ablation, fairness
python scripts/make_synthetic.py -n 500       # benchmark corpus (seeded)
python scripts/run_benchmark.py               # latency curve
python scripts/make_injected_fixture.py       # injection test fixture
python -m pytest tests/ -q                    # all 60 tests
python scripts/build_notebook.py --execute    # regenerate notebook with outputs
```

## 3 · Push to GitHub

```bash
git init && git add -A && git commit -m "Millennium BD candidate intelligence platform"
git branch -M main
git remote add origin https://github.com/<you>/millennium-bd-platform.git
git push -u origin main
```

`.env` is gitignored. Confirm before pushing: `git status --porcelain | grep -c '\.env$'`
must print `0`.

## 4 · Deploy on Streamlit Community Cloud

1. share.streamlit.io → **New app**
2. Repository: your repo · Branch: `main` · Main file path: `app.py`
3. **Advanced settings → Secrets**, paste exactly:
   ```toml
   DEMO_MODE = "1"
   ```
   No API key is needed — the deployed app replays `data/llm_cache/`.
4. Deploy. First build takes ~3 minutes (fastembed downloads the ONNX model on first
   query and caches it).

## 5 · Wire the URL into the deliverables

Once live, put the URL in **both** places:

```bash
# 1. the notebook (appears twice: hero link at the top, and section 15)
sed -i '' 's|https://REPLACE-ME.streamlit.app|https://YOUR-APP.streamlit.app|g' \
    scripts/build_notebook.py
python scripts/build_notebook.py --execute

# 2. the README
sed -i '' 's|_(paste your Streamlit Community Cloud URL here.*)_|<https://YOUR-APP.streamlit.app>|' \
    README.md
```

Then **open the link in a private window** and confirm it loads before submitting.

## 6 · Pre-submission checks

- [ ] `python -m pytest tests/ -q` — all pass, nothing skipped
- [ ] Notebook: Restart & Run All, clean, outputs committed
- [ ] `data/exports/candidates.json` and `candidates.csv` present and valid
- [ ] Live link opens in a private window
- [ ] Search, Candidate, Requisition, Analytics and System pages all render
- [ ] `git status` shows no `.env`
- [ ] Core demo runs with every optional flag off:
      `ENABLE_SEMANTIC=0 ENABLE_LLM_QUERY=0 ENABLE_INJECTION_SCAN=0 streamlit run app.py`
