.PHONY: install validate features train explain report test all lint format

PYTHON ?= python
CONFIG ?= configs/project_config.yaml
# macOS XGBoost requires libomp; project ships a local symlink helper under .libs/
export DYLD_LIBRARY_PATH := $(CURDIR)/.libs:$(DYLD_LIBRARY_PATH)

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	@mkdir -p .libs
	@if [ ! -e .libs/libomp.dylib ]; then \
	  if [ -f /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libomp.dylib ]; then \
	    ln -sf /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libomp.dylib .libs/libomp.dylib; \
	  elif [ -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]; then \
	    ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib .libs/libomp.dylib; \
	  else \
	    echo "WARNING: libomp.dylib not found. On macOS run: brew install libomp"; \
	  fi; \
	fi

validate:
	$(PYTHON) -c "from src.pipeline import run_pipeline; print('use make all for full run')"
	$(PYTHON) -c "from src.config import load_config, resolve_paths; from src.data_loader import summarize_raw_catalog; c=load_config('$(CONFIG)'); p=resolve_paths(c); print(summarize_raw_catalog(p.raw))"

features:
	$(PYTHON) -c "from src.config import load_config, resolve_paths, set_global_seed, get_seed; from src.data_loader import load_btcirt, save_parquet; from src.data_validation import audit_timestamp_gaps; from src.preprocessing import preprocess; from src.feature_engineering import engineer_features; c=load_config('$(CONFIG)'); set_global_seed(get_seed(c)); p=resolve_paths(c); df=load_btcirt(p.raw,c); g=audit_timestamp_gaps(df.sort_values('timestamp')); clean,_=preprocess(df,c,g); feat=engineer_features(clean,c); save_parquet(feat,p.processed); print('features', feat.shape)"

train:
	$(PYTHON) -m src.pipeline --config $(CONFIG)

explain:
	$(PYTHON) -m src.pipeline --config $(CONFIG)

report:
	$(PYTHON) -c "from src.config import load_config, resolve_paths; from src.reporting import write_final_report, write_readme; import json; from pathlib import Path; c=load_config('$(CONFIG)'); p=resolve_paths(c); s=json.loads((p.metrics/'pipeline_summary.json').read_text()) if (p.metrics/'pipeline_summary.json').exists() else {'pipeline_completed': False}; write_final_report(p,s); write_readme(p,s)"

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

all: install test train
	@echo "Full pipeline finished"
