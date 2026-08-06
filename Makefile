.PHONY: archive validate features study-a study-b study-c explain report test all lint format install

PYTHON ?= python
CONFIG ?= configs/project_config.yaml
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

archive:
	@mkdir -p reports/archive/pre_research_redesign
	@echo "Archive directory: reports/archive/pre_research_redesign (do not overwrite prior archive blindly)"

validate:
	$(PYTHON) -c "from src.config import load_config, resolve_paths; from src.data_loader import summarize_raw_catalog; c=load_config('$(CONFIG)'); p=resolve_paths(c); print(summarize_raw_catalog(p.raw))"

features:
	$(PYTHON) -c "from src.config import load_config, resolve_paths, set_global_seed, get_seed; from src.data_loader import load_btcirt, save_parquet; from src.data_validation import audit_timestamp_gaps; from src.preprocessing import preprocess; from src.feature_engineering import engineer_features; c=load_config('$(CONFIG)'); set_global_seed(get_seed(c)); p=resolve_paths(c); df=load_btcirt(p.raw,c); g=audit_timestamp_gaps(df.sort_values('timestamp')); clean,_=preprocess(df,c,g); feat=engineer_features(clean,c); save_parquet(feat,p.processed); print('features', feat.shape)"

study-a:
	./scripts/run_study_a.sh

study-b:
	./scripts/run_study_b.sh

study-c:
	./scripts/run_study_c.sh

explain report train:
	$(PYTHON) -m src.pipeline --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests
	black --check src tests

format:
	black src tests
	ruff check --fix src tests

all: test
	./scripts/run_pipeline.sh
	@echo "Full redesigned pipeline finished"
