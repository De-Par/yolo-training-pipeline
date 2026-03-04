PYTHON ?= python3

install:
	$(PYTHON) -m pip install -r requirements.txt

download-fashionpedia:
	./scripts/download_datasets.sh --dataset fashionpedia --out-dir $(or $(OUT_DIR),data/raw)

download-deepfashion2:
	./scripts/download_datasets.sh --dataset deepfashion2 --out-dir $(or $(OUT_DIR),data/raw)

download-datasets:
	./scripts/download_datasets.sh --dataset all --out-dir $(or $(OUT_DIR),data/raw)

check:
	bash -n scripts/setup_env.sh
	bash -n scripts/download_datasets.sh
	bash -n scripts/download_models.sh
	$(PYTHON) -m py_compile tools/*.py
	$(PYTHON) tools/convert_coco_to_yolo.py --help >/dev/null
	$(PYTHON) tools/convert_deepfashion2_to_yolo.py --help >/dev/null
	$(PYTHON) tools/run_pipeline.py --help >/dev/null
	$(PYTHON) tools/train_yolo.py --help >/dev/null

fashionpedia-pipeline:
	$(PYTHON) tools/run_pipeline.py \
		--dataset fashionpedia \
		--raw-root $(RAW_ROOT) \
		--workdir data/processed \
		--model $(MODEL) \
		--epochs $(or $(EPOCHS),100) \
		--imgsz $(or $(IMGSZ),640) \
		--batch $(or $(BATCH),16) \
		--device $(or $(DEVICE),cpu) \
		--name $(or $(NAME),yolo-fashionpedia)

deepfashion2-pipeline:
	$(PYTHON) tools/run_pipeline.py \
		--dataset deepfashion2 \
		--raw-root $(RAW_ROOT) \
		--workdir data/processed \
		--model $(MODEL) \
		--epochs $(or $(EPOCHS),100) \
		--imgsz $(or $(IMGSZ),640) \
		--batch $(or $(BATCH),16) \
		--device $(or $(DEVICE),cpu) \
		--name $(or $(NAME),yolo-deepfashion2)
