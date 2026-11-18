# exl3xpu: EXL3 inference on Intel Arc (Battlemage) via vLLM XPU.
FROM intel/llm-scaler-vllm:0.26.0-b2@sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b
# Build from the repository root:  docker build -f xpu/docker/Dockerfile -t exl3xpu .
COPY core /opt/trellis-core
RUN pip install --no-deps --no-build-isolation -e /opt/trellis-core
WORKDIR /opt/exl3xpu
COPY xpu/pyproject.toml ./
COPY xpu/exl3xpu ./exl3xpu
COPY xpu/csrc ./csrc
COPY xpu/scripts ./scripts
COPY xpu/models ./models
COPY xpu/bench ./bench
COPY xpu/tests ./tests
RUN bash scripts/build_ext.sh && pip install --no-deps --no-build-isolation -e . \
 && python3 -c "from importlib.metadata import entry_points as e; assert any(x.name=='exl3xpu' for x in e(group='vllm.general_plugins'))"
ENV MODELS_DIR=/models
# Inside a container the API must listen on all of the container's interfaces for a published port (-p) to reach
# it. serve.py then refuses to start without an API key (SERVE_API_KEY_FILE) unless given --no-api-key; publish
# the port to the host's loopback only (-p 127.0.0.1:8000:8000), and with --network host pass --host 127.0.0.1.
ENV SERVE_HOST=0.0.0.0
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["models/qwen3.8-27b-exl3-4.00bpw", "--gpu", "0"]
