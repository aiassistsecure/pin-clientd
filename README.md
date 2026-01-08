# PIN Client Daemon (pin-clientd) v2.2.0

The PIN Client Daemon connects your local LLM inference server to the AiAssist P2P Inference Network.

## Quick Start

1. Register as an operator at https://aiassist.net/pin/join
2. Copy the config template and add your credentials
3. Run the daemon: `./pin-clientd --config config.json`

## CLI Options

```bash
./pin-clientd [OPTIONS]

Options:
  -c, --config <FILE>     Config file path [default: config.json]
  -l, --log-level <LEVEL> Log level (trace, debug, info, warn, error) [default: info]
  -n, --threads <NUM>     Number of concurrent inference threads [default: 1]
  -h, --help              Print help
  -V, --version           Print version
```

### Multi-threaded Inference

Use `-n` to enable parallel request processing:

```bash
# Process up to 4 requests concurrently
./pin-clientd -c config.json -n 4
```

Recommended values based on hardware:
- CPU-only: 1-2 threads
- Single GPU: 2-4 threads  
- Multi-GPU: threads per GPU × number of GPUs

**Important: Ollama requires additional configuration for parallel requests.**

By default, Ollama processes requests sequentially. To enable parallel inference:

```bash
# Set before starting Ollama
export OLLAMA_NUM_PARALLEL=4
ollama serve
```

Or add to your systemd service file:
```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
```

| Backend | Parallel Support |
|---------|------------------|
| Ollama | Requires `OLLAMA_NUM_PARALLEL` env var |
| vLLM | ✅ Native (no config needed) |
| TGI | ✅ Native (no config needed) |
| LMStudio | ✅ Native (no config needed) |

Match `OLLAMA_NUM_PARALLEL` to your daemon `-n` value for optimal throughput.

## Build

```bash
./build.sh
```

## Configuration

### Config Structure

```json
{
  "clientId": "op_your_operator_id",
  "apiSecret": "your_api_secret",
  "nodes": [
    {
      "alias": "GPU-1",
      "inferenceUri": "http://localhost:11434",
      "apiMode": "ollama",
      "region": "us-east",
      "capacity": 10
    }
  ]
}
```

### Root Fields

| Field | Required | Description |
|-------|----------|-------------|
| `clientId` | Yes | Your operator ID (starts with `op_`) |
| `apiSecret` | Yes | Your API secret from registration |
| `nodes` | Yes | Array of node configurations (at least one required) |

### Node Fields (All Required)

| Field | Description |
|-------|-------------|
| `alias` | Friendly name for this node |
| `inferenceUri` | LLM server URL (e.g., `http://localhost:11434`) |
| `apiMode` | API format: `ollama` or `openai` |
| `region` | Geographic region (see table below) |
| `capacity` | Max concurrent requests |

## API Modes

### Ollama Mode

Use `"apiMode": "ollama"` for standard Ollama installations.

- Model discovery: `GET /api/tags`
- Chat endpoint: `POST /api/chat`
- Default port: 11434

```json
{
  "alias": "ollama-node",
  "inferenceUri": "http://localhost:11434",
  "apiMode": "ollama",
  "region": "us-east",
  "capacity": 5
}
```

### OpenAI Mode

Use `"apiMode": "openai"` for OpenAI-compatible APIs:
- vLLM
- text-generation-inference (TGI)
- LMStudio
- LocalAI
- Any OpenAI-compatible server

- Model discovery: `GET /v1/models`
- Chat endpoint: `POST /v1/chat/completions`

```json
{
  "alias": "vllm-node",
  "inferenceUri": "http://localhost:8000",
  "apiMode": "openai",
  "region": "us-west",
  "capacity": 20
}
```

## Regions

Choose the region closest to your server's physical location.

| Region ID | Name | Example Use Case |
|-----------|------|------------------|
| `us-east` | US East | AWS us-east-1, NYC, Virginia |
| `us-west` | US West | AWS us-west-2, California, Oregon |
| `eu-west` | EU West | AWS eu-west-1, Ireland, UK |
| `eu-central` | EU Central | AWS eu-central-1, Frankfurt, Amsterdam |
| `asia-pacific` | Asia Pacific | AWS ap-northeast-1, Tokyo, Singapore |
| `global` | Global (Any) | Multi-region or unknown location |

## Multi-Node Configuration

Register multiple nodes with different backends:

```json
{
  "clientId": "op_abc123",
  "apiSecret": "secret_xyz",
  "nodes": [
    {
      "alias": "ollama-node",
      "inferenceUri": "http://localhost:11434",
      "apiMode": "ollama",
      "region": "us-east",
      "capacity": 10
    },
    {
      "alias": "vllm-node",
      "inferenceUri": "http://localhost:8000",
      "apiMode": "openai",
      "region": "us-east",
      "capacity": 20
    },
    {
      "alias": "lmstudio-node",
      "inferenceUri": "http://192.168.1.100:1234",
      "apiMode": "openai",
      "region": "us-west",
      "capacity": 5
    }
  ]
}
```

## Running the Daemon

```bash
# Basic usage
./pin-clientd --config config.json

# With debug logging
RUST_LOG=debug ./pin-clientd --config config.json

# Specify log level
./pin-clientd --config config.json --log-level info
```

## Interview System

When your daemon connects, the server sends interview prompts to verify LLM quality. The daemon automatically:

1. Receives test prompts from the server
2. Runs them against your local LLM
3. Reports timing metrics (TTFT, tokens/sec)
4. Gets assigned a quality tier

Quality tiers affect routing priority:
- `verified` - Highest priority (>90% accuracy, >20 tok/s)
- `standard` - Normal priority (>70% accuracy, >10 tok/s)
- `slow` - Budget tier (>70% accuracy, <10 tok/s)
- `failed` - Blocked from production (<70% accuracy)

## Install as Service

```bash
sudo mkdir -p /opt/pin-clientd
sudo cp target/release/pin-clientd /opt/pin-clientd/
sudo cp config.json /opt/pin-clientd/
sudo cp pin-clientd.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pin-clientd
sudo systemctl start pin-clientd
```

## View Logs

```bash
journalctl -u pin-clientd -f
```

## Example Configurations

| File | Description |
|------|-------------|
| `config.example.json` | Basic single-node template |
| `config.ollama.example.json` | Ollama-specific example |
| `config.openai.example.json` | OpenAI-compatible API example |
| `config.multi-node.example.json` | Multi-node with mixed backends |

## License

MIT License - AiAssist Secure
