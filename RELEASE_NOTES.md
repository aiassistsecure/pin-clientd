# PIN Release Notes
**by Interchained LLC**

---

## Version 2.2.0 (January 2026)

### New Features

#### Operator Price Control
Operators can now set their own inference pricing per node via the `pricePerThousandTokens` configuration field. This enables competitive marketplace dynamics where operators price their compute resources based on hardware quality, model performance, and regional availability.

```json
{
  "alias": "GPU-1",
  "inferenceUri": "http://localhost:11434",
  "apiMode": "ollama",
  "region": "us-east",
  "capacity": 10,
  "pricePerThousandTokens": 0.001
}
```

- Default price: $0.001 per 1,000 tokens
- Price is set per-node, allowing different rates for different hardware
- Pricing is transmitted during node registration and stored server-side

#### Multi-Node Support
Single daemon instances can now register multiple inference backends simultaneously. This allows operators to:
- Run Ollama and vLLM side-by-side
- Connect nodes across different machines on a local network
- Manage heterogeneous GPU clusters from one config file

#### Dual API Mode Support
The daemon now supports two API formats for maximum compatibility:

| Mode | Endpoint Format | Compatible Backends |
|------|-----------------|---------------------|
| `ollama` | `/api/chat`, `/api/tags` | Ollama |
| `openai` | `/v1/chat/completions`, `/v1/models` | vLLM, TGI, LMStudio, LocalAI |

### Quality Assurance

#### Interview System
All nodes undergo automated quality interviews upon connection:

**Standard Track (Ollama mode):**
- 5 prompts testing factual accuracy, instruction following, and math
- Thresholds: 90% accuracy, 20 tok/s for `verified` tier

**Advanced Track (OpenAI mode):**
- 7 prompts including multi-step reasoning, code generation, and complex math
- Thresholds: 95% accuracy, 30 tok/s for `verified` tier

**Minimum Quality Floor:**
All operators must achieve at least **10 tok/s** - slower systems are automatically rejected.

**Quality Tiers:**
| Tier | Routing Priority | Requirements |
|------|------------------|--------------|
| `verified` | Highest | >90% accuracy, >20 tok/s |
| `standard` | Normal | >70% accuracy, >10 tok/s |
| `failed` | Blocked | <70% accuracy or <10 tok/s |

#### Custom Interview Model
Operators can now specify which model to use for quality interviews via the `interviewModel` config field:

```json
{
  "alias": "GPU-1",
  "inferenceUri": "http://localhost:11434",
  "apiMode": "ollama",
  "region": "us-east",
  "capacity": 10,
  "interviewModel": "phi3:medium"
}
```

This enables:
- Testing with newer models not in the benchmark list
- Proving capability with your best-performing model
- Supporting custom or fine-tuned models

If not specified, the system defaults to:
- **Ollama mode**: A recognized benchmark model (llama3:8b, mistral:7b, etc.)
- **OpenAI mode**: The first model in your list

### Performance

#### Multi-Threaded Inference
New `-n` flag enables concurrent request processing:

```bash
./pin-clientd -c config.json -n 4
```

Recommended thread counts:
- CPU-only: 1-2 threads
- Single GPU: 2-4 threads
- Multi-GPU: threads × GPUs

### Operator Experience

#### Automatic Node Registration
Nodes are automatically registered or updated based on alias matching. Duplicate aliases receive automatic suffixes (a, b, c) to prevent conflicts.

#### Wallet Configuration
Operators can configure BSC wallet addresses for USDT payouts:

```json
{
  "clientId": "op_xxx",
  "apiSecret": "pin_sk_xxx",
  "payoutAddress": "0x_your_bsc_wallet",
  "nodes": [...]
}
```

#### Improved Logging
Enhanced log output with clear status indicators:
- `[NODE]` - Node registration events
- `[WALLET]` - Payout address updates
- `[INTERVIEW]` - Quality assessment progress
- `[INFERENCE]` - Request processing

### Security

- WebSocket-only architecture (no public endpoint exposure required)
- API key authentication with `pin_sk_` prefixed secrets
- No inbound ports needed on operator machines
- TLS encryption for all server communication

### Breaking Changes

None. Existing configurations remain compatible. The `pricePerThousandTokens` field is optional and defaults to $0.001.

---

## Getting Started

1. Register at https://aiassist.net/pin/operator
2. Download the daemon from GitHub releases
3. Create your config file with pricing
4. Run: `./pin-clientd -c config.json`

## Support

- GitHub: https://github.com/aiassistsecure/pin-clientd
- Documentation: https://aiassist.net/pin

---

*PIN Network - Decentralized AI Inference Powered by the Community*

**Interchained LLC** | https://interchained.org
