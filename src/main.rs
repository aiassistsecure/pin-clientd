use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Semaphore};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

static RUNNING: AtomicBool = AtomicBool::new(true);
static TOTAL_REQUESTS: AtomicU64 = AtomicU64::new(0);

#[derive(Parser, Debug)]
#[command(name = "pin-clientd")]
#[command(about = "PIN Client Daemon - Headless P2P Inference Network Node")]
#[command(version = "2.3.0")]
struct Args {
    #[arg(short, long, default_value = "config.json")]
    config: PathBuf,

    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[arg(short = 'n', long = "threads", default_value = "1", help = "Number of concurrent inference threads")]
    threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NodeConfig {
    alias: String,
    inference_uri: String,
    api_mode: String,
    region: String,
    capacity: u32,
    #[serde(default = "default_price")]
    price_per_thousand_tokens: f64,
    #[serde(default)]
    interview_model: Option<String>,
}

fn default_price() -> f64 {
    0.001
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Config {
    client_id: String,
    api_secret: String,
    nodes: Vec<NodeConfig>,
    #[serde(default)]
    payout_address: Option<String>,
    #[serde(default = "default_server_url")]
    server_url: String,
    #[serde(default = "default_reconnect_delay")]
    reconnect_delay_secs: u64,
}

fn default_server_url() -> String {
    "wss://aiassist.net/api/v1/pin/ws".to_string()
}

fn default_reconnect_delay() -> u64 {
    5
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[allow(non_camel_case_types)]
enum ServerMessage {
    AUTH_SUCCESS { operator_id: String, node_id: Option<String>, message: String },
    ERROR { message: String },
    PING,
    HEARTBEAT_ACK,
    MODEL_LIST_ACK,
    REGISTER_NODE_ACK { node_id: String, alias: String, models: Vec<String>, created: bool, message: String },
    UPDATE_WALLET_ACK { success: bool, message: String },
    INFERENCE_REQUEST { request_id: String, payload: InferencePayload },
    INTERVIEW_REQUEST { interview_id: String, node_id: Option<String>, model: String, prompts: Vec<InterviewPrompt>, timeout_ms: u32 },
    INTERVIEW_COMPLETE { interview_id: String, node_id: Option<String>, tier: String, accuracy: f32, tokens_per_sec: f32, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InterviewPrompt {
    id: String,
    prompt: String,
    max_tokens: u32,
}

#[derive(Debug, Serialize)]
struct InterviewResult {
    #[serde(rename = "type")]
    msg_type: String,
    interview_id: String,
    model: String,
    results: Vec<PromptResult>,
}

#[derive(Debug, Serialize)]
struct PromptResult {
    prompt_id: String,
    response: String,
    ttft_ms: u32,
    total_ms: u32,
    tokens_generated: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct InferencePayload {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    // The server has always sent these; serde dropped them on the floor
    // because they were not declared, so every PIN request ran at the
    // backend's own defaults and the caller's temperature / max_tokens were
    // silently ignored.
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<u32>,
}

/// Generation knobs, threaded to whichever backend serves the request.
#[derive(Debug, Clone, Copy, Default)]
struct GenOpts {
    temperature: Option<f64>,
    max_tokens: Option<u32>,
}

impl GenOpts {
    /// Ollama takes these under `options`, and names the token cap
    /// `num_predict`. Omitted entirely when nothing was requested, so the
    /// model's own defaults still apply.
    fn ollama_options(&self) -> Option<serde_json::Value> {
        let mut map = serde_json::Map::new();
        if let Some(t) = self.temperature {
            map.insert("temperature".into(), serde_json::json!(t));
        }
        if let Some(n) = self.max_tokens {
            map.insert("num_predict".into(), serde_json::json!(n));
        }
        if map.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(map))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct AuthMessage {
    #[serde(rename = "type")]
    msg_type: String,
    client_id: String,
    timestamp: String,
    signature: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClientMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<Vec<String>>,
}

/// One token (or small run of tokens) on its way to the consumer.
///
/// Streaming is purely ADDITIVE to the existing protocol: chunks are emitted
/// only when the server sets `payload.stream`, and the SAME terminal
/// `INFERENCE_RESPONSE` is still sent, carrying the assembled message and the
/// real usage counts. A server that ignores chunks behaves exactly as before,
/// and billing keeps happening once off the final message -- never off a
/// chunk. No capability negotiation is needed in either direction.
#[derive(Debug, Serialize)]
struct ChunkMessage {
    #[serde(rename = "type")]
    msg_type: String,
    request_id: String,
    index: u32,
    delta: String,
}

impl ChunkMessage {
    fn new(request_id: &str, index: u32, delta: String) -> Self {
        Self {
            msg_type: "INFERENCE_CHUNK".to_string(),
            request_id: request_id.to_string(),
            index,
            delta,
        }
    }
}

/// One NDJSON line from Ollama `/api/chat` with `stream: true`.
#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

/// One `data:` frame from an OpenAI-compatible SSE stream.
#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAIStreamChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    #[serde(default)]
    delta: OpenAIStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIStreamDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct RegisterNodeMessage {
    #[serde(rename = "type")]
    msg_type: String,
    alias: String,
    models: Vec<String>,
    capacity: u32,
    region: String,
    #[serde(rename = "pricePerThousandTokens")]
    price_per_thousand_tokens: f64,
    #[serde(rename = "interviewModel", skip_serializing_if = "Option::is_none")]
    interview_model: Option<String>,
}

#[derive(Debug, Serialize)]
struct UpdateWalletMessage {
    #[serde(rename = "type")]
    msg_type: String,
    payout_address: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: ChatMessage,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
    model: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaModelsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaModel {
    name: String,
}

fn compute_signature(client_id: &str, timestamp: &str, api_secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_secret.as_bytes());
    let secret_hash = hex::encode(hasher.finalize());

    let mut sig_hasher = Sha256::new();
    sig_hasher.update(format!("{}{}{}", client_id, timestamp, secret_hash).as_bytes());
    hex::encode(sig_hasher.finalize())
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelInfo {
    id: String,
}

async fn get_ollama_models(base_url: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));

    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama: {}", e))?;

    let data: OllamaModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    Ok(data.models.iter().map(|m| m.name.clone()).collect())
}

async fn get_openai_models(base_url: &str) -> Result<Vec<String>, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));

    let response = client
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to OpenAI-compatible API: {}", e))?;

    let data: OpenAIModelsResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    Ok(data.data.iter().map(|m| m.id.clone()).collect())
}

async fn get_models(base_url: &str, api_mode: &str) -> Result<Vec<String>, String> {
    match api_mode {
        "openai" => get_openai_models(base_url).await,
        _ => get_ollama_models(base_url).await,
    }
}

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

async fn chat_completion_ollama(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    opts: GenOpts,
) -> Result<OpenAIResponse, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));

    let request = OllamaChatRequest {
        model: model.to_string(),
        messages,
        stream: Some(false),
        options: opts.ollama_options(),
    };

    let response = client
        .post(&url)
        .json(&request)
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Ollama error {}: {}", status, body));
    }

    let ollama_resp: OllamaChatResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

    let prompt_tokens = ollama_resp.prompt_eval_count.unwrap_or(0);
    let completion_tokens = ollama_resp.eval_count.unwrap_or(0);

    Ok(OpenAIResponse {
        model: ollama_resp.model,
        choices: vec![OpenAIChoice {
            index: 0,
            message: ollama_resp.message,
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(OpenAIUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
    })
}

async fn chat_completion_openai(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    opts: GenOpts,
) -> Result<OpenAIResponse, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let request = OpenAIChatRequest {
        model: model.to_string(),
        messages,
        stream: Some(false),
        temperature: opts.temperature,
        max_tokens: opts.max_tokens,
    };

    let response = client
        .post(&url)
        .json(&request)
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("OpenAI request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI error {}: {}", status, body));
    }

    response
        .json()
        .await
        .map_err(|e| format!("Failed to parse OpenAI response: {}", e))
}

async fn chat_completion(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    api_mode: &str,
    opts: GenOpts,
) -> Result<OpenAIResponse, String> {
    match api_mode {
        "openai" => chat_completion_openai(base_url, model, messages, opts).await,
        _ => chat_completion_ollama(base_url, model, messages, opts).await,
    }
}

/// Streaming variant. Emits an `INFERENCE_CHUNK` per delta through `tx` and
/// returns the assembled `OpenAIResponse` so the caller still sends the usual
/// terminal `INFERENCE_RESPONSE`.
///
/// A send failure on `tx` means the WebSocket writer is gone; generation is
/// abandoned at that point rather than burning GPU on output nobody will read.
async fn chat_completion_stream(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    api_mode: &str,
    opts: GenOpts,
    request_id: &str,
    tx: &mpsc::UnboundedSender<String>,
) -> Result<OpenAIResponse, String> {
    match api_mode {
        "openai" => {
            chat_completion_stream_openai(base_url, model, messages, opts, request_id, tx).await
        }
        _ => chat_completion_stream_ollama(base_url, model, messages, opts, request_id, tx).await,
    }
}

fn send_chunk(
    tx: &mpsc::UnboundedSender<String>,
    request_id: &str,
    index: u32,
    delta: String,
) -> Result<(), String> {
    let msg = ChunkMessage::new(request_id, index, delta);
    let json = serde_json::to_string(&msg)
        .map_err(|e| format!("Failed to encode chunk: {}", e))?;
    tx.send(json)
        .map_err(|_| "WebSocket writer closed mid-stream".to_string())
}

async fn chat_completion_stream_ollama(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    opts: GenOpts,
    request_id: &str,
    tx: &mpsc::UnboundedSender<String>,
) -> Result<OpenAIResponse, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));

    let request = OllamaChatRequest {
        model: model.to_string(),
        messages,
        stream: Some(true),
        options: opts.ollama_options(),
    };

    let response = client
        .post(&url)
        .json(&request)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .map_err(|e| format!("Ollama request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Ollama error {}: {}", status, body));
    }

    let mut stream = response.bytes_stream();
    // Ollama emits newline-delimited JSON. A chunk boundary can land mid-line,
    // so hold the partial line in `buf` rather than parsing per network read.
    let mut buf = String::new();
    let mut content = String::new();
    let mut resolved_model = model.to_string();
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut index = 0u32;

    while let Some(item) = stream.next().await {
        let bytes = item.map_err(|e| format!("Ollama stream error: {}", e))?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(nl) = buf.find('\n') {
            let line: String = buf.drain(..=nl).collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let chunk: OllamaStreamChunk = match serde_json::from_str(line) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Skipping unparseable Ollama stream line: {}", e);
                    continue;
                }
            };
            if let Some(m) = chunk.model {
                resolved_model = m;
            }
            if let Some(msg) = chunk.message {
                if !msg.content.is_empty() {
                    content.push_str(&msg.content);
                    send_chunk(tx, request_id, index, msg.content)?;
                    index += 1;
                }
            }
            if chunk.done {
                prompt_tokens = chunk.prompt_eval_count.unwrap_or(prompt_tokens);
                completion_tokens = chunk.eval_count.unwrap_or(completion_tokens);
            }
        }
    }

    Ok(OpenAIResponse {
        model: resolved_model,
        choices: vec![OpenAIChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Some(OpenAIUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
    })
}

async fn chat_completion_stream_openai(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
    opts: GenOpts,
    request_id: &str,
    tx: &mpsc::UnboundedSender<String>,
) -> Result<OpenAIResponse, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    // `stream_options.include_usage` is how an OpenAI-compatible server is
    // asked to put real token counts in the final frame. Servers that don't
    // support it ignore the field, and usage stays zero -- the same outcome as
    // a missing `usage` on the non-streaming path today. Counts are never
    // estimated: billing on a guess would be worse than billing on zero.
    let mut request = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if let Some(t) = opts.temperature {
        request["temperature"] = serde_json::json!(t);
    }
    if let Some(n) = opts.max_tokens {
        request["max_tokens"] = serde_json::json!(n);
    }

    let response = client
        .post(&url)
        .json(&request)
        .timeout(Duration::from_secs(600))
        .send()
        .await
        .map_err(|e| format!("OpenAI request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI error {}: {}", status, body));
    }

    let mut stream = response.bytes_stream();
    let mut buf = String::new();
    let mut content = String::new();
    let mut resolved_model = model.to_string();
    let mut usage: Option<OpenAIUsage> = None;
    let mut finish_reason: Option<String> = None;
    let mut index = 0u32;

    while let Some(item) = stream.next().await {
        let bytes = item.map_err(|e| format!("OpenAI stream error: {}", e))?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(nl) = buf.find('\n') {
            let line: String = buf.drain(..=nl).collect();
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue; // SSE comments and blank separators
            }
            let data = line["data:".len()..].trim();
            if data == "[DONE]" {
                continue;
            }
            let chunk: OpenAIStreamChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Skipping unparseable SSE frame: {}", e);
                    continue;
                }
            };
            if let Some(m) = chunk.model {
                resolved_model = m;
            }
            if chunk.usage.is_some() {
                usage = chunk.usage;
            }
            if let Some(choice) = chunk.choices.into_iter().next() {
                if choice.finish_reason.is_some() {
                    finish_reason = choice.finish_reason;
                }
                if let Some(delta) = choice.delta.content {
                    if !delta.is_empty() {
                        content.push_str(&delta);
                        send_chunk(tx, request_id, index, delta)?;
                        index += 1;
                    }
                }
            }
        }
    }

    Ok(OpenAIResponse {
        model: resolved_model,
        choices: vec![OpenAIChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: finish_reason.or_else(|| Some("stop".to_string())),
        }],
        usage,
    })
}

async fn run_interview_prompt(
    base_url: &str,
    model: &str,
    prompt: &InterviewPrompt,
    api_mode: &str,
) -> PromptResult {
    let start = std::time::Instant::now();
    
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: prompt.prompt.clone(),
    }];
    
    // The interview always specified a token cap; it was being dropped with
    // everything else. Honour it now so a slow node is not judged on an
    // unbounded generation.
    let result = chat_completion(
        base_url,
        model,
        messages,
        api_mode,
        GenOpts { temperature: None, max_tokens: Some(prompt.max_tokens) },
    )
    .await;
    let total_ms = start.elapsed().as_millis() as u32;
    
    match result {
        Ok(resp) => {
            let content = resp.choices.first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();
            let tokens = resp.usage.as_ref()
                .map(|u| u.completion_tokens)
                .unwrap_or(0);
            
            PromptResult {
                prompt_id: prompt.id.clone(),
                response: content,
                ttft_ms: total_ms / 2,  // Approximate TTFT
                total_ms,
                tokens_generated: tokens,
                error: None,
            }
        }
        Err(e) => PromptResult {
            prompt_id: prompt.id.clone(),
            response: String::new(),
            ttft_ms: 0,
            total_ms,
            tokens_generated: 0,
            error: Some(e),
        },
    }
}

async fn execute_interview(
    base_url: &str,
    interview_id: &str,
    model: &str,
    prompts: Vec<InterviewPrompt>,
    api_mode: &str,
) -> InterviewResult {
    info!("[INTERVIEW] Starting interview {} with {} prompts on model {} ({})", 
        interview_id, prompts.len(), model, api_mode);
    
    let mut results = Vec::new();
    
    for (i, prompt) in prompts.iter().enumerate() {
        info!("[INTERVIEW] Running prompt {}/{}: {}", i + 1, prompts.len(), prompt.id);
        let result = run_interview_prompt(base_url, model, prompt, api_mode).await;
        
        if result.error.is_some() {
            warn!("[INTERVIEW] Prompt {} failed: {:?}", prompt.id, result.error);
        } else {
            info!("[INTERVIEW] Prompt {} completed: {} tokens in {}ms", 
                prompt.id, result.tokens_generated, result.total_ms);
        }
        
        results.push(result);
    }
    
    info!("[INTERVIEW] Interview {} complete with {} results", interview_id, results.len());
    
    InterviewResult {
        msg_type: "INTERVIEW_RESULT".to_string(),
        interview_id: interview_id.to_string(),
        model: model.to_string(),
        results,
    }
}

async fn run_connection(config: &Config, max_threads: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Connecting to PIN server: {}", config.server_url);
    info!("Inference threads: {}", max_threads);

    let (ws_stream, _) = connect_async(&config.server_url).await?;
    let (mut write, mut read) = ws_stream.split();
    
    let semaphore = Arc::new(Semaphore::new(max_threads));
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .to_string();

    let signature = compute_signature(&config.client_id, &timestamp, &config.api_secret);

    let auth_msg = AuthMessage {
        msg_type: "AUTH".to_string(),
        client_id: config.client_id.clone(),
        timestamp,
        signature,
    };

    write
        .send(Message::Text(serde_json::to_string(&auth_msg)?))
        .await?;
    info!("Sent AUTH message for {}", config.client_id);

    let mut node_endpoints: std::collections::HashMap<String, (String, String)> = std::collections::HashMap::new();
    for node in &config.nodes {
        node_endpoints.insert(node.alias.clone(), (node.inference_uri.clone(), node.api_mode.clone()));
    }

    while RUNNING.load(Ordering::SeqCst) {
        tokio::select! {
            response_json = rx.recv() => {
                if let Some(json) = response_json {
                    if let Err(e) = write.send(Message::Text(json)).await {
                        error!("Failed to send response: {}", e);
                    }
                }
            }
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<ServerMessage>(&text) {
                            Ok(server_msg) => {
                                match server_msg {
                                    ServerMessage::AUTH_SUCCESS { operator_id, node_id: _, message } => {
                                        info!("Authenticated! Operator: {}", operator_id);
                                        info!("{}", message);

                                        // Update wallet address if configured
                                        if let Some(ref payout_addr) = config.payout_address {
                                            if !payout_addr.is_empty() {
                                                info!("Updating payout wallet: {}...{}", &payout_addr[..6.min(payout_addr.len())], &payout_addr[payout_addr.len().saturating_sub(4)..]);
                                                let wallet_msg = UpdateWalletMessage {
                                                    msg_type: "UPDATE_WALLET".to_string(),
                                                    payout_address: payout_addr.clone(),
                                                };
                                                if let Err(e) = write.send(Message::Text(serde_json::to_string(&wallet_msg)?)).await {
                                                    error!("Failed to update wallet: {}", e);
                                                }
                                            }
                                        }

                                        // Register each configured node with the server
                                        // Each node may have its own endpoint and API mode
                                        for node_config in &config.nodes {
                                            info!("Registering node: {} (region: {}, capacity: {}, endpoint: {}, mode: {})", 
                                                node_config.alias, node_config.region, node_config.capacity, 
                                                node_config.inference_uri, node_config.api_mode);
                                            
                                            let models = match get_models(&node_config.inference_uri, &node_config.api_mode).await {
                                                Ok(m) => m,
                                                Err(e) => {
                                                    error!("Failed to get models for {} ({}): {}", node_config.alias, node_config.api_mode, e);
                                                    vec![]
                                                }
                                            };
                                            
                                            if models.is_empty() {
                                                warn!("No models found for node {} - check endpoint {}", node_config.alias, node_config.inference_uri);
                                            } else {
                                                info!("Node {} has {} models: {:?}", node_config.alias, models.len(), models);
                                            }
                                            
                                            let register_msg = RegisterNodeMessage {
                                                msg_type: "REGISTER_NODE".to_string(),
                                                alias: node_config.alias.clone(),
                                                models: models.clone(),
                                                capacity: node_config.capacity,
                                                region: node_config.region.clone(),
                                                price_per_thousand_tokens: node_config.price_per_thousand_tokens,
                                                interview_model: node_config.interview_model.clone(),
                                            };
                                            
                                            if let Err(e) = write.send(Message::Text(serde_json::to_string(&register_msg)?)).await {
                                                error!("Failed to register node {}: {}", node_config.alias, e);
                                            }
                                        }
                                        
                                        info!("Registered {} node(s) with PIN network", config.nodes.len());
                                    }
                                    ServerMessage::REGISTER_NODE_ACK { node_id, alias, models, created, message } => {
                                        let status = if created { "registered" } else { "updated" };
                                        info!("[NODE] {} {} (ID: {}) with {} models", status.to_uppercase(), alias, node_id, models.len());
                                        info!("[NODE] {}", message);
                                    }
                                    ServerMessage::ERROR { message } => {
                                        error!("Server error: {}", message);
                                        return Err(message.into());
                                    }
                                    ServerMessage::PING => {
                                        let pong = ClientMessage {
                                            msg_type: "PONG".to_string(),
                                            request_id: None,
                                            result: None,
                                            error: None,
                                            models: None,
                                        };
                                        let _ = write.send(Message::Text(serde_json::to_string(&pong)?)).await;
                                    }
                                    ServerMessage::HEARTBEAT_ACK | ServerMessage::MODEL_LIST_ACK => {}
                                    ServerMessage::UPDATE_WALLET_ACK { success, message } => {
                                        if success {
                                            info!("[WALLET] {}", message);
                                        } else {
                                            warn!("[WALLET] Failed: {}", message);
                                        }
                                    }
                                    ServerMessage::INTERVIEW_REQUEST { interview_id, node_id, model, prompts, timeout_ms: _ } => {
                                        let node_label = node_id.as_deref().unwrap_or("operator");
                                        info!("[INTERVIEW] Received interview for {} - model {} ({} prompts)", 
                                            node_label, model, prompts.len());
                                        
                                        let (uri, mode) = match node_endpoints.get(node_label) {
                                            Some((u, m)) => (u.clone(), m.clone()),
                                            None => {
                                                let first = config.nodes.first().unwrap();
                                                (first.inference_uri.clone(), first.api_mode.clone())
                                            }
                                        };
                                        
                                        let interview_result = execute_interview(&uri, &interview_id, &model, prompts, &mode).await;
                                        
                                        if let Err(e) = write.send(Message::Text(serde_json::to_string(&interview_result)?)).await {
                                            error!("[INTERVIEW] Failed to send result: {}", e);
                                        } else {
                                            info!("[INTERVIEW] Result sent to server for {}", node_label);
                                        }
                                    }
                                    ServerMessage::INTERVIEW_COMPLETE { interview_id: _, node_id, tier, accuracy, tokens_per_sec, reason } => {
                                        let node_label = node_id.as_deref().unwrap_or("operator");
                                        info!("=====================================");
                                        info!("[INTERVIEW] Quality Tier Assigned for {}!", node_label);
                                        info!("  Tier: {}", tier.to_uppercase());
                                        info!("  Accuracy: {:.1}%", accuracy);
                                        info!("  Speed: {:.1} tokens/sec", tokens_per_sec);
                                        info!("  Reason: {}", reason);
                                        info!("=====================================");
                                        
                                        if tier == "failed" {
                                            error!("Node {} failed quality check - connection will be closed", node_label);
                                        }
                                    }
                                    ServerMessage::INFERENCE_REQUEST { request_id, payload } => {
                                        let count = TOTAL_REQUESTS.fetch_add(1, Ordering::SeqCst) + 1;
                                        
                                        let first_node = config.nodes.first().unwrap();
                                        let uri = first_node.inference_uri.clone();
                                        let mode = first_node.api_mode.clone();
                                        let model = payload.model.clone();
                                        let messages = payload.messages;
                                        let stream = payload.stream;
                                        let opts = GenOpts {
                                            temperature: payload.temperature,
                                            max_tokens: payload.max_tokens,
                                        };

                                        info!("[#{}] Inference request: {} ({}) via {}{} [queued]",
                                            count, request_id, model, mode,
                                            if stream { " streaming" } else { "" });
                                        
                                        let sem = semaphore.clone();
                                        let tx = tx.clone();
                                        
                                        tokio::spawn(async move {
                                            let _permit = sem.acquire().await.expect("semaphore closed");
                                            
                                            info!("[#{}] Starting inference for {}", count, request_id);
                                            let result = if stream {
                                                chat_completion_stream(
                                                    &uri, &model, messages, &mode, opts,
                                                    &request_id, &tx,
                                                )
                                                .await
                                            } else {
                                                chat_completion(&uri, &model, messages, &mode, opts)
                                                    .await
                                            };

                                            let response = match result {
                                                Ok(openai_resp) => {
                                                    let usage = openai_resp.usage.as_ref();
                                                    let prompt_tokens = usage.map(|u| u.prompt_tokens).unwrap_or(0);
                                                    let completion_tokens = usage.map(|u| u.completion_tokens).unwrap_or(0);
                                                    
                                                    info!("[#{}] Completed successfully ({}+{} tokens)", count, prompt_tokens, completion_tokens);
                                                    ClientMessage {
                                                        msg_type: "INFERENCE_RESPONSE".to_string(),
                                                        request_id: Some(request_id),
                                                        result: Some(serde_json::to_value(openai_resp).unwrap()),
                                                        error: None,
                                                        models: None,
                                                    }
                                                }
                                                Err(e) => {
                                                    error!("[#{}] Failed: {}", count, e);
                                                    ClientMessage {
                                                        msg_type: "INFERENCE_ERROR".to_string(),
                                                        request_id: Some(request_id),
                                                        result: None,
                                                        error: Some(e),
                                                        models: None,
                                                    }
                                                }
                                            };

                                            if let Ok(json) = serde_json::to_string(&response) {
                                                let _ = tx.send(json);
                                                info!("[#{}] Response queued for send", count);
                                            }
                                        });
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse server message: {} - {}", e, text);
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        info!("Server closed connection");
                        break;
                    }
                    Some(Err(e)) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    None => {
                        info!("WebSocket stream ended");
                        break;
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(30)) => {
                let heartbeat = ClientMessage {
                    msg_type: "HEARTBEAT".to_string(),
                    request_id: None,
                    result: None,
                    error: None,
                    models: None,
                };
                if write.send(Message::Text(serde_json::to_string(&heartbeat)?)).await.is_err() {
                    warn!("Failed to send heartbeat");
                    break;
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&args.log_level)),
        )
        .init();

    println!();
    println!("     █████╗ ██╗ █████╗ ███████╗    ██████╗ ██╗███╗   ██╗");
    println!("    ██╔══██╗██║██╔══██╗██╔════╝    ██╔══██╗██║████╗  ██║");
    println!("    ███████║██║███████║███████╗    ██████╔╝██║██╔██╗ ██║");
    println!("    ██╔══██║██║██╔══██║╚════██║    ██╔═══╝ ██║██║╚██╗██║");
    println!("    ██║  ██║██║██║  ██║███████║    ██║     ██║██║ ╚████║");
    println!("    ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝╚═╝  ╚═══╝");
    println!();
    println!("    PIN Client Daemon v2.1.0 - https://AiAssist.net");
    println!();

    let config_path = &args.config;
    info!("Loading config from: {:?}", config_path);

    let config_str = match std::fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to read config file {:?}: {}", config_path, e);
            error!("Create config.json with: clientId, apiSecret, nodes");
            error!("  Each node requires: alias, inferenceUri, apiMode, region, capacity");
            std::process::exit(1);
        }
    };

    let config: Config = match serde_json::from_str(&config_str) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed to parse config: {}", e);
            std::process::exit(1);
        }
    };

    info!("Operator ID: {}", config.client_id);
    info!("Nodes configured: {}", config.nodes.len());
    for node in &config.nodes {
        info!("  - {} | {} | {} | capacity: {}", 
            node.alias, node.inference_uri, node.api_mode, node.capacity);
    }
    
    if config.nodes.is_empty() {
        error!("No nodes configured! Add at least one node to the 'nodes' array.");
        std::process::exit(1);
    }

    ctrlc::set_handler(move || {
        info!("Shutdown signal received");
        RUNNING.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    info!("Concurrent inference threads: {}", args.threads);
    
    while RUNNING.load(Ordering::SeqCst) {
        match run_connection(&config, args.threads).await {
            Ok(_) => {
                if RUNNING.load(Ordering::SeqCst) {
                    info!("Reconnecting in {}s...", config.reconnect_delay_secs);
                    tokio::time::sleep(Duration::from_secs(config.reconnect_delay_secs)).await;
                }
            }
            Err(e) => {
                error!("Connection error: {}", e);
                if RUNNING.load(Ordering::SeqCst) {
                    info!("Reconnecting in {}s...", config.reconnect_delay_secs);
                    tokio::time::sleep(Duration::from_secs(config.reconnect_delay_secs)).await;
                }
            }
        }
    }

    info!("Shutdown complete. Total requests: {}", TOTAL_REQUESTS.load(Ordering::SeqCst));
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Capture the request body a caller sent, alongside serving a response.
    fn serve_once_capturing(pieces: Vec<String>) -> (String, std::sync::mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (btx, brx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 8192];
            let n = sock.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = req.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            let _ = btx.send(body);
            let full: String = pieces.concat();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                full.len()
            );
            sock.write_all(head.as_bytes()).unwrap();
            sock.write_all(full.as_bytes()).unwrap();
        });
        (format!("http://{}", addr), brx)
    }

    /// Serve one HTTP response whose body is written in the given pieces.
    /// Splitting a JSON line across pieces is the whole point: a naive
    /// parser that treats each network read as a complete line loses tokens.
    fn serve_once(pieces: Vec<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut discard = [0u8; 4096];
            let _ = sock.read(&mut discard);
            let body: String = pieces.concat();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            sock.write_all(head.as_bytes()).unwrap();
            for piece in pieces {
                sock.write_all(piece.as_bytes()).unwrap();
                sock.flush().unwrap();
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        format!("http://{}", addr)
    }

    fn drain(rx: &mut mpsc::UnboundedReceiver<String>) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        while let Ok(s) = rx.try_recv() {
            out.push(serde_json::from_str(&s).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn ollama_stream_emits_chunks_and_assembles_final_message() {
        let l1 = r#"{"model":"muse-local:latest","message":{"role":"assistant","content":"Hel"},"done":false}"#;
        let l2 = r#"{"model":"muse-local:latest","message":{"role":"assistant","content":"lo"},"done":false}"#;
        let l3 = r#"{"model":"muse-local:latest","message":{"role":"assistant","content":" world"},"done":false}"#;
        let l4 = r#"{"model":"muse-local:latest","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":7,"eval_count":3}"#;

        // Deliberately split l2 mid-JSON across two network writes.
        let (a, b) = l2.split_at(20);
        let base = serve_once(vec![
            format!("{}\n{}", l1, a),
            format!("{}\n{}\n{}\n", b, l3, l4),
        ]);

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let resp = chat_completion_stream_ollama(
            &base,
            "muse-local:latest",
            vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            GenOpts::default(),
            "pin_req_test",
            &tx,
        )
        .await
        .expect("stream should succeed");

        let chunks = drain(&mut rx);
        assert_eq!(chunks.len(), 3, "one chunk per non-empty delta");
        assert_eq!(chunks[0]["type"], "INFERENCE_CHUNK");
        assert_eq!(chunks[0]["request_id"], "pin_req_test");
        assert_eq!(chunks[0]["index"], 0);
        assert_eq!(chunks[1]["index"], 1);
        assert_eq!(chunks[2]["index"], 2);

        let deltas: String = chunks
            .iter()
            .map(|c| c["delta"].as_str().unwrap())
            .collect();
        assert_eq!(deltas, "Hello world", "split line must not lose tokens");

        // The terminal response is unchanged in shape and carries real usage,
        // so billing keeps working off the final message.
        assert_eq!(resp.choices[0].message.content, "Hello world");
        assert_eq!(deltas, resp.choices[0].message.content);
        let usage = resp.usage.expect("usage from the done frame");
        assert_eq!(usage.prompt_tokens, 7);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn openai_sse_stream_parses_deltas_done_and_usage() {
        let base = serve_once(vec![
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n".to_string(),
            "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n".to_string(),
            "data: {\"model\":\"m\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n".to_string(),
            "data: [DONE]\n\n".to_string(),
        ]);

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let resp = chat_completion_stream_openai(
            &base,
            "m",
            vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            GenOpts::default(),
            "pin_req_sse",
            &tx,
        )
        .await
        .expect("stream should succeed");

        let chunks = drain(&mut rx);
        assert_eq!(chunks.len(), 2, "[DONE] and the usage-only frame are not chunks");
        assert_eq!(resp.choices[0].message.content, "Hello");
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = resp.usage.expect("usage frame");
        assert_eq!(usage.total_tokens, 7);
    }

    #[tokio::test]
    async fn stream_aborts_when_the_websocket_writer_is_gone() {
        let base = serve_once(vec![
            format!("{}\n", r#"{"message":{"role":"assistant","content":"x"},"done":false}"#),
            format!("{}\n", r#"{"message":{"role":"assistant","content":"y"},"done":true}"#),
        ]);

        let (tx, rx) = mpsc::unbounded_channel::<String>();
        drop(rx); // consumer disconnected mid-generation

        let err = chat_completion_stream_ollama(
            &base,
            "m",
            vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            GenOpts::default(),
            "pin_req_dead",
            &tx,
        )
        .await
        .expect_err("must not keep generating for a consumer that vanished");
        assert!(err.contains("closed mid-stream"), "got: {}", err);
    }

    #[tokio::test]
    async fn non_success_status_is_surfaced_not_swallowed() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut discard = [0u8; 4096];
            let _ = sock.read(&mut discard);
            let body = "model not found";
            let _ = sock.write_all(
                format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                )
                .as_bytes(),
            );
        });

        let (tx, _rx) = mpsc::unbounded_channel::<String>();
        let err = chat_completion_stream_ollama(
            &format!("http://{}", addr),
            "nope",
            vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            GenOpts::default(),
            "pin_req_404",
            &tx,
        )
        .await
        .expect_err("404 must be an error");
        assert!(err.contains("404"), "got: {}", err);
    }

    #[tokio::test]
    async fn generation_options_reach_ollama() {
        // Regression: InferencePayload did not declare temperature/max_tokens,
        // so serde dropped them and every request silently ran at the
        // backend's defaults -- the playground's slider did nothing.
        let done = r#"{"message":{"role":"assistant","content":"ok"},"done":true,"prompt_eval_count":1,"eval_count":1}"#;
        let (base, body_rx) = serve_once_capturing(vec![format!("{}\n", done)]);

        let (tx, _rx) = mpsc::unbounded_channel::<String>();
        chat_completion_stream_ollama(
            &base,
            "m",
            vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            GenOpts { temperature: Some(0.15), max_tokens: Some(64) },
            "pin_req_opts",
            &tx,
        )
        .await
        .expect("stream ok");

        let body = body_rx.recv_timeout(Duration::from_secs(5)).expect("request body");
        let sent: serde_json::Value = serde_json::from_str(&body).expect("json body");
        assert_eq!(sent["options"]["temperature"], 0.15);
        assert_eq!(sent["options"]["num_predict"], 64);
        assert_eq!(sent["stream"], true);
    }

    #[test]
    fn empty_options_are_omitted_so_model_defaults_apply() {
        assert!(GenOpts::default().ollama_options().is_none());
        let o = GenOpts { temperature: None, max_tokens: Some(8) };
        let v = o.ollama_options().unwrap();
        assert_eq!(v["num_predict"], 8);
        assert!(v.get("temperature").is_none());
    }
}
