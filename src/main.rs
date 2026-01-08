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
#[command(version = "2.2.0")]
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
    "wss://aiassist-secure.replit.app/api/v1/pin/ws".to_string()
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
}

async fn chat_completion_ollama(
    base_url: &str,
    model: &str,
    messages: Vec<ChatMessage>,
) -> Result<OpenAIResponse, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));

    let request = OllamaChatRequest {
        model: model.to_string(),
        messages,
        stream: Some(false),
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
) -> Result<OpenAIResponse, String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

    let request = OpenAIChatRequest {
        model: model.to_string(),
        messages,
        stream: Some(false),
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
) -> Result<OpenAIResponse, String> {
    match api_mode {
        "openai" => chat_completion_openai(base_url, model, messages).await,
        _ => chat_completion_ollama(base_url, model, messages).await,
    }
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
    
    let result = chat_completion(base_url, model, messages, api_mode).await;
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
                                        
                                        info!("[#{}] Inference request: {} ({}) via {} [queued]", count, request_id, model, mode);
                                        
                                        let sem = semaphore.clone();
                                        let tx = tx.clone();
                                        
                                        tokio::spawn(async move {
                                            let _permit = sem.acquire().await.expect("semaphore closed");
                                            
                                            info!("[#{}] Starting inference for {}", count, request_id);
                                            let result = chat_completion(&uri, &model, messages, &mode).await;

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
