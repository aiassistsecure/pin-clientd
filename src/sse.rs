//! PIN SSE transport — persistent downlink, independent HTTP uplink.
//!
//! Additive to the existing WebSocket path. The server selects per operator;
//! both run simultaneously during rollout. The signature contract is identical
//! to the WebSocket AUTH frame.

use crate::compute_signature;
use crate::Config;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Semaphore};

#[derive(Debug, Deserialize)]
#[expect(dead_code)]
struct SseAuthSuccess {
    #[serde(rename = "operatorId")]
    operator_id: String,
    #[serde(rename = "leaseUntil")]
    lease_until: f64,
    #[serde(rename = "streamToken")]
    stream_token: String,
}

/// One SSE event parsed from the downlink stream.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event_type: String,
    pub data: String,
}

/// Connect the persistent SSE downlink and return a channel of events.
///
/// The caller owns the event receiver; the connection lives until either the
/// server closes the stream or the caller drops the receiver.
pub async fn connect_downlink(
    config: &Config,
    _semaphore: Arc<Semaphore>,
) -> Result<(String, mpsc::UnboundedReceiver<SseEvent>), String> {
    let base = config.server_url.trim_end_matches("/ws");
    let sse_url = format!("{}/stream/connect", base.replace("wss://", "https://").replace("ws://", "http://"));

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock: {e}"))?
        .as_secs()
        .to_string();
    let signature = compute_signature(&config.client_id, &timestamp, &config.api_secret);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let response = client
        .get(&sse_url)
        .header("X-PIN-Client-Id", &config.client_id)
        .header("X-PIN-Timestamp", &timestamp)
        .header("X-PIN-Signature", &signature)
        .header("Accept", "text/event-stream")
        .send()
        .await
        .map_err(|e| format!("SSE connect: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("SSE connect HTTP {status}: {body}"));
    }

    let stream_token = response
        .headers()
        .get("X-PIN-Stream-Token")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .ok_or_else(|| "SSE response missing X-PIN-Stream-Token".to_string())?;

    let (event_tx, event_rx) = mpsc::unbounded_channel::<SseEvent>();
    let stream_token_clone = stream_token.clone();

    tokio::spawn(async move {
        use tokio_stream::StreamExt;
        let mut stream = response.bytes_stream();
        let mut buf = String::new();
        let mut event_type = String::new();
        let mut data = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("SSE downlink read error: {e}");
                    break;
                }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(idx) = buf.find("\n\n") {
                let frame = buf[..idx].to_string();
                buf = buf[idx + 2..].to_string();

                for line in frame.lines() {
                    if let Some(rest) = line.strip_prefix("event: ") {
                        event_type = rest.to_string();
                    } else if let Some(rest) = line.strip_prefix("data: ") {
                        if !data.is_empty() {
                            data.push('\n');
                        }
                        data.push_str(rest);
                    }
                }

                if !event_type.is_empty() {
                    let _ = event_tx.send(SseEvent {
                        event_type: std::mem::take(&mut event_type),
                        data: std::mem::take(&mut data),
                    });
                }
            }
        }
        tracing::info!("SSE downlink closed for stream_token={}", stream_token_clone);
    });

    Ok((stream_token, event_rx))
}

/// Sign and POST a JSON body to an SSE uplink endpoint.
async fn post_uplink(
    config: &Config,
    stream_token: &str,
    path: &str,
    body: &impl Serialize,
) -> Result<(), String> {
    let base = config.server_url.trim_end_matches("/ws");
    let url = format!(
        "{}/stream/{}",
        base.replace("wss://", "https://").replace("ws://", "http://"),
        path
    );

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock: {e}"))?
        .as_secs()
        .to_string();
    let signature = compute_signature(&config.client_id, &timestamp, &config.api_secret);

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let response = client
        .post(&url)
        .header("X-PIN-Client-Id", &config.client_id)
        .header("X-PIN-Timestamp", &timestamp)
        .header("X-PIN-Signature", &signature)
        .header("X-PIN-Stream-Token", stream_token)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("SSE uplink {path}: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("SSE uplink {path} HTTP {status}: {text}"));
    }

    Ok(())
}

/// Send a heartbeat lease renewal.
pub async fn heartbeat(
    config: &Config,
    stream_token: &str,
    models: Option<&[String]>,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct Empty {}
    post_uplink(config, stream_token, "heartbeat", &Empty {}).await?;
    if let Some(models) = models {
        let header_value = models.join(",");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("system clock: {e}"))?
            .as_secs()
            .to_string();
        let signature = compute_signature(&config.client_id, &timestamp, &config.api_secret);
        let base = config.server_url.trim_end_matches("/ws");
        let url = format!(
            "{}/stream/heartbeat",
            base.replace("wss://", "https://").replace("ws://", "http://")
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        let response = client
            .post(&url)
            .header("X-PIN-Client-Id", &config.client_id)
            .header("X-PIN-Timestamp", &timestamp)
            .header("X-PIN-Signature", &signature)
            .header("X-PIN-Stream-Token", stream_token)
            .header("X-PIN-Models", &header_value)
            .send()
            .await
            .map_err(|e| format!("SSE heartbeat models: {e}"))?;
        if !response.status().is_success() {
            return Err(format!("SSE heartbeat models HTTP {}", response.status()));
        }
    }
    Ok(())
}

/// Push an inference result to the server.
pub async fn inference_result(
    config: &Config,
    stream_token: &str,
    request_id: &str,
    result: &serde_json::Value,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct ResultBody<'a> {
        request_id: &'a str,
        result: &'a serde_json::Value,
    }
    post_uplink(
        config,
        stream_token,
        "result",
        &ResultBody { request_id, result },
    )
    .await
}

/// Push a streamed inference chunk to the server.
pub async fn inference_chunk(
    config: &Config,
    stream_token: &str,
    request_id: &str,
    index: u32,
    delta: &str,
    kind: &str,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct ChunkBody<'a> {
        request_id: &'a str,
        index: u32,
        delta: &'a str,
        kind: &'a str,
    }
    post_uplink(
        config,
        stream_token,
        "chunk",
        &ChunkBody { request_id, index, delta, kind },
    )
    .await
}
