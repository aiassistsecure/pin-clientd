//! PIN SSE transport — the live operator path.
//!
//! Persistent HTTP downlink for jobs and control. Independent authenticated
//! POSTs for heartbeat, node registration, inference chunks, and results.
//! WebSocket stays in main.rs as the compatibility fallback.
//!
//! The signature contract is identical to the WebSocket AUTH frame.

use crate::compute_signature;
use crate::Config;
use serde::Serialize;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

/// One SSE event parsed from the downlink stream.
#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event_type: String,
    pub data: String,
}

impl SseEvent {
    /// Payload the server put on the wire. Comment/keepalive frames have no data.
    pub fn payload(&self) -> &str {
        if self.data.is_empty() {
            &self.event_type
        } else {
            &self.data
        }
    }
}

fn http_base(server_url: &str) -> String {
    server_url
        .trim_end_matches("/ws")
        .replace("wss://", "https://")
        .replace("ws://", "http://")
}

fn now_secs() -> Result<String, String> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock: {e}"))?
        .as_secs()
        .to_string())
}

/// A live SSE session: one downlink reader plus signed HTTP uplinks.
pub struct Session {
    config: Config,
    stream_token: String,
    events: mpsc::UnboundedReceiver<SseEvent>,
}

impl Session {
    /// Open the persistent downlink. No total request timeout — a 60s deadline
    /// would close an otherwise healthy stream the first time inference ran long.
    pub async fn connect(config: &Config) -> Result<Self, String> {
        let sse_url = format!("{}/stream/connect", http_base(&config.server_url));
        let timestamp = now_secs()?;
        let signature = compute_signature(&config.client_id, &timestamp, &config.api_secret);

        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
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
        let token_for_log = stream_token.clone();
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
                    if !event_type.is_empty() || !data.is_empty() {
                        let _ = event_tx.send(SseEvent {
                            event_type: std::mem::take(&mut event_type),
                            data: std::mem::take(&mut data),
                        });
                    }
                }
            }
            tracing::info!("SSE downlink closed for stream_token={token_for_log}");
        });

        Ok(Self {
            config: config.clone(),
            stream_token,
            events: event_rx,
        })
    }

    pub fn token_preview(&self) -> &str {
        let n = self.stream_token.len().min(8);
        &self.stream_token[..n]
    }

    pub async fn recv(&mut self) -> Option<SseEvent> {
        self.events.recv().await
    }

    async fn post(&self, path: &str, body: &impl Serialize) -> Result<(), String> {
        let url = format!("{}/stream/{path}", http_base(&self.config.server_url));
        let timestamp = now_secs()?;
        let signature = compute_signature(&self.config.client_id, &timestamp, &self.config.api_secret);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        let response = client
            .post(&url)
            .header("X-PIN-Client-Id", &self.config.client_id)
            .header("X-PIN-Timestamp", &timestamp)
            .header("X-PIN-Signature", &signature)
            .header("X-PIN-Stream-Token", &self.stream_token)
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

    /// Renew the application lease. Optionally re-assert the advertised model set.
    pub async fn heartbeat(&self, models: Option<&[String]>) -> Result<(), String> {
        #[derive(Serialize)]
        struct Empty {}
        self.post("heartbeat", &Empty {}).await?;
        let Some(models) = models else { return Ok(()); };
        if models.is_empty() {
            return Ok(());
        }

        let url = format!("{}/stream/heartbeat", http_base(&self.config.server_url));
        let timestamp = now_secs()?;
        let signature = compute_signature(&self.config.client_id, &timestamp, &self.config.api_secret);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        let response = client
            .post(&url)
            .header("X-PIN-Client-Id", &self.config.client_id)
            .header("X-PIN-Timestamp", &timestamp)
            .header("X-PIN-Signature", &signature)
            .header("X-PIN-Stream-Token", &self.stream_token)
            .header("X-PIN-Models", models.join(","))
            .send()
            .await
            .map_err(|e| format!("SSE heartbeat models: {e}"))?;
        if !response.status().is_success() {
            return Err(format!("SSE heartbeat models HTTP {}", response.status()));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn register_node(
        &self,
        alias: &str,
        models: &[String],
        capacity: u32,
        region: &str,
        price_per_thousand_tokens: f64,
        interview_model: Option<&str>,
        api_mode: &str,
    ) -> Result<(), String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct RegisterBody<'a> {
            alias: &'a str,
            models: &'a [String],
            capacity: u32,
            region: &'a str,
            price_per_thousand_tokens: f64,
            #[serde(skip_serializing_if = "Option::is_none")]
            interview_model: Option<&'a str>,
            api_mode: &'a str,
        }
        self.post(
            "register",
            &RegisterBody {
                alias,
                models,
                capacity,
                region,
                price_per_thousand_tokens,
                interview_model,
                api_mode,
            },
        )
        .await
    }

    pub async fn inference_result(
        &self,
        request_id: &str,
        result: &serde_json::Value,
    ) -> Result<(), String> {
        #[derive(Serialize)]
        struct ResultBody<'a> {
            request_id: &'a str,
            result: &'a serde_json::Value,
        }
        self.post("result", &ResultBody { request_id, result }).await
    }

    pub async fn inference_chunk(
        &self,
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
        self.post(
            "chunk",
            &ChunkBody {
                request_id,
                index,
                delta,
                kind,
            },
        )
        .await
    }

    /// Route one worker JSON payload (chunk, result, interview, TTS) over HTTP.
    pub async fn send_uplink_json(&self, json: &str) -> Result<(), String> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("uplink json: {e}"))?;
        let msg_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let request_id = parsed
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        match msg_type {
            "INFERENCE_CHUNK" => {
                let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                let delta = parsed.get("delta").and_then(|v| v.as_str()).unwrap_or("");
                let kind = parsed.get("kind").and_then(|v| v.as_str()).unwrap_or("content");
                self.inference_chunk(request_id, index, delta, kind).await
            }
            "INFERENCE_RESPONSE"
            | "INFERENCE_ERROR"
            | "TTS_RESPONSE"
            | "TTS_ERROR"
            | "INTERVIEW_RESULT" => {
                // SEND THE PAYLOAD, NOT THE ENVELOPE.
                //
                // This passed `&parsed` -- the whole message, `type` and
                // `request_id` included -- as the `result` field, so the
                // server received a completion wrapped twice. It unwrapped
                // once, read `choices` off the envelope that remained, got
                // `[]`, and the caller got:
                //
                //     Inference result received: ['request_id','result','type']
                //     Provider API error: list index out of range
                //     POST /v1/chat/completions 500
                //
                // The WebSocket path never had this bug: the server unwraps
                // the frame there, so exactly one envelope was ever intended.
                //
                // Messages carrying their payload inline rather than under
                // `result` still send the whole object, minus the routing
                // fields the server supplies itself.
                self.inference_result(request_id, &uplink_payload(&parsed)).await
            }
            other => Err(format!("unhandled SSE uplink type {other}")),
        }
    }
}

/// The payload to put in a result uplink's `result` field.
///
/// Pure, and separate from the request, because the bug it fixes was a
/// one-word mistake that a test could have caught: the caller passed the whole
/// parsed envelope, so the server received the completion wrapped twice, read
/// `choices` off the outer envelope, got `[]`, and answered 500.
///
/// A message that nests its payload under `result` contributes that. One that
/// carries its payload inline contributes itself, minus the routing fields the
/// server already has from the request.
pub fn uplink_payload(parsed: &serde_json::Value) -> serde_json::Value {
    match parsed.get("result") {
        Some(inner) => inner.clone(),
        None => {
            let mut stripped = parsed.clone();
            if let Some(obj) = stripped.as_object_mut() {
                obj.remove("type");
                obj.remove("request_id");
            }
            stripped
        }
    }
}

/// True when this daemon should prefer SSE over WebSocket.
pub fn prefers_sse(config: &Config) -> bool {
    config.transport != "ws"
}
