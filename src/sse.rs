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

/// Signed HTTP uplinks for one live session. Cheap to clone (`reqwest::Client`
/// is reference-counted internally), so the uplink can be moved onto its own
/// task while the main loop keeps reading the downlink.
///
/// THE CLIENT IS BUILT ONCE. Every uplink method used to call
/// `reqwest::Client::builder().build()` per POST, which discards the connection
/// pool and pays a fresh TLS handshake for every streamed token. Measured
/// against the live gateway: 63.6 ms cold vs 49.1 ms on a warm connection.
#[derive(Clone)]
pub struct Uplink {
    config: Config,
    stream_token: String,
    client: reqwest::Client,
}

/// A live SSE session: one downlink reader plus signed HTTP uplinks.
pub struct Session {
    uplink: Uplink,
    events: mpsc::UnboundedReceiver<SseEvent>,
}

impl Session {
    /// A handle for the uplink half, for code that sends but never receives.
    ///
    /// This is what lets results and chunks leave on a dedicated task instead
    /// of inside the `select!` that also owns heartbeat and job intake.
    pub fn uplink(&self) -> Uplink {
        self.uplink.clone()
    }
}

impl Session {
    /// Open the persistent downlink. No total request timeout — a 60s deadline
    /// would close an otherwise healthy stream the first time inference ran long.
    pub async fn connect(config: &Config) -> Result<Self, String> {
        let sse_url = format!("{}/stream/connect", http_base(&config.server_url));
        let timestamp = now_secs()?;
        let signature = compute_signature(&config.client_id, &timestamp, &config.api_secret);

        // ONE client for the whole session: the downlink GET and every
        // subsequent uplink POST share this pool, so uplinks reuse a warm
        // connection instead of handshaking per token.
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
            uplink: Uplink {
                config: config.clone(),
                stream_token,
                client,
            },
            events: event_rx,
        })
    }

    pub fn token_preview(&self) -> &str {
        self.uplink.token_preview()
    }

    pub async fn recv(&mut self) -> Option<SseEvent> {
        self.events.recv().await
    }

    /// Renew the application lease. Optionally re-assert the advertised models.
    pub async fn heartbeat(&self, models: Option<&[String]>) -> Result<(), String> {
        self.uplink.heartbeat(models).await
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
        self.uplink
            .register_node(
                alias,
                models,
                capacity,
                region,
                price_per_thousand_tokens,
                interview_model,
                api_mode,
            )
            .await
    }
}

impl Uplink {
    pub fn token_preview(&self) -> &str {
        let n = self.stream_token.len().min(8);
        &self.stream_token[..n]
    }

    async fn post(&self, path: &str, body: &impl Serialize) -> Result<(), String> {
        let url = format!("{}/stream/{path}", http_base(&self.config.server_url));
        let timestamp = now_secs()?;
        let signature = compute_signature(&self.config.client_id, &timestamp, &self.config.api_secret);
        let response = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(30))
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
        let response = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(30))
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
        self.send_uplink_json_value(&parsed).await
    }

    /// Same routing, for a payload that is already decoded — the batching path
    /// parses once and coalesces before sending, so it must not re-serialise
    /// just to be re-parsed here.
    pub async fn send_uplink_json_value(&self, parsed: &serde_json::Value) -> Result<(), String> {
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
            | "INTERVIEW_RESULT" => self.inference_result(request_id, parsed).await,
            other => Err(format!("unhandled SSE uplink type {other}")),
        }
    }
}

/// True when this daemon should prefer SSE over WebSocket.
pub fn prefers_sse(config: &Config) -> bool {
    config.transport != "ws"
}

/// Upper bound on coalesced delta bytes in a single chunk POST. Keeps a very
/// long generation from assembling one enormous body.
pub const MAX_COALESCED_DELTA_BYTES: usize = 32 * 1024;

/// (request_id, kind, delta) when `v` is an INFERENCE_CHUNK, else None.
fn chunk_parts(v: &serde_json::Value) -> Option<(&str, &str, &str)> {
    if v.get("type").and_then(|t| t.as_str()) != Some("INFERENCE_CHUNK") {
        return None;
    }
    Some((
        v.get("request_id").and_then(|x| x.as_str()).unwrap_or(""),
        v.get("kind").and_then(|x| x.as_str()).unwrap_or("content"),
        v.get("delta").and_then(|x| x.as_str()).unwrap_or(""),
    ))
}

/// Fold `next` into `acc` when both are chunks of the same request and kind.
///
/// WHY THIS IS SAFE, AND WHY IT IS NOT REORDERING ANYTHING. The gateway's
/// `push_chunk` appends each delta onto the consumer's queue in ARRIVAL order;
/// `index` rides along as metadata and is never used to reassemble. So sending
/// "Hel" then "lo" is indistinguishable from sending "Hello" with the first
/// index. Only ADJACENT chunks are merged, and only when request_id and kind
/// both match, so a `thinking`→`content` transition still lands as its own
/// POST and two interleaved requests never blend.
///
/// Returns true when the fold happened; `next` is untouched and must still be
/// sent when it returns false.
fn try_fold_chunk(acc: &mut serde_json::Value, next: &serde_json::Value) -> bool {
    let (Some((a_req, a_kind, a_delta)), Some((b_req, b_kind, b_delta))) =
        (chunk_parts(acc), chunk_parts(next))
    else {
        return false;
    };
    if a_req != b_req || a_kind != b_kind {
        return false;
    }
    if a_delta.len() + b_delta.len() > MAX_COALESCED_DELTA_BYTES {
        return false;
    }
    let merged = format!("{a_delta}{b_delta}");
    acc["delta"] = serde_json::Value::String(merged);
    true
}

/// Drain a batch of queued uplink messages into the fewest POST bodies that
/// carry identical information.
///
/// This is the fix for the real cost: ONE HTTP ROUND TRIP PER TOKEN, awaited
/// in the same `select!` that owned the 15 s heartbeat. A 1417-token stream
/// meant 1417 sequential POSTs (~70 s at a measured 49 ms warm), during which
/// no heartbeat could go out. AiAS's reaper evicts an operator whose heartbeat
/// is older than 60 s, so a busy operator removed ITSELF from the online set
/// and the gateway then answered "No operators available" for a model that
/// node was actively serving.
///
/// Non-chunk messages (results, TTS, interviews) always stand alone.
pub fn coalesce_uplink_batch(batch: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    for msg in batch {
        // Not a `match` guard: a binding is immutable until the guard ends, so
        // the fold cannot borrow `acc` mutably from inside one.
        let folded = match out.last_mut() {
            Some(acc) => try_fold_chunk(acc, &msg),
            None => false,
        };
        if !folded {
            out.push(msg);
        }
    }
    out
}

#[cfg(test)]
mod coalesce_tests {
    use super::*;
    use serde_json::json;

    fn chunk(req: &str, index: u32, delta: &str, kind: &str) -> serde_json::Value {
        json!({"type":"INFERENCE_CHUNK","request_id":req,"index":index,
               "delta":delta,"kind":kind})
    }

    #[test]
    fn adjacent_chunks_of_one_request_become_one_post() {
        let out = coalesce_uplink_batch(vec![
            chunk("r1", 0, "Hel", "content"),
            chunk("r1", 1, "lo", "content"),
            chunk("r1", 2, " world", "content"),
        ]);
        assert_eq!(out.len(), 1, "three chunks collapse to one POST");
        assert_eq!(out[0]["delta"], "Hello world");
        assert_eq!(out[0]["index"], 0, "keeps the first index");
    }

    #[test]
    fn a_thousand_chunks_collapse() {
        let batch: Vec<_> = (0..1000).map(|i| chunk("r1", i, "x", "content")).collect();
        let out = coalesce_uplink_batch(batch);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["delta"].as_str().unwrap().len(), 1000);
    }

    #[test]
    fn thinking_and_content_never_blend() {
        let out = coalesce_uplink_batch(vec![
            chunk("r1", 0, "reason", "thinking"),
            chunk("r1", 1, "answer", "content"),
        ]);
        assert_eq!(out.len(), 2, "kind transition forces a boundary");
        assert_eq!(out[0]["kind"], "thinking");
        assert_eq!(out[1]["kind"], "content");
    }

    #[test]
    fn two_requests_never_blend() {
        let out = coalesce_uplink_batch(vec![
            chunk("r1", 0, "a", "content"),
            chunk("r2", 0, "b", "content"),
            chunk("r1", 1, "c", "content"),
        ]);
        assert_eq!(out.len(), 3, "interleaved requests stay separate");
        assert_eq!(out[0]["request_id"], "r1");
        assert_eq!(out[1]["request_id"], "r2");
        assert_eq!(out[2]["request_id"], "r1");
    }

    #[test]
    fn results_are_never_folded_into_chunks() {
        let result = json!({"type":"INFERENCE_RESPONSE","request_id":"r1",
                            "result":{"ok":true}});
        let out = coalesce_uplink_batch(vec![
            chunk("r1", 0, "a", "content"),
            result.clone(),
            chunk("r1", 1, "b", "content"),
        ]);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1]["type"], "INFERENCE_RESPONSE");
    }

    #[test]
    fn two_results_stay_two_posts() {
        let a = json!({"type":"INFERENCE_RESPONSE","request_id":"r1"});
        let b = json!({"type":"INFERENCE_RESPONSE","request_id":"r2"});
        assert_eq!(coalesce_uplink_batch(vec![a, b]).len(), 2);
    }

    #[test]
    fn the_byte_cap_forces_a_boundary() {
        let big = "x".repeat(MAX_COALESCED_DELTA_BYTES - 1);
        let out = coalesce_uplink_batch(vec![
            chunk("r1", 0, &big, "content"),
            chunk("r1", 1, "yy", "content"),
        ]);
        assert_eq!(out.len(), 2, "cap respected rather than one giant body");
    }

    #[test]
    fn no_delta_is_lost_or_reordered() {
        // The property that matters: concatenating the coalesced output must
        // equal concatenating the input, in order.
        let batch: Vec<_> = (0..200)
            .map(|i| chunk("r1", i, &format!("<{i}>"), "content"))
            .collect();
        let expected: String = (0..200).map(|i| format!("<{i}>")).collect();
        let got: String = coalesce_uplink_batch(batch)
            .iter()
            .map(|m| m["delta"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn an_empty_batch_is_empty() {
        assert!(coalesce_uplink_batch(vec![]).is_empty());
    }
}
