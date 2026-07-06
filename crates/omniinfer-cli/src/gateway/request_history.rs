use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde_json::{Map, Value, json};

const MAX_STRING_CHARS: usize = 12_000;
const MAX_HISTORY_LINES_SCANNED: usize = 10_000;

#[derive(Debug, Clone)]
pub(super) struct RequestHistoryRecord {
    pub(super) admin_id: Option<String>,
    pub(super) auth_kind: String,
    pub(super) method: String,
    pub(super) path: String,
    pub(super) model: Option<String>,
    pub(super) backend: Option<String>,
    pub(super) status: u16,
    pub(super) latency_ms: u64,
    pub(super) usage: Option<Value>,
    pub(super) metrics: Option<Value>,
    pub(super) request: Value,
    pub(super) response: Option<Value>,
    pub(super) error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct RequestHistoryQuery {
    pub(super) limit: usize,
    pub(super) model: Option<String>,
    pub(super) admin: Option<String>,
    pub(super) status: Option<String>,
}

pub(super) fn enabled() -> bool {
    std::env::var("OMNIINFER_REQUEST_HISTORY")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
        .unwrap_or(true)
}

pub(super) fn append_record(history_dir: PathBuf, record: RequestHistoryRecord) -> Result<()> {
    std::fs::create_dir_all(&history_dir)?;
    let now = unix_seconds();
    let entry = json!({
        "id": make_history_id(now),
        "ts": format_unix_seconds(now),
        "ts_unix": now,
        "admin_id": record.admin_id,
        "auth_kind": record.auth_kind,
        "method": record.method,
        "path": record.path,
        "model": record.model,
        "backend": record.backend,
        "status": record.status,
        "latency_ms": record.latency_ms,
        "usage": record.usage,
        "omniinfer_metrics": record.metrics,
        "request": sanitize_value(&record.request),
        "response": record.response.as_ref().map(sanitize_value),
        "error": record.error,
    });
    let path = history_dir.join(format!("{}.jsonl", date_utc(now)));
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, &entry)?;
    file.write_all(b"\n")?;
    Ok(())
}

pub(super) fn query_records(history_dir: &Path, query: RequestHistoryQuery) -> Result<Value> {
    let limit = query.limit.clamp(1, 500);
    let mut data = Vec::new();
    let mut scanned = 0usize;
    for path in history_files_newest_first(history_dir)? {
        let file = File::open(path)?;
        let mut lines = BufReader::new(file)
            .lines()
            .collect::<std::io::Result<Vec<_>>>()?;
        lines.reverse();
        for line in lines {
            if scanned >= MAX_HISTORY_LINES_SCANNED || data.len() >= limit {
                break;
            }
            scanned += 1;
            let Ok(entry) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if !matches_query(&entry, &query) {
                continue;
            }
            data.push(entry);
        }
        if scanned >= MAX_HISTORY_LINES_SCANNED || data.len() >= limit {
            break;
        }
    }
    Ok(json!({
        "object": "list",
        "data": data,
        "scanned": scanned,
        "limit": limit,
    }))
}

pub(super) fn get_record(history_dir: &Path, id: &str) -> Result<Option<Value>> {
    if id.trim().is_empty() {
        return Ok(None);
    }
    for path in history_files_newest_first(history_dir)? {
        let file = File::open(path)?;
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let Ok(entry) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if entry.get("id").and_then(Value::as_str) == Some(id) {
                return Ok(Some(entry));
            }
        }
    }
    Ok(None)
}

fn history_files_newest_first(history_dir: &Path) -> Result<Vec<PathBuf>> {
    if !history_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = std::fs::read_dir(history_dir)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("jsonl"))
        .collect::<Vec<_>>();
    paths.sort();
    paths.reverse();
    Ok(paths)
}

fn matches_query(entry: &Value, query: &RequestHistoryQuery) -> bool {
    if let Some(model) = query.model.as_deref()
        && entry.get("model").and_then(Value::as_str) != Some(model)
    {
        return false;
    }
    if let Some(admin) = query.admin.as_deref()
        && entry.get("admin_id").and_then(Value::as_str) != Some(admin)
    {
        return false;
    }
    if let Some(status) = query.status.as_deref() {
        let code = entry.get("status").and_then(Value::as_u64).unwrap_or(0);
        match status {
            "error" if code < 400 => return false,
            "ok" | "success" if !(200..400).contains(&code) => return false,
            other => {
                if let Ok(expected) = other.parse::<u64>()
                    && code != expected
                {
                    return false;
                }
            }
        }
    }
    true
}

fn sanitize_value(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sanitized = Map::new();
            for (key, value) in object {
                if is_sensitive_key(key) {
                    sanitized.insert(key.clone(), json!({"omitted": "sensitive"}));
                } else {
                    sanitized.insert(key.clone(), sanitize_value(value));
                }
            }
            Value::Object(sanitized)
        }
        Value::Array(items) => Value::Array(items.iter().map(sanitize_value).collect()),
        Value::String(text) => sanitize_string(text),
        _ => value.clone(),
    }
}

fn sanitize_string(text: &str) -> Value {
    if let Some((mime, encoded)) = parse_data_url(text) {
        return json!({
            "omitted": "data_url",
            "mime": mime,
            "encoded_chars": encoded.len(),
            "approx_bytes": encoded.len().saturating_mul(3) / 4,
        });
    }
    if text.chars().count() > MAX_STRING_CHARS {
        let truncated = text.chars().take(MAX_STRING_CHARS).collect::<String>();
        return json!({
            "truncated": true,
            "chars": text.chars().count(),
            "prefix": truncated,
        });
    }
    Value::String(text.to_string())
}

fn parse_data_url(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("data:")?;
    let (mime, data) = rest.split_once(";base64,")?;
    if mime.starts_with("image/") || mime.starts_with("audio/") || mime.starts_with("video/") {
        Some((mime, data))
    } else {
        None
    }
}

fn is_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    normalized.contains("authorization")
        || normalized.contains("api_key")
        || normalized.contains("apikey")
        || normalized.contains("access_token")
        || normalized.contains("secret")
        || normalized.contains("password")
}

fn make_history_id(now: u64) -> String {
    format!("hist_{now:x}_{:016x}", rand::random::<u64>())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn format_unix_seconds(seconds: u64) -> String {
    let (year, month, day) = date_parts_utc(seconds);
    let sod = seconds % 86_400;
    let hour = sod / 3_600;
    let minute = (sod % 3_600) / 60;
    let second = sod % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn date_utc(seconds: u64) -> String {
    let (year, month, day) = date_parts_utc(seconds);
    format!("{year:04}-{month:02}-{day:02}")
}

fn date_parts_utc(seconds: u64) -> (i32, u32, u32) {
    civil_from_days((seconds / 86_400) as i64)
}

fn civil_from_days(days_since_epoch: i64) -> (i32, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}

pub(super) fn query_from_pairs(pairs: BTreeMap<String, String>) -> RequestHistoryQuery {
    RequestHistoryQuery {
        limit: pairs
            .get("limit")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(50),
        model: pairs
            .get("model")
            .filter(|value| !value.is_empty())
            .cloned(),
        admin: pairs
            .get("admin")
            .filter(|value| !value.is_empty())
            .cloned(),
        status: pairs
            .get("status")
            .filter(|value| !value.is_empty())
            .cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_data_urls_and_sensitive_keys() {
        let value = json!({
            "Authorization": "Bearer secret",
            "messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
        });
        let sanitized = sanitize_value(&value);
        assert_eq!(sanitized["Authorization"]["omitted"], "sensitive");
        assert_eq!(
            sanitized["messages"][0]["content"][0]["image_url"]["url"]["omitted"],
            "data_url"
        );
    }

    #[test]
    fn formats_unix_date() {
        assert_eq!(format_unix_seconds(0), "1970-01-01T00:00:00Z");
        assert_eq!(date_utc(1_704_067_200), "2024-01-01");
    }
}
