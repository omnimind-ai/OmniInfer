use super::*;

pub(super) fn print_warnings(payload: &Value) {
    if let Some(warnings) = payload.get("warnings").and_then(Value::as_array) {
        for warning in warnings.iter().filter_map(Value::as_str).take(2) {
            notice(&format!("Advisor: {warning}"), NoticeKind::Warning);
        }
    }
}

pub(super) fn print_help() {
    print_section("Help", "Conversation commands");
    let rows = [
        ("/backend", "switch the selected runtime"),
        ("/model", "load a different managed model"),
        (
            "/think",
            "toggle thinking mode; use /think on or /think off",
        ),
        (
            "/reasoning",
            "toggle visible reasoning; use /reasoning on or /reasoning off",
        ),
        ("/status", "show backend, model, and context usage"),
        ("/clear", "clear the terminal and redraw the chat header"),
        ("/help", "show this command reference"),
        ("/exit", "stop the OmniInfer service and leave the TUI"),
    ];
    for (name, description) in rows {
        print_kv(name, description);
    }
    println!();
}

pub(super) fn select_menu(
    title: &str,
    subtitle: &str,
    items: &[MenuItem],
    default_index: usize,
) -> Result<Option<usize>> {
    if items.is_empty() {
        return Ok(None);
    }
    print_section(title, subtitle);
    for (index, item) in items.iter().enumerate() {
        let marker = if item.selected { "*" } else { " " };
        let details = if item.details.is_empty() {
            String::new()
        } else {
            format!("  {}", item.details.join(" | "))
        };
        println!("{:>2}. [{}] {}{}", index + 1, marker, item.label, details);
    }
    println!("Press Enter to keep the default, or type q to cancel.");
    loop {
        let choice = prompt_default("Select", &(default_index + 1).to_string())?;
        if matches!(
            choice.trim().to_ascii_lowercase().as_str(),
            "q" | "quit" | "cancel" | "esc"
        ) {
            return Ok(None);
        }
        if let Ok(index) = choice.trim().parse::<usize>()
            && (1..=items.len()).contains(&index)
        {
            return Ok(Some(index - 1));
        }
        notice("Invalid selection.", NoticeKind::Warning);
    }
}

pub(super) fn prompt_default(label: &str, default: &str) -> Result<String> {
    if default.is_empty() {
        print!("{label}: ");
    } else {
        print!("{label} [{default}]: ");
    }
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let text = input.trim();
    if text.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(text.to_string())
    }
}

pub(super) fn print_header(title: &str, subtitle: &str) {
    println!("{title}");
    println!("{subtitle}");
    println!("{}", "-".repeat(64));
    println!();
}

pub(super) fn print_section(title: &str, subtitle: &str) {
    println!("{title}");
    if !subtitle.is_empty() {
        println!("{subtitle}");
    }
    println!("{}", "-".repeat(title.len().max(24)));
}

pub(super) fn print_chat_header(session: &ChatSession) {
    print_section("Chat", &format!("Backend: {}", session.backend));
    println!("Commands: /backend /model /think /reasoning /status /clear /help /exit");
    println!();
}

pub(super) fn print_kv(label: &str, value: &str) {
    println!("  {label}: {value}");
}

#[derive(Debug, Clone, Copy)]
pub(super) enum NoticeKind {
    Success,
    Warning,
}

pub(super) fn notice(message: &str, kind: NoticeKind) {
    let prefix = match kind {
        NoticeKind::Success => "ok",
        NoticeKind::Warning => "warn",
    };
    println!("  {prefix}: {message}");
}

pub(super) fn clear_screen() {
    print!("\x1b[2J\x1b[H");
    let _ = io::stdout().flush();
}

pub(super) fn is_interactive() -> bool {
    use std::io::IsTerminal;
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

pub(super) fn format_gib(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_f64)
        .map(|value| format!("{value:.2} GiB"))
        .unwrap_or_else(|| "-".to_string())
}

pub(super) fn memory_breakdown_text(breakdown: &Value) -> Option<String> {
    let fields = [
        ("weights", "weights_gib"),
        ("mmproj", "mmproj_gib"),
        ("kv", "kv_cache_gib"),
        ("act", "activation_gib"),
        ("runtime", "runtime_overhead_gib"),
    ];
    let parts = fields
        .iter()
        .filter_map(|(label, key)| {
            breakdown
                .get(*key)
                .and_then(Value::as_f64)
                .map(|value| format!("{label} {value:.2} GiB"))
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join(" | "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_breakdown_text_uses_expected_labels() {
        let breakdown = serde_json::json!({
            "weights_gib": 2.55,
            "kv_cache_gib": 0.25,
            "runtime_overhead_gib": 0.5
        });
        assert_eq!(
            memory_breakdown_text(&breakdown).as_deref(),
            Some("weights 2.55 GiB | kv 0.25 GiB | runtime 0.50 GiB")
        );
    }
}
