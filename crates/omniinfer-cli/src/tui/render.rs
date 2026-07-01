use super::*;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{self, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ModelMenuItem {
    pub(super) label: String,
    pub(super) quant: String,
    pub(super) size: String,
    pub(super) fit: String,
    pub(super) backend: String,
    pub(super) evidence: String,
    pub(super) selected: bool,
}

pub(super) fn select_model_menu(
    title: &str,
    subtitle: &str,
    items: &[ModelMenuItem],
    default_index: usize,
) -> Result<Option<usize>> {
    if items.is_empty() {
        return Ok(None);
    }
    let _terminal_mode = ModelPickerTerminalMode::enter()?;
    let mut selected = default_index.min(items.len().saturating_sub(1));
    let mut number_buffer = String::new();
    loop {
        redraw_model_menu(title, subtitle, items, selected, &number_buffer)?;
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
                number_buffer.clear();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(items.len().saturating_sub(1));
                number_buffer.clear();
            }
            KeyCode::PageUp => {
                selected = selected.saturating_sub(10);
                number_buffer.clear();
            }
            KeyCode::PageDown => {
                selected = (selected + 10).min(items.len().saturating_sub(1));
                number_buffer.clear();
            }
            KeyCode::Home => {
                selected = 0;
                number_buffer.clear();
            }
            KeyCode::End => {
                selected = items.len().saturating_sub(1);
                number_buffer.clear();
            }
            KeyCode::Enter => {
                if let Some(index) = buffered_model_index(&number_buffer, items.len()) {
                    println!();
                    return Ok(Some(index));
                }
                println!();
                return Ok(Some(selected));
            }
            KeyCode::Backspace => {
                number_buffer.pop();
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                println!();
                return Ok(None);
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                anyhow::bail!("Interrupted.");
            }
            KeyCode::Char(ch)
                if key.modifiers.is_empty() && ch.is_ascii_digit() && number_buffer.len() < 4 =>
            {
                number_buffer.push(ch);
                if let Some(index) = buffered_model_index(&number_buffer, items.len()) {
                    selected = index;
                }
            }
            _ => {}
        }
    }
}

pub(super) fn format_model_menu_with_cursor(
    items: &[ModelMenuItem],
    selected_index: Option<usize>,
    number_buffer: &str,
) -> String {
    let width = terminal::size()
        .ok()
        .map(|(width, _)| width as usize)
        .unwrap_or(120);
    format_model_menu_for_width(items, selected_index, number_buffer, width)
}

fn format_model_menu_for_width(
    items: &[ModelMenuItem],
    selected_index: Option<usize>,
    number_buffer: &str,
    terminal_width: usize,
) -> String {
    let columns = model_menu_columns(terminal_width);
    let mut output = String::new();
    output.push_str(&format!(
        "{:<2} {:>3}  {:<3} {:<model_width$} {:<quant_width$} {:>8} {:<6} {:<backend_width$} {:<evidence_width$}\n",
        "",
        "#",
        "Sel",
        "Model",
        "Quant",
        "Size",
        "Fit",
        "Backend",
        "Evidence",
        model_width = columns.model,
        quant_width = columns.quant,
        backend_width = columns.backend,
        evidence_width = columns.evidence,
    ));
    output.push_str(&format!(
        "{:<2} {:>3}  {:<3} {:<model_width$} {:<quant_width$} {:>8} {:<6} {:<backend_width$} {:<evidence_width$}\n",
        "",
        "---",
        "---",
        "-".repeat(columns.model),
        "-".repeat(columns.quant),
        "-".repeat(8),
        "-".repeat(6),
        "-".repeat(columns.backend),
        "-".repeat(columns.evidence),
        model_width = columns.model,
        quant_width = columns.quant,
        backend_width = columns.backend,
        evidence_width = columns.evidence,
    ));
    for (index, item) in items.iter().enumerate() {
        output.push_str(&format!(
            "{:<2} {:>3}  {:<3} {:<model_width$} {:<quant_width$} {:>8} {:<6} {:<backend_width$} {:<evidence_width$}\n",
            if selected_index == Some(index) { ">" } else { "" },
            index + 1,
            if item.selected { "*" } else { "" },
            truncate_cell(&item.label, columns.model),
            truncate_cell(&fallback_dash(&item.quant), columns.quant),
            truncate_cell(&fallback_dash(&item.size), 8),
            truncate_cell(&fallback_dash(&item.fit), 6),
            truncate_cell(&fallback_dash(&item.backend), columns.backend),
            truncate_cell(&fallback_dash(&item.evidence), columns.evidence),
            model_width = columns.model,
            quant_width = columns.quant,
            backend_width = columns.backend,
            evidence_width = columns.evidence,
        ));
    }
    output.push('\n');
    output.push_str(
        "Use Up/Down or j/k to move, PgUp/PgDn to jump, Enter to load, q/Esc to cancel.\n",
    );
    if number_buffer.is_empty() {
        output.push_str("Select: ");
    } else {
        output.push_str(&format!("Select: {number_buffer}"));
    }
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelMenuColumns {
    model: usize,
    quant: usize,
    backend: usize,
    evidence: usize,
}

fn model_menu_columns(terminal_width: usize) -> ModelMenuColumns {
    // Fixed columns and separators consume 47 cells:
    // cursor, number, selected marker, size, fit, and inter-column spaces.
    let available = terminal_width.saturating_sub(47);
    let evidence = 11.min(available.saturating_sub(18));
    let backend = 18.min(available.saturating_sub(evidence + 14));
    let quant = 7.min(available.saturating_sub(evidence + backend + 8));
    let model = available.saturating_sub(evidence + backend + quant).max(16);
    ModelMenuColumns {
        model,
        quant: quant.max(5),
        backend: backend.max(8),
        evidence: evidence.max(6),
    }
}

fn redraw_model_menu(
    title: &str,
    subtitle: &str,
    items: &[ModelMenuItem],
    selected: usize,
    number_buffer: &str,
) -> Result<()> {
    let mut stdout = io::stdout();
    execute!(
        stdout,
        cursor::MoveTo(0, 0),
        terminal::Clear(ClearType::All)
    )?;
    let mut screen = String::new();
    screen.push_str(title);
    screen.push_str("\r\n");
    if !subtitle.is_empty() {
        screen.push_str(subtitle);
        screen.push_str("\r\n");
    }
    screen.push_str(&"-".repeat(title.len().max(24)));
    screen.push_str("\r\n");
    screen.push_str(
        &format_model_menu_with_cursor(items, Some(selected), number_buffer).replace('\n', "\r\n"),
    );
    print!("{screen}");
    io::stdout().flush()?;
    Ok(())
}

fn buffered_model_index(number_buffer: &str, item_count: usize) -> Option<usize> {
    let raw = number_buffer.parse::<usize>().ok()?;
    if (1..=item_count).contains(&raw) {
        Some(raw - 1)
    } else {
        None
    }
}

struct ModelPickerTerminalMode;

impl ModelPickerTerminalMode {
    fn enter() -> Result<Self> {
        let mut stdout = io::stdout();
        terminal::enable_raw_mode()?;
        execute!(stdout, EnterAlternateScreen, cursor::Hide)?;
        Ok(Self)
    }
}

impl Drop for ModelPickerTerminalMode {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = execute!(
            stdout,
            terminal::Clear(ClearType::All),
            cursor::MoveTo(0, 0),
            cursor::Show,
            LeaveAlternateScreen
        );
        let _ = terminal::disable_raw_mode();
    }
}

fn fallback_dash(value: &str) -> String {
    if value.trim().is_empty() {
        "-".to_string()
    } else {
        value.to_string()
    }
}

fn truncate_cell(value: &str, width: usize) -> String {
    let mut chars = value.chars();
    let mut text = String::new();
    for _ in 0..width {
        let Some(ch) = chars.next() else {
            return text;
        };
        text.push(ch);
    }
    if chars.next().is_none() || width <= 3 {
        return text;
    }
    let mut truncated = text
        .chars()
        .take(width.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
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

    #[test]
    fn format_model_menu_renders_core_columns() {
        let items = [ModelMenuItem {
            label: "qwen/Qwen3.5-4B-Q4_K_M.gguf".to_string(),
            quant: "Q4_K_M".to_string(),
            size: "2.31 GiB".to_string(),
            fit: "good".to_string(),
            backend: "llama.cpp-linux-cuda".to_string(),
            evidence: "direct/high".to_string(),
            selected: true,
        }];
        let table = format_model_menu_for_width(&items, None, "", 140);
        assert!(table.contains("Model"));
        assert!(table.contains("Q4_K_M"));
        assert!(table.contains("llama.cpp"));
        assert!(table.contains("direct/high"));
        assert!(table.contains("*"));
    }

    #[test]
    fn format_model_menu_marks_cursor_and_prompt_buffer() {
        let items = [
            ModelMenuItem {
                label: "first.gguf".to_string(),
                quant: "Q4_K_M".to_string(),
                size: "2 GiB".to_string(),
                fit: "good".to_string(),
                backend: "llama.cpp-linux-cuda".to_string(),
                evidence: "direct/high".to_string(),
                selected: false,
            },
            ModelMenuItem {
                label: "second.gguf".to_string(),
                quant: "Q8_0".to_string(),
                size: "4 GiB".to_string(),
                fit: "good".to_string(),
                backend: "llama.cpp-linux-cuda".to_string(),
                evidence: "direct/high".to_string(),
                selected: true,
            },
        ];
        let table = format_model_menu_for_width(&items, Some(1), "2", 120);
        assert!(table.contains(">    2  *"));
        assert!(table.contains("Select: 2"));
        assert!(table.contains("Up/Down"));
    }

    #[test]
    fn buffered_model_index_uses_one_based_numbers() {
        assert_eq!(buffered_model_index("1", 3), Some(0));
        assert_eq!(buffered_model_index("3", 3), Some(2));
        assert_eq!(buffered_model_index("0", 3), None);
        assert_eq!(buffered_model_index("4", 3), None);
    }

    #[test]
    fn model_menu_columns_fit_narrow_terminals() {
        let columns = model_menu_columns(100);
        assert!(columns.model >= 16);
        assert!(columns.quant >= 5);
        assert!(columns.backend >= 8);
        assert!(columns.evidence >= 6);
        assert!(columns.model + columns.quant + columns.backend + columns.evidence <= 53);
    }

    #[test]
    fn truncate_cell_marks_long_values() {
        assert_eq!(truncate_cell("abcdefg", 6), "abc...");
        assert_eq!(truncate_cell("abc", 4), "abc");
    }
}
