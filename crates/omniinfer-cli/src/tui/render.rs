use super::*;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    style::{Color, Stylize, style},
    terminal::{self, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};

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
    pub(super) provider: String,
    pub(super) quant: String,
    pub(super) disk: String,
    pub(super) ctx: String,
    pub(super) fit: String,
    pub(super) backend: String,
    pub(super) evidence: String,
    pub(super) selected: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct ModelMenuContext {
    pub(super) hardware_lines: Vec<String>,
    pub(super) backend_line: String,
}

pub(super) fn select_model_menu(
    title: &str,
    subtitle: &str,
    items: &[ModelMenuItem],
    default_index: usize,
    context: &ModelMenuContext,
) -> Result<Option<usize>> {
    if items.is_empty() {
        return Ok(None);
    }
    let _terminal_mode = ModelPickerTerminalMode::enter()?;
    let mut selected = default_index.min(items.len().saturating_sub(1));
    let mut number_buffer = String::new();
    loop {
        redraw_model_menu(title, subtitle, context, items, selected, &number_buffer)?;
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

fn format_model_menu_for_width(
    items: &[ModelMenuItem],
    selected_index: Option<usize>,
    number_buffer: &str,
    terminal_width: usize,
) -> String {
    let columns = model_menu_columns(terminal_width);
    let mut output = String::new();
    output.push_str(&model_menu_row(
        "",
        "#",
        "Sel",
        "Model",
        &columns.header_values(),
        &columns,
    ));
    output.push('\n');
    output.push_str(&model_menu_row(
        "",
        "───",
        "───",
        &"─".repeat(columns.model),
        &columns.separator_values(),
        &columns,
    ));
    output.push('\n');
    for (index, item) in items.iter().enumerate() {
        output.push_str(&model_menu_row(
            if selected_index == Some(index) {
                ">"
            } else {
                ""
            },
            &(index + 1).to_string(),
            if item.selected { "*" } else { "" },
            &item.label,
            &columns.item_values(item),
            &columns,
        ));
        output.push('\n');
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

fn format_model_menu_screen_for_width(
    title: &str,
    subtitle: &str,
    context: &ModelMenuContext,
    items: &[ModelMenuItem],
    selected_index: Option<usize>,
    number_buffer: &str,
    terminal_width: usize,
) -> String {
    let content_width = panel_content_width(terminal_width);
    let mut output = String::new();
    output.push_str(&accent_line(&format_panel(
        "OmniInfer",
        "",
        &model_context_lines(context, terminal_width),
        terminal_width,
        PanelBodyMode::Wrapped,
    )));
    output.push('\n');
    output.push_str(&accent_line(&format_panel(
        title,
        subtitle,
        &format_model_menu_for_width(items, selected_index, number_buffer, content_width)
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>(),
        terminal_width,
        PanelBodyMode::Preformatted,
    )));
    output
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelMenuColumns {
    model: usize,
    optional: &'static [ModelMenuColumn],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelMenuColumn {
    kind: ModelMenuColumnKind,
    label: &'static str,
    width: usize,
    align: CellAlign,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelMenuColumnKind {
    Provider,
    Quant,
    Disk,
    Ctx,
    Fit,
    Backend,
    Evidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellAlign {
    Left,
    Right,
}

const MENU_COLUMNS_FULL: &[ModelMenuColumn] = &[
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Provider,
        label: "Provider",
        width: 12,
        align: CellAlign::Left,
    },
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Quant,
        label: "Quant",
        width: 8,
        align: CellAlign::Left,
    },
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Disk,
        label: "Disk",
        width: 8,
        align: CellAlign::Right,
    },
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Ctx,
        label: "Ctx",
        width: 6,
        align: CellAlign::Right,
    },
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Fit,
        label: "Fit",
        width: 6,
        align: CellAlign::Left,
    },
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Backend,
        label: "Backend",
        width: 18,
        align: CellAlign::Left,
    },
    ModelMenuColumn {
        kind: ModelMenuColumnKind::Evidence,
        label: "Evidence",
        width: 14,
        align: CellAlign::Left,
    },
];

impl ModelMenuColumns {
    fn header_values(&self) -> Vec<String> {
        self.optional
            .iter()
            .map(|column| column.label.to_string())
            .collect()
    }

    fn separator_values(&self) -> Vec<String> {
        self.optional
            .iter()
            .map(|column| "─".repeat(column.width))
            .collect()
    }

    fn item_values(&self, item: &ModelMenuItem) -> Vec<String> {
        self.optional
            .iter()
            .map(|column| match column.kind {
                ModelMenuColumnKind::Provider => fallback_dash(&item.provider),
                ModelMenuColumnKind::Quant => fallback_dash(&item.quant),
                ModelMenuColumnKind::Disk => fallback_dash(&item.disk),
                ModelMenuColumnKind::Ctx => fallback_dash(&item.ctx),
                ModelMenuColumnKind::Fit => fallback_dash(&item.fit),
                ModelMenuColumnKind::Backend => fallback_dash(&item.backend),
                ModelMenuColumnKind::Evidence => fallback_dash(&item.evidence),
            })
            .collect()
    }
}

fn model_menu_columns(terminal_width: usize) -> ModelMenuColumns {
    let optional = match terminal_width {
        0..=34 => &MENU_COLUMNS_FULL[..0],
        35..=47 => &MENU_COLUMNS_FULL[..1],
        48..=57 => &MENU_COLUMNS_FULL[..2],
        58..=65 => &MENU_COLUMNS_FULL[..3],
        66..=73 => &MENU_COLUMNS_FULL[..4],
        74..=91 => &MENU_COLUMNS_FULL[..5],
        92..=113 => &MENU_COLUMNS_FULL[..6],
        _ => MENU_COLUMNS_FULL,
    };
    let mut visible = optional;
    while terminal_width.saturating_sub(model_menu_fixed_width(visible)) < 8 && !visible.is_empty()
    {
        visible = &visible[..visible.len() - 1];
    }
    let model = terminal_width
        .saturating_sub(model_menu_fixed_width(visible))
        .max(4);
    ModelMenuColumns {
        model,
        optional: visible,
    }
}

fn model_menu_fixed_width(optional: &[ModelMenuColumn]) -> usize {
    let prefix = 12;
    prefix
        + optional
            .iter()
            .map(|column| 1 + column.width)
            .sum::<usize>()
}

fn model_menu_row(
    cursor: &str,
    number: &str,
    selected: &str,
    model: &str,
    values: &[String],
    columns: &ModelMenuColumns,
) -> String {
    let mut row = format!(
        "{:<2} {:>3}  {:<3} {:<model_width$}",
        truncate_cell_plain(cursor, 2),
        truncate_cell_plain(number, 3),
        truncate_cell_plain(selected, 3),
        truncate_cell_plain(model, columns.model),
        model_width = columns.model,
    );
    for (column, value) in columns.optional.iter().zip(values.iter()) {
        row.push(' ');
        let value = truncate_cell_plain(value, column.width);
        match column.align {
            CellAlign::Left => row.push_str(&format!("{value:<width$}", width = column.width)),
            CellAlign::Right => row.push_str(&format!("{value:>width$}", width = column.width)),
        }
    }
    row
}

fn redraw_model_menu(
    title: &str,
    subtitle: &str,
    context: &ModelMenuContext,
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
    let width = terminal::size()
        .ok()
        .map(|(width, _)| width as usize)
        .unwrap_or(120);
    let screen = format_model_menu_screen_for_width(
        title,
        subtitle,
        context,
        items,
        Some(selected),
        number_buffer,
        width,
    )
    .replace('\n', "\r\n");
    print!("{screen}");
    io::stdout().flush()?;
    Ok(())
}

fn model_context_lines(context: &ModelMenuContext, _terminal_width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    lines.extend(context.hardware_lines.iter().cloned());
    if !context.backend_line.is_empty() {
        lines.push(context.backend_line.clone());
    }
    if lines.is_empty() {
        lines.push("System details unavailable".to_string());
    }
    lines
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PanelBodyMode {
    Wrapped,
    Preformatted,
}

fn format_panel(
    title: &str,
    subtitle: &str,
    lines: &[String],
    terminal_width: usize,
    body_mode: PanelBodyMode,
) -> String {
    let content_width = panel_content_width(terminal_width);
    let panel_width = content_width + 4;
    let top = panel_top(title, panel_width);
    let bottom = format!("└{}┘", "─".repeat(panel_width.saturating_sub(2)));
    let mut output = String::new();
    output.push_str(&top);
    output.push('\n');
    if !subtitle.is_empty() {
        for line in wrap_text(subtitle, content_width) {
            output.push_str(&format!("│ {:<content_width$} │\n", line));
        }
        output.push_str(&format!("│ {:<content_width$} │\n", ""));
    }
    for line in lines {
        match body_mode {
            PanelBodyMode::Wrapped => {
                for wrapped in wrap_text(line, content_width) {
                    output.push_str(&format!("│ {:<content_width$} │\n", wrapped));
                }
            }
            PanelBodyMode::Preformatted => {
                output.push_str(&format!(
                    "│ {:<content_width$} │\n",
                    truncate_cell_plain(line, content_width)
                ));
            }
        }
    }
    output.push_str(&bottom);
    output
}

fn panel_top(title: &str, panel_width: usize) -> String {
    let content = format!(" {} ", truncate_cell(title, panel_width.saturating_sub(8)));
    let left = 2;
    let right = panel_width.saturating_sub(content.len() + left + 2);
    format!("┌{}{}{}┐", "─".repeat(left), content, "─".repeat(right))
}

fn panel_content_width(terminal_width: usize) -> usize {
    terminal_width.saturating_sub(4).clamp(12, 220)
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.len() <= width {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if word.len() > width {
            if !current.is_empty() {
                lines.push(current);
                current = String::new();
            }
            lines.push(truncate_cell(word, width));
        } else if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn accent_line(text: &str) -> String {
    let accent = Color::Rgb {
        r: 139,
        g: 92,
        b: 246,
    };
    text.lines()
        .map(|line| {
            if line.starts_with('┌') || line.starts_with('└') {
                style(line).with(accent).to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
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

fn truncate_cell_plain(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_model_menu_renders_core_columns() {
        let items = [ModelMenuItem {
            label: "qwen/Qwen3.5-4B-Q4_K_M.gguf".to_string(),
            provider: "qwen".to_string(),
            quant: "Q4_K_M".to_string(),
            disk: "2.31 GiB".to_string(),
            ctx: "32k".to_string(),
            fit: "good".to_string(),
            backend: "llama.cpp-linux-cuda".to_string(),
            evidence: "direct/high".to_string(),
            selected: true,
        }];
        let table = format_model_menu_for_width(&items, None, "", 140);
        assert!(table.contains("Model"));
        assert!(table.contains("Provider"));
        assert!(table.contains("Q4_K_M"));
        assert!(table.contains("32k"));
        assert!(table.contains("llama.cpp"));
        assert!(table.contains("direct"));
        assert!(table.contains("*"));
    }

    #[test]
    fn model_menu_screen_renders_context_panels() {
        let context = ModelMenuContext {
            hardware_lines: vec![
                "Host: linux x86_64 | CPU: 24 threads | RAM: 80.0 GiB free / 125.0 GiB total"
                    .to_string(),
                "GPU: NVIDIA RTX 3090 x8 | best free GPU 0: 23.0 GiB free / 24.0 GiB total"
                    .to_string(),
            ],
            backend_line: "Backend: llama.cpp-linux-cuda (installed, compatible)".to_string(),
        };
        let items = [ModelMenuItem {
            label: "qwen/model.gguf".to_string(),
            provider: "qwen".to_string(),
            quant: "Q4_K_M".to_string(),
            disk: "2 GiB".to_string(),
            ctx: "32k".to_string(),
            fit: "good".to_string(),
            backend: "llama.cpp-linux-cuda".to_string(),
            evidence: "direct/high".to_string(),
            selected: true,
        }];
        let screen = format_model_menu_screen_for_width(
            "OmniInfer",
            "Local model picker",
            &context,
            &items,
            Some(0),
            "",
            120,
        );
        assert!(screen.contains("Host: linux"));
        assert!(screen.contains("Backend: llama.cpp-linux-cuda"));
        assert!(screen.contains("Provider"));
        assert!(screen.contains('┌'));
        assert!(screen.contains('└'));
        assert!(!screen.contains("+--"));
    }

    #[test]
    fn format_model_menu_marks_cursor_and_prompt_buffer() {
        let items = [
            ModelMenuItem {
                label: "first.gguf".to_string(),
                provider: "local".to_string(),
                quant: "Q4_K_M".to_string(),
                disk: "2 GiB".to_string(),
                ctx: "32k".to_string(),
                fit: "good".to_string(),
                backend: "llama.cpp-linux-cuda".to_string(),
                evidence: "direct/high".to_string(),
                selected: false,
            },
            ModelMenuItem {
                label: "second.gguf".to_string(),
                provider: "local".to_string(),
                quant: "Q8_0".to_string(),
                disk: "4 GiB".to_string(),
                ctx: "128k".to_string(),
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
        let narrow = model_menu_columns(34);
        assert!(narrow.optional.is_empty());
        assert!(narrow.model >= 8);

        let medium = model_menu_columns(76);
        assert!(
            medium
                .optional
                .iter()
                .any(|column| column.kind == ModelMenuColumnKind::Fit)
        );
        assert!(
            !medium
                .optional
                .iter()
                .any(|column| column.kind == ModelMenuColumnKind::Backend)
        );

        let wide = model_menu_columns(160);
        assert_eq!(wide.optional.len(), MENU_COLUMNS_FULL.len());
        let row = model_menu_row("", "1", "", "model", &wide.header_values(), &wide);
        assert!(row.len() <= 160);
    }

    #[test]
    fn truncate_cell_marks_long_values() {
        assert_eq!(truncate_cell("abcdefg", 6), "abc...");
        assert_eq!(truncate_cell("abc", 4), "abc");
        assert_eq!(truncate_cell_plain("abcdefg", 6), "abcdef");
    }
}
