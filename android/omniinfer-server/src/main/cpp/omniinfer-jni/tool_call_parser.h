#pragma once

// Table-driven tool call parser for MNN backend.
// Extracts tool calls from raw model output and returns OpenAI-compatible JSON.
// Adding a new model format: add a parse function + one entry in kFormats[].

#include <string>
#include <sstream>

namespace omniinfer {
namespace tool_parser {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Escape a string for embedding in JSON output.
static std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else out += c;
  }
  return out;
}

// Trim leading/trailing whitespace.
static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\n\r");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\n\r");
  return s.substr(a, b - a + 1);
}

// ---------------------------------------------------------------------------
// Qwen XML format (Qwen3.5, Qwen2.5):
//   <tool_call>
//   <function=function_name>
//   <parameter=key>
//   value
//   </parameter>
//   </function>
//   </tool_call>
// ---------------------------------------------------------------------------

static std::string parse_qwen_tool_calls(const std::string& output) {
  std::ostringstream ss;
  int count = 0;
  size_t pos = 0;

  while (pos < output.size()) {
    size_t tc_start = output.find("<tool_call>", pos);
    if (tc_start == std::string::npos) break;
    size_t tc_end = output.find("</tool_call>", tc_start);
    if (tc_end == std::string::npos) break;

    std::string block = output.substr(tc_start + 11, tc_end - tc_start - 11);

    // Extract function name: <function=name>
    size_t fn_start = block.find("<function=");
    if (fn_start == std::string::npos) { pos = tc_end + 12; continue; }
    size_t fn_name_start = fn_start + 10;
    size_t fn_name_end = block.find('>', fn_name_start);
    if (fn_name_end == std::string::npos) { pos = tc_end + 12; continue; }
    std::string name = block.substr(fn_name_start, fn_name_end - fn_name_start);

    // Extract parameters: <parameter=key>\nvalue\n</parameter>
    std::ostringstream args;
    args << "{";
    size_t param_pos = fn_name_end;
    bool first_param = true;
    while (param_pos < block.size()) {
      size_t p_start = block.find("<parameter=", param_pos);
      if (p_start == std::string::npos) break;
      size_t key_start = p_start + 11;
      size_t key_end = block.find('>', key_start);
      if (key_end == std::string::npos) break;
      std::string key = block.substr(key_start, key_end - key_start);

      size_t val_start = key_end + 1;
      if (val_start < block.size() && block[val_start] == '\n') val_start++;
      size_t p_end = block.find("</parameter>", val_start);
      if (p_end == std::string::npos) break;
      std::string value = block.substr(val_start, p_end - val_start);
      while (!value.empty() && value.back() == '\n') value.pop_back();

      if (!first_param) args << ",";
      args << "\"" << json_escape(key) << "\":";

      // Heuristic: if value looks like JSON literal, emit raw; otherwise quote as string.
      std::string tv = trim(value);
      if (!tv.empty() && (tv[0] == '{' || tv[0] == '[' || tv[0] == '-' ||
          (tv[0] >= '0' && tv[0] <= '9') ||
          tv == "true" || tv == "false" || tv == "null")) {
        args << tv;
      } else {
        args << "\"" << json_escape(value) << "\"";
      }
      first_param = false;
      param_pos = p_end + 12;
    }
    args << "}";

    if (!name.empty()) {
      if (count == 0) ss << "{\"tool_calls\":[";
      else ss << ",";
      ss << "{\"id\":\"call_" << count << "\","
         << "\"type\":\"function\","
         << "\"function\":{\"name\":\"" << json_escape(name) << "\","
         << "\"arguments\":" << args.str() << "}}";
      count++;
    }
    pos = tc_end + 12;
  }

  if (count == 0) return "";
  ss << "]}";
  return ss.str();
}

// ---------------------------------------------------------------------------
// JSON format (generic): <tool_call>\n{"name":"...","arguments":{...}}\n</tool_call>
// Some models output JSON objects inside tool_call tags.
// ---------------------------------------------------------------------------

static std::string extract_json_string(const std::string& json, const std::string& key) {
  std::string token = "\"" + key + "\"";
  size_t kp = json.find(token);
  if (kp == std::string::npos) return "";
  size_t cp = json.find(':', kp + token.size());
  if (cp == std::string::npos) return "";
  size_t p = cp + 1;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) p++;
  if (p >= json.size() || json[p] != '"') return "";
  p++;
  std::string out;
  bool esc = false;
  while (p < json.size()) {
    char ch = json[p++];
    if (esc) { out.push_back(ch == 'n' ? '\n' : ch == 't' ? '\t' : ch); esc = false; continue; }
    if (ch == '\\') { esc = true; continue; }
    if (ch == '"') break;
    out.push_back(ch);
  }
  return out;
}

static std::string extract_json_balanced(const std::string& json, size_t pos) {
  if (pos >= json.size()) return "";
  char open = json[pos];
  char close = (open == '{') ? '}' : (open == '[') ? ']' : '\0';
  if (!close) return "";
  int depth = 0;
  bool in_str = false;
  for (size_t i = pos; i < json.size(); i++) {
    char c = json[i];
    if (in_str) { if (c == '"' && i > 0 && json[i - 1] != '\\') in_str = false; continue; }
    if (c == '"') { in_str = true; continue; }
    if (c == open) depth++;
    else if (c == close) { depth--; if (depth == 0) return json.substr(pos, i - pos + 1); }
  }
  return "";
}

static std::string extract_json_raw(const std::string& json, const std::string& key) {
  std::string token = "\"" + key + "\"";
  size_t kp = json.find(token);
  if (kp == std::string::npos) return "";
  size_t cp = json.find(':', kp + token.size());
  if (cp == std::string::npos) return "";
  size_t p = cp + 1;
  while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) p++;
  if (p >= json.size()) return "";
  if (json[p] == '{' || json[p] == '[') return extract_json_balanced(json, p);
  return "";
}

static std::string parse_json_tool_calls(const std::string& output) {
  std::ostringstream ss;
  int count = 0;
  size_t pos = 0;

  while (pos < output.size()) {
    size_t tc_start = output.find("<tool_call>", pos);
    if (tc_start == std::string::npos) break;
    size_t tc_end = output.find("</tool_call>", tc_start);
    if (tc_end == std::string::npos) break;

    std::string body = output.substr(tc_start + 11, tc_end - tc_start - 11);
    std::string name = extract_json_string(body, "name");
    std::string arguments = extract_json_raw(body, "arguments");
    if (arguments.empty()) arguments = "{}";

    if (!name.empty()) {
      if (count == 0) ss << "{\"tool_calls\":[";
      else ss << ",";
      ss << "{\"id\":\"call_" << count << "\","
         << "\"type\":\"function\","
         << "\"function\":{\"name\":\"" << json_escape(name) << "\","
         << "\"arguments\":" << arguments << "}}";
      count++;
    }
    pos = tc_end + 12;
  }

  if (count == 0) return "";
  ss << "]}";
  return ss.str();
}

// ---------------------------------------------------------------------------
// Hunyuan format:
//   <tool_call>function_name
//   ```json
//   {"key": "value"}
//   ```
//   </tool_call>
// ---------------------------------------------------------------------------

static std::string parse_hunyuan_tool_calls(const std::string& output) {
  std::ostringstream ss;
  int count = 0;
  size_t pos = 0;

  while (pos < output.size()) {
    size_t tc_start = output.find("<tool_call>", pos);
    if (tc_start == std::string::npos) break;
    size_t tc_end = output.find("</tool_call>", tc_start);
    if (tc_end == std::string::npos) break;

    std::string body = output.substr(tc_start + 11, tc_end - tc_start - 11);
    // First line (after trim) is the function name.
    std::string trimmed = trim(body);
    size_t nl = trimmed.find('\n');
    if (nl == std::string::npos) { pos = tc_end + 12; continue; }
    std::string name = trim(trimmed.substr(0, nl));

    // Arguments between ```json and ```.
    std::string args = "{}";
    size_t json_start = trimmed.find("```json");
    size_t json_end = (json_start != std::string::npos)
        ? trimmed.find("```", json_start + 7) : std::string::npos;
    if (json_start != std::string::npos && json_end != std::string::npos) {
      args = trim(trimmed.substr(json_start + 7, json_end - json_start - 7));
      if (args.empty()) args = "{}";
    }

    if (!name.empty()) {
      if (count == 0) ss << "{\"tool_calls\":[";
      else ss << ",";
      ss << "{\"id\":\"call_" << count << "\","
         << "\"type\":\"function\","
         << "\"function\":{\"name\":\"" << json_escape(name) << "\","
         << "\"arguments\":" << args << "}}";
      count++;
    }
    pos = tc_end + 12;
  }

  if (count == 0) return "";
  ss << "]}";
  return ss.str();
}

// ---------------------------------------------------------------------------
// Dispatch: detect format and call the right parser.
// ---------------------------------------------------------------------------

// Try each registered format. Returns OpenAI-compatible JSON or empty string.
static std::string parse_tool_calls(const std::string& output) {
  if (output.find("<tool_call>") == std::string::npos) return "";
  // Qwen3.5 XML: has <function= tag
  if (output.find("<function=") != std::string::npos) {
    std::string result = parse_qwen_tool_calls(output);
    if (!result.empty()) return result;
  }
  // Hunyuan: has ```json inside tool_call block
  if (output.find("```json") != std::string::npos) {
    std::string result = parse_hunyuan_tool_calls(output);
    if (!result.empty()) return result;
  }
  // Qwen3 JSON fallback
  return parse_json_tool_calls(output);
}

}  // namespace tool_parser
}  // namespace omniinfer
