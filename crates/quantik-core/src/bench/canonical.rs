//! Python-compatible canonical JSON encoding.
//!
//! Dataset checksums are sha256 over the exact bytes Python produces with
//! `json.dumps(payload, sort_keys=True, separators=(",", ":"))`. Two spots
//! diverge from serde_json's encoder and are reimplemented here:
//!
//! - floats: Python uses `repr(float)` (shortest round-trip, scientific
//!   notation when the decimal exponent is < -4 or >= 16, two-digit
//!   exponent with sign), where ryu prefers positional notation;
//! - strings: Python defaults to `ensure_ascii=True`, escaping any
//!   non-ASCII character as `\uXXXX` (surrogate pairs above the BMP).
//!
//! Map keys are already sorted because `serde_json::Map` is a `BTreeMap`
//! (byte order == code-point order for the ASCII keys used by the schema).

use serde_json::Value;

/// Format an f64 exactly like CPython's `repr(float)`.
pub fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }

    // `{:e}` gives the shortest round-trip digits in scientific form,
    // e.g. "4.7e-5", "3e1", "-1.75e-4".
    let sci = format!("{x:e}");
    let (mantissa, exp_str) = sci.split_once('e').expect("LowerExp always emits an e");
    let exp: i32 = exp_str.parse().expect("valid exponent");
    let neg = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();

    let body = if (-4..16).contains(&exp) {
        // Positional notation.
        if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if digits.len() > int_len {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            } else {
                // Pad with zeros and force a trailing ".0" like Python.
                format!("{}{}.0", digits, "0".repeat(int_len - digits.len()))
            }
        } else {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        }
    } else {
        // Scientific notation: single leading digit, no trailing ".0",
        // sign always present, exponent at least two digits.
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", m, sign, exp.abs())
    };

    if neg {
        format!("-{body}")
    } else {
        body
    }
}

/// Escape a string like Python's `json.dumps` with `ensure_ascii=True`.
fn escape_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if c.is_ascii() => out.push(c),
            c => {
                let code = c as u32;
                if code <= 0xFFFF {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    // Surrogate pair.
                    let v = code - 0x10000;
                    out.push_str(&format!(
                        "\\u{:04x}\\u{:04x}",
                        0xD800 + (v >> 10),
                        0xDC00 + (v & 0x3FF)
                    ));
                }
            }
        }
    }
    out.push('"');
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&python_float_repr(n.as_f64().expect("finite float")));
            }
        }
        Value::String(s) => escape_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                escape_string(key, out);
                out.push(':');
                write_value(item, out);
            }
            out.push('}');
        }
    }
}

/// Serialize `value` exactly as Python's
/// `json.dumps(value, sort_keys=True, separators=(",", ":"))`.
pub fn canonical_json(value: &Value) -> String {
    let mut out = String::new();
    write_value(value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        // (input, CPython repr) pairs.
        let cases: [(f64, &str); 14] = [
            (0.000047, "4.7e-05"),
            (0.0001, "0.0001"),
            (0.000175, "0.000175"),
            (7.959543, "7.959543"),
            (30.0, "30.0"),
            (1.0, "1.0"),
            (-0.5, "-0.5"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (1e22, "1e+22"),
            (1e16, "1e+16"),
            (9999999999999998.0, "9999999999999998.0"),
            (0.0, "0.0"),
            (1.414, "1.414"),
        ];
        for (x, expected) in cases {
            assert_eq!(python_float_repr(x), expected, "for {x}");
        }
    }

    #[test]
    fn canonical_matches_python_layout() {
        let value = json!({
            "b": [1, 2.5, null, true],
            "a": {"nested": "ok", "t": 4.7e-5},
        });
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"nested":"ok","t":4.7e-05},"b":[1,2.5,null,true]}"#
        );
    }

    #[test]
    fn non_ascii_escaped_like_python() {
        // CPython: json.dumps({"s": "héllo — 🎉"}, sort_keys=True,
        //                     separators=(",", ":"))
        let value = json!({"s": "héllo — 🎉"});
        assert_eq!(
            canonical_json(&value),
            "{\"s\":\"h\\u00e9llo \\u2014 \\ud83c\\udf89\"}"
        );
    }
}
