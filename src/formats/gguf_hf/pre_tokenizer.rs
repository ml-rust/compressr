//! `tokenizer.ggml.pre`: the name llama.cpp gives a BPE pre-tokenizer.
//!
//! llama.cpp names a pre-tokenizer by fingerprint, not by its regex. Its
//! converter encodes a fixed text with the checkpoint's own tokenizer, hashes
//! the id list with SHA-256, and looks the hash up in a table of known models.
//! compressr does the same with splintr, which reproduces HuggingFace
//! `tokenizers` ids. An unknown fingerprint yields no name, and the caller
//! omits the key: llama.cpp then falls back to its default pre-tokenizer.

use super::pre_hashes::{CHKTXT, PRE_TOKENIZER_HASHES};
use sha2::{Digest, Sha256};

/// The boundary tokens HuggingFace's `encode` adds around the content ids.
#[derive(Debug, Clone, Copy, Default)]
pub struct Boundary {
    /// The BOS id, when the tokenizer prepends one.
    pub bos: Option<u32>,
    /// The EOS id, when the tokenizer appends one.
    pub eos: Option<u32>,
}

/// The SHA-256 of Python's `str()` of an id list, as lowercase hex.
pub fn fingerprint(ids: &[u32]) -> String {
    let text = format!(
        "[{}]",
        ids.iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// llama.cpp's name for the fingerprint `hash`, when its table has one.
pub fn name_for_fingerprint(hash: &str) -> Option<&'static str> {
    PRE_TOKENIZER_HASHES
        .iter()
        .find(|(h, _)| *h == hash)
        .map(|(_, name)| *name)
}

/// The pre-tokenizer name for a `tokenizer.json`, or `None` when splintr
/// cannot load it or llama.cpp's table has no entry for its fingerprint.
///
/// The ids are hashed twice: once with `boundary` around splintr's content
/// ids, and once with splintr's own special-token template. Either match
/// names the tokenizer, because a match means the ids equal those of the
/// reference model the table entry was made from.
pub fn pre_tokenizer_name(tokenizer_json: &[u8], boundary: Boundary) -> Option<&'static str> {
    let tokenizer = match splintr::from_json_bytes(tokenizer_json) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("splintr cannot load tokenizer.json ({e}); tokenizer.ggml.pre is omitted");
            return None;
        }
    };
    let content = tokenizer.encode_raw(CHKTXT);
    let mut framed = Vec::with_capacity(content.len() + 2);
    framed.extend(boundary.bos);
    framed.extend_from_slice(&content);
    framed.extend(boundary.eos);
    let templated = tokenizer.encode(CHKTXT);
    for ids in [&framed, &templated] {
        if let Some(name) = name_for_fingerprint(&fingerprint(ids)) {
            return Some(name);
        }
    }
    log::warn!(
        "llama.cpp's pre-tokenizer table has no entry for this tokenizer (fingerprint {}); \
         tokenizer.ggml.pre is omitted, so llama.cpp uses its default pre-tokenizer",
        fingerprint(&framed)
    );
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python: `sha256(str([128000, 198]).encode()).hexdigest()`.
    #[test]
    fn fingerprint_hashes_the_python_list_text() {
        assert_eq!(
            fingerprint(&[128000, 198]),
            "6ffa2674f2ca8cee47d6710bf8a3668f93e637fb00fb9246651a84af6972d0bb"
        );
    }

    #[test]
    fn the_table_names_llama3() {
        assert_eq!(
            name_for_fingerprint(
                "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5"
            ),
            Some("llama-bpe")
        );
        assert_eq!(name_for_fingerprint("0"), None);
    }
}
