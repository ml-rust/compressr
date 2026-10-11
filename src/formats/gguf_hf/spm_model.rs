//! A SentencePiece vocabulary (`tokenizer.ggml.model = "llama"`) from
//! `tokenizer.model`.
//!
//! `tokenizer.model` is a protobuf `ModelProto`. Its field 1 repeats one
//! `SentencePiece` message per id: `piece` (field 1, string), `score`
//! (field 2, float) and `type` (field 3, enum). Nothing else in the file
//! feeds the GGUF vocabulary, so this reader decodes those fields and skips
//! the rest by wire type. The token list then follows llama.cpp's
//! `_create_vocab_sentencepiece`: `added_tokens.json` and the
//! `added_tokens_decoder` of `tokenizer_config.json` override the pieces.

use super::bpe_vocab::{looks_special, token_type};
use super::special_vocab::read_json_object;
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;

/// The `llama` token list, scores and token types.
#[derive(Debug, Clone, PartialEq)]
pub struct SpmVocab {
    /// Token text by id.
    pub tokens: Vec<String>,
    /// Score by id: SentencePiece's merge rank.
    pub scores: Vec<f32>,
    /// [`token_type`] by id.
    pub token_types: Vec<i32>,
}

/// One `SentencePiece` message.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// The piece text.
    pub piece: String,
    /// The score, 0 when the message omits it.
    pub score: f32,
    /// The `Type` enum, NORMAL (1) when the message omits it.
    pub kind: u64,
}

/// The pieces of a `ModelProto`, in id order.
///
/// # Errors
/// The bytes are not a well-formed protobuf message.
pub fn read_pieces(bytes: &[u8]) -> Result<Vec<Piece>> {
    let mut pieces = Vec::new();
    let mut reader = Wire::new(bytes);
    while let Some((field, wire)) = reader.key()? {
        if field == 1 && wire == 2 {
            pieces.push(read_piece(reader.bytes()?)?);
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(pieces)
}

fn read_piece(bytes: &[u8]) -> Result<Piece> {
    let mut piece = Piece {
        piece: String::new(),
        score: 0.0,
        kind: 1,
    };
    let mut reader = Wire::new(bytes);
    while let Some((field, wire)) = reader.key()? {
        match (field, wire) {
            (1, 2) => {
                piece.piece = String::from_utf8(reader.bytes()?.to_vec())
                    .context("tokenizer.model holds a piece that is not UTF-8")?;
            }
            (2, 5) => piece.score = f32::from_le_bytes(reader.fixed4()?),
            (3, 0) => piece.kind = reader.varint()?,
            _ => reader.skip(wire)?,
        }
    }
    Ok(piece)
}

/// A cursor over protobuf wire bytes.
struct Wire<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Wire<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let Some(&byte) = self.bytes.get(self.pos) else {
                bail!("tokenizer.model ends inside a varint at byte {}", self.pos);
            };
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!(
            "tokenizer.model holds a varint wider than 64 bits at byte {}",
            self.pos
        )
    }

    fn key(&mut self) -> Result<Option<(u64, u64)>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        Ok(Some((key >> 3, key & 7)))
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).filter(|e| *e <= self.bytes.len());
        let Some(end) = end else {
            bail!(
                "tokenizer.model ends inside a {len}-byte field at byte {}",
                self.pos
            );
        };
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.varint()?).context("tokenizer.model field length")?;
        self.take(len)
    }

    fn fixed4(&mut self) -> Result<[u8; 4]> {
        let b = self.take(4)?;
        Ok([b[0], b[1], b[2], b[3]])
    }

    fn skip(&mut self, wire: u64) -> Result<()> {
        match wire {
            0 => self.varint().map(drop),
            1 => self.take(8).map(drop),
            2 => self.bytes().map(drop),
            5 => self.take(4).map(drop),
            other => bail!("tokenizer.model uses protobuf wire type {other}, which it never holds"),
        }
    }
}

/// llama.cpp's token type for a SentencePiece `Type`: a USER_DEFINED piece
/// is NORMAL, because `SentencePieceProcessor` has no test that matches it.
fn piece_type(kind: u64) -> i32 {
    match kind {
        2 => token_type::UNKNOWN,
        3 => token_type::CONTROL,
        5 => token_type::UNUSED,
        6 => token_type::BYTE,
        _ => token_type::NORMAL,
    }
}

/// The `llama` vocabulary of the checkpoint in `dir`.
///
/// `vocab_size` is `config.json`'s, when stated. Without it the size is the
/// piece count. Ids past the pieces are `[PAD<id>]`, score -10000, UNUSED.
///
/// # Errors
/// `tokenizer.model` cannot be read or parsed, or a JSON override file is
/// malformed.
pub fn read_spm_vocab(dir: &Path, vocab_size: Option<u32>) -> Result<SpmVocab> {
    let path = dir.join("tokenizer.model");
    let bytes =
        std::fs::read(&path).with_context(|| format!("Failed to read {}", path.display()))?;
    let pieces =
        read_pieces(&bytes).with_context(|| format!("Failed to parse {}", path.display()))?;
    let size = vocab_size.map_or(pieces.len(), |v| v as usize);
    let mut vocab = SpmVocab {
        tokens: (0..size).map(|i| format!("[PAD{i}]")).collect(),
        scores: vec![-10000.0; size],
        token_types: vec![token_type::UNUSED; size],
    };
    if pieces.len() > size {
        log::warn!(
            "tokenizer.model has {} pieces; ids from {size} are ignored",
            pieces.len()
        );
    }
    for (id, piece) in pieces.into_iter().take(size).enumerate() {
        vocab.tokens[id] = piece.piece;
        vocab.scores[id] = piece.score;
        vocab.token_types[id] = piece_type(piece.kind);
    }
    if let Some(added) = read_json_object(&dir.join("added_tokens.json"))? {
        for (token, id) in added {
            match id.as_u64().map(|i| i as usize).filter(|i| *i < size) {
                Some(id) => {
                    vocab.tokens[id] = token;
                    vocab.scores[id] = -1000.0;
                    vocab.token_types[id] = token_type::USER_DEFINED;
                }
                None => log::warn!("added_tokens.json gives '{token}' the id {id}; ignored"),
            }
        }
    }
    let decoder = read_json_object(&dir.join("tokenizer_config.json"))?.and_then(|c| {
        c.get("added_tokens_decoder")
            .and_then(Value::as_object)
            .cloned()
    });
    for (id, data) in decoder.into_iter().flatten() {
        let Some(id) = id.parse::<usize>().ok().filter(|i| *i < size) else {
            log::warn!("tokenizer_config.json added token id {id} is outside the vocabulary");
            continue;
        };
        let Some(content) = data.get("content").and_then(Value::as_str) else {
            continue;
        };
        let special = data.get("special").and_then(Value::as_bool) == Some(true);
        if special || looks_special(content) {
            vocab.tokens[id] = content.to_string();
            vocab.token_types[id] = token_type::CONTROL;
        } else {
            vocab.tokens[id] = content.replace('\u{2581}', " ");
            vocab.token_types[id] = token_type::USER_DEFINED;
        }
        vocab.scores[id] = -1000.0;
    }
    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn piece(text: &str, score: f32, kind: Option<u64>) -> Vec<u8> {
        let mut body = vec![0x0a];
        varint(text.len() as u64, &mut body);
        body.extend_from_slice(text.as_bytes());
        body.push(0x15);
        body.extend_from_slice(&score.to_le_bytes());
        if let Some(kind) = kind {
            body.push(0x18);
            varint(kind, &mut body);
        }
        let mut out = vec![0x0a];
        varint(body.len() as u64, &mut out);
        out.extend(body);
        out
    }

    /// A `ModelProto` with three pieces and a trainer spec to skip.
    pub(crate) fn model_proto() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(piece("<unk>", 0.0, Some(2)));
        out.extend(piece("<s>", 0.0, Some(3)));
        out.extend(piece("\u{2581}the", -1.5, None));
        out.extend(piece("<0x0A>", 0.0, Some(6)));
        out.extend(piece("<tool>", 0.0, Some(4)));
        // Field 2, a nested message the reader skips.
        out.extend([0x12, 0x03, 0x08, 0x96, 0x01]);
        out
    }

    #[test]
    fn pieces_decode_and_unknown_fields_are_skipped() {
        let pieces = read_pieces(&model_proto()).unwrap();
        assert_eq!(pieces.len(), 5);
        assert_eq!(pieces[2].piece, "\u{2581}the");
        assert_eq!(pieces[2].score, -1.5);
        assert_eq!(pieces[2].kind, 1);
        assert!(read_pieces(&[0x0a, 0x05, 0x01]).is_err());
    }

    #[test]
    fn vocab_follows_llama_cpp_types_and_padding() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.model"), model_proto()).unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder": {"5": {"content": "[INST]", "special": true}}}"#,
        )
        .unwrap();
        let vocab = read_spm_vocab(dir.path(), Some(7)).unwrap();
        assert_eq!(
            vocab.tokens,
            [
                "<unk>",
                "<s>",
                "\u{2581}the",
                "<0x0A>",
                "<tool>",
                "[INST]",
                "[PAD6]"
            ]
        );
        use token_type::*;
        assert_eq!(
            vocab.token_types,
            [UNKNOWN, CONTROL, NORMAL, BYTE, NORMAL, CONTROL, UNUSED]
        );
        assert_eq!(vocab.scores, [0.0, 0.0, -1.5, 0.0, 0.0, -1000.0, -10000.0]);
    }
}
