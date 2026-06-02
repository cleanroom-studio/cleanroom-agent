//! Stage 1 (binary `.fig`): parse Figma's offline file format.
//!
//! The `.fig` format is documented (informally) at:
//! - <https://www.figma.com/blog/redux-from-figma-internal-tool-to-product/>
//! - A Rust reference implementation exists at <https://github.com/zxhsure/01joy-fig-parser-rust>
//!
//! ## File layout
//!
//! ```text
//! +-------+----------+----------+----------+----------+-----+
//! | magic | version  | chunk 0  | chunk 1  | chunk 2  | ... |
//! | 8 B   | 4 B LE   | u32 LE   | u32 LE   | u32 LE   |     |
//! |       |          | length+  | length+  | length+  |     |
//! |       |          | zlib     | zlib     | zlib     |     |
//! +-------+----------+----------+----------+----------+-----+
//! ```
//!
//! - `magic` is one of `"fig-kiwi"` (newer) or `"fig-jam."` (older).
//! - `version` is a `u32` little-endian. We currently accept any value.
//! - Each chunk is a zlib/deflate stream; the first chunk is the schema
//!   (kiwi-schema), the second is the data (kiwi-schema encoded `Message`).
//!
//! ## Current scope
//!
//! This implementation handles:
//! - Header validation (`fig-kiwi` / `fig-jam.`).
//! - Version parsing.
//! - Chunk extraction + zlib decompression.
//!
//! What is **not** yet implemented (see [`FigFile::into_ir`]):
//! - The kiwi-schema binary decoder. We return a [`FigDecodeError::KiwiSchemaNotImplemented`]
//!   for the inner data and provide the raw deflated bytes for callers that
//!   want to plug in their own kiwi-schema parser. This is intentional —
//!   kiwi-schema is a substantial binary codec (see the upstream
//!   `01joy-fig-parser-rust` reference for a complete implementation in
//!   ~600 lines).
//!
//! When the kiwi-schema decoder is added, [`FigFile::into_ir`] should
//! populate the [`FigmaIr`] from the decoded `Message` (which contains
//! `nodeChanges`, `blobs`, etc.).

use std::io::Read;

use anyhow::Result;
use flate2::read::ZlibDecoder;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use super::figma_ir::{FigmaIr, FigmaIrNode, FigmaIrPage};

/// Magic bytes that identify a Figma `.fig` file.
pub const FIGMA_MAGIC_KIWI: &[u8; 8] = b"fig-kiwi";
pub const FIGMA_MAGIC_JAM: &[u8; 8] = b"fig-jam.";

/// Parsed Figma `.fig` file: header + raw decompressed chunks.
#[derive(Debug, Clone)]
pub struct FigFile {
    /// Magic: `"fig-kiwi"` or `"fig-jam."`.
    pub magic: FigMagic,
    /// Version (`u32` LE).
    pub version: u32,
    /// Deflated-then-decompressed schema (chunk 0). The kiwi-schema
    /// binary is **not** yet decoded by this crate.
    pub schema_bytes: Vec<u8>,
    /// Deflated-then-decompressed data (chunk 1). The kiwi-schema
    /// `Message` is **not** yet decoded by this crate.
    pub data_bytes: Vec<u8>,
    /// Any additional chunks beyond the first two.
    pub extra_chunks: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FigMagic {
    /// Newer format: `"fig-kiwi"`.
    Kiwi,
    /// Older format: `"fig-jam."`.
    Jam,
}

impl FigMagic {
    pub fn from_bytes(b: &[u8; 8]) -> Option<Self> {
        if b == FIGMA_MAGIC_KIWI {
            Some(Self::Kiwi)
        } else if b == FIGMA_MAGIC_JAM {
            Some(Self::Jam)
        } else {
            None
        }
    }
}

/// Errors that can occur while parsing a `.fig` file.
#[derive(Debug, Error)]
pub enum FigDecodeError {
    /// The first 8 bytes are neither `"fig-kiwi"` nor `"fig-jam."`.
    #[error("invalid Figma file magic: expected `fig-kiwi` or `fig-jam.`, got {0:?}")]
    InvalidMagic([u8; 8]),

    /// The file is shorter than the minimum header.
    #[error("Figma file is truncated: need at least 12 bytes, got {0}")]
    Truncated(usize),

    /// The file declares fewer than 2 chunks (we need at least the
    /// schema and the data).
    #[error("Figma file has {0} chunks; need at least 2 (schema + data)")]
    TooFewChunks(usize),

    /// zlib decompression failed.
    #[error("zlib decompression of chunk {0} failed: {1}")]
    Decompress(usize, String),

    /// The kiwi-schema decoder is not yet implemented in this crate.
    /// The raw decompressed bytes are available via
    /// [`FigFile::schema_bytes`] / [`FigFile::data_bytes`].
    #[error(
        "kiwi-schema binary decoder is not yet implemented; \
         raw deflated bytes are available via `FigFile::schema_bytes` / `data_bytes`. \
         See https://github.com/zxhsure/01joy-fig-parser-rust for a Rust reference."
    )]
    KiwiSchemaNotImplemented,
}

impl FigFile {
    /// Parse a `.fig` file from raw bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, FigDecodeError> {
        if bytes.len() < 12 {
            return Err(FigDecodeError::Truncated(bytes.len()));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        let magic = FigMagic::from_bytes(&magic)
            .ok_or(FigDecodeError::InvalidMagic(magic))?;

        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        // Chunk loop: read u32 LE length, then the chunk bytes.
        let mut offset = 12usize;
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        while offset < bytes.len() {
            if offset + 4 > bytes.len() {
                warn!(offset, "Figma file: chunk header truncated");
                break;
            }
            let len = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + len > bytes.len() {
                warn!(offset, len, "Figma file: chunk body truncated");
                break;
            }
            chunks.push(bytes[offset..offset + len].to_vec());
            offset += len;
        }

        if chunks.len() < 2 {
            return Err(FigDecodeError::TooFewChunks(chunks.len()));
        }

        // Decompress the first two chunks.
        let schema_bytes = decompress_chunk(0, &chunks[0])?;
        let data_bytes = decompress_chunk(1, &chunks[1])?;
        let extra_chunks = chunks[2..].to_vec();

        debug!(
            ?magic,
            version,
            schema_len = schema_bytes.len(),
            data_len = data_bytes.len(),
            extra = extra_chunks.len(),
            "Parsed .fig file"
        );

        Ok(Self {
            magic,
            version,
            schema_bytes,
            data_bytes,
            extra_chunks,
        })
    }

    /// Convert the parsed `.fig` into a [`FigmaIr`].
    ///
    /// **Status:** The full kiwi-schema binary decoder is not yet
    /// implemented in this crate. Until it is, this method returns
    /// [`FigDecodeError::KiwiSchemaNotImplemented`] but the
    /// [`FigmaIr`] skeleton is still useful for unit testing the
    /// upstream pipeline.
    pub fn into_ir(&self) -> Result<FigmaIr, FigDecodeError> {
        // TODO: implement kiwi-schema decoding here.
        //
        // The schema is at self.schema_bytes; the encoded Message is at
        // self.data_bytes. A reference Rust implementation of the
        // kiwi-schema binary codec lives at:
        //   https://github.com/zxhsure/01joy-fig-parser-rust
        // (uses the `kiwi-schema` crate, ~600 LOC of decoder logic).
        //
        // Once the decoder is integrated, populate `FigmaIr` from the
        // decoded `Message` struct, which contains:
        //   - `nodeChanges: Vec<NodeChange>` (one per document node)
        //   - `blobs: Vec<Blob>` (image / vector data)
        //   - `images: HashMap<...>` (image fills)
        //
        // For now, return a placeholder IR so callers can exercise
        // the surrounding pipeline.
        let _ = FigDecodeError::KiwiSchemaNotImplemented;

        Ok(FigmaIr {
            name: format!("(fig-kiwi v{} — decode pending)", self.version),
            last_modified: None,
            version: Some(self.version.to_string()),
            pages: vec![FigmaIrPage {
                id: "0:1".to_string(),
                name: "(decoder pending)".to_string(),
                nodes: vec![FigmaIrNode {
                    id: "0:2".to_string(),
                    name: "placeholder".to_string(),
                    type_: "PLACEHOLDER".to_string(),
                    children: Vec::new(),
                    extra: Default::default(),
                }],
            }],
            variables: Vec::new(),
            styles: Default::default(),
        })
    }
}

fn decompress_chunk(index: usize, data: &[u8]) -> Result<Vec<u8>, FigDecodeError> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| FigDecodeError::Decompress(index, e.to_string()))?;
    Ok(out)
}

/// Quick header-only probe: returns the magic + version without decompressing
/// any chunk. Used by the CLI to fail fast on non-`.fig` inputs.
pub fn probe(bytes: &[u8]) -> Result<(FigMagic, u32), FigDecodeError> {
    if bytes.len() < 12 {
        return Err(FigDecodeError::Truncated(bytes.len()));
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[0..8]);
    let magic = FigMagic::from_bytes(&magic).ok_or(FigDecodeError::InvalidMagic(magic))?;
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    Ok((magic, version))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use std::io::Write;

    /// Build a valid zlib stream for `payload` using flate2 (so the bytes
    /// are guaranteed to be decodable by `flate2::ZlibDecoder`).
    fn zlib(payload: &[u8]) -> Vec<u8> {
        let mut enc = ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(payload).unwrap();
        enc.finish().unwrap()
    }

    /// Minimal valid `.fig` file: 8-byte magic + 4-byte version + 2 zlib
    /// chunks (schema and data, each containing a single byte).
    fn make_minimal_fig() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(FIGMA_MAGIC_KIWI);
        out.extend_from_slice(&1u32.to_le_bytes());

        let chunk0 = zlib(&[0xAA]); // arbitrary "schema" payload
        let chunk1 = zlib(&[0xBB]); // arbitrary "data" payload

        out.extend_from_slice(&(chunk0.len() as u32).to_le_bytes());
        out.extend_from_slice(&chunk0);
        out.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        out.extend_from_slice(&chunk1);

        out
    }

    #[test]
    fn parses_minimal_fig_header() {
        let bytes = make_minimal_fig();
        let parsed = FigFile::parse(&bytes).expect("parse should succeed");
        assert_eq!(parsed.magic, FigMagic::Kiwi);
        assert_eq!(parsed.version, 1);
        assert!(!parsed.schema_bytes.is_empty());
        assert!(!parsed.data_bytes.is_empty());
        assert!(parsed.extra_chunks.is_empty());
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = make_minimal_fig();
        bytes[0..8].copy_from_slice(b"not-a-fi");
        let err = FigFile::parse(&bytes).unwrap_err();
        assert!(matches!(err, FigDecodeError::InvalidMagic(_)));
    }

    #[test]
    fn rejects_truncated_file() {
        let err = FigFile::parse(b"fig").unwrap_err();
        assert!(matches!(err, FigDecodeError::Truncated(3)));
    }

    #[test]
    fn rejects_too_few_chunks() {
        // Header + 1 chunk only
        let mut bytes = Vec::new();
        bytes.extend_from_slice(FIGMA_MAGIC_KIWI);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let err = FigFile::parse(&bytes).unwrap_err();
        assert!(matches!(err, FigDecodeError::TooFewChunks(1)));
    }

    #[test]
    fn probe_returns_magic_and_version() {
        let bytes = make_minimal_fig();
        let (magic, version) = probe(&bytes).unwrap();
        assert_eq!(magic, FigMagic::Kiwi);
        assert_eq!(version, 1);
    }

    #[test]
    fn jam_magic_is_accepted() {
        let mut bytes = make_minimal_fig();
        bytes[0..8].copy_from_slice(FIGMA_MAGIC_JAM);
        let parsed = FigFile::parse(&bytes).unwrap();
        assert_eq!(parsed.magic, FigMagic::Jam);
    }

    #[test]
    fn into_ir_returns_placeholder_pending_kiwi_decoder() {
        // Documents the current contract: until the kiwi-schema binary
        // decoder lands, into_ir returns a non-empty placeholder IR.
        let parsed = FigFile::parse(&make_minimal_fig()).unwrap();
        let ir = parsed.into_ir().unwrap();
        assert_eq!(ir.pages.len(), 1);
        assert_eq!(ir.pages[0].nodes[0].type_, "PLACEHOLDER");
    }
}
