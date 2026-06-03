-- Phase 1.5: add a free-form text column to `fingerprints` for the
-- LLM's narrative explanation of "why these three hashes disagree".
--
-- The existing `fingerprints` table holds three SHA-256 hashes
-- (sdef_hash / db_hash / code_hash) and lets us detect a mismatch
-- via the partial index `idx_fingerprints_inconsistent`. Before
-- 1.5 the diagnostic on a mismatch was a bare log line; the
-- `consistency_llm` module (Phase 1.5) asks the LLM to write a
-- short explanation, stored here, that the human reviewer (or
-- the dashboard) can read.
--
-- Default NULL: 99% of the time the hashes agree and there's no
-- explanation to record. We only fill this column when the
-- consistency checker invokes the LLM.
--
-- Idempotent: `IF NOT EXISTS` so re-running this migration is a
-- no-op (mirrors the pattern in 009 / 010 / 011).

ALTER TABLE fingerprints ADD COLUMN llm_explanation TEXT;
