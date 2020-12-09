
# Bootleg Change Log
 
## [Unreleased] - 2020-12-09
 
### Added
- @lorr1: Distributed support for dumping of embeddings for `dump_preds` or `dump_embs`.
 
### Changed
 - @lorr1: Upgraded to Transformers 4.0.0 and updated models by adding `position_ids` (see conversion [script](bootleg/utils/preprocessing/convert_to_trans4_0.py))
 - @lorr1: Better logging in data prep.
 
### Fixed
 - @lorr1: Fixed some issues with punctuation and translating word spans to subword token spans.