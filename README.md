# Amazon ML Challenge — Business Entity Resolution

`entity_resolution.ipynb` is a single, self-contained Kaggle notebook that goes from the raw TSVs to the two submission files.

## Run on Kaggle
1. Add the challenge dataset (the folder with `train/` and `test/`) as notebook input. The notebook finds `train_source1.tsv` anywhere under `/kaggle/input`, or you can set `ER_DATA_DIR`.
2. Turn internet on (only needed for `pip install rapidfuzz sparse_dot_topn anyascii`), then **Run All**.
3. Outputs are written to `/kaggle/working/output/`:
   - `matching_results.tsv` is the leaderboard file.
   - `candidate_pairs.tsv` is the exact candidate set scored by the model.

The last cell checks the submission format. Intermediate results are cached in `/kaggle/working/work/`, so rerunning skips finished stages.

For a quick smoke test, set `ER_SAMPLE=0.03` to run on a 3% slice of the data in about 1 minute.

## Method
1. **Normalisation**
   - Romanises native scripts (Devanagari, Tamil, Kannada, …) and strips accents.
   - Removes junk prefixes and honorifics, and splits DBA / "t/a" aliases and legal suffixes (Pvt/Ltd/LLC/SARL/EURL…).
   - Expands street-type abbreviations (US / India / France), standardises state names, and strips zero padding from house numbers.
   - Uses lexicons mined from training pairs only (e.g. `प्राइवेट → private`, native state names) and splits glued names or domains (`americancoalition.com → american coalition`).
2. **Blocking**, per country (open set of labels), from the S2/S3 side. It unions three sparse TF-IDF top-k channels:
   - name + address words
   - name words
   - name character 3-grams

   Very frequent tokens are left out of retrieval. The exact cosines are kept as features, and the candidates are pruned to about 5.6 per record, with about 97% pair recall on a train sample.
3. **Matcher**: LightGBM on about 60 features:
   - rapidfuzz name and address similarities
   - alias, legal-suffix and house-number agreement
   - name commonness
   - context margins against the other candidates on both the S2/S3 and S1 side

   Country is deliberately not a feature, so the model transfers to France.
4. **Decoding**: every S2/S3 record belongs to at most one S1, so it is linked only to its argmax S1 when p ≥ τ. τ (or an expected-F0.5 prefix rule) is tuned on held-out S1s with the exact macro-F0.5 metric, singletons included.

Only the provided data is used; there are no external lookups or APIs. All libraries are MIT, Apache or BSD licensed.
