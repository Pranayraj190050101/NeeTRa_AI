# Retrieval Quality Report — Class 11 Biology NCERT Index

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Index: `F:\Project\neet_ai_tutor\vectordb\neet_index`
- top_k=5, score_threshold=0.15

## 10 Sanity Checks

| # | Query (truncated) | Expected file | Top file | Top score | Expected hit? | Results |
|---|---|---|---|---|---|---|
| 1 | What characteristics define and distinguish livin... | `kebo101.pdf` | `kebo101.pdf` | 0.651 | YES | 3 |
| 2 | Describe the five kingdom classification of organ... | `kebo102.pdf` | `kebo102.pdf` | 0.811 | YES | 3 |
| 3 | What is the economic importance of algae and gymn... | `kebo103.pdf` | `kebo103.pdf` | 0.570 | YES | 3 |
| 4 | Explain the different types of epithelial tissues... | `kebo107.pdf` | `kebo107.pdf` | 0.636 | YES | 3 |
| 5 | What are the stages of the mitotic cell cycle in ... | `kebo110.pdf` | `kebo110.pdf` | 0.804 | YES | 3 |
| 6 | Explain the light reactions and the Calvin cycle ... | `kebo111.pdf` | `kebo111.pdf` | 0.722 | YES | 3 |
| 7 | Describe glycolysis and the Krebs cycle in cellul... | `kebo112.pdf` | `kebo112.pdf` | 0.680 | YES | 3 |
| 8 | How does gas exchange occur in the human respirat... | `kebo114.pdf` | `kebo114.pdf` | 0.719 | YES | 3 |
| 9 | What are the formed elements of human blood and t... | `kebo115.pdf` | `kebo115.pdf` | 0.680 | YES | 3 |
| 10 | Explain the structure of the nephron and the proc... | `kebo116.pdf` | `kebo116.pdf` | 0.752 | YES | 3 |

## Metadata Verification (top chunk of each query)

| # | chapter (metadata) | ncert_subject | class | source | subject | all keys present |
|---|---|---|---|---|---|---|
| 1 | The Living World | Biology | 11 | NCERT | biology | OK |
| 2 | 2.1 Kingdom Monera | Biology | 11 | NCERT | biology | OK |
| 3 | 3.1 Algae | Biology | 11 | NCERT | biology | OK |
| 4 | 7.1 Organ and Organ | Biology | 11 | NCERT | biology | OK |
| 5 | 10.1 Cell Cycle | Biology | 11 | NCERT | biology | OK |
| 6 | Photosynthesis in Higher | Biology | 11 | NCERT | biology | OK |
| 7 | 12.1 Do Plants | Biology | 11 | NCERT | biology | OK |
| 8 | Breathing and Exchange | Biology | 11 | NCERT | biology | OK |
| 9 | 15.1 Blood | Biology | 11 | NCERT | biology | OK |
| 10 | 16.1 Human | Biology | 11 | NCERT | biology | OK |

## Aggregate

- Queries returning >=1 result: 10/10
- Top hit == expected file: 10/10
- Chunks with full NCERT metadata set: 10/10
- Chunks with correct subject/class/source values: 10/10

## Top passages (evidence)

**1. Biology is the science of life forms and living processes. The living world comprises an amazing diversity of living organisms. Early man could easily perceive the difference betwe ...**

**2. BIOLOGICAL CLASSIFICATION 11 1111 1111 R.H. Whittaker (1969) proposed a Five Kingdom Classification. Five Kingdom Classification. Five Kingdom Classification. Five Kingdom Classifi ...**

**3. 26 BIOLOGY Algae are useful to man in a variety of ways. At least a half of the total carbon dioxide fixation on earth is carried out by algae through photosynthesis. Being photosy ...**

**4. substances perform a specific function. Such an organisation is called tissue. You may be surprised to know that all complex animals consist of only four basic types of tissues. Th ...**

**5. CELL CYCLE AND CELL DIVISION 129 The stages through which a cell passes from one division to the next is called the cell cycle. Cell cycle is divided into two phases called (i) Int ...**

**6. PHOTOSYNTHESIS IN HIGHER PLANTS 151 SUMMARY Green plants make their own food by photosynthesis. During this process carbon dioxide from the atmosphere is taken in by leaves through ...**

**7. 2. What are respiratory substrates? Name the most common respiratory substrate. 3. Give the schematic representation of glycolysis? 4. What are the main steps in aerobic respiratio ...**

**8. 191BREATHING AND EXCHANGE OF GASES SUMMARY Cells utilise oxygen for metabolism and produce energy along with substances like carbon dioxide which is harmful. Animals have evolved d ...**

**9. 194 BIOLOGY Fibrinogens are needed for clotting or coagulation of blood. Globulins primarly are involved in defense mechanisms of the body and the albumins help in osmotic balance. ...**

**10. the loop of Henle is very long and runs deep into the medulla. These nephrons are called juxta medullary nephrons. The efferent arteriole emerging from the glomerulus forms a fine  ...**

## Findings & Honest Assessment

**What works (verified end-to-end on the real index):**

- Ingestion ran the NCERT-aware pipeline with real embeddings (`all-MiniLM-L6-v2`): 20 files → 259 pages → 780 chunks, 0 failures.
- `subject`, `class` and `source` metadata are **100% correct** on every sampled chunk (`Biology` / `11` / `NCERT`, with the retrieval-critical lowercase `subject=biology` preserved).
- All 10 sanity queries returned results, and every top hit was the **expected chapter file** (10/10) with healthy cosine scores (0.57–0.81). The evidence passages are unambiguously on-topic, so **semantic retrieval quality is strong**.
- The non-chapter file `kebo1ps.pdf` (book preface) was correctly processed but excluded from the 19 detected chapters.

**Known defect — chapter *title* extraction (flagged, NOT patched, per requirement 8):**

- The `chapter` field is only partially correct. Clean for openers that carry an inline/ALL-CAPS heading (`The Living World`, `Cell: The Unit of Life`, `Breathing and Exchange`, `Photosynthesis in Higher`), but **truncated to a subsection for ~12 of 19 files** (e.g. `2.1 Kingdom Monera` for *Biological Classification*, `12.1 Do Plants` for *Respiration in Plants*, `16.1 Human` for *Excretory Products*).
- Root cause: these are old-scheme NCERT chapter *part* PDFs that open mid-chapter; the running header/`Chapter N:` pattern is captured before the real ALL-CAPS chapter title, and headings wrap across lines so the capture is cut short.
- Impact: **none on retrieval** (search ranks on page-text embeddings, not the `chapter` label), but it degrades report readability and any future chapter-name filter/display. Fixing it requires editing `rag/ncert.py` (title-extraction priority), which is out of scope for this run.

**Minor:** one page of `kebo108.pdf` was skipped by pypdf ("maximum number of tokens per stream") — non-fatal, the file still indexed 18 pages. Reproducible in future runs by installing `fontTools` to silence CFF warnings (cosmetic only).
