"""Generate synthetic clinical notes for the IST-3 Text-Enriched DGP.

Usage:
    uv run python -m air_cate.dgp.generate_notes [--concurrency 50] [--resume]

Requires: pip install anthropic (or uv sync --extra text)
"""

import asyncio
import gzip
import json
import re
import time
from pathlib import Path

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────
CONCURRENCY = 50
MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0.8
MAX_TOKENS = 200
CHECKPOINT_EVERY = 5000

ARTIFACT_DIR = Path(__file__).parent / "artifacts_text"
TVAE_CSV = Path(__file__).parents[3] / "IST_VAE" / "sampled_data.csv"

# ── Mappings ──────────────────────────────────────────────────────────
TRAJECTORY = {
    0: "good outcome (alive and independent at 6 months)",
    1: "poor outcome (death or severe dependency at 6 months)",
}

RCONSC_MAP = {"F": "fully conscious", "D": "drowsy", "U": "unconscious"}

DEFICIT_COLS = [
    ("RDEF1", "facial weakness"),
    ("RDEF2", "arm/hand weakness"),
    ("RDEF3", "leg/foot weakness"),
    ("RDEF4", "dysphasia"),
    ("RDEF5", "hemianopia"),
    ("RDEF6", "visuospatial deficit"),
    ("RDEF7", "brainstem/cerebellar signs"),
    ("RDEF8", "other deficit"),
]

STYPE_MAP = {
    "TACS": "total anterior circulation syndrome (TACS)",
    "PACS": "partial anterior circulation syndrome (PACS)",
    "LACS": "lacunar syndrome (LACS)",
    "POCS": "posterior circulation syndrome (POCS)",
    "OTH": "other stroke subtype",
}

SYSTEM_PROMPT = (
    "You are a hospital physician writing a brief admission note for a stroke "
    "patient. Write ONLY the clinical note — no headers, labels, preamble, or "
    "formatting. Use realistic medical shorthand and abbreviations where natural "
    '(e.g., "hx", "pmhx", "o/e", "L MCA territory"). The note must read as a '
    "pre-treatment assessment written before any intervention."
)

PROGNOSTIC_RE = re.compile(
    r"\b(prognosis|prognostic|expected survival|mortality risk|"
    r"likely to (die|survive|recover)|life expectancy|"
    r"poor outcome|good outcome|fatal(?:ity)?|terminal)\b",
    re.IGNORECASE,
)


# ── Prompt construction ──────────────────────────────────────────────

def build_user_prompt(row: pd.Series) -> str:
    """Build the user prompt for a single patient row."""
    sex_str = "male" if row["SEX"] == "M" else "female"
    age = int(row["AGE"])
    onset_hours = int(row["RDELAY"])
    conscious = RCONSC_MAP.get(row["RCONSC"], "unknown")
    sbp = int(row["RSBP"])

    # Neurological deficits
    present, uncertain = [], []
    for col, desc in DEFICIT_COLS:
        if row[col] == "Y":
            present.append(desc)
        elif row[col] == "C":
            uncertain.append(desc)
    parts = []
    if present:
        parts.append(", ".join(present))
    if uncertain:
        parts.append("unable to assess " + ", ".join(uncertain))
    deficits_str = "; ".join(parts) if parts else "none identified"

    # Medical history
    hx = []
    if row["RATRIAL"] == "Y":
        hx.append("atrial fibrillation")
    if row["aspirin"] == 1:
        hx.append("prior aspirin use")
    if row["heparin"] == 1:
        hx.append("prior heparin use")
    hx_str = ", ".join(hx) if hx else "no significant past medical history"

    # Imaging
    imaging = (
        "visible ischaemic changes on CT"
        if row["RVISINF"] == "Y"
        else "no visible infarct on CT"
    )

    stype = STYPE_MAP.get(row["STYPE"], str(row["STYPE"]))
    trajectory = TRAJECTORY[int(row["deathdep"])]

    return (
        "Write an admission note for this patient. The note should naturally "
        "reflect the clinical severity implied by all the information below. "
        "Do NOT explicitly state prognosis, expected survival, outcome, or "
        "trajectory.\n\n"
        f"Demographics: {age}-year-old {sex_str}.\n"
        f"Presentation: acute ischaemic stroke, {onset_hours} hours from "
        f"symptom onset. Conscious level: {conscious}. "
        f"Systolic BP: {sbp} mmHg. Classification: {stype}.\n"
        f"Neurological deficits: {deficits_str}.\n"
        f"Medical history: {hx_str}.\n"
        f"Imaging: {imaging}.\n"
        f"Clinical trajectory (for calibrating severity of language only — "
        f"do NOT mention in the note): {trajectory}.\n\n"
        "Write 3–5 sentences."
    )


# ── Async note generation ────────────────────────────────────────────

async def generate_note(client, semaphore, row, row_idx, max_retries=3):
    """Generate a single note with retries and prognostic filter."""
    user_prompt = build_user_prompt(row)

    for attempt in range(max_retries):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                note = response.content[0].text.strip()

                if PROGNOSTIC_RE.search(note) and attempt < max_retries - 1:
                    continue  # retry without prognostic language

                return row_idx, note

            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return row_idx, f"[ERROR: {e}]"

    return row_idx, "[ERROR: max retries]"


async def main(concurrency: int = CONCURRENCY, resume: bool = False):
    import anthropic
    import httpx

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACT_DIR / "clinical_notes.json.gz"
    checkpoint_path = ARTIFACT_DIR / "notes_checkpoint.json"

    # Load data
    print("Loading sampled_data.csv...")
    df = pd.read_csv(TVAE_CSV)
    n_total = len(df)
    print(f"  {n_total} rows loaded")

    # Resume from checkpoint
    notes: dict[str, str] = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            notes = json.load(f)
        print(f"  Resumed: {len(notes)} notes from checkpoint")

    remaining = [i for i in range(n_total) if str(i) not in notes]
    print(f"  {len(remaining)} notes to generate")

    if not remaining:
        print("All notes already generated!")
        _save_final(notes, n_total, output_path)
        return

    # Disable SSL verification (needed behind Stanford proxy)
    http_client = httpx.AsyncClient(verify=False)
    client = anthropic.AsyncAnthropic(http_client=http_client)
    semaphore = asyncio.Semaphore(concurrency)

    t0 = time.time()
    done = 0

    for batch_start in range(0, len(remaining), CHECKPOINT_EVERY):
        batch = remaining[batch_start : batch_start + CHECKPOINT_EVERY]

        tasks = [
            generate_note(client, semaphore, df.iloc[i], i) for i in batch
        ]
        results = await asyncio.gather(*tasks)

        for idx, note in results:
            notes[str(idx)] = note
            done += 1

        # Checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(notes, f)

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - done) / rate if rate > 0 else float("inf")
        print(
            f"  [{done}/{len(remaining)}] {rate:.1f} notes/sec, "
            f"ETA: {eta / 60:.1f} min"
        )

    _save_final(notes, n_total, output_path)
    await http_client.aclose()

    elapsed = time.time() - t0
    n_errors = sum(1 for i in range(n_total) if notes[str(i)].startswith("[ERROR"))
    print(f"\nDone! {n_total} notes generated in {elapsed / 60:.1f} min")
    print(f"  Errors: {n_errors}")
    print(f"  Saved to {output_path}")

    if n_errors == 0 and checkpoint_path.exists():
        checkpoint_path.unlink()


def _save_final(notes, n_total, output_path):
    """Save notes in order as compressed JSON list."""
    ordered = [notes[str(i)] for i in range(n_total)]
    with gzip.open(output_path, "wt") as f:
        json.dump(ordered, f)
    print(f"Saved {n_total} notes to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate clinical notes for IST-3 Text DGP"
    )
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    args = parser.parse_args()
    asyncio.run(main(concurrency=args.concurrency, resume=args.resume))
