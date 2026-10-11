# MP3 Demo Video Run Sheet (2–3 minutes)

The video must show three distinct behaviors: a cited answer, a question whose answer is absent from the corpus, and a question whose answer is in the corpus but missing from the retrieved excerpts. Use the live `ask` command for each example. Do not show `.env`, API keys, or account dashboards.

## Before recording

- Ensure the collection has been ingested and `.env` contains working credentials.
- In PowerShell, load `.env` and start the assistant:

```powershell
Get-Content .env | ForEach-Object {
	if ($_ -match '^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
		Set-Item -Path "Env:$($matches[1])" -Value $matches[2].Trim('"')
	}
}
python .\mp3_grounded_qa.py ask
```

Default behavior for this demo is top-five retrieval (`k=5`) and `RETRIEVAL_THRESHOLD=0.50`.

## Recording script (about 2:30)

### 1. Intro (15 seconds)

Say: "This assistant answers from five Sherlock story summaries. I’ll show a cited answer, a question outside the corpus, and a case where the corpus has the answer but retrieval misses the supporting excerpt."

### 2. Cited answer (35 seconds)

Ask:

`In A Scandal in Bohemia, who is Irene Adler and what does the King of Bohemia want from her?`

Show the answer, inline `[n]` marker, and matching `05_scandal_in_bohemia.txt` source. Say that the citation number maps to a retrieved excerpt.

### 3. Answer absent from corpus (25 seconds)

Ask:

`Who is Professor Moriarty, and how does he meet his end?`

Show `I don't know based on the provided documents.` and the below-threshold abstention diagnostic. Explain that Moriarty is not in these five stories, so refusing is correct.

### 4. Answer in corpus, evidence missed by retrieval (55 seconds)

Ask:

`What was the exact title Holmes used for Irene Adler?`

Show the same `I don't know` response and the retrieved source scores. Explain: "The corpus does contain this fact, but the top-five excerpts did not retrieve the supporting line, so the assistant has no evidence to answer from." Briefly show the opening of `corpus/05_scandal_in_bohemia.txt` in the editor, where it says, `To Sherlock Holmes she is always THE woman.` Point out that this supporting phrase was absent from the retrieved context. Do not imply the corpus lacks the answer; this is a retrieval failure/false abstention.

### 5. Close (20 seconds)

Show `mp3_eval.txt` briefly. Say that its metrics are from the fixed evaluation questions, while the Irene Adler title query is a separate diagnostic example demonstrating a remaining retrieval limitation. The assistant correctly abstains rather than guessing, but retrieval quality should improve so supported facts are found more consistently.

## Recording notes

- Keep the terminal and editor at a readable font size; close unrelated tabs and hide the `.env` tab.
- After the diagnostic third question, show the corpus sentence only as verification; do not type or display credentials.
- On Windows, start/stop screen recording with `Win+Alt+R` (Xbox Game Bar) if available, or use Clipchamp/OBS.