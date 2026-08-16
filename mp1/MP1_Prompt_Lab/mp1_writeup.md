# MP1 Writeup — Simple Summary

This is a short, easy-to-read summary of the prompt strategy comparison.

## 1) Which strategy won?
- Few-shot did best. It had the highest average score (2.8 out of 3) and was fast. The other strategies were close (about 2.6 out of 3).

## 2) What surprised me?
- The parser worked for every response — we could get a JSON object each time, even for messy snippets.
- Few-shot beat chain-of-thought here. Giving 2 example pairs helped the model more than asking it to "think step-by-step."

## 3) Which strategy would I use for my project?
- Start with few-shot. It is simple and works well for extracting fields. Make 2–4 clear examples and test on a few snippets.

## 4) If I had one more day, what would I try?
- Try different few-shot examples to see which help most.
- Test on more snippets (20–50) to check how well it generalizes.
- Try combining two strategies (for example, run few-shot and structured and pick the most common answer).

---

How to reproduce
- Results are in `results/mp1_results.jsonl` and `results/mp1_comparison.md`.
- The notebook used `gpt-4o-mini` with temperature 0.0 and a judge model for extra scoring.

Date: 2026-08-15
