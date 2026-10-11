# MP3 Reflection

## Grounding & abstention
I measured the maximum retrieval scores before choosing `RETRIEVAL_THRESHOLD=0.50`: the lowest answerable score was 0.528, Moriarty scored 0.482, and the out-of-domain water question scored 0.144. This separates the tested positives from negatives, though the 0.046 gap between the closest cases is narrow and should not be treated as a universal cutoff.

The first live run used three retrieved chunks and grounded only 2/6 answerable questions. Inspection showed that the source story was present but the excerpts often omitted a needed fact. Retrieving five chunks and clarifying that the model may combine evidence across excerpts raised grounded response rate to 6/6 while all three should-abstain questions remained refused. The threshold, larger context, and prompt clarification were evaluated together in the final run, so this small experiment does not isolate the effect of each change.

## Citations
The validator recognizes bracketed integer markers and classifies them as valid only when they refer to one of the retrieved sources. The offline selftest checks real markers and a fabricated `[3]` when only two sources exist; the fabricated marker is reported as invalid. In the final run, all six answerable responses had a valid citation to the expected story and no invalid marker according to the evaluator.

That metric does not verify that a cited passage semantically supports every claim. I manually checked the demo answers against their excerpts; a stronger evaluation would inspect claim-level support for every answer, not just marker range and expected source.

I also tested a separate retrieval-miss example for the demo: "What was the exact title Holmes used for Irene Adler?" The corpus contains the line "To Sherlock Holmes she is always THE woman," but that phrase was absent from the five retrieved excerpts. The assistant abstained even though the top score (0.611) exceeded the 0.50 gate, because the generated context did not contain the answer. This is a false abstention caused by retrieval/context selection, and it is not included in the fixed evaluation metrics above.

## Reading your eval table
The final table covered six answerable questions and three should-abstain questions. Retrieval hit rate was 1.0, grounded-and-cited rate was 1.0, abstention accuracy was 1.0, and hallucination rate was 0.0. On this small set, every answerable question was answered with a valid citation to its expected story, and every should-abstain question was refused. The 0.50 cutoff is close to the lowest positive score, so I would test it on a larger and more varied question set before relying on it beyond this corpus.

## One surprise
The evaluation separates retrieval success from answer success. The initial hit rate was already 1.0 while grounded rate was only 0.3333; finding the right story did not mean the top three excerpts contained the needed facts. Increasing context to five excerpts and allowing evidence to be combined addressed that mismatch, and the final grounded rate reached 1.0.
