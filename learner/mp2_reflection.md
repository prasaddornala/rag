# MP2 Reflection

## What worked

The structure-aware chunker worked well for this corpus because each chunk
retains its story title and section heading. The dense Qdrant search and BM25
search complemented each other: dense retrieval handled semantic wording while
BM25 preserved exact names and terms. Reciprocal Rank Fusion gave both signals a
role before the cross-encoder selected the final evidence.

## What didn't work

My first chunking loop could stall when a paragraph was longer than the
remaining space in the current chunk. I fixed it by emitting the current chunk,
carrying its 80-character overlap forward, and ensuring that the remaining text
always advanced. I also found that using only three final chunks could omit a
needed fact: the question about Vincent Spaulding retrieved the right story but
initially missed the section identifying him as John Clay.

## What I'd change

With more time, I would cache the parsed chunks and BM25 index instead of
rebuilding them for every question. I would also evaluate several candidate and
final-k values systematically, recording source-match, fact coverage, latency,
and cost. For this small corpus, increasing the final evidence set from k=3 to
k=5 preserved the relevant identity section without adding excessive context.

## One surprise

I expected the cross-encoder to identify every obviously relevant fact, but it
ranked the "The Identity of the Assistant" section below broader sections about
the Red-Headed League. That made the hybrid design useful rather than cosmetic:
the lexical BM25 signal and a larger final evidence set helped retain the exact
fact needed for a grounded answer. The supplied validation also showed that
source-match is a more reliable retrieval diagnostic than simple substring-based
fact matching, because correct answers can use different wording.