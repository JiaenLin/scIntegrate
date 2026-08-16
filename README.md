# scIntegrate

**Is integration needed?** — and what would each method cost.

scIntegrate compares integration methods **including doing nothing**, measures mixing against
what each correction cost in retained structure, draws every method at one scale, ranks them under a weighting you set, and
**withholds a recommendation when the design makes one unsafe**.

That refusal is the design. Whether integration is needed is a question about what the study is
for: a batch effect that must be removed before a composition claim is the same signal another
question depends on, and nothing this tool measures knows which question is being asked.

```bash
scintegrate assess --h5ad joint.h5ad --out results/03_integrate \
    --batch-key sample --label-key cell_type --methods none,harmony,bbknn
```

## It recommends — and refuses to, when a recommendation would be wrong

`--bio-factor age --bio-factor diet` declares what the study is actually about. Each is tested
for **nesting inside the batch key**, and the composite score becomes a recommendation only when
none is nested.

The score is the scIB convention: `w_bio` on biological conservation (kNN retention + label
coherence), the rest on batch removal (mixing as a ratio to chance, **clipped at 1.0** — past
chance is over-correction, not better integration). `--w-bio` defaults to 0.6 and is exposed
because **that weight is the value judgement**, not a constant of nature.

The refusal is the part worth reading. If every batch carries exactly one value of `age`, then
correcting the batch removes the age difference with it — and the score will rank that method
first, because it mixed the batches, which is what it was asked to do and what destroys the
comparison. **A number that ranks confidently in exactly the case where it is wrong is worse
than no number.** So the ranking is still printed, still labelled as "what each method did", and
the recommendation is withheld with the blocking factor named.

## Three ideas it is built on

**`none` is a method, and it is never optional.** A comparison that omits the un-integrated case
can only answer which correction is strongest — never whether any was warranted. If you leave it
out of `--methods`, it is put back.

**Mixing is measured against chance, not against 100%.** The share of foreign neighbours you
would get from random mixing depends on how big the batches are: for two equal batches it is
0.50, for a 90/10 split it is 0.18. A raw "72% foreign" means nothing without the number it is
being compared to, so the report gives a ratio to chance where **1.00× is fully mixed**.

**Mixing is only ever reported beside what it cost.** Any method can score perfectly on mixing by
destroying the structure — every neighbourhood becomes a random sample of libraries. So retention
of the original kNN is reported next to it. Higher mixing with lower retention is a trade, not a
victory.

## Every panel at the same scale

No integration decision should be made on metrics alone. The report draws each method coloured by
cell type, by batch, and with one batch highlighted against everything else — all sharing one set
of axis limits, because per-panel autoscaling makes a dispersed method look compact.

A number can say a population was mixed. Only the picture distinguishes *aligned with its
counterparts* from *dispersed everywhere*.

## Install

```bash
git clone https://github.com/JiaenLin/scIntegrate.git && cd scIntegrate
pip install -e '.[run]'                       # anndata + scanpy + matplotlib
pip install -e '.[run,harmony,bbknn]'         # and the methods you want compared
python tests/test_assess.py
```

The core is numpy + scikit-learn. A method whose package is absent is **named in the report** as
not compared, rather than silently dropped — a method missing from a comparison changes what the
comparison means.

## What it cannot tell you

Nothing here establishes that a batch effect is technical. Where a biological factor is
confounded with a batch factor — age with chemistry, say — a method that removes "batch" may be
removing age, and no mixing statistic can separate them. The figures show you what was moved;
they cannot tell you whether it should have been.
