"""scIntegrate — is integration needed, which method, and one object carrying the answer.

Three steps, in the order the questions actually arrive:

    scintegrate doctor      what is installed, and what each absence costs you
    scintegrate assess      is integration needed? per cell type and per group, nothing trained
    scintegrate integrate   run the methods, score with scIB, name a default, write one object

It compares methods INCLUDING doing nothing, measures every correction against what it cost in
retained structure, draws every method at one scale, names a default embedding on the scIB total,
and states separately - computed from the design, not assumed - what that embedding may be used
for.
"""
__version__ = "0.3.0"
