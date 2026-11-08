# ELEC5305 project requirements (supplied by the student, 8 November 2026)

Deliver working code downloadable through GitHub, a research-style written report
(typically 10 pages including figures), and a brief video explaining the signal
processing code and demonstrating it in action. Demonstrations and sound examples
are encouraged. Submission is due at the end of Week 13; an exact calendar deadline
has not been supplied.

The report must motivate one research question; analyse relevant publications;
justify methods; present reproducible tests and clear figures with interpretation;
and discuss what was learned, limitations, unsuccessful aspects and future work.
The intended scope is reproduction of an existing approach followed by a small,
well-justified modification, rather than a large research programme.

| Criterion | Weight |
|---|---:|
| Originality and personal contribution | 10% |
| Command of subject | 10% |
| Introduction | 5% |
| Literature review | 15% |
| Methodology | 15% |
| Results | 10% |
| Discussion and conclusion | 25% |
| Presentation, format and references | 10% |

Distinction descriptors emphasize analytical comparison of literature and models,
a justified method derived from strengths and weaknesses of previous work,
precise analysis linked to the research question, interpretation grounded in theory,
limitations and ways to address them, and a cohesive, concise report.

## Application to this project

One question links overlap, mixture phase and vocal-separation errors. The original
power-IRM oracle and true-magnitude/mixture-phase oracle are reproduced. A clipped
phase-sensitive mask and a nonnegative unbounded version isolate mask constraints.
These are established oracle concepts, not newly invented algorithms. The project
contribution is the paired musical experiment, gain intervention, verified phase
identity, and analysis of when the reference models agree with the oracle trends.

The publication discussion must distinguish training data, inference context and
metric definitions. Learned models do not require true stems at inference; oracle
methods do. Quantitative scores are not subjective quality measurements. No
personal listening observations or formal listener ratings may be invented.

A second iteration evaluates surrounding inference context on every retained natural excerpt, scoring the same centre samples. It tests a concrete short-input failure without retraining.
