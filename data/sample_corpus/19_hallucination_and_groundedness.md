# Hallucination Mitigation and Groundedness in LLMs

A prominent operational barrier to deploying large language models in enterprise contexts is hallucination—the generation of statements that contradict verifiable external facts or source documents.

## Forms of Hallucination

Hallucination in natural language generation manifests in two primary ways:

1. **Intrinsic Hallucination**: The generated text directly contradicts evidence explicitly present in the input context (e.g. source context states "Net profit was $4M" while generation states "Net profit dropped to zero").
2. **Extrinsic Hallucination**: The generated text introduces facts, entities, or claims that are absent from the source context (e.g. fabricating hypothetical product features not described in provided documentation). While sometimes benign if factually true in reality, extrinsic hallucinations are hazardous in audit-regulated environments requiring strict provenance.

## Quantitative Groundedness Metrics

Measuring groundedness requires comparing the generated tokens directly against the retrieved reference passages:

- **Token Overlap / Containment**: Evaluates the proportion of substantive non-stopword tokens in the generated response that appear in the retrieved context documents.
- **Citation Precision**: Calculates the fraction of inline citations (e.g. `[1]`, `[2]`) that genuinely support the specific claim they accompany.
- **Refusal Accuracy**: Measures how reliably the system produces an explicit abstention ("I don't know based on the provided documents") when the retrieved context lacks sufficient evidence to answer the user query.

## Architectural Safeguards

To minimize hallucination, production RAG pipelines enforce several constraints:
- System instructions strictly forbidding speculation and demanding explicit attribution.
- Score gating: if the maximum retrieval confidence score falls below a safety threshold $\tau$, the pipeline immediately terminates generation and returns a standard refusal string.
- Verification agents: lightweight secondary models check generated claims against retrieved citations before final delivery to the end user.
