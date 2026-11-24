# 04 — Safety and Evaluation

## Safety model

Safety is a system property, not a single prompt, regex, or model capability. KatAI uses multiple independent checks and a conservative failure policy.

## Scope enforcement

The system must distinguish:
- a genuine English-learning request;
- a non-English-learning request written in English;
- an unrelated task that merely contains an English word;
- a mixed request with separable English-learning and unrelated parts;
- an ambiguous request.

For clearly out-of-scope requests, do not invoke the language model. Return exactly:

```text
Hiện tại, chúng tôi không hỗ trợ các môn học khác ngoài Tiếng Anh. Xin cảm ơn
```

Test both Vietnamese and English requests, paraphrases, spelling errors, slang, and attempts to disguise another subject as an English exercise.

## Safety categories

The policy and evaluation set should cover:
- age-appropriate explanations of sensitive topics in legitimate language-learning contexts;
- requests for dangerous instructions or wrongdoing;
- harassment, hateful or abusive language;
- sexual or exploitative content;
- self-harm-related content;
- privacy, personal data, and credential disclosure;
- attempts to override system policy or extract hidden instructions.

The model should not provide dangerous operational instructions. For sensitive language-learning vocabulary, it may provide brief, neutral definitions when safe and relevant, without graphic or actionable detail.

## Layered controls

1. Request validation and rate limiting.
2. Input risk screening.
3. Scope decision.
4. Task-specific generation constraints.
5. Output safety and scope screening.
6. Logging/monitoring that minimizes sensitive content.
7. Regression testing on every policy or model change.

No single layer is treated as infallible. A classifier can make mistakes; measure them.

## Evaluation datasets

Maintain separate, versioned sets for:
- in-scope positive examples;
- out-of-scope negative examples;
- mixed-intent examples;
- ambiguous examples;
- safety examples;
- adversarial/prompt-injection examples;
- output-quality examples by task and learner level.

Avoid training/test leakage. Keep a held-out set that is not used to tune rules or model weights.

## Metrics

### Scope
- False accept: out-of-scope request incorrectly allowed.
- False reject: valid English-learning request incorrectly blocked.
- Precision and recall by task and language.
- Results by Vietnamese/English and mixed-language phrasing.

False accepts are particularly important because they undermine the product's strict scope. Still, reducing false accepts must not make ordinary English vocabulary or grammar questions unusable.

### Learning quality
- correctness;
- clarity and age appropriateness;
- relevance to the requested task;
- quality of examples;
- calibrated uncertainty;
- rate of fabricated rules or unsupported claims.

### Safety
- unsafe completion rate on the safety evaluation set;
- policy bypass success rate;
- output screening catch rate;
- regression failures by category.

### Operations
- latency percentiles;
- CPU and memory use;
- timeout and error rate;
- request volume and rate-limit behavior.

## Release gates

Do not enable a task in production until:
- scope tests for that task pass;
- safety tests pass;
- held-out quality reaches a threshold agreed before evaluation;
- limitations are documented;
- rollback is possible.

Do not claim that the system is perfectly safe. Publish the tested boundaries and known failure modes.
