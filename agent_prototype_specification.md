No clarification is necessary for a first prototype. I would deliberately keep **v0 narrower than the eventual research system**: one persistent reasoning agent, one target LLM, pre- and post-generation screening, conversation-level state, structured tool calling, and ToxinPred2 as the first biological tool. Do **not** implement MTSA training yet; design the interfaces so that trajectory-level alignment can be added later.

The specification below is grounded in the four papers. Meng & Zhang motivates pre/post runtime guarding and tool orchestration; MTSA motivates treating conversation history as safety-relevant state; BioSecBench-Refusal motivates distinguishing underlying intent from risky-looking vocabulary; and SPIKE motivates modality-specific biological tools because text reasoning alone may not adequately assess biological artifacts.    

# Technical Specification: Prototype Multi-Turn Biosafety Defense Agent

## 1. Objective and Scope

### 1.1 Prototype objective

Build an inference-time defense layer positioned between a user and an arbitrary target LLM:

```text
                  ┌──────────────────────────────────────┐
                  │      BIOSAFETY DEFENSE AGENT         │
                  │                                      │
User ───────────► │  Pre-generation safety reasoning     │
                  │          │                           │
                  │          ├──► Biological tools       │
                  │          │      └─ ToxinPred2        │
                  │          ▼                           │
                  │     Policy decision                  │
                  └──────────┬───────────────────────────┘
                             │ ALLOW
                             ▼
                       ┌────────────┐
                       │ Target LLM │
                       └─────┬──────┘
                             │ candidate response
                             ▼
                  ┌──────────────────────────────────────┐
                  │      BIOSAFETY DEFENSE AGENT         │
                  │                                      │
                  │  Post-generation safety reasoning    │
                  │          │                           │
                  │          ├──► Biological tools       │
                  │          │      └─ ToxinPred2        │
                  │          ▼                           │
                  │     Policy decision                  │
                  └──────────┬───────────────────────────┘
                             │
                  ┌──────────┼──────────────┐
                  ▼          ▼              ▼
                ALLOW      REFUSE       SAFE_RESPONSE
```

The critical difference from a conventional input classifier is that the defense agent receives the **conversation trajectory**, not merely the latest user message.

For turn \(t\), define:

$$
H_t = \{u_1,a_1,u_2,a_2,\ldots,u_t\}
$$

where \(u_i\) is a user message and \(a_i\) is the response actually delivered to the user.

The pre-generation decision becomes approximately:

$$
D_t^{pre}=G(H_t,T_t)
$$

where \(T_t\) represents optional evidence from tools.

Following target generation \(y_t\):

$$
D_t^{post}=G(H_t,y_t,T'_t)
$$

This extends Meng & Zhang's pre-guard/model/post-guard pipeline by making \(G\) a **stateful reasoning-and-tool-use agent** rather than primarily a fixed collection of detectors. Meng & Zhang explicitly place guards on both sides of the target LLM, while MTSA models safety as history-dependent across previous dialogue turns.  

### 1.2 What v0 should demonstrate

The prototype is successful if it can:

1. maintain an entire conversation;
2. inspect the new user request in context;
3. infer an evolving user objective;
4. produce a structured biosafety risk assessment;
5. recognize when a biological sequence requires specialized assessment;
6. invoke ToxinPred2 automatically;
7. incorporate the result into its assessment;
8. decide whether a request reaches the target LLM;
9. independently inspect the target LLM's candidate response;
10. prevent the candidate from reaching the user when necessary;
11. persist safety state across subsequent turns;
12. produce machine-readable logs explaining each decision.

**High detection performance is explicitly not a v0 requirement.**

---

# 2. Design Principles

The implementation should preserve six conceptual separations.

| Concept A               | Must remain distinct from      |
| ----------------------- | ------------------------------ |
| Suspicious terminology  | Underlying user intent         |
| Current-turn risk       | Trajectory-level risk          |
| User intent             | Technical capability requested |
| Input risk              | Output risk                    |
| LLM semantic assessment | Biological-tool evidence       |
| Risk assessment         | Policy/action decision         |

These distinctions follow directly from the combined literature. BioSecBench shows that legitimate biological tasks can trigger safeguards more readily than concealed hazards, while SPIKE demonstrates that refusal behavior and functional biological risk can diverge.  

---

# 3. Prototype Components

The minimum system should contain seven components:

```text
biosafety-defense/
│
├── gateway/
│   └── conversation_controller.py
│
├── defense_agent/
│   ├── agent.py
│   ├── prompts.py
│   ├── schemas.py
│   └── policy.py
│
├── models/
│   ├── reasoning_backend.py
│   └── target_backend.py
│
├── memory/
│   ├── conversation_store.py
│   └── safety_state.py
│
├── tools/
│   ├── registry.py
│   ├── sequence_parser.py
│   └── toxinpred2.py
│
├── logging/
│   └── audit_logger.py
│
├── configs/
│   ├── models.yaml
│   ├── policy.yaml
│   └── tools.yaml
│
├── tests/
│   ├── unit/
│   └── scenarios/
│
└── app.py
```

For v0, avoid unnecessary agent frameworks. Plain Python with explicit control flow will be easier to debug than a large LangChain/LangGraph-style abstraction.

---

# 4. Reasoning Model

## 4.1 Requirements

The defense model needs:

* instruction following;
* reasonable multi-turn reasoning;
* reliable structured output;
* sufficient context length;
* local deployment;
* deterministic or near-deterministic inference;
* tool-selection capability.

The prototype does **not** require biological generation capability.

A 7B-14B instruct model is sufficient for proving the architecture.

Examples of suitable model classes include contemporary Qwen/Llama/Mistral-family instruct checkpoints available to you on the HPC. The exact checkpoint should be treated as a configuration parameter rather than embedded into the architecture.

For development:

```yaml
reasoning_model:
  backend: vllm
  model: "<local-instruct-checkpoint>"
  tensor_parallel_size: 1
  temperature: 0.0
  max_tokens: 1500
  max_context_tokens: 32768
```

Use temperature 0 or very low temperature initially. Safety decisions need reproducibility more than creativity.

## 4.2 Serving

Run the reasoning model as an independent service:

```text
GPU node
│
├── vLLM reasoning-agent server
│
├── optional target-model server
│
└── CPU/GPU biological tools
```

Expose the reasoning model through an OpenAI-compatible local endpoint if using vLLM.

This gives:

```python
reasoner.chat(messages, response_format=...)
```

without coupling the application to a specific inference engine.

---

# 5. Conversation Representation

This is one of the most important implementation decisions.

Do not pass only:

```json
{
  "message": "current user message"
}
```

Pass a structured trajectory:

```json
{
  "conversation_id": "abc123",
  "turn": 5,

  "history": [
    {
      "turn": 1,
      "role": "user",
      "content": "..."
    },
    {
      "turn": 1,
      "role": "assistant",
      "content": "..."
    }
  ],

  "current_user_message": "...",

  "previous_safety_state": {...}
}
```

MTSA's central methodological contribution is precisely that safety should be conditioned on the state produced by previous dialogue, rather than solely the current query. 

---

# 6. Persistent Safety State

Do not force the reasoning model to reconstruct everything from raw history every turn.

Maintain a compact structured state.

Example:

```json
{
  "conversation_id": "abc123",
  "turn": 5,

  "inferred_intent": {
    "summary": "User appears to be conducting legitimate protein analysis.",
    "confidence": 0.62
  },

  "risk_state": {
    "current_turn_risk": 0.18,
    "trajectory_risk": 0.24,
    "artifact_risk": 0.00
  },

  "observed_capabilities": [
    "protein_sequence_analysis"
  ],

  "trajectory_signals": [],

  "biological_entities": [],

  "tool_evidence": [],

  "previous_decisions": [
    "ALLOW",
    "ALLOW",
    "ALLOW"
  ]
}
```

The model receives both:

```text
raw conversation history
+
previous structured safety state
```

and returns an updated state.

### Why both?

Raw history prevents errors in the generated summary from becoming irreversible.

Structured state provides a compact persistent representation for long conversations.

---

# 7. Intent Model

For the prototype, **do not train a separate intent classifier**.

Use the reasoning model itself.

The agent should answer several distinct questions:

```text
1. What is the user's apparent immediate objective?

2. What is the likely cumulative objective across the conversation?

3. Has the apparent objective changed?

4. Are the individual requests being decomposed into capabilities?

5. Does the current request become more concerning when previous
   requests are considered?

6. Is there evidence of legitimate scientific context?

7. Is there evidence of harmful intent?

8. How certain is the inference?
```

Critically, absence of benign context is **not** evidence of malicious intent.

The output should therefore support:

```text
BENIGN
AMBIGUOUS
CONCERNING
MALICIOUS
UNKNOWN
```

rather than simply:

```text
SAFE / UNSAFE
```

This follows the BioSecBench lesson that risky-looking biological language and underlying biological risk are not equivalent. 

---

# 8. Risk Representation

Avoid one scalar as the only internal representation.

Use several dimensions.

```json
{
  "intent": {
    "class": "AMBIGUOUS",
    "confidence": 0.67
  },

  "request_risk": {
    "level": "MEDIUM",
    "score": 0.48
  },

  "trajectory_risk": {
    "level": "MEDIUM",
    "score": 0.55
  },

  "artifact_risk": {
    "level": "UNKNOWN",
    "score": null
  },

  "uncertainty": 0.31
}
```

For v0, scores can simply be model-generated values from 0 to 1.

They do **not** need to be calibrated probabilities.

Document this explicitly:

> Scores represent ordinal model confidence/risk estimates and must not be interpreted as calibrated probabilities.

---

# 9. Reasoning-Agent Output Schema

Never parse free-form prose to determine safety decisions.

Require JSON.

Recommended schema:

```json
{
  "analysis_summary": "Short safety-relevant summary.",

  "current_user_intent": {
    "classification": "BENIGN|AMBIGUOUS|CONCERNING|MALICIOUS|UNKNOWN",
    "summary": "...",
    "confidence": 0.0
  },

  "trajectory_analysis": {
    "summary": "...",
    "risk_score": 0.0,
    "signals": [
      {
        "turn": 3,
        "type": "ESCALATION",
        "description": "..."
      }
    ]
  },

  "biological_content": {
    "present": true,
    "artifact_types": [
      "PROTEIN_SEQUENCE"
    ]
  },

  "tool_requests": [
    {
      "tool": "toxinpred2",
      "input_ref": "artifact_1",
      "reason": "Protein sequence requires domain-specific assessment."
    }
  ],

  "risk_assessment": {
    "input_risk": 0.0,
    "trajectory_risk": 0.0,
    "artifact_risk": null,
    "overall_risk": 0.0,
    "uncertainty": 0.0
  },

  "recommended_action": "ALLOW|REFUSE|SAFE_RESPONSE|REVIEW",

  "state_update": {
    "intent_summary": "...",
    "risk_trend": "DECREASING|STABLE|INCREASING",
    "relevant_observations": []
  }
}
```

---

# 10. Do Not Expose Hidden Reasoning

The model should internally reason but return only a **concise safety rationale**.

Do not design the application around storing unrestricted chain-of-thought.

Store:

```json
{
  "rationale": "Risk increased because the current request depends on capabilities requested in turns 2 and 3."
}
```

not lengthy private reasoning traces.

This reduces storage burden and avoids coupling your system to verbose model-specific reasoning behavior.

---

# 11. Tool Architecture

The reasoning model should not execute arbitrary commands.

Expose an explicit tool registry.

```python
class Tool:
    name: str
    description: str

    def validate_input(self, data):
        ...

    def run(self, data):
        ...

    def normalize_output(self, result):
        ...
```

Registry:

```python
TOOLS = {
    "toxinpred2": ToxinPred2Tool(...)
}
```

Later:

```text
sequence_validator
structure_predictor
similarity_search
pathogenicity_classifier
biosecurity_database_lookup
```

can be added without changing the agent architecture.

---

# 12. Sequence Extraction

Before calling ToxinPred2, create a deterministic parser.

Do **not** ask the LLM to copy a sequence into a tool argument if it can be extracted programmatically.

Pipeline:

```text
text
 ↓
candidate sequence detector
 ↓
normalization
 ↓
validation
 ↓
artifact object
```

Example internal object:

```json
{
  "artifact_id": "artifact_1",
  "type": "PROTEIN_SEQUENCE",
  "source": "target_output",
  "length": 183,
  "normalized_sequence": "<sequence>",
  "validation": {
    "alphabet_valid": true,
    "length_valid": true
  }
}
```

The sequence itself should not be duplicated unnecessarily in logs.

Use hashes:

```text
sha256(sequence)
```

for audit correlation.

---

# 13. ToxinPred2 Integration

SPIKE uses ToxinPred2 because it can operate directly on amino-acid sequences and reports strong sensitivity on their ground-truth comparison; importantly, the paper treats it as one computational signal rather than proof of real-world toxicity. 

Your wrapper should preserve this distinction.

Interface:

```python
class ToxinPred2Tool:

    def predict(self, sequence: str) -> ToxinPrediction:
        ...
```

Normalized return object:

```json
{
  "tool": "toxinpred2",
  "version": "...",

  "artifact_id": "artifact_1",

  "prediction": {
    "label": "TOXIN",
    "score": 0.87
  },

  "status": "SUCCESS",

  "interpretation": {
    "type": "COMPUTATIONAL_SCREEN",
    "experimental_validation": false
  }
}
```

The reasoning agent should see something like:

```text
ToxinPred2:
Predicted class: toxin
Confidence/score: 0.87
Interpretation: computational screening result only.
```

It should **not** receive:

> "This sequence is definitely toxic."

SPIKE explicitly cautions that computational screening does not establish experimental functionality, potency, delivery, or real-world harm. 

---

# 14. Tool Invocation Loop

Use an explicit bounded loop.

```python
MAX_TOOL_ROUNDS = 3

assessment = reasoner.assess(context)

for _ in range(MAX_TOOL_ROUNDS):

    if not assessment.tool_requests:
        break

    observations = tool_executor.run(
        assessment.tool_requests
    )

    context.tool_observations.extend(observations)

    assessment = reasoner.assess(context)
```

Never allow recursive unlimited tool invocation.

For v0:

```text
max tool calls / safety evaluation = 3
timeout / tool = 30-60 s
```

Tool failure must not silently produce `SAFE`.

---

# 15. Tool Failure Semantics

Return explicit errors:

```json
{
  "tool": "toxinpred2",
  "status": "FAILED",
  "error_type": "TIMEOUT"
}
```

The policy engine can then choose:

```text
ALLOW
REVIEW
REFUSE
```

according to configuration.

Do not implement:

```python
try:
    result = toxinpred(...)
except:
    result = "not toxin"
```

A failed safety tool is **unknown evidence**, not negative evidence.

---

# 16. Pre-Generation Pipeline

At every user turn:

```text
Receive user input
      │
      ▼
Append provisional user turn
      │
      ▼
Extract biological artifacts
      │
      ▼
Load conversation + safety state
      │
      ▼
Defense reasoning pass #1
      │
      ├── no tools ───────────────┐
      │                           │
      └── tools requested         │
             │                    │
             ▼                    │
        execute tools             │
             │                    │
             ▼                    │
      Defense reasoning pass #2 ◄─┘
             │
             ▼
       Policy engine
        │          │
      ALLOW      BLOCK
        │          │
        ▼          ▼
    Target LLM   Safe response
```

---

# 17. Target LLM Isolation

The target model must never communicate directly with the user.

Enforce:

```text
User
 ↓
Defense
 ↓
Target
 ↓
Defense
 ↓
User
```

not:

```text
User
 ↓
Defense ───► Target ───► User
```

The target's response is always a **candidate response** until the post-generation guard approves it.

---

# 18. Post-Generation Pipeline

After the target generates:

```text
candidate_response
       │
       ▼
extract biological artifacts
       │
       ▼
semantic safety reasoning
       │
       ├──► ToxinPred2 when appropriate
       │
       ▼
integrate tool evidence
       │
       ▼
policy decision
```

The post-guard should receive:

```json
{
  "conversation_history": "...",
  "current_user_request": "...",
  "pre_guard_assessment": "...",
  "candidate_target_response": "...",
  "extracted_artifacts": [],
  "tool_results": []
}
```

This lets the model ask:

> Is this output problematic **given what the user has been trying to accomplish?**

rather than assessing the output without context.

---

# 19. Artifact-Triggered Tool Routing

For v0, make routing hybrid rather than entirely agent-driven.

Use deterministic detection:

```python
artifacts = sequence_parser.extract(text)

if artifacts:
    available_tools.append("toxinpred2")
```

Then allow the reasoning model to decide whether it needs the tool.

For **post-generation protein sequences**, I would make ToxinPred2 invocation automatic in v0:

```python
if artifact.type == "PROTEIN_SEQUENCE":
    run_toxinpred2 = True
```

This is simpler and safer than relying on the reasoning model to remember to call it.

Thus:

```text
Agent-driven routing
+
mandatory deterministic routing for selected artifact classes
```

---

# 20. Policy Engine

Keep policy outside the LLM.

The LLM produces:

```text
risk evidence
+
recommended action
```

The deterministic policy engine makes the actual decision.

For example:

```python
def decide(a):

    if a.tool_failure_on_required_check:
        return REVIEW

    if a.intent == "MALICIOUS" and a.intent_confidence >= 0.8:
        return REFUSE

    if a.overall_risk >= 0.8:
        return REFUSE

    if a.overall_risk >= 0.5:
        return SAFE_RESPONSE

    return ALLOW
```

These thresholds are **prototype placeholders**, not literature-derived calibrated values.

Put them in configuration:

```yaml
policy:
  refuse_threshold: 0.80
  safe_response_threshold: 0.50
  review_on_required_tool_failure: true
```

BioSecBench is especially relevant here: technical risk assessment and the normative decision to refuse should not be treated as identical problems. 

---

# 21. Intervention Types

Support four states from the beginning:

### `ALLOW`

Send candidate output unchanged.

### `REFUSE`

Do not call the target model during pre-screening, or suppress its candidate response during post-screening.

### `SAFE_RESPONSE`

Allow a separate constrained response path.

For v0 this can simply be:

```text
I can't assist with that request. I can help with a safer,
non-operational discussion of the topic instead.
```

Do not have the target model "sanitize its own unsafe response" yet.

### `REVIEW`

Internal fail-closed/fail-open testing state for uncertainty and tool errors.

During development, expose this in logs even if the UI maps it to refusal.

---

# 22. Defense Agent System Prompt

A good initial system prompt should establish the agent as a **risk assessor**, not a maximally conservative refusal model.

Conceptually:

> You are a biosafety risk-assessment agent operating between a user and another language model.
>
> Evaluate the current interaction using the entire conversation history.
>
> Distinguish biological subject matter from harmful intent. Legitimate scientific discussion must not be classified as malicious merely because it contains pathogens, toxins, genetic engineering, or other dual-use terminology.
>
> Infer the user's immediate objective and the objective emerging across the conversation. Consider whether requests that are individually benign combine into a concerning trajectory.
>
> Assess both intent and the capabilities/information being requested.
>
> When specialized biological artifacts are present, use available tools rather than assuming that natural-language reasoning can determine their biological properties.
>
> Treat tool outputs as evidence, not unquestionable truth.
>
> Return only the specified structured schema.

Then provide explicit definitions for:

```text
BENIGN
AMBIGUOUS
CONCERNING
MALICIOUS
UNKNOWN
```

and the available tools.

---

# 23. Two-Pass Agent Reasoning

Use at least two logical phases inside the prompt.

### Phase A — understand

Determine:

```text
current objective
conversation objective
risk evolution
relevant biological entities
available artifacts
uncertainty
```

### Phase B — decide what evidence is missing

Determine:

```text
Can I assess this using context alone?

Is there a biological artifact?

Which available tool can resolve uncertainty?

Is the tool actually relevant?
```

Then issue tool requests.

After tools return, run the same model again with:

```text
TOOL OBSERVATIONS
```

and require the final risk decision.

---

# 24. Safety-State Update

After every completed turn:

```python
state = update_state(
    previous_state,
    pre_assessment,
    post_assessment,
    final_action
)
```

Important: update the history with **what the user actually received**, not the blocked target response.

Keep blocked candidate outputs in the audit log, but not in the conversational transcript presented as assistant messages.

---

# 25. Long Conversations

Initially, pass the complete conversation while it fits.

When context becomes large:

```text
recent raw turns
+
structured safety state
+
older safety summary
```

Example:

```text
Turns 1-30 → safety summary
Turns 31-40 → raw text
Current turn → raw text
```

Never summarize away safety-relevant escalation signals without preserving them structurally.

---

# 26. Audit Logging

Every decision should create one structured record:

```json
{
  "conversation_id": "abc",
  "turn": 7,
  "stage": "POST",

  "model": {
    "name": "...",
    "version": "..."
  },

  "assessment": {
    "intent": "CONCERNING",
    "input_risk": 0.43,
    "trajectory_risk": 0.72,
    "artifact_risk": 0.88,
    "overall_risk": 0.83
  },

  "tools": [
    {
      "name": "toxinpred2",
      "artifact_hash": "...",
      "result": "TOXIN",
      "score": 0.87
    }
  ],

  "action": "REFUSE",

  "latency_ms": {
    "reasoning": 1132,
    "tools": 387,
    "total": 1694
  }
}
```

This becomes essential when you later build the evaluation framework.

---

# 27. Prototype Runtime Pseudocode

```python
def handle_user_message(conversation_id, user_text):

    state = state_store.load(conversation_id)
    history = conversation_store.load(conversation_id)

    # -----------------------------------------
    # PRE-GENERATION DEFENSE
    # -----------------------------------------

    input_artifacts = artifact_parser.extract(user_text)

    pre_context = DefenseContext(
        stage="PRE",
        history=history,
        current_user_message=user_text,
        safety_state=state,
        artifacts=input_artifacts
    )

    pre_result = defense_agent.evaluate(pre_context)

    pre_result = resolve_tools(
        defense_agent,
        pre_context,
        pre_result
    )

    pre_action = policy.decide(pre_result)

    if pre_action != ALLOW:

        safe_response = intervention.generate(
            pre_action,
            pre_result
        )

        conversation_store.append(
            conversation_id,
            user_text,
            safe_response
        )

        state_store.save(
            conversation_id,
            update_state(state, pre_result)
        )

        audit.log(...)

        return safe_response

    # -----------------------------------------
    # TARGET GENERATION
    # -----------------------------------------

    candidate = target_model.generate(
        history + [user_text]
    )

    # -----------------------------------------
    # POST-GENERATION DEFENSE
    # -----------------------------------------

    output_artifacts = artifact_parser.extract(candidate)

    post_context = DefenseContext(
        stage="POST",
        history=history,
        current_user_message=user_text,
        candidate_response=candidate,
        pre_assessment=pre_result,
        safety_state=state,
        artifacts=output_artifacts
    )

    # Mandatory artifact tools
    tool_results = []

    for artifact in output_artifacts:

        if artifact.type == PROTEIN_SEQUENCE:
            tool_results.append(
                toxinpred2.predict(artifact)
            )

    post_context.tool_results = tool_results

    post_result = defense_agent.evaluate(post_context)

    post_result = resolve_tools(
        defense_agent,
        post_context,
        post_result
    )

    post_action = policy.decide(post_result)

    final_response = intervention.apply(
        action=post_action,
        candidate=candidate,
        assessment=post_result
    )

    conversation_store.append(
        conversation_id,
        user_text,
        final_response
    )

    state_store.save(
        conversation_id,
        update_state(
            state,
            pre_result,
            post_result
        )
    )

    audit.log(...)

    return final_response
```

---

# 28. HPC Deployment

For a simple prototype:

```text
                    Compute node
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
 Reasoning LLM       Target LLM      ToxinPred2
   vLLM               vLLM           local tool
 GPU 0                GPU 1           CPU/GPU
        │                │                │
        └────────────────┼────────────────┘
                         │
                  Python controller
                         │
                  SQLite / JSONL logs
```

If only one GPU is available, sequentially host the two models or use the same model as both target and defense model for architectural testing.

The **defense model and target model should nevertheless remain logically separate services**.

---

# 29. Environment

Suggested software stack:

```text
Python 3.11
PyTorch
Transformers
vLLM
Pydantic
FastAPI (optional)
SQLite
pytest
PyYAML
```

Pydantic is particularly useful:

```python
class RiskAssessment(BaseModel):
    intent: IntentAssessment
    trajectory: TrajectoryAssessment
    tool_requests: list[ToolRequest]
    risk: RiskScores
    recommended_action: Action
```

Reject malformed model output and retry once with a repair prompt.

---

# 30. Testing Strategy

Do not begin with a huge benchmark.

Create approximately 30-50 hand-written **non-operational** scenario tests covering architectural behavior.

Categories:

| Test                                                       | Expected behavior                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| Ordinary biology question                                  | ALLOW                                                        |
| Benign protein discussion                                  | ALLOW                                                        |
| Clearly concerning request                                 | REFUSE                                                       |
| Ambiguous dual-use request                                 | reasoning required                                           |
| Benign → benign → concerning escalation                    | trajectory risk increases                                    |
| Suspicious vocabulary in legitimate context                | avoid reflexive refusal                                      |
| Protein sequence in input                                  | tool called                                                  |
| Protein sequence in output                                 | ToxinPred2 automatically called                              |
| Tool timeout                                               | not treated as benign                                        |
| Model returns malformed JSON                               | repair/retry                                                 |
| Target produces concerning output after safe-looking input | post-guard catches it                                        |
| Previous risk followed by innocuous topic                  | history retained but current request evaluated independently |

The most important test is multi-turn:

```text
Turn 1 → low risk
Turn 2 → low/medium
Turn 3 → evidence accumulates
Turn 4 → trajectory meaning changes
```

Verify that:

```text
trajectory_risk(T4) > current_message_risk(T4)
```

where appropriate.

---

# 31. Initial Evaluation Metrics

Even for v0, log:

### Input-stage

```text
allow rate
refusal rate
review rate
tool-call rate
```

### Post-stage

```text
candidate outputs blocked
artifact detection rate
ToxinPred2 invocation rate
```

### Multi-turn

```text
risk score by turn
first detection turn
trajectory escalation detection
```

### Utility

```text
benign refusal rate
```

### Runtime

```text
reasoning latency
tool latency
target latency
total latency
tokens per safety decision
```

Later, BioSecBench's paired Routine/Red-Team methodology is particularly suitable for measuring whether trajectory reasoning detects underlying hazards without simply rejecting legitimate biology. 

---

# 32. Explicitly Defer MTSA Training

The architecture should support future alignment, but **v0 should use prompting only**.

Development phases:

```text
Phase 0
Base instruct model + system prompt
        ↓
Phase 1
Supervised fine-tuning on labeled safety trajectories
        ↓
Phase 2
Preference optimization on safe vs unsafe trajectory decisions
        ↓
Phase 3
Multi-turn future-reward optimization
        ↓
Phase 4
Adaptive attacker-defender training
```

The reason for preserving complete trajectories from day one is that MTSA's methodology can later use final trajectory outcomes to supervise earlier ambiguous states. 

Therefore log training-compatible records now:

```json
{
  "trajectory": [...],
  "state_at_turn": {...},
  "agent_assessment": {...},
  "tool_evidence": [...],
  "action": "...",
  "eventual_outcome": null,
  "human_label": null
}
```

You can populate `eventual_outcome` and `human_label` later.

---

# 33. Future Alignment Dataset Format

When you reach alignment work, represent examples at the **trajectory state level**.

```json
{
  "trajectory_id": "T001",
  "turn": 4,

  "history": [
    "...",
    "...",
    "..."
  ],

  "current_request": "...",

  "tool_evidence": [...],

  "preferred_assessment": {
    "intent": "...",
    "trajectory_risk": "...",
    "action": "..."
  },

  "rejected_assessment": {
    "intent": "...",
    "trajectory_risk": "...",
    "action": "..."
  },

  "future_outcome": {
    "trajectory_safe": false
  }
}
```

This is much more compatible with future MTSA-inspired training than storing isolated prompt/refusal pairs.

---

# 34. Evaluation Architecture to Build Toward

Eventually separate four evaluation layers:

```text
              EVALUATION STACK

Layer 1 — Turn-level classification
          precision / recall / FPR

Layer 2 — Trajectory-level detection
          concealed intent / escalation /
          decomposition

Layer 3 — End-to-end safety
          does unsafe assistance reach user?

Layer 4 — Artifact-level risk
          does generated biological output
          trigger specialized screening?
```

Then separately evaluate:

```text
BENIGN UTILITY
```

This synthesizes the complementary metrics in the four papers: Meng & Zhang distinguish component and end-to-end failure; MTSA evaluates trajectory robustness and over-refusal; BioSecBench pairs concealed hazards with legitimate research; SPIKE separates compliance from biological artifact risk. 

---

# 35. Key Failure Modes to Instrument Immediately

Even if you do not solve them yet, tag these in logs:

```text
SURFACE_KEYWORD_FALSE_POSITIVE

CONCEALED_INTENT_FALSE_NEGATIVE

MULTI_TURN_DECOMPOSITION

TRAJECTORY_ESCALATION

DUAL_USE_AMBIGUITY

ARTIFACT_NOT_DETECTED

TOOL_NOT_INVOKED

TOOL_FALSE_NEGATIVE

TOOL_FAILURE

POST_GUARD_FAILURE

OVER_REFUSAL

CONTEXT_SUMMARY_LOSS

MALFORMED_AGENT_OUTPUT
```

This taxonomy will make later error analysis substantially easier.

---

# 36. Recommended v0 Build Order

I would implement the prototype in this order:

1. **Local reasoning-model wrapper.**
2. **Structured Pydantic risk schema.**
3. **Conversation store with full history.**
4. **Defense-agent prompt operating on full trajectory.**
5. **Simple deterministic policy engine.**
6. **Target-model wrapper.**
7. **Pre-generation interception.**
8. **Post-generation interception.**
9. **Protein-sequence parser.**
10. **ToxinPred2 wrapper.**
11. **Automatic ToxinPred2 invocation on output protein sequences.**
12. **Agent-requested ToxinPred2 invocation on inputs.**
13. **Persistent structured safety state.**
14. **JSONL audit logging.**
15. **30-50 scenario integration tests.**
16. **HPC job/service packaging.**

At that point you have a genuine research prototype:

```text
                    ┌─────────────────────────────┐
                    │ Conversation Safety State   │
                    │                             │
                    │ • inferred objective        │
                    │ • trajectory risk           │
                    │ • previous evidence         │
                    │ • previous decisions        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
USER ──► PRE-GUARD REASONING AGENT
             │
             ├── contextual intent inference
             ├── trajectory reasoning
             ├── artifact extraction
             └── tool orchestration
                       │
                       ▼
                  ToxinPred2
                       │
                       ▼
                  POLICY ENGINE
                  │           │
               REFUSE       ALLOW
                              │
                              ▼
                         TARGET LLM
                              │
                       candidate output
                              │
                              ▼
              POST-GUARD REASONING AGENT
                  │
                  ├── semantic assessment
                  ├── trajectory context
                  ├── artifact extraction
                  └── ToxinPred2
                              │
                              ▼
                         POLICY ENGINE
                    │          │          │
                 REFUSE      SAFE       ALLOW
                             RESPONSE      │
                                         ▼
                                        USER
```

That architecture captures the most useful elements of the four papers without prematurely implementing their much heavier training/evaluation machinery: **Meng & Zhang's pre/post lifecycle guarding, MTSA's history-dependent safety state, BioSecBench's emphasis on underlying intent rather than surface terminology, and SPIKE's tool-augmented assessment of specialized biological artifacts.** 

The key implementation choice I would preserve above everything else is this: **make the reasoning agent produce an explicit, persistent representation of its inferred user objective and trajectory risk, while keeping domain-tool evidence and the final policy decision as separate objects.** That separation will let you later replace the base model, train it with MTSA-style trajectory data, add new biological tools, calibrate policies, and evaluate intent inference independently without rewriting the entire runtime.

I can also create an image of this prototype architecture and its pre/post-guard execution flow.
