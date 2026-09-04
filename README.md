# Multi-Turn Biosafety Defense Agent (v0 Prototype)

An inference-time, multi-turn biosecurity defense layer positioned between users and target language models. 

The system implements defense-in-depth through **lifecycle pre- and post-generation screening**, **conversation-level persistent safety state**, **deterministic policy enforcement**, **biological artifact extraction**, and **integrated biological screening tools (ToxinPred2)**.

---

## 🔬 System Architecture

```text
                                  ┌─────────────────────────────┐
                                  │ Conversation Safety State   │
                                  │                             │
                                  │ • Inferred Objective        │
                                  │ • Trajectory Risk           │
                                  │ • Tool Evidence             │
                                  │ • Decision History          │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
USER ─────────────────────────────► PRE-GUARD REASONING AGENT
                                           │
                                           ├── Contextual Intent Inference
                                           ├── Multi-Turn Trajectory Reasoning
                                           ├── Biological Sequence Extraction
                                           └── Tool Orchestration (ToxinPred2)
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
                                                ├── Semantic Risk Assessment
                                                ├── Trajectory Context Evaluation
                                                ├── Sequence Artifact Extraction
                                                └── Automatic ToxinPred2 Screening
                                                           │
                                                           ▼
                                                      POLICY ENGINE
                                                  │          │          │
                                               REFUSE      SAFE       ALLOW
                                                           RESPONSE     │
                                                                        ▼
                                                                      USER
```

---

## 🌟 Key Features

1. **Multi-Turn Trajectory Conditioning:**
   Unlike single-prompt classifiers, safety decisions are conditioned on the entire dialogue history $H_t$ and a structured persistent safety state, detecting subtle multi-turn capability accumulation and escalation.

2. **Decoupled Deterministic Policy Engine:**
   The reasoning LLM outputs risk scores and observations; an independent deterministic engine enforces policy rules and thresholds (`ALLOW`, `REFUSE`, `SAFE_RESPONSE`, `REVIEW`).

3. **Deterministic Biological Sequence Extraction:**
   Programmatically extracts, validates (IUPAC standard/extended amino acids), and normalizes protein sequences from FASTA headers, code blocks, and raw text.

4. **Biological Tool Integration (ToxinPred2):**
   * **Pre-Guard:** Agent-requested computational toxicity screening on user-provided sequences.
   * **Post-Guard:** Mandatory automatic screening on any model-generated protein sequences before delivery to the user.
   * **Explicit Failure Handling:** Tool errors and timeouts return `FAILED` status and are routed to `REVIEW` / fail-closed paths rather than silently assumed safe.

5. **Target Model Isolation:**
   The target model communicates only with the defense layer. Candidate outputs are evaluated and held until cleared by the post-guard.

6. **Audit & Future Alignment Logging:**
   * **Structured Audit Telemetry (`logs/audit.jsonl`):** Records turn-level latencies, risk scores, artifact SHA-256 hashes, and failure taxonomy tags.
   * **MTSA-Compatible Trajectories (`logs/trajectories.jsonl`):** Preserves full multi-turn safety trajectories formatted for future trajectory-level preference optimization and fine-tuning.

---

## 📁 Repository Structure

```text
biosafety-agent/
├── configs/
│   ├── models.yaml                      # Reasoning & target backend configurations
│   ├── policy.yaml                      # Decision thresholds & intervention messages
│   └── tools.yaml                       # Tool registry, parser, & timeout settings
│
├── biosafety_defense/
│   ├── __init__.py
│   ├── factory.py                       # Component wiring & initialization helper
│   ├── defense_agent/
│   │   ├── agent.py                     # Pre- and post-guard reasoning agent
│   │   ├── policy.py                    # Deterministic policy engine & interventions
│   │   ├── prompts.py                   # Pre/post-guard system prompts & formatters
│   │   └── schemas.py                   # Pydantic data schemas
│   ├── gateway/
│   │   └── conversation_controller.py   # Runtime orchestration controller
│   ├── memory/
│   │   ├── conversation_store.py        # Dialogue history storage & windowing
│   │   └── safety_state.py              # Persistent structured safety state tracking
│   ├── models/
│   │   ├── reasoning_backend.py         # vLLM / OpenAI client with JSON repair loop
│   │   └── target_backend.py            # Target LLM client wrapper
│   ├── tools/
│   │   ├── registry.py                  # Tool coordinator & execution loop
│   │   ├── sequence_parser.py           # Protein sequence extractor & IUPAC validator
│   │   └── toxinpred2.py                # ToxinPred2 computational screening wrapper
│   └── audit/
│       └── audit_logger.py              # JSONL audit & MTSA trajectory logger
│
├── hpc/
│   ├── start_single_vllm.sh             # Launch single vLLM server on remote GPUs
│   ├── start_vllm_background.sh         # Launch dual vLLM servers in background
│   ├── stop_vllm_services.sh            # Terminate running vLLM server processes
│   └── launch_vllm_services.sh          # SLURM multi-GPU batch submission script
│
├── tests/
│   ├── conftest.py                      # Pytest fixtures
│   └── unit/
│       ├── test_memory.py               # Dialogue and state persistence tests
│       ├── test_policy.py               # Deterministic policy engine tests
│       ├── test_sequence_parser.py      # Sequence extraction and validation tests
│       └── test_toxinpred2.py           # ToxinPred2 screening and failure tests
│
├── app.py                               # Interactive CLI session runner
├── requirements.txt                     # Project dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/biosafety-agent.git
cd biosafety-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Unit Tests

Verify that all components (sequence parser, policy engine, memory store, and screening tools) pass:

```bash
pytest tests/unit/ -v
```

---

## 💻 Running the Interactive Agent

### Mode A: Standalone Simulated Mode (No GPUs Required)

By default, the prototype can run in simulated mode for offline testing and development:

```bash
python app.py
```

* Type `state` at any turn to inspect the active persistent safety state.
* Type `exit` to end the session.

---

### Mode B: Connected to Remote GPU vLLM Server (Live Models)

To run live neural inference using a remote GPU server (e.g. multi-GPU RTX PRO 6000 / A100):

#### 1. On the Remote GPU Machine:
Start the vLLM server:
```bash
cd biosafety-agent
bash hpc/start_single_vllm.sh
```

#### 2. On Your Local Laptop:
Open an SSH tunnel forwarding port 8000:
```bash
ssh -N -L 8000:localhost:8000 username@your-server-ip
```

#### 3. Update Configuration:
In [`configs/models.yaml`](file:///c:/Users/User/OneDrive/Desktop/NUS/AIS/Projects/biosafety-agent/configs/models.yaml), verify `backend` is set to `"vllm"`:
```yaml
reasoning_model:
  backend: "vllm"
  model: "deepseek-v4-flash"
  base_url: "http://localhost:8000/v1"
  temperature: 0.0

target_model:
  backend: "vllm"
  model: "deepseek-v4-flash"
  base_url: "http://localhost:8000/v1"
  temperature: 0.7
```

#### 4. Launch the Interactive Agent:
```bash
python app.py
```

---

## 📊 Telemetry and Logs

During execution, the defense agent automatically writes telemetry to the `logs/` directory:

* **`logs/audit.jsonl`**: Machine-readable log of every pre- and post-guard evaluation, including latency breakdown, risk assessments, detected biological artifacts, tool screening scores, and decision actions.
* **`logs/trajectories.jsonl`**: Full multi-turn conversational trajectories with intermediate safety states and tool evidence, formatted for future MTSA-style preference optimization.

---

## 📚 Theoretical Foundations

This prototype synthesizes methodologies and lessons from four core papers in LLM biosecurity:

* **Meng & Zhang (2025):** Pre- and post-generation lifecycle defense-in-depth and guard isolation.
* **MTSA (Guo et al., 2025):** Conditioning safety on multi-turn dialogue trajectories and history-dependent safety states.
* **BioSecBench-Refusal (Wintermute et al., 2026):** Decoupling underlying harmful intent from legitimate dual-use scientific terminology.
* **SPIKE (Quan et al., 2026):** Modality-specific biological screening tools (ToxinPred2) to evaluate functional biological artifacts beyond natural-language safety classifiers.
