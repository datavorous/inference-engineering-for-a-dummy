# The Beginner's Complete Guide to LLM Inference Engineering
### For the intern who just walked in and knows nothing yet

---

## How to Use This Document

Read it top to bottom. Every section builds on the last. By the end you will understand what the team is doing, why the infrastructure is set up the way it is, what the problems are, and how the work you will do fits into the bigger picture.

---

## Part 1: What Is This Team Working On?

The LTRC (Language Technologies Research Centre) team at IIIT Hyderabad runs its own GPU clusters - real machines with graphics cards in them, sitting in a server room, not rented from Amazon or Google. The goal is to **train and serve large language models (LLMs)** on this hardware.

An LLM is a neural network (think: a very large mathematical function) that generates text. ChatGPT, Gemini, and Claude are all LLMs. To make one work you need to:

1. **Train** it - run the model over huge amounts of text, adjusting billions of numbers (called parameters or weights) until the model gets good at predicting the next word.
2. **Serve** it - take the trained model and let people (or programs) send it prompts and get responses back in real time.

Both operations are extremely computationally expensive. That is why you need GPUs.

---

## Part 2: Understanding GPUs (Without an EE Degree)

A regular CPU (the main chip in your laptop) is great at doing one or two complex things at a time. A GPU is great at doing millions of simple things simultaneously. Neural networks are basically giant grids of numbers being multiplied together - which GPUs are perfect for.

```mermaid
graph LR
    A[CPU<br/>4-64 cores<br/>great at complex sequential work] -->|slow for ML| C[Your Model]
    B[GPU<br/>thousands of cores<br/>great at parallel matrix math] -->|fast for ML| C
```

The team's clusters have NVIDIA GPUs. Each GPU has its own **VRAM** (Video RAM) - fast memory that sits right on the card. The model weights must fit in VRAM to run efficiently.

---

## Part 3: The Two Clusters - Ada and Turing

The team has access to two GPU clusters. Think of a cluster as a large collection of computers ("nodes") connected by a fast network, each node having several GPUs.

```mermaid
graph TB
    subgraph Ada["Ada Cluster (~90 nodes)"]
        direction TB
        A1[~40 nodes<br/>GTX 1080 GPUs<br/>11GB VRAM each<br/>older CUDA only]
        A2[~50 nodes<br/>RTX 2080/3080 GPUs<br/>11GB VRAM each<br/>better CUDA support]
    end

    subgraph Turing["Turing Cluster (pay-per-use)"]
        direction TB
        T1[LS40 / RTX6000 GPUs<br/>48GB VRAM each<br/>much more powerful]
    end

    Ada -->|free but constrained| U[Your Research Work]
    Turing -->|paid but capable| U
```

### What "11GB VRAM" means in practice

A model is stored as numbers. How many bytes those numbers take depends on **precision** (how many bits each number uses):

| Precision | Bits per number | Memory for a 7B parameter model |
|---|---|---|
| FP32 (full precision) | 32 bits = 4 bytes | ~28 GB |
| FP16 (half precision) | 16 bits = 2 bytes | ~14 GB |
| INT8 (8-bit integer) | 8 bits = 1 byte | ~7 GB |
| INT4 (4-bit integer) | 4 bits = 0.5 bytes | ~3.5 GB |

With 4 GPUs at 11GB each = 44GB total. This means:
- **7B params at FP32** → needs 28GB → fits across 4 GPUs ✓
- **10B params at FP16** → needs 20GB → fits across 4 GPUs ✓
- **70B params at FP16** → needs 140GB → does NOT fit ✗

The Turing cluster's 48GB cards are about 4× more capable per card.

---

## Part 4: The Two Phases of LLM Inference (Prefill vs Decode)

When you send a prompt to an LLM and it generates a response, there are two distinct phases happening under the hood:

```mermaid
sequenceDiagram
    participant You
    participant GPU
    You->>GPU: "Tell me about black holes" (prompt = 6 tokens)
    Note over GPU: PREFILL PHASE<br/>Process ALL 6 input tokens at once<br/>Compute-heavy, fast
    GPU->>GPU: Store KV cache (memory of what was seen)
    loop For each output token
        GPU->>GPU: DECODE PHASE<br/>Generate ONE new token<br/>Reads KV cache + weights<br/>Memory-bandwidth-heavy, slow
        GPU->>You: "Black" ... "holes" ... "are" ...
    end
```

**Why this matters:** These two phases have different bottlenecks, so optimizing them requires different techniques. Prefill is limited by raw compute speed (TFLOPS). Decode is limited by how fast you can read data from memory (memory bandwidth, measured in TB/s).

---

## Part 5: The 7 Layers of the Inference Stack

Think of inference as a stack of layers, like floors of a building. Each floor depends on the one below it.

```mermaid
graph BT
    L1["Layer 1: Hardware & Silicon<br/>(GPUs, memory, interconnects)"]
    L2["Layer 2: Kernels & Low-Level Compute<br/>(CUDA programs, FlashAttention)"]
    L3["Layer 3: Frameworks & Model Runtime<br/>(PyTorch, compilers)"]
    L4["Layer 4: Inference Engines<br/>(vLLM, SGLang, TensorRT-LLM)"]
    L5["Layer 5: Distributed Orchestration<br/>(multi-GPU, multi-node)"]
    L6["Layer 6: Serving Infrastructure<br/>(APIs, autoscaling, monitoring)"]
    L7["Layer 7: Model Optimization<br/>(quantization, distillation, MoE)"]

    L1 --> L2 --> L3 --> L4 --> L5 --> L6
    L7 -.->|affects all layers| L4

    style L1 fill:#2d2d2d,color:#fff
    style L4 fill:#1a3a5c,color:#fff
    style L7 fill:#3a1a5c,color:#fff
```

### Layer 1: Hardware & Silicon

**What it is in plain English:** The physical chips. GPUs, their memory (called HBM - High Bandwidth Memory), and the cables connecting them.

**The analogy:** This is like the road network of a city. Everything else (cars, trucks, buses) depends on how wide and fast the roads are. You cannot make traffic move faster than the road allows.

**Key concept - the Roofline Model:** Every GPU has two fundamental limits:
- **Compute ceiling:** How many math operations per second (TFLOPS)
- **Memory bandwidth ceiling:** How fast data moves from memory to cores (TB/s)

No software trick can exceed these physical limits.

**Relevant to our cluster:**
- GTX 1080: old, limited CUDA version support, only 11GB VRAM
- RTX 2080/3080: better, but still 11GB VRAM
- RTX 6000 (Turing): 48GB VRAM, modern CUDA, much more capable

**Impact on you:** When a model run crashes with "CUDA out of memory", this layer is why. The fix is either a smaller model, lower precision, or spreading across more GPUs.

---

### Layer 2: Kernels and Low-Level Compute

**What it is in plain English:** A "kernel" is a function that runs on the GPU - thousands of threads executing it simultaneously. When PyTorch does a matrix multiplication, it calls a kernel. When attention is computed, it calls a kernel.

**The analogy:** If hardware is the road, kernels are the engine design. Two cars on the same road can have very different speeds depending on how efficient their engines are.

**The biggest example - FlashAttention:**

Standard attention (the mechanism that makes transformers work) has a problem: it creates a huge intermediate matrix in slow GPU memory.

```mermaid
graph LR
    subgraph Standard["Standard Attention (slow)"]
        direction LR
        Q[Q matrix] --> S["S = QKᵀ<br/>(written to slow HBM)"]
        K[K matrix] --> S
        S --> P["P = softmax(S)<br/>(written to slow HBM)"]
        P --> O["O = PV<br/>final output"]
        V[V matrix] --> O
    end
```

```mermaid
graph LR
    subgraph Flash["FlashAttention (fast)"]
        direction LR
        Q2[Q matrix] --> T["Tiles processed in<br/>fast on-chip SRAM<br/>never written to slow HBM"]
        K2[K matrix] --> T
        V2[V matrix] --> T
        T --> O2["Final output<br/>(only this written to HBM)"]
    end
```

FlashAttention does the same math but 2-8× faster because it avoids expensive memory round-trips. This is a kernel-level optimization - same hardware, smarter program.

---

### Layer 3: Frameworks and Model Runtime

**What it is in plain English:** PyTorch is the "operating system" for AI models. You write your model in Python, and PyTorch handles translating that to GPU kernel calls, managing memory, and running backwards passes for training.

**The analogy:** If kernels are the engine, the framework is the transmission system - it decides which gear to use, when to accelerate, how to distribute power.

**Two modes of execution:**

```mermaid
flowchart LR
    A[Python model code] -->|Eager Mode| B["Execute each operation<br/>immediately as code runs<br/>Flexible, easy to debug<br/>Slower"]
    A -->|torch.compile| C["Capture the whole computation<br/>graph, optimize it,<br/>then execute<br/>Faster, harder to debug"]
```

**Key vocabulary:**
- **Autograd:** PyTorch automatically computes gradients for training (you don't do it by hand)
- **CUDA Graph:** Record a sequence of GPU operations once, then replay it very fast
- **Operator Fusion:** Combine two small operations into one large one to avoid overhead

---

### Layer 4: Inference Engines

**What it is in plain English:** An inference engine is a specialized system for serving LLMs to many users at once efficiently. vLLM is the most popular one. Think of it as the "restaurant manager" that decides which orders to batch together and how to use the kitchen (GPU) as efficiently as possible.

**The naive problem - why you need an engine:**

Without an engine, you process one request at a time:

```mermaid
gantt
    title Naive serving: one request at a time
    dateFormat X
    axisFormat %s

    section GPU (wasted!)
    Request 1 (prefill + decode) :0, 5
    GPU IDLE :5, 7
    Request 2 (prefill + decode) :7, 12
    GPU IDLE :12, 14
```

**With continuous batching (what vLLM does):**

```mermaid
gantt
    title Continuous batching: GPU always busy
    dateFormat X
    axisFormat %s

    section GPU utilization
    Req1 prefill :0, 2
    Req1+Req2 decode together :2, 4
    Req1+Req2+Req3 decode :4, 6
    Req2+Req3 decode (Req1 finished) :6, 8
```

As soon as one request finishes, another joins the batch mid-flight. The GPU is almost never idle.

**PagedAttention - solving the memory fragmentation problem:**

The KV cache (the model's "working memory" during generation) has variable size per request. Allocating a fixed block for each request wastes memory (like reserving a whole hotel floor for one guest).

```mermaid
graph TB
    subgraph Old["Old approach: static allocation"]
        R1[Request 1: reserved 10K tokens<br/>uses 3K → 7K wasted]
        R2[Request 2: reserved 10K tokens<br/>uses 8K → 2K wasted]
        R3["Request 3: can't fit!<br/>(memory 'full' even though 9K wasted)"]
    end

    subgraph New["PagedAttention: dynamic paging"]
        B1[Block 1: 256 tokens → Request 1]
        B2[Block 2: 256 tokens → Request 1]
        B3[Block 3: 256 tokens → Request 2]
        B4[Block 4: 256 tokens → Request 2]
        B5[Block 5: 256 tokens → Request 3 ← fits now!]
    end
```

**Key metrics you will care about:**
- **TTFT:** Time to First Token - how long until the user sees the first word
- **ITL / TPOT:** Inter-Token Latency - how long between each subsequent word
- **Throughput:** How many tokens per second the system produces total

---

### Layer 5: Distributed Orchestration

**What it is in plain English:** When a model is too big for one GPU, or you want to serve millions of users, you need to coordinate many GPUs. This layer decides how to split the work and how GPUs communicate.

**The four ways to split a model across GPUs:**

```mermaid
graph TB
    subgraph TP["Tensor Parallelism (TP)"]
        direction LR
        G1[GPU 1: left half of weight matrix]
        G2[GPU 2: right half of weight matrix]
        G1 <-->|all-reduce sync needed| G2
    end

    subgraph PP["Pipeline Parallelism (PP)"]
        direction LR
        G3[GPU 1: Layers 1-16]
        G4[GPU 2: Layers 17-32]
        G3 -->|activations| G4
    end

    subgraph EP["Expert Parallelism (EP) for MoE models"]
        direction LR
        G5[GPU 1: Expert 1,2,3]
        G6[GPU 2: Expert 4,5,6]
    end

    subgraph CP["Context Parallelism (CP) for long inputs"]
        direction LR
        G7[GPU 1: tokens 1-4096]
        G8[GPU 2: tokens 4097-8192]
    end
```

**Why "cross-node" communication is the hard part:**

Within a single node (4 GPUs), NVIDIA's NVLink connects them at ~600 GB/s. Between nodes, you rely on InfiniBand or Ethernet, which is ~10-200 GB/s. This bottleneck is exactly why the Ada cluster struggles:

```mermaid
flowchart LR
    subgraph Node1["Node 1 (gnode01)"]
        G1[GPU 0] <-->|NVLink ~600GB/s| G2[GPU 1]
        G2 <-->|NVLink| G3[GPU 2]
        G3 <-->|NVLink| G4[GPU 3]
    end
    subgraph Node2["Node 2 (gnode02)"]
        G5[GPU 0] <-->|NVLink| G6[GPU 1]
    end
    Node1 <-->|InfiniBand ~100GB/s<br/>10x slower<br/>CURRENT BOTTLENECK| Node2
```

This is why the meeting notes say "distribute to multiple GPUs across gnodes: not feasible due to network bottleneck."

---

### Layer 6: Serving Infrastructure and Production

**What it is in plain English:** Once your model is running, you need to make it into a real service - with an API endpoint, load balancing, health checks, cost controls, and the ability to handle sudden spikes in traffic.

**The anatomy of a production serving setup:**

```mermaid
graph TB
    Users[Users / Client Apps] -->|HTTPS requests| LB[Load Balancer<br/>routes to healthy instances]
    LB --> S1[Serving Instance 1<br/>vLLM + GPU]
    LB --> S2[Serving Instance 2<br/>vLLM + GPU]
    LB --> S3[Serving Instance 3<br/>vLLM + GPU]

    S1 & S2 & S3 --> Mon[Monitoring<br/>latency, errors, GPU util]
    Mon --> AS[Autoscaler<br/>add/remove instances based on traffic]

    S1 & S2 & S3 --> GR[Guardrails<br/>safety / content filters]
```

**Key vocabulary:**
- **SLA/SLO:** Service Level Agreement/Objective - the promise you make about latency and uptime (e.g., "99% of requests complete in under 2 seconds")
- **P95/P99 latency:** The response time that 95%/99% of requests are faster than - used instead of average because averages hide the slowest users
- **Blue-Green Deployment:** Run old and new versions simultaneously; switch traffic over only when new version is healthy - zero downtime updates
- **Streaming:** Sending tokens to the user one-by-one as they are generated (what gives ChatGPT that typewriter effect)

---

### Layer 7: Model Optimization Techniques

**What it is in plain English:** Before any of the above layers, you can make the model itself smaller and faster - without training a new one from scratch. This is called post-training optimization.

**The precision ladder - trading accuracy for speed:**

```mermaid
graph LR
    FP32["FP32<br/>32-bit float<br/>full accuracy<br/>4 bytes/param"] -->|quantize| FP16["FP16<br/>16-bit float<br/>tiny accuracy loss<br/>2 bytes/param"]
    FP16 -->|quantize| INT8["INT8<br/>8-bit integer<br/>small accuracy loss<br/>1 byte/param"]
    INT8 -->|quantize| INT4["INT4<br/>4-bit integer<br/>noticeable accuracy loss<br/>0.5 bytes/param"]
```

A 7B model at FP32 = ~28GB. The same model at INT4 = ~3.5GB. That is 8× smaller, and can fit on a single 11GB GPU card on Ada.

**Other key techniques:**

| Technique | What it does | Why it helps |
|---|---|---|
| Knowledge Distillation | Train a small "student" model to mimic a large "teacher" model | Smaller model, similar quality |
| MoE (Mixture of Experts) | Model has many sub-networks ("experts"), only 2-3 activate per token | Same parameter count, less compute per token |
| GQA (Grouped Query Attention) | Multiple attention heads share key/value projections | Smaller KV cache, faster decode |
| LoRA / PEFT | Fine-tune only a tiny fraction of parameters | Cheap customization without full retraining |

---

## Part 6: The Problems with Our Cluster Right Now

This section maps the infrastructure issues from the meeting notes to the technical concepts above.

```mermaid
mindmap
  root((LTRC Cluster Problems))
    Storage
      Home dir only 30GB persistent
      Share1 only 100GB persistent
      1TB tmp has 7-day auto-delete
      No shared model cache → repeated downloads
    Compute
      Ada GPUs old 1080s, limited CUDA
      Cross-node network bottleneck InfiniBand underperforming
      4-day wall-time job limit
      Less than 10 percent GPU utilization
    Dependencies
      Different PyTorch versions across nodes
      Libraries in home dir eating limited space
      No containerization Docker in place
    Access
      VPN and SSH setup needed for everyone
      Email access pending
```

### The Storage Problem in Detail

```mermaid
graph TB
    subgraph Storage["Storage available per user"]
        Home["home directory<br/>30GB - persistent<br/>⚠️ PyTorch alone fills this"]
        Share1["share1<br/>100GB - persistent<br/>⚠️ Only place for model cache"]
        Tmp["tmp (NAS)<br/>1TB - AUTO-DELETED after 7 days<br/>⚠️ Slow network access"]
    end

    Home -->|scp over network = slow| Tmp
    Share1 -->|scp over network = slow| Tmp
```

**The recommendation** from the team: Buy a NAS (Network Attached Storage) device, replicate it to Ada and Turing, and use it as a shared model/data cache. This would eliminate the repeated downloading of the same models by every student.

### The Compute Problem in Detail

| Scenario | Ada 1080 (4 GPUs, 44GB total) | Turing RTX6000 (4 GPUs, 192GB total) |
|---|---|---|
| 8B param model FP16 training | 4 days (if no OOM) | 1 day |
| 70B param model | Impossible | Just barely fits |
| Flash-attention optimization | Not supported | Supported |
| FP8 / mxfp4 | Not supported | Not supported (needs H/B series) |

---

## Part 7: The Work Plan - What You Will Actually Do

This is the sequence of tasks from the April 14 meeting, mapped to the inference stack layers above:

```mermaid
flowchart TD
    T1["Step 1: Cluster Access<br/>Get email, VPN, SSH working<br/>📍 Admin work - just do it"]
    T2["Step 2: Containerization<br/>Set up Docker with pinned dependencies<br/>📍 Layer 3 - Frameworks"]
    T3["Step 3: Single-node training<br/>1 GPU → 4 GPUs with torchrun<br/>📍 Layer 1+2 - Hardware + Kernels"]
    T4["Step 4: Multi-node training<br/>Fix InfiniBand bottleneck<br/>📍 Layer 5 - Distributed Orchestration"]
    T5["Step 5: Data pipeline<br/>Batch feeding, HuggingFace cache management<br/>📍 Layer 3 - Frameworks"]
    T6["Step 6: LLM Serving as API<br/>Run vLLM, expose endpoint<br/>📍 Layer 4+6 - Engines + Serving"]
    T7["Step 7: Benchmarking<br/>Measure throughput and latency<br/>📍 All layers"]
    T8["Step 8: Optimization<br/>Mixed precision, gradient compression<br/>📍 Layer 7 - Model Optimization"]
    T9["Step 9: Kubernetes<br/>Wrap everything in K8s<br/>📍 Layer 5+6 - Orchestration + Serving"]

    T1 --> T2 --> T3 --> T4
    T3 --> T5
    T4 --> T6
    T5 --> T6
    T6 --> T7
    T7 --> T8
    T8 --> T9
```

### Your Day One

SSH into one node. Run:
```bash
nvidia-smi                         # check GPUs are visible
python -c "import torch; print(torch.cuda.is_available())"  # check PyTorch
```

Then run a toy training script on one GPU. Once that works, containerize it with Docker. That is the foundation everything else builds on.

---

## Part 8: Key Vocabulary Reference

### GPU Memory Terms

| Term | Meaning |
|---|---|
| VRAM / GPU memory | Memory on the GPU card itself (fast, expensive, limited) |
| HBM | High Bandwidth Memory - the stacked RAM on modern GPUs |
| SRAM / Shared memory | Very fast on-chip buffer inside the GPU (tiny, ~20MB) |
| OOM | Out of Memory - your model is too big for the VRAM |

### Training Terms

| Term | Meaning |
|---|---|
| Parameters / weights | The numbers inside a model that are learned during training |
| Gradient | The direction and magnitude to adjust each weight |
| Batch | A group of examples processed together in one forward pass |
| NCCL | NVIDIA's library for GPU-to-GPU communication during training |
| DDP | DistributedDataParallel - each GPU gets a copy of the model, gradients averaged across GPUs |
| FSDP | Fully Sharded Data Parallel - model weights split across GPUs (more memory efficient than DDP) |
| ZeRO | Microsoft's optimization that shards optimizer state, gradients, and weights |

### Inference Terms

| Term | Meaning |
|---|---|
| Prefill | Processing the input prompt - compute-heavy |
| Decode | Generating output tokens one at a time - memory-bandwidth-heavy |
| KV Cache | Stored attention key/value tensors - avoids recomputing past tokens |
| Continuous Batching | Dynamically grouping multiple requests at the token level |
| Speculative Decoding | A fast "draft" model proposes tokens; the main model verifies in parallel |
| TTFT | Time to First Token |
| ITL | Inter-Token Latency |

### Infrastructure Terms

| Term | Meaning |
|---|---|
| Node / gnode | One physical machine (with its GPUs) in a cluster |
| NVLink | NVIDIA's fast cable connecting GPUs within a node (~600 GB/s) |
| InfiniBand | Fast network between nodes (~100 GB/s - much slower than NVLink) |
| SLURM | Job scheduler software used on HPC clusters to queue and run jobs |
| Docker | Containerization tool - packages your code + dependencies into a portable image |
| Kubernetes (K8s) | Orchestrates many Docker containers across many machines |
| NAS | Network Attached Storage - a shared disk accessible over the network |
| Wall-time | Maximum time a job is allowed to run before the scheduler kills it |

---

## Part 9: How Everything Connects - The Full Picture

```mermaid
graph TB
    subgraph Physical["Physical Layer"]
        GPU1["Ada GPUs<br/>11GB VRAM each<br/>1080/2080/3080"]
        GPU2["Turing GPUs<br/>48GB VRAM each<br/>LS40/RTX6000"]
        IB["InfiniBand<br/>inter-node network<br/>⚠️ currently bottlenecked"]
        NAS_hw["Proposed NAS<br/>shared model/data cache"]
    end

    subgraph Software["Software Layer"]
        Docker["Docker containers<br/>pinned dependencies<br/>portable across nodes"]
        PyTorch["PyTorch<br/>model definition<br/>training loops"]
        NCCL["NCCL<br/>gradient sync across GPUs"]
        vLLM["vLLM<br/>efficient inference engine<br/>continuous batching + PagedAttention"]
    end

    subgraph Serving["Serving Layer"]
        API["REST API endpoint<br/>OpenAI-compatible<br/>accessible to students"]
        K8s["Kubernetes<br/>scheduling, scaling, health checks"]
    end

    Physical --> Software --> Serving

    GPU1 & GPU2 -->|run| Docker
    Docker -->|contains| PyTorch
    PyTorch -->|uses| NCCL
    NCCL -->|syncs across| IB
    Docker -->|contains| vLLM
    vLLM -->|exposes| API
    API -->|managed by| K8s
    NAS_hw -->|feeds models to| Docker
```

---

## Part 10: Suggested Learning Path

If you are a complete beginner, do these in order:

1. **GPU basics:** Watch "But what is a neural network?" by 3Blue1Brown. Then watch NVIDIA's intro to CUDA.
2. **PyTorch:** Work through the official PyTorch 60-minute blitz tutorial.
3. **Distributed training:** Read HuggingFace's "Efficient Training on Multiple GPUs" guide.
4. **vLLM:** Read the PagedAttention blog post. Run `python -m vllm.entrypoints.openai.api_server --model <small-model>`.
5. **Docker:** Do Docker's official "Get Started" tutorial. Containerize a PyTorch script.
6. **Profiling:** Run `nvidia-smi dmon` while a model runs. Understand what GPU utilization and memory usage mean.

Once you have done all of that, you will understand 80% of the work happening in this project.
