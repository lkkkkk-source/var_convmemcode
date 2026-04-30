# Learned Local Prior Architecture

This figure summarizes the third innovation point as implemented in the project:
an auxiliary patch-level realism prior is trained separately, then used at
sampling time to rerank multiple VAR candidates.

```mermaid
flowchart LR
  subgraph Train["Training the local prior"]
    R["Real knitting images"] --> C["Random patch crop\nP x P, default 64 x 64"]
    F["Generated images from baseline VAR"] --> C
    C --> S["PatchRealismScorer"]
    S --> L["BCE real/fake loss"]
    L --> W["patch-local-prior-best.pth"]
  end

  subgraph Infer["Sampling-time reranking"]
    Y["Class label y"] --> G["v2.2 VAR_convMem generator"]
    G --> X1["Candidate image 1"]
    G --> X2["Candidate image 2"]
    G --> XM["Candidate image M"]
    X1 --> P1["Sample K local patches"]
    X2 --> P2["Sample K local patches"]
    XM --> PM["Sample K local patches"]
    W -. frozen weights .-> SF["Frozen PatchRealismScorer"]
    P1 --> SF
    P2 --> SF
    PM --> SF
    SF --> A["Average patch realism score"]
    A --> B["Argmax over candidates"]
    B --> O["Selected output image"]
  end
```

## PatchRealismScorer detail

```mermaid
flowchart LR
  I["Input patch\nB x 3 x P x P"] --> B1["Conv 3x3, stride 2\nBN, LeakyReLU"]
  B1 --> B2["Conv 3x3, stride 2\nBN, LeakyReLU"]
  B2 --> B3["Conv 3x3, stride 2\nBN, LeakyReLU"]
  B3 --> B4["Conv 3x3, stride 1\nBN, LeakyReLU"]
  B4 --> GAP["Global average pooling"]
  GAP --> MLP["Linear, LeakyReLU, Linear"]
  MLP --> LOGIT["Realism logit"]
```

## Paper wording

The learned local prior estimates local texture realism from randomly sampled
image patches. During generation, the base VAR_convMem model samples multiple
candidate images for the same class label. Each candidate is decomposed into
local patches and scored by the frozen prior; the candidate with the highest
average patch realism score is selected as the final output.

Formula:

```text
s(I_j) = (1 / K) * sum_{k=1..K} f_theta(P_k(I_j))
I* = argmax_j s(I_j)
```

Code locations:

- `models/patch_realism_scorer.py`: lightweight CNN scorer.
- `train_patch_local_prior.py`: real/fake patch-level prior training.
- `test_var_convmem.py`: candidate generation, patch scoring, and reranking.
