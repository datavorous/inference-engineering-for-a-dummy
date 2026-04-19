<img src="media/banner.png">

## inference engineering

A personal research workspace for learning and documenting inference engineering work at the [Language Technologies Research Center](https://ltrc.iiit.ac.in/).

The field is new, and doesn't have structured and well organized information on the internet. This is my attempt to document everything: from day 0.

All my actions are [logged](log.md). Problems which I was told to tackle are in [here](knowledge/problems.md), and will be updated accordingly.

Right now, my work is on:  
1. infra/ops: env setup, storage, docker, InfiniBand, nas, kubernetes
2. serving: vLLM deployment, API exposure

In future (incomplete list):
- optimization techniques
- kernel experiments

The model quality side (kernels, quantization, distillation) comes later once the infra is stable enough to actually experiment on.

> [!NOTE]
> It was decided that all work will be done on the **Turing cluster** exclusively. Turing is a homogeneous cluster i.e. every node is equipped with **LS40 / RTX 6000** GPU cards (48GB VRAM each).
> I have used Claude Code and Opus 4.7 to aid the documentation process.

## micro-log

1. creating the prompts [1](prompts/what.txt) [2](prompts/clarity.txt)
2. exploring all the abstraction layers in "inference engineering" [1](knowledge/layers.md)
3. gathered the documents related to the "current state of infrastructure", and the proposed "changes" to be made. [files hidden from public]
4. used claude code to reference the docs + meeting notes and build a ["primer"](knowledge/primer.md) containing all sorts of definitions and mermaid diagrams to convey the entire status information in a very accessible manner. 
    - allowed web access to search for NVIDIA's docs, vLLM docs etc. **[NEEDS MANUAL VERIFICATION]**
5. generated the list of ["problems"](knowledge/init_problems.md) that we need to tackle initially
6. summarised everything upto this [understanding WHAT we have + minimal setup] in [stage0.md](summary/stage0.md). in short:
    - need to setup dev environment (bare metal test + run a small model end to end)
    - build install scripts to fix dependency hellhole
        + have correct paths to allow caching among users
        + additionally have some verification scripts.
    - create a setup repo
    - dockerize it eventually, push docker images. 
    - expected result: dependecy errors are avoided, libraries with pinned version(s) are used with minimal setup headache, users start using the `/share1` directory, HF models get cached. 
        + > FINAL: anyone can can clone, run the setup script, pull the image, and run inference. 
    

## references

1. https://www.spheron.network/blog/inference-engineering-guide-2026/
2. https://inference-engineering.com/index.html
3. https://github.com/1duo/awesome-ai-infrastructures
4. https://inferenceengineering.tech/paths
5. https://www.baseten.co/inference-engineering/digital-download
6. https://github.com/sytelus/Model_Inference_Deployment
7. https://github.com/sytelus/awesome-inference

8. docker files: https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/
