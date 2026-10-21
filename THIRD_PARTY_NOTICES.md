# DeepSeek Reference

The architectural reference is DeepSeek-AI's DeepSeek-V4.1-Flash release:
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

Revision: `314113f8a7dadb388215396ba5fbee2c610a04d6`.
The local sparse checkout is in `references/DeepSeek-V4.1-Flash`.
No released model weights are required or downloaded.

The Hyper-Connection, Engram gating, shared-latent attention, indexer and MoE
implementations are adapted from the architecture and reference implementation.
The CPU/SM75-compatible training operations are maintained in this project.
The upstream inference entry point uses inference-mode functions and FP8/FP4
kernels; it is not a pretraining program. CED encoder-source global KV and bounded
replay follow section 2.2 and section 3.2.2 of the upstream technical report.

## Upstream License

MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
