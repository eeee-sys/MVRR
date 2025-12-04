# MVRR on Public Benchmarks

### 🔮 Table of Content
[CG-Bench](#cg-bench-mini)
[Video-MME](#video-mme-wo-subs)
[MLVU](#mlvu)
[LVBench](#lvbench)
[LongVideoBench](#longvideobench)

> Click on the datasets to jump to the tables.

### CG-Bench (mini):

| Method               | Size   | long-acc. | 
|----------------------|:------:|:---------:|
| Video-LLaVA          | 7B     | 16.2      |
| VideoLLaMA           | 7B     | 18.4      |
| VideoChat2           | 7B     | 19.3      |
| ST-LLM               | 7B     | 23.8      |
| ShareGPT4Video       | 16B    | 26.7      |
| Chat-UniVi-v1.5      | 13B    | 25.9      |
| LongVA               | 7B     | 28.7      |
| LLaVA-OV             | 7B     | 31.1      |
| Video-CCAM           | 14B    | 29.7      |
| Kangaroo             | 8B     | 30.2      |
| VideoMind (Ours)     | 2B     | 31.0      |
| **MVRR (Ours)**      | 2B     | 31.9      |



### Video-MME (w/o subs):

| Model                | Size | Long |
|----------------------|:----:|:----:|
| Video-LLaVA          | 7B   | 37.8 |
| TimeChat             | 7B   | 32.1 |
| MovieChat            | 7B   | 33.4 |
| PLLaVA               | 34B  | 34.7 |
| VideoChat-TPO        | 7B   | 41.0 |
| LongVA               | 7B   | 46.2 |
| VideoMind            | 2B   | 45.4 |
| **MVRR** (Ours)      | 2B   | 46.6 |



### LongVideoBench:

| Method               | Size | Acc  | 
|----------------------|:----:|:----:|
| Idefics2             | 8B   | 49.7 | 
| Phi-3-Vision         | 4B   | 49.6 |
| Mantis-Idefics2      | 8B   | 47.0 | 
| Mantis-BakLLaVA      | 7B   | 43.7 | 
| VideoMind            | 2B   | 48.8 | 
| **MVRR (Ours)**      | 2B   | 51.4 | 

<!--
##### The following Benchmarks are only used during early exploration, but we still show them to facilitate future research.

### MLVU:

| Model                | Size | M-Avg |
|----------------------|:----:|:-----:|
| Video-LLaVA          | 7B   | 29.3  |
| TimeChat             | 7B   | 30.9  |
| MovieChat            | 7B   | 25.8  |
| PLLaVA               | 34B  | 53.6  |
| VideoChat-TPO        | 7B   | 54.7  |
| LongVA               | 7B   | 56.3  |
| VideoMind            | 2B   | 58.7  |
| **MVRR** (Ours)      | 2B   | 58.9  |

### LVBench:

| Model                | Size | Overall |
|----------------------|:----:|:-------:|
| Gemini-1.5-Pro       | –    | 33.1    |
| GPT-4o               | –    | 30.8    |
| Video-LLaVA          | 7B   | –       |
| TimeChat             | 7B   | 22.3    |
| MovieChat            | 7B   | 22.5    |
| PLLaVA               | 34B  | 26.1    |
| VideoChat-TPO        | 7B   | –       |
| LongVA               | 7B   | –       |
| **VideoMind** (Ours) | 2B   | 35.4    |
| **VideoMind** (Ours) | 7B   | 40.8    |
-->

