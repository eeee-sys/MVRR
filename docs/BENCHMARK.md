# MVRR on Public Benchmarks

### 🔮 Table of Content
[CG-Bench](#cg-bench-mini)
[LongVideoBench](#longvideobench)
[LVBench](#lvbench)


> Click on the datasets to jump to the tables.

### CG-Bench (mini):

| Method               | Size   | long-acc. | mIou | rec.@IoU | acc.@IoU |
|----------------------|:------:|:---------:|:----:|:--------:|:--------:|
| Video-LLaVA          | 7B     | 16.2      | 1.13 |   1.96   |  0.59    |
| VideoLLaMA           | 7B     | 18.4      | 1.21 |   1.87   |  0.84    |
| VideoChat2           | 7B     | 19.3      | 1.28 |   1.98   |  0.94    |
| ST-LLM               | 7B     | 23.8      | 2.23 |   2.86   |  1.13    |
| ShareGPT4Video       | 16B    | 26.7      | 1.85 |   2.65   |  1.01    |
| Chat-UniVi-v1.5      | 13B    | 25.9      | 2.07 |   2.53   |  1.21    |
| LongVA               | 7B     | 28.7      | 2.94 |   3.86   |  1.78    |
| LLaVA-OV             | 7B     | 31.1      | 1.63 |   1.78   |  1.08    |
| Video-CCAM           | 14B    | 29.7      | 2.63 |   3.48   |  1.83    |
| Kangaroo             | 8B     | 30.2      | 2.56 |   2.81   |  1.94    |
| VideoMind (Ours)     | 2B     | 31.0      | 5.94 |   8.50   |  4.02    |
| **MVRR (Ours)**      | 2B     | 31.8      | 6.07 |   8.73   |  4.03    |



### LongVideoBench val (900,3600]:

| Method               | Size | Acc  | 
|----------------------|:----:|:----:| 
| Mantis-BakLLaVA      | 7B   | 38.7 | 
| VideoMind            | 2B   | 41.7 | 
| **MVRR (Ours)**      | 2B   | 42.2 | 


### LVBench:

| Model                | Size | Overall |
|----------------------|:----:|:-------:|
| Gemini-1.5-Pro       | –    | 33.1    |
| GPT-4o               | –    | 30.8    |
| TimeChat             | 7B   | 22.3    |
| MovieChat            | 7B   | 22.5    |
| PLLaVA               | 34B  | 26.1    |
| VideoMind            | 2B   | 35.4    |
| **MVRR** (Ours)      | 2B   | 35.9    |
