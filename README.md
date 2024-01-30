# SpecExec
This repository contains the supplementary code for the paper "SpecExec: Massively Parallel Speculative Decoding For Interactive LLM Inference on Consumer Devices".

## Launching experiements

The main experiments script is `run_exp.py`.
By default, it uses: 
- `TheBloke/Llama-2-7B-Chat-GPTQ` for draft model
- `TheBloke/Llama-2-70B-chat-GPTQ` for the target model.
- `--temperature 1.0 --top_p 1.0`
- `oasst` dataset for prompts
- no offloading (activated by `--offload` argument)


SpecExec on OpenAssistant data (`--gen_type SX2`):
`python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SX2 --max_budget="16, 32, 64, 128, 256, 512, 1024"  --max_beam_len=256 --max_n_beams=128 --max_branch_width=256 --n_tests=100 --exp_name="SX_sample"`

SpecInfer on OpenAssistant data (`--gen_type SI`):
`python run_exp.py --temperature 0.6 --top_p 0.9 --gen_type SI --max_beam_len="8, 16, 32, 64, 128, 256, 512" --max_n_beams=1 --exp_name="SI_sample"`

Benchmarking on WikiText data:
`python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SX2 --max_budget="16, 32, 64, 128, 256, 512, 1024"  --max_beam_len=256 --max_n_beams=128 --max_branch_width=256 --n_tests=100 --model_0="TheBloke/Llama-2-7B-GPTQ" --model_1="TheBloke/Llama-2-70B-GPTQ" --dataset="wikitext" --exp_name="SX_wikitext"`

For offloaded inference, add `--offload`:
`python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SX2 --max_budget="16, 32, 64, 128, 256, 512, 1024"  --max_beam_len=256 --max_n_beams=128 --max_branch_width=256 --n_tests=100 --offload --exp_name="SX_sample_offload"`

Ablation test with different models:
`python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SX2 --model_0 "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" --model_1 "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" --max_n_beams=128 --max_budget="16,32,64,128,256,512,1024,2048,4096" --max_beam_len=256  --n_tests=100 --exp_name="SX2_mixtral"`

Benchmark run without speculative decodng, use `--zero`:
`python run_exp.py --top_p 0.9 --temperature 0.6 --zero`
`python run_exp.py --top_p 0.9 --temperature 0.6 --offload --zero`
`python run_exp.py --model_0 "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" --model_1 "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" --offload --zero`

During the run, the script will log individual test results to stdout and to the logfile located in ./logs/[exp_name]. In the end, the summary result is displayed as a table:
```
--------------------------------------------------------------------------------------------------
     S U M M A R Y
--------------------------------------------------------------------------------------------------
   max_n_beams  max_beam_len  max_budget  max_branch_width min_log_prob  gen_rate  gen_speed
0          128            16          64               256         None     10.20       8.44
1          128            16         256               256         None     13.00       9.54
2          128           256          64               256         None     10.50       6.19
3          128           256         256               256         None     16.56       6.11
--------------------------------------------------------------------------------------------------
```

Here, `gen_rate` represents the average number of tokens accepted per draft tree and `gen_speed` is the average number of tokens generated per second.
