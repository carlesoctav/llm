# Loss implementation benchmark
- Generated (UTC): `2026-02-25 10:59:12Z`
## Qwen3-0.6B
- Config: `/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py`
- Sweep: packing=True, max_length ∈ {512, 1024, 2048, 4096, 8096}, loss_impl ∈ {xla_chunked}
- Steps: warmup=0, timed=30
- global_batch_size: `8`

| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | xla_chunked | ok | 109.9 | 0.2843 | 3848 | 1.354e+04 | 3.7 |
| 1024 | xla_chunked | ok | 109.5 | 0.6437 | 8105 | 1.259e+04 | 5.3 |
| 2048 | xla_chunked | ok | 122.7 | 1.863 | 16120 | 8650 | 8.6 |
| 4096 | xla_chunked | ok | 116.5 | 6.38 | 32495 | 5093 | 16.2 |
| 8096 | xla_chunked | error | - | - | - | - | - |

### Raw results
```json
[
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 109.9245564439334,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8224477153271437s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 3.7 GB, Output size: 0.7 GB, Temp size: 2.3 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 109.9245564439334, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 512, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 2.3, \"total_gb\": 3.7}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 0.28429281886977453, \"timed_steps\": 30, \"tokens_per_s\": 13535.340130285338, \"tokens_per_step\": 3848, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 33797.78it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 512,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 2.3,
      "total_gb": 3.7
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 0.28429281886977453,
    "timed_steps": 30,
    "tokens_per_s": 13535.340130285338,
    "tokens_per_step": 3848,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 109.50476232776418,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8706962992437184s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 5.3 GB, Output size: 0.7 GB, Temp size: 3.9 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 109.50476232776418, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 1024, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 3.9, \"total_gb\": 5.3}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 0.6436766972299666, \"timed_steps\": 30, \"tokens_per_s\": 12591.725061478068, \"tokens_per_step\": 8105, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 33130.36it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 1024,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 3.9,
      "total_gb": 5.3
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 0.6436766972299666,
    "timed_steps": 30,
    "tokens_per_s": 12591.725061478068,
    "tokens_per_step": 8105,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 122.68410331616178,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.827127723954618s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 8.6 GB, Output size: 0.7 GB, Temp size: 7.2 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 122.68410331616178, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 2048, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 7.2, \"total_gb\": 8.6}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 1.863478508234645, \"timed_steps\": 30, \"tokens_per_s\": 8650.488819037244, \"tokens_per_step\": 16120, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 34183.41it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 2048,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 7.2,
      "total_gb": 8.6
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 1.863478508234645,
    "timed_steps": 30,
    "tokens_per_s": 8650.488819037244,
    "tokens_per_step": 16120,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 116.49632606096566,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8202015850692987s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 16.2 GB, Output size: 0.7 GB, Temp size: 14.8 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 116.49632606096566, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 4096, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 14.8, \"total_gb\": 16.2}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 6.380338155850768, \"timed_steps\": 30, \"tokens_per_s\": 5092.9902469514245, \"tokens_per_step\": 32495, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 40721.40it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 4096,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 14.8,
      "total_gb": 16.2
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 6.380338155850768,
    "timed_steps": 30,
    "tokens_per_s": 5092.9902469514245,
    "tokens_per_step": 32495,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "ExceptionGroup: all implementations failed (1 sub-exception)",
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "log_tail": "model loaded at 0.8275487339124084s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"ExceptionGroup: all implementations failed (1 sub-exception)\", \"exp_name\": \"8_xla_chunked\", \"loss_impl\": \"xla_chunked\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"error\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 40760.97it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 8 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "error",
    "timed_steps": 30,
    "warmup_steps": 0
  }
]
```


---

## Qwen3-0.6B (2026-02-25 12:11:06Z) V(32K)
- Config: `/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py`
- Sweep: packing=True, max_length ∈ {512, 1024, 2048, 4096, 8096}, loss_impl ∈ {xla_chunked, reference}
- Steps: warmup=0, timed=30
- global_batch_size: `8`
- optimizer: `sgd`

| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | xla_chunked | ok | 141.1 | 0.2863 | 3848 | 1.344e+04 | 3.7 |
| 512 | reference | ok | 138.3 | 0.2048 | 3848 | 1.879e+04 | 4.8 |
| 1024 | xla_chunked | ok | 142.8 | 0.6635 | 8105 | 1.222e+04 | 5.3 |
| 1024 | reference | ok | 148.8 | 0.4686 | 8105 | 1.729e+04 | 7.3 |
| 2048 | xla_chunked | ok | 153.7 | 1.862 | 16120 | 8658 | 8.6 |
| 2048 | reference | ok | 144 | 1.484 | 16120 | 1.087e+04 | 12.1 |
| 4096 | xla_chunked | ok | 149.3 | 6.404 | 32495 | 5074 | 16.2 |
| 4096 | reference | ok | 143.8 | 5.636 | 32495 | 5766 | 21.9 |
| 8096 | xla_chunked | error | - | - | - | - | - |
| 8096 | reference | oom | - | - | - | - | - |

### Raw results
```json
[
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 141.09501889673993,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8319290340878069s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 3.7 GB, Output size: 0.7 GB, Temp size: 2.3 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 141.09501889673993, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 512, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 2.3, \"total_gb\": 3.7}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 0.28627799306996166, \"timed_steps\": 30, \"tokens_per_s\": 13441.48028542177, \"tokens_per_step\": 3848, \"warmup_steps\": 0}\n\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 39089.51it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 512,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 2.3,
      "total_gb": 3.7
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 0.28627799306996166,
    "timed_steps": 30,
    "tokens_per_s": 13441.48028542177,
    "tokens_per_step": 3848,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "compile_time_s": 138.30573433404788,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_reference",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8194167562760413s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 4.8 GB, Output size: 0.7 GB, Temp size: 3.4 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"compile_time_s\": 138.30573433404788, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_reference\", \"global_batch_size\": 8, \"loss_impl\": \"reference\", \"max_length\": 512, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 3.4, \"total_gb\": 4.8}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 0.2047662667464465, \"timed_steps\": 30, \"tokens_per_s\": 18792.157815548875, \"tokens_per_step\": 3848, \"warmup_steps\": 0}\n\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 13551.87it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 512,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 3.4,
      "total_gb": 4.8
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 0.2047662667464465,
    "timed_steps": 30,
    "tokens_per_s": 18792.157815548875,
    "tokens_per_step": 3848,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 142.8139767688699,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8112585763446987s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 5.3 GB, Output size: 0.7 GB, Temp size: 3.9 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 142.8139767688699, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 1024, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 3.9, \"total_gb\": 5.3}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 0.6634975600366791, \"timed_steps\": 30, \"tokens_per_s\": 12215.568659441557, \"tokens_per_step\": 8105, \"warmup_steps\": 0}\n\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 41241.93it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 1024,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 3.9,
      "total_gb": 5.3
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 0.6634975600366791,
    "timed_steps": 30,
    "tokens_per_s": 12215.568659441557,
    "tokens_per_step": 8105,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "compile_time_s": 148.7647270541638,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_reference",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8228560551069677s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 7.3 GB, Output size: 0.7 GB, Temp size: 5.9 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"compile_time_s\": 148.7647270541638, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_reference\", \"global_batch_size\": 8, \"loss_impl\": \"reference\", \"max_length\": 1024, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 5.9, \"total_gb\": 7.3}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 0.46863997347342473, \"timed_steps\": 30, \"tokens_per_s\": 17294.72614111014, \"tokens_per_step\": 8105, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 41201.41it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 1024,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 5.9,
      "total_gb": 7.3
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 0.46863997347342473,
    "timed_steps": 30,
    "tokens_per_s": 17294.72614111014,
    "tokens_per_step": 8105,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 153.70739338640124,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.816641584970057s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 8.6 GB, Output size: 0.7 GB, Temp size: 7.2 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 153.70739338640124, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 2048, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 7.2, \"total_gb\": 8.6}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 1.861925997181485, \"timed_steps\": 30, \"tokens_per_s\": 8657.701769244246, \"tokens_per_step\": 16120, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 55701.25it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 2048,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 7.2,
      "total_gb": 8.6
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 1.861925997181485,
    "timed_steps": 30,
    "tokens_per_s": 8657.701769244246,
    "tokens_per_step": 16120,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "compile_time_s": 144.04881080286577,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_reference",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8196958061307669s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 12.1 GB, Output size: 0.7 GB, Temp size: 10.7 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"compile_time_s\": 144.04881080286577, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_reference\", \"global_batch_size\": 8, \"loss_impl\": \"reference\", \"max_length\": 2048, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 10.7, \"total_gb\": 12.1}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 1.4836304370313882, \"timed_steps\": 30, \"tokens_per_s\": 10865.239481238117, \"tokens_per_step\": 16120, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 43062.67it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 2048,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 10.7,
      "total_gb": 12.1
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 1.4836304370313882,
    "timed_steps": 30,
    "tokens_per_s": 10865.239481238117,
    "tokens_per_step": 16120,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "compile_time_s": 149.26682420494035,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8066014158539474s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 16.2 GB, Output size: 0.7 GB, Temp size: 14.8 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"compile_time_s\": 149.26682420494035, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_xla_chunked\", \"global_batch_size\": 8, \"loss_impl\": \"xla_chunked\", \"max_length\": 4096, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 14.8, \"total_gb\": 16.2}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 6.4039853225306915, \"timed_steps\": 30, \"tokens_per_s\": 5074.184021889482, \"tokens_per_step\": 32495, \"warmup_steps\": 0}\n\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 33797.78it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 4096,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 14.8,
      "total_gb": 16.2
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 6.4039853225306915,
    "timed_steps": 30,
    "tokens_per_s": 5074.184021889482,
    "tokens_per_step": 32495,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "compile_time_s": 143.79905621893704,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "device_backend": "tpu",
    "device_count": 4,
    "exit_code": 0,
    "exp_name": "8_reference",
    "global_batch_size": 8,
    "log_tail": "model loaded at 0.8180377059616148s \nNo custom chat_template provided; using default chat formatting.\nTotal memory size: 21.9 GB, Output size: 0.7 GB, Temp size: 20.5 GB, Argument size: 0.7 GB, Host temp size: 0.0 GB.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"compile_time_s\": 143.79905621893704, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"device_backend\": \"tpu\", \"device_count\": 4, \"exp_name\": \"8_reference\", \"global_batch_size\": 8, \"loss_impl\": \"reference\", \"max_length\": 4096, \"memory\": {\"argument_gb\": 0.7, \"host_temp_gb\": 0.0, \"output_gb\": 0.7, \"temp_gb\": 20.5, \"total_gb\": 21.9}, \"model_id\": \"Qwen/Qwen3-0.6B\", \"optimizer\": \"sgd\", \"packing\": true, \"status\": \"ok\", \"step_time_s\": 5.635932644487669, \"timed_steps\": 30, \"tokens_per_s\": 5765.682815919093, \"tokens_per_step\": 32495, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 56148.65it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 5 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 4096,
    "memory": {
      "argument_gb": 0.7,
      "host_temp_gb": 0.0,
      "output_gb": 0.7,
      "temp_gb": 20.5,
      "total_gb": 21.9
    },
    "model_id": "Qwen/Qwen3-0.6B",
    "optimizer": "sgd",
    "packing": true,
    "status": "ok",
    "step_time_s": 5.635932644487669,
    "timed_steps": 30,
    "tokens_per_s": 5765.682815919093,
    "tokens_per_step": 32495,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "ExceptionGroup: all implementations failed (1 sub-exception)",
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "log_tail": "model loaded at 0.819512443151325s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"ExceptionGroup: all implementations failed (1 sub-exception)\", \"exp_name\": \"8_xla_chunked\", \"loss_impl\": \"xla_chunked\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"error\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 49578.06it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 8 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "error",
    "timed_steps": 30,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>",
    "exit_code": 0,
    "exp_name": "8_reference",
    "log_tail": "model loaded at 0.8022861848585308s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>\", \"exp_name\": \"8_reference\", \"loss_impl\": \"reference\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"oom\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 31230.86it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "oom",
    "timed_steps": 30,
    "warmup_steps": 0
  }
]
```


---

## Qwen3-0.6B (2026-02-25 12:52:51Z)
- Config: `/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py`
- Sweep: packing=True, max_length ∈ {8096}, loss_impl ∈ {xla_chunked, reference}
- Steps: warmup=0, timed=30
- global_batch_size: `8`
- optimizer: `sgd`

| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |
|---:|---|---|---:|---:|---:|---:|---:|
| 8096 | xla_chunked | error | - | - | - | - | - |
| 8096 | reference | oom | - | - | - | - | - |

### Raw results
```json
[
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 32768
    },
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "ExceptionGroup: all implementations failed (1 sub-exception)",
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "log_tail": "model loaded at 0.8281633933074772s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 32768}, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"ExceptionGroup: all implementations failed (1 sub-exception)\", \"exp_name\": \"8_xla_chunked\", \"loss_impl\": \"xla_chunked\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"error\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 26563.04it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 8 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "error",
    "timed_steps": 30,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>",
    "exit_code": 0,
    "exp_name": "8_reference",
    "log_tail": "model loaded at 0.8201564941555262s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>\", \"exp_name\": \"8_reference\", \"loss_impl\": \"reference\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"oom\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 33314.57it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "oom",
    "timed_steps": 30,
    "warmup_steps": 0
  }
]
```


---

## Qwen3-0.6B (2026-02-25 13:29:05Z)
- Config: `/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py`
- Sweep: packing=True, max_length ∈ {8096}, loss_impl ∈ {xla_chunked, reference}
- Steps: warmup=0, timed=30
- global_batch_size: `8`
- optimizer: `sgd`

| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |
|---:|---|---|---:|---:|---:|---:|---:|
| 8096 | xla_chunked | error | - | - | - | - | - |
| 8096 | reference | oom | - | - | - | - | - |

### Raw results
```json
[
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 8192
    },
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "ExceptionGroup: all implementations failed (1 sub-exception)",
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "log_tail": "model loaded at 0.8484425307251513s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 8192}, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"ExceptionGroup: all implementations failed (1 sub-exception)\", \"exp_name\": \"8_xla_chunked\", \"loss_impl\": \"xla_chunked\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"error\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 32640.50it/s]\nWarning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\nWARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 8 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "error",
    "timed_steps": 30,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>",
    "exit_code": 0,
    "exp_name": "8_reference",
    "log_tail": "model loaded at 0.8199789342470467s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>\", \"exp_name\": \"8_reference\", \"loss_impl\": \"reference\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"oom\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 37282.70it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "oom",
    "timed_steps": 30,
    "warmup_steps": 0
  }
]
```


---

## Qwen3-0.6B (2026-02-25 13:43:15Z)
- Config: `/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py`
- Sweep: packing=True, max_length ∈ {8096}, loss_impl ∈ {xla_chunked, reference}
- Steps: warmup=0, timed=30
- global_batch_size: `8`
- optimizer: `sgd`

| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |
|---:|---|---|---:|---:|---:|---:|---:|
| 8096 | xla_chunked | error | - | - | - | - | - |
| 8096 | reference | oom | - | - | - | - | - |

### Raw results
```json
[
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 8192
    },
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "ExceptionGroup: all implementations failed (1 sub-exception)",
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "log_tail": "model loaded at 0.8077079257927835s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 8192}, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"ExceptionGroup: all implementations failed (1 sub-exception)\", \"exp_name\": \"8_xla_chunked\", \"loss_impl\": \"xla_chunked\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"error\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 13319.48it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 8 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "error",
    "timed_steps": 30,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>",
    "exit_code": 0,
    "exp_name": "8_reference",
    "log_tail": "model loaded at 0.8532106908969581s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>\", \"exp_name\": \"8_reference\", \"loss_impl\": \"reference\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"oom\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 16946.68it/s]\n/mnt/carles/llm/src/jaxformers/data/training.py:170: UserWarning: Shuffling a MapDataset may not yield optimal performance due to memory-mapped access. If shuffling is important for your workflow, please pre-shuffle the dataset.\n  warnings.warn(\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "oom",
    "timed_steps": 30,
    "warmup_steps": 0
  }
]
```


---

## Qwen3-0.6B (2026-02-25 13:47:53Z)
- Config: `/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py`
- Sweep: packing=True, max_length ∈ {8096}, loss_impl ∈ {xla_chunked, reference}
- Steps: warmup=0, timed=30
- global_batch_size: `8`
- optimizer: `sgd`

| max_length | loss_impl | status | compile_time_s | step_time_s | tokens/step | tokens/s | total_mem_gb |
|---:|---|---|---:|---:|---:|---:|---:|
| 8096 | xla_chunked | error | - | - | - | - | - |
| 8096 | reference | oom | - | - | - | - | - |

### Raw results
```json
[
  {
    "batch_size": 8,
    "block_sizes": {
      "b": 1024,
      "h": 512,
      "v": 8192
    },
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "ExceptionGroup: all implementations failed (1 sub-exception)",
    "exit_code": 0,
    "exp_name": "8_xla_chunked",
    "log_tail": "model loaded at 0.8142566452734172s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": {\"b\": 1024, \"h\": 512, \"v\": 8192}, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"ExceptionGroup: all implementations failed (1 sub-exception)\", \"exp_name\": \"8_xla_chunked\", \"loss_impl\": \"xla_chunked\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"error\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 10451.79it/s]\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 8 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "xla_chunked",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "error",
    "timed_steps": 30,
    "warmup_steps": 0
  },
  {
    "batch_size": 8,
    "block_sizes": null,
    "config_path": "/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py",
    "error": "JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>",
    "exit_code": 0,
    "exp_name": "8_reference",
    "log_tail": "model loaded at 0.8169348048977554s \nNo custom chat_template provided; using default chat formatting.\nBENCH_RESULT {\"batch_size\": 8, \"block_sizes\": null, \"config_path\": \"/mnt/carles/llm/src/jaxformers/config/config_qwen_0_6b_loss_bench.py\", \"error\": \"JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=39362363392) would exceed memory (size=34359738368) :: #allocation13856 [shape = 'f32[64768,151936]{1,0:T(8,128)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of fusion.10958.remat@{}'] :: <no-hlo-instruction>\", \"exp_name\": \"8_reference\", \"loss_impl\": \"reference\", \"max_length\": 8096, \"optimizer\": \"sgd\", \"status\": \"oom\", \"timed_steps\": 30, \"warmup_steps\": 0}\n\n\nFetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]\nFetching 10 files: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 10/10 [00:00<00:00, 20651.42it/s]\n/home/carlesoctav/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 4 leaked shared_memory objects to clean up at shutdown\n  warnings.warn('resource_tracker: There appear to be %d '",
    "loss_impl": "reference",
    "max_length": 8096,
    "optimizer": "sgd",
    "status": "oom",
    "timed_steps": 30,
    "warmup_steps": 0
  }
]
```
