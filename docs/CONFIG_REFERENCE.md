# Canonical 설정 참고서

Model 구조는 model JSON(JavaScript Object Notation)에, context level과 tail probability는 estimator JSON에 둔다. 나머지 runtime 값은 명령행 인터페이스(command-line interface, CLI)에서만 받는다.

## 인자 전달 흐름

실제 호출 순서는 다음과 같다.

```text
torchrun_main.py
  -> umcg.training.runner.main(arguments)
  -> umcg.cli.arguments.parse_runtime_config(arguments)
  -> RuntimeConfig
  -> umcg.training.runner.run(config)
```

`runner.py`가 `ArgumentParser`를 받는 것은 아니다. `src/umcg/cli/arguments.py`가 유일한 parser 원본이고, `runner.py`는 파싱이 끝난 `RuntimeConfig`를 받는다. `runner.py` 시작부의 `SUPPORTED_RUNTIME_ARGUMENTS`는 전체 field를 기능별로 빠르게 보는 색인이며 자동 시험이 parser와의 일치를 보장한다.

현재 실행 환경의 전체 help는 다음 명령으로 확인한다.

```bash
python torchrun_main.py --help
```

## 실제 학습 CLI

### 분산 실행과 model

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--distributed_backend` | 예 | 없음 | `ddp`, `fsdp2`, `zero` | 항상 | 분산 runtime 선택 | 허용값 밖이면 parser 종료 |
| `--zero_stage` | 조건부 | 없음 | `1`, `2`, `3` | `zero`에서 필수 | DeepSpeed ZeRO stage | `zero`에서 누락하거나 다른 backend에서 지정하면 model 생성 전 종료 |
| `--model_backend` | 예 | 없음 | `huggingface`, `reference` | 항상 | model 구현 선택 | 허용값 밖이면 parser 종료 |
| `--model_config` | 예 | 없음 | 경로 | 항상 | Canonical LLaMA 구조 JSON | 파일·schema·최대 context가 맞지 않으면 종료 |
| `--tokenizer` | 아니요 | `t5-base` | Hub 이름 또는 경로 | 항상 | C4 tokenization | model vocabulary·EOS·PAD와 다르면 종료 |
| `--tokenizer_revision` | 아니요 | `main` | revision | 항상 | 시작할 때 immutable commit으로 해석 | revision을 해석하지 못하면 종료 |

### 정밀도와 attention

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--precision` | 예 | 없음 | `float32`, `float16`, `bfloat16`, `float8` | 항상 | autocast·scaler·저정밀 module 경로 선택 | backend·optimizer와 금지된 조합이면 model 생성 전 종료 |
| `--attention_backend` | 아니요 | `automatic` | `automatic`, `flash_attention_4`, `flash_attention_3`, `flash_attention_2`, `pytorch_sdpa`, `eager` | 항상 | Attention kernel 선택 | 명시 backend가 probe에 실패하면 fallback 없이 종료 |

`automatic`에서만 사용 가능한 후보를 실제 forward·backward로 검사한 뒤 fallback할 수 있다.

### Gradient estimator

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--estimator_config` | 예 | 없음 | 경로 | 항상 | Context level과 tail probability (Q_k)를 제공 | schema, level, 확률이 계약과 다르면 종료 |
| `--gradient_estimator` | 예 | 없음 | `full`, `russian_roulette` | 항상 | 실제 gradient 계산 방식 선택 | 허용값 밖이면 parser 종료 |

두 인자는 겹치지 않는다.

- `full`도 어떤 최대 context를 계산할지 알아야 하므로 estimator JSON이 필요하다. 단, JSON의 (Q_k)는 무시하고 항상 가장 긴 level을 실행한다.
- `russian_roulette`는 같은 JSON의 (Q_k=P(N\ge k))로 최대 level (N)을 표본 추출하고 correction을 (1/Q_k)로 가중한다.
- 따라서 동일한 calibration JSON을 full과 Russian Roulette 비교에 사용해도 full gradient는 바뀌지 않는다.

Q=1 안전 템플릿은 수학적 정확성 기준선이다. `russian_roulette_smoke_4096.json`의 `[1, 0.5, 0.25, 0.125]`는 분기 동작 확인용이며 효율성 주장이 없다. 최종 실험은 독립 감사를 통과한 calibration 출력만 사용한다.

### Optimizer와 scheduler

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--optimizer` | 예 | 없음 | `adamw`, `adam`, `sgd`, `sgdm`, `muon`, `adamw_8bit` | 항상 | Optimizer 선택 | backend·precision과 금지된 조합이면 종료 |
| `--scheduler` | 예 | 없음 | `cosine`, `linear` | 항상 | Update 단위 학습률 schedule | 허용값 밖이면 parser 종료 |
| `--learning_rate` | 예 | 없음 | 양수 | 항상 | 최대 학습률 | 0 이하면 종료 |
| `--beta1` | 아니요 | `0.9` | `[0,1)` | Adam 계열 | 첫 moment decay | 범위를 벗어나면 종료 |
| `--beta2` | 아니요 | `0.95` | `[0,1)` | Adam 계열 | 둘째 moment decay | 범위를 벗어나면 종료 |
| `--epsilon` | 아니요 | `1e-8` | 양수 | 지원 optimizer | 수치 안정화 | 0 이하면 종료 |
| `--weight_decay` | 아니요 | `0.1` | 0 이상 | 지원 optimizer | Weight decay | 음수면 종료 |
| `--momentum` | 조건부 | `sgdm=0.9`, `muon=0.95` | `[0,1)` | `sgdm`, `muon`만 | Momentum | `sgd`나 다른 optimizer에서 지정하면 종료 |
| `--gradient_clip_norm` | 아니요 | 없음 | 양수 | 선택 | Global gradient norm 제한 | 0 이하면 종료 |

### Batch, 학습, 평가, 저장 주기

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--batch_size` | 예 | 없음 | 양의 정수 또는 `auto` | 항상 | Rank당 parent microbatch 크기 | 전체 batch를 나눌 수 없거나 probe가 실패하면 종료 |
| `--total_batch_size` | 예 | 없음 | 양의 정수 | 항상 | 한 optimizer update의 전체 parent 수 | world size와 microbatch로 나누어지지 않으면 종료 |
| `--num_training_steps` | 예 | 없음 | 양의 정수 | 항상 | Optimizer update 수 | 0 이하면 종료 |
| `--warmup_steps` | 예 | 없음 | `0..num_training_steps` | 항상 | 학습률 warmup update 수 | 범위를 벗어나면 종료 |
| `--eval_every` | 예 | 없음 | 양의 정수 | 항상 | 평가 간 update 수 | 0 이하면 종료 |
| `--eval_parent_batches` | 아니요 | `32` | 양의 정수 | 평가 | 각 rank가 처리할 validation batch 수 | 0 이하면 종료 |
| `--save_every` | 예 | 없음 | 양의 정수 | 항상 | Checkpoint 저장 간 update 수 | 0 이하면 종료 |
| `--save_at_step` | 아니요 | 없음 | `1..num_training_steps`의 정수, 반복 가능 | 지정 시 | 주기와 별개로 해당 update의 checkpoint 저장 | 범위 밖이거나 중복이면 model 생성 전 종료 |

Accumulation은 다음 식으로 결정된다.

```text
total_batch_size = batch_size * world_size * accumulation_steps
```

예를 들어 GPU 두 개, `batch_size=2`, `eval_parent_batches=32`이면 각 rank가 64 parent, 전체가 128 parent를 최대 context에서 평가한다. 모든 rank의 유효 token loss 합과 유효 target 수를 각각 더해 token 가중 평균을 구하고 `perplexity = exp(validation_loss)`를 계산한다. 평가 stream은 매번 같은 validation 시작점에서 다시 만들어진다.

### C4 자료

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--c4_source` | 예 | 없음 | `streaming`, `local`, `local_raw` | 항상 | C4 입력 방식 선택 | 허용값 밖이면 parser 종료 |
| `--c4_repo` | 아니요 | `allenai/c4` | Hub 이름 | `streaming` | Streaming dataset 위치 | 해석하지 못하면 종료 |
| `--c4_revision` | 아니요 | `main` | revision | 모든 source에서 기록 | Upstream source 식별 | streaming revision을 해석하지 못하면 종료 |
| `--c4_local_path` | 조건부 | 없음 | 경로 | `local`, `local_raw`에서 필수 | Prepared parent 또는 원본 gzip root | 누락하거나 streaming에서 지정하면 종료 |
| `--workers` | 아니요 | `1` | 양의 정수 | 항상 | Rank당 논리 C4 iterator 수 | 0 이하거나 shard 수보다 많으면 종료 |

| Source | 학습 중 CPU 작업 | 저장공간 | 네트워크 특성 | Exact resume |
|---|---|---|---|---|
| `streaming` | 원격 row 해석과 tokenization | 별도 local 복제 없음 | 외부 네트워크 영향 | IterableDataset state 저장 |
| `local_raw` | gzip 해제, JSON 해석, tokenization | 압축 원본만 유지 | 압축 전송량은 작지만 CPU 작업 필요 | 파일 manifest와 iterator state 저장 |
| `local` | 정수 token JSONL 해석 | 사전 tokenized 자료가 추가로 필요 | 읽는 byte가 커질 수 있지만 tokenizer 작업 없음 | Shard·line 위치 저장 |

`local_raw`는 일반적으로 CPU 작업이 더 많지만 `local` 결과는 압축 원본보다 훨씬 커질 수 있으므로 전체 시간이 반드시 짧아지는 것은 아니다. 동일 저장장치에서 data-wait 시간을 측정해 결정한다.

`workers`는 CPU core, thread, process 수가 아니다. 현재 구현은 여러 논리 iterator를 한 process에서 round-robin으로 호출하므로 tokenization을 동시에 수행하지 않는다. `workers=1`이 안전한 기본값이다. 실제 병렬 tokenization에는 별도 producer process, 제한된 prefetch queue, checkpoint 가능한 queue·iterator 상태가 필요하다.

### 실행 성능, 출력, 추적, 재시작

| 인자 | 필수 | 기본값 | 허용값 | 적용 조건 | 실제 동작 | 잘못 사용하면 |
|---|---:|---|---|---|---|---|
| `--save_dir` | 예 | 없음 | 경로 | 항상 | Metrics와 checkpoint root | 새 실행에서 비어 있지 않으면 종료 |
| `--seed` | 예 | 없음 | 정수 | 항상 | Model, data, estimator seed | 형식이 정수가 아니면 parser 종료 |
| `--use_torch_compile` | 아니요 | 꺼짐 | flag | 선택 | Training model compile | Compile 실패 시 종료 |
| `--compile_mode` | 아니요 | `default` | `default`, `reduce-overhead`, `max-autotune` | compile 사용 시 | Compile 최적화 수준 | 허용값 밖이면 parser 종료 |
| `--activation_checkpointing` | 아니요 | 꺼짐 | flag | 선택 | Backward에서 activation 재계산 | 지원하지 않는 model 경로면 종료 |
| `--wandb_mode` | 아니요 | `online` | `online`, `offline`, `disabled` | 추적 | Weights & Biases 기록 방식 | 초기화 실패 시 종료 |
| `--wandb_project` | 아니요 | `umcg-pretraining` | 이름 | 추적 | Project 이름 | 서비스가 거부하면 종료 |
| `--wandb_entity` | 아니요 | 없음 | 이름 | 추적 | 선택적 entity | 서비스가 거부하면 종료 |
| `--name` | 예 | 없음 | 이름 | 항상 | Run 식별자 | 누락하면 parser 종료 |
| `--continue_from` | 아니요 | 없음 | native checkpoint 경로 | Resume | Model·optimizer·scheduler·data·난수 상태 전체 재개 | 현재 runtime 계약과 다르면 종료 |
| `--initial_weights` | 아니요 | 없음 | exported weight 경로 | 새 run | Weight만 불러오고 나머지는 새로 시작 | `continue_from`과 함께 쓰면 parser 종료 |

## Calibration CLI

Calibration은 optimizer update를 수행하지 않는다. 전체 인자는 `python calibrate_main.py --help`로 확인한다.

Canonical full-coordinate 설정은 다음과 같다.

| 인자 | 기본값 | 역할 |
|---|---:|---|
| `--logical_parent_batch_size` | `128` | 하나의 gradient 관측값을 이루는 두 rank 합계 parent 수 |
| `--max_parent_batch_size_per_gpu` | `64` | rank당 먼저 시도할 physical parent 수 |
| `--memory_limit_fraction` | `0.85` | 정확 경로가 사용할 수 있는 rank 최대 VRAM 비율 |
| `--measurement_parent_batches` | `64` | Gradient 기하와 level 비용을 측정하는 논리 batch 수 |
| `--selection_parent_batches` | `32` | 측정 자료와 document가 겹치지 않는 일정 선택 논리 batch 수 |
| `--audit_parent_batches` | `32` | 일정을 고정한 뒤 한 번만 사용하는 독립 감사 논리 batch 수 |
| `--timing_parent_batches` | `8` | 측정 split 앞부분에서 CUDA 시간을 기록할 논리 batch 수 |
| `--timing_repeats` | `1` | 각 논리 batch·context의 forward/backward 시간 측정 횟수 |
| `--bootstrap_parent_resamples` | `10000` | 감사 효율비의 95% 신뢰구간을 위한 논리 batch 재표본 수 |
| `--parent_cache` | 출력 옆 경로 | 세 split의 token과 document 배정을 고정해 재사용 |
| `--activation_checkpointing` | 켜짐 | Backward에서 activation을 재계산해 full-gradient 보유 공간 확보 |

`64/32/32`는 문서 64/32/32개가 아니라 논리 batch 수다. 논리 batch가 128이므로 실제 parent sample 수는 각각 `8192/4096/4096`, 합계 `16384`개다. 각 논리 batch는 rank 0의 64개와 rank 1의 64개를 하나의 전역 valid-token 평균 gradient로 합친다. Physical 64가 VRAM 85% 관문을 넘으면 rank별 32개씩 두 번 계산해 합치되, 논리 gradient의 parent 128개는 바꾸지 않는다.

350M model의 모든 parameter gradient 좌표를 그대로 사용한다. 각 논리 batch에서 `G_512`, `G_1024`, `G_2048`, `G_4096`의 4×4 Gram 행렬을 FP64 dot product로 갱신하고 해당 batch gradient를 폐기한다. CountSketch, 특이값 분해(Singular Value Decomposition, SVD), 다른 투영은 사용하지 않는다. Level Gram과 직접 차분한 correction Gram이 일치하는지도 독립 계산으로 검사한다.

후보는 `Q_512=1`이고 나머지가 `{1,.75,.5,.25,.125}`에 속하는 단조감소 일정 35개다. 각 일정의 네 가지 가능한 최대 level을 확률로 정확히 열거하므로 후보 선택에 Monte Carlo 표본은 쓰지 않는다. 감사 효율 신뢰구간에만 parent-batch bootstrap을 사용한다.

GPU 비용 `C_k`는 측정 split의 앞 8개 논리 batch에서 context별 한 번씩 측정한 두 rank CUDA 시간 중 큰 값의 평균이다. 이전 `--parent_batches`와 `--batch_size`는 각각 측정 batch 수와 physical 상한의 호환 alias다. CountSketch용 `--sketch_dimension`과 후보 Monte Carlo용 `--monte_carlo_samples`는 제거되어 전달하면 즉시 종료한다. 독립 감사가 실패하면 진단 report만 쓰고 estimator JSON은 만들지 않는다.

`--locked_estimator PATH --diagnostic_only --initial_weights PATH`를 함께 사용하면 중간 checkpoint의 gradient drift를 재측정하되 원래 tail schedule을 바꾸지 않는다.

## Estimator JSON

허용 field는 정확히 다섯 개다.

```json
{
  "schema_version": 1,
  "context_levels": [512, 1024, 2048, 4096],
  "tail_probabilities": [1.0, 1.0, 1.0, 1.0],
  "sampling": "shared_global_microbatch",
  "source": {"type": "safe_template"}
}
```

지원 context preset은 `[128,256,512,1024]`와 `[512,1024,2048,4096]`다. Tail probability의 첫 값은 `1.0`이고, 모든 값은 `(0,1]` 범위에서 단조감소해야 한다.

## Model JSON

Model JSON은 canonical LLaMA schema를 사용한다. 알 수 없는 field는 무시하지 않는다. `vocab_size`, `eos_token_id`, `pad_token_id`는 tokenizer와 같아야 하며 `max_position_embeddings`는 가장 긴 context 이상이어야 한다.

`lm_head_chunk_size`는 sequence 방향 language-model head chunk 크기다. 전체 vocabulary logits를 한 번에 만들지 않기 위한 값이다.

## 조합 검사

- DDP: 여섯 optimizer, 네 precision 경로 구현
- FSDP2: Muon 제외, 네 precision 경로 구현
- ZeRO-1·2·3: AdamW 8-bit와 FP8 제외
- Muon과 AdamW 8-bit: FP8 제외

실제 지원 표시는 외부 서버 검증 보고서가 있어야 한다.
