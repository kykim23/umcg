# Canonical 설정

Model 구조는 model JSON에 둔다. Context level과 tail probability는 estimator JSON에 둔다. 나머지 runtime 값은 CLI에서만 받는다.

## 실제 학습 CLI

### 분산과 model

- `--distributed_backend {ddp,fsdp2,zero}`: 필수
- `--zero_stage {1,2,3}`: `zero`에서만 필수
- `--model_backend {huggingface,reference}`: 필수
- `--model_config PATH`: 필수
- `--tokenizer NAME`: 기본 `t5-base`
- `--tokenizer_revision REVISION`: 기본 `main`

### Precision과 attention

- `--precision {float32,float16,bfloat16,float8}`: 필수
- `--attention_backend {automatic,flash_attention_4,flash_attention_3,flash_attention_2,pytorch_sdpa,eager}`: 기본 `automatic`

명시한 attention backend는 실패해도 다른 backend로 내려가지 않는다. 자동 선택에서만 fallback한다.

### Estimator

- `--estimator_config PATH`: 필수
- `--gradient_estimator {full,russian_roulette}`: 필수

기본 estimator JSON은 안전한 `Q=1` 설정이다. 효율적인 확률은 calibration 결과 또는 사용자가 검증한 JSON으로 넣는다.

### Optimizer와 scheduler

- `--optimizer {adamw,adam,sgd,sgdm,muon,adamw_8bit}`: 필수
- `--scheduler {cosine,linear}`: 필수
- `--learning_rate FLOAT`: 필수
- `--beta1 FLOAT`: 기본 `0.9`
- `--beta2 FLOAT`: 기본 `0.95`
- `--epsilon FLOAT`: 기본 `1e-8`
- `--weight_decay FLOAT`: 기본 `0.1`
- `--momentum FLOAT`: `sgdm` 또는 `muon`에서 선택
- `--gradient_clip_norm FLOAT`: 선택

`sgd`는 momentum을 받지 않는다. `sgdm`의 기본 momentum은 `0.9`다. Muon의 기본 momentum은 `0.95`다.

### Batch와 step

- `--batch_size INTEGER|auto`: 필수
- `--total_batch_size INTEGER`: 필수
- `--num_training_steps INTEGER`: 필수
- `--warmup_steps INTEGER`: 필수
- `--eval_every INTEGER`: 필수
- `--eval_parent_batches INTEGER`: 기본 `32`
- `--save_every INTEGER`: 필수

계산식은 다음과 같다.

```text
total_batch_size = batch_size * world_size * accumulation_steps
```

나누어떨어지지 않으면 model 생성 전에 종료한다.

### Data

- `--c4_source {streaming,local,local_raw}`: 필수
- `--c4_repo NAME`: 기본 `allenai/c4`
- `--c4_revision REVISION`: 기본 `main`
- `--c4_local_path PATH`: `local`과 `local_raw`에서 필수
- `--workers INTEGER`: rank별 논리 worker 수, 기본 `1`

Streaming C4와 tokenizer revision은 시작 시 immutable Hub commit으로 고정된다.

`local`은 `prepare_c4_main.py`가 만든 tokenized parent manifest를 읽는다. `local_raw`는 C4 원본 `c4-train.*.json.gz`와 `c4-validation.*.json.gz`를 직접 streaming한다. 후자는 파일 목록과 크기의 SHA-256 및 사용자가 지정한 upstream revision을 실행 설정에 저장한다.

### 실행과 기록

- `--save_dir PATH`: 필수
- `--seed INTEGER`: 필수
- `--use_torch_compile`: 선택
- `--compile_mode {default,reduce-overhead,max-autotune}`: 기본 `default`
- `--activation_checkpointing`: 선택
- `--wandb_mode {online,offline,disabled}`: 기본 `online`
- `--wandb_project NAME`: 기본 `umcg-pretraining`
- `--wandb_entity NAME`: 선택
- `--name NAME`: 필수

### 재시작

- `--continue_from PATH`: native state 전체를 정확히 이어감
- `--initial_weights PATH`: 공통 weight에서 새 run을 시작함

두 값은 함께 쓸 수 없다.

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

Context preset은 다음 두 개다.

- `[128, 256, 512, 1024]`
- `[512, 1024, 2048, 4096]`

Tail probability는 첫 값이 `1.0`이어야 한다. 값은 양수이고 단조 감소해야 한다.

## Model JSON

Model JSON은 저장소의 canonical LLaMA schema를 사용한다. 두 backend가 같은 파일을 읽는다. 알 수 없는 field는 무시하지 않고 즉시 종료한다.

`vocab_size`, `eos_token_id`, `pad_token_id`는 실제 tokenizer와 정확히 같아야 한다. 자동 embedding resize는 없다. `max_position_embeddings`는 가장 긴 context 이상이어야 한다.

`lm_head_chunk_size`는 sequence 방향 LM head chunk 크기다. 이 값은 전체 vocabulary logits tensor가 한 번에 만들어지는 것을 막는다.

## 조합 검사

- DDP: 여섯 optimizer, 네 precision 경로 구현
- FSDP2: Muon 제외, 네 precision 경로 구현
- ZeRO-1·2·3: AdamW 8-bit와 FP8 제외
- Muon과 AdamW 8-bit: FP8 제외

실제 지원 표시는 외부 서버 검증 보고서가 있어야 한다.
