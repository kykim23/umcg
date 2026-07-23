# UMCG Production Pretraining

이 저장소는 C4 pretraining을 위한 독립 코드베이스다.

Full gradient와 Russian Roulette UMCG를 같은 학습 경로에서 비교한다. 이전 연구 저장소의 코드나 package를 import하지 않는다.

## 공개 진입점

공개 진입점은 아래 여섯 개뿐이다.

| 목적 | 파일 |
|---|---|
| 실제 학습 | `torchrun_main.py` |
| tail probability calibration | `calibrate_main.py` |
| 최대 microbatch 검사 | `vram_check_main.py` |
| 공통 FP32 weight export | `export_weights_main.py` |
| CPU 또는 단일 GPU smoke test | `smoke_main.py` |
| local C4 전처리 | `prepare_c4_main.py` |

Python package는 내부 구현용이다. 학습은 항상 `torchrun_main.py`로 시작한다.

학습 인자는 `src/umcg/cli/arguments.py`에서 한 번만 정의한다. `runner.py`는 parser가 아니라 파싱된 `RuntimeConfig`를 받는다. 기능별 전체 인자와 기본값은 다음 명령 및 `docs/CONFIG_REFERENCE.md`에서 확인한다.

```bash
python torchrun_main.py --help
```

## 설치

Python 3.12, CUDA 12.8용 PyTorch 2.11, Transformers 5.14를 기준으로 한다.

Triton, DeepSpeed, FlashAttention을 사용하는 환경에는 C/C++ compiler와 binutils가 필요하다. 실행 전 build tool이 현재 shell의 `PATH`에서 확인되어야 한다.

```bash
pip install -e '.[dev,distributed,low-precision]'
```

이 서버의 재현 환경은 `requirements-implementation.txt`의 정확한 버전을 사용한다.

DeepSpeed를 쓰지 않으면 `distributed` extra를 생략할 수 있다. FP8 또는 AdamW 8-bit를 쓰지 않으면 `low-precision` extra를 생략할 수 있다.

## 실제 C4 학습

DDP 4-GPU 예시는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=8 \
torchrun --standalone --nproc_per_node 4 torchrun_main.py \
  --distributed_backend ddp \
  --model_backend huggingface \
  --model_config configs/model/llama_350m_t5_4096.json \
  --tokenizer t5-base \
  --tokenizer_revision main \
  --precision bfloat16 \
  --attention_backend automatic \
  --estimator_config configs/estimator/russian_roulette_safe_4096.json \
  --gradient_estimator russian_roulette \
  --optimizer adamw \
  --scheduler cosine \
  --learning_rate 3e-4 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0.1 \
  --batch_size auto \
  --total_batch_size 512 \
  --num_training_steps 60000 \
  --warmup_steps 6000 \
  --eval_every 1000 \
  --save_every 20000 \
  --save_at_step 30000 \
  --save_dir checkpoint/umcg_350m_rr \
  --c4_source streaming \
  --c4_repo allenai/c4 \
  --c4_revision main \
  --workers 1 \
  --seed 777 \
  --use_torch_compile \
  --activation_checkpointing \
  --wandb_mode online \
  --name umcg_350m_rr_seed777
```

`russian_roulette_safe_4096.json`은 모든 tail probability가 `Q=1`인 수학적 정확성 기준선이다. 별도 GPU 실행의 bitwise equality를 뜻하지는 않는다. 효율적인 Russian Roulette 실험에는 `calibrate_main.py` 결과나 사용자가 검증한 estimator JSON을 사용한다.

`russian_roulette_smoke_4096.json`의 `[1, 0.5, 0.25, 0.125]`는 level 선택과 correction 경로만 확인하는 미검증 smoke 일정이다. 효율성 주장이 없으며 최종 실험에는 측정 64, 선택 32, 독립 감사 32를 통과한 calibration 결과를 사용한다.

이미 내려받은 원본 C4 `json.gz` shard를 직접 읽을 때에는 다음 인자를 사용한다.

```bash
--c4_source local_raw \
--c4_local_path /home/ubuntu/data/c4_en/en \
--c4_revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2
```

`local_raw`는 원본을 복제하거나 사전 토큰화하지 않는다. 정렬된 train·validation shard 목록과 파일 크기의 SHA-256을 실행 계약과 checkpoint에 고정한다. 학습 중 gzip 해제와 tokenization을 수행하므로 CPU 비용은 `local`보다 크지만, 미리 tokenization한 자료의 추가 저장·네트워크 비용은 들지 않는다.

FSDP2는 한 값만 바꾼다.

```bash
--distributed_backend fsdp2
```

DeepSpeed ZeRO는 stage까지 지정한다.

```bash
--distributed_backend zero --zero_stage 1
--distributed_backend zero --zero_stage 2
--distributed_backend zero --zero_stage 3
```

`batch_size`는 rank별 문서 수다. `total_batch_size`는 전체 optimizer update의 문서 수다. 실제 accumulation 수는 두 값과 world size에서 계산된다.

`--save_at_step`은 `--save_every`와 별개인 정확한 milestone 저장 옵션이며 여러 번 지정할 수 있다. 예를 들어 `--save_at_step 30 --save_at_step 300`은 두 update를 추가로 보존한다.

`batch_size=auto`는 전체 batch의 divisor만 시험한다. 최대 context에서 실제 backend로 forward와 backward를 실행한다. 모든 rank가 GPU memory 90% 이하인 가장 큰 값을 고른다. 선택값은 checkpoint에 고정된다.

## C4 문서 경계

C4 English row 하나를 document 하나로 취급한다. 서로 다른 row의 token은 절대 한 parent에 합치지 않는다.

한 row는 special token 없이 tokenize한다. 실제 document 끝에만 EOS를 붙인다. 최대 context 길이로 겹치지 않게 자른다. 마지막 parent만 오른쪽 PAD를 갖는다.

9,000-token row와 4,096 preset은 다음 parent를 만든다.

- `0..4095`
- `4096..8191`
- `8192..8999`, EOS, 오른쪽 PAD

각 sample에는 document SHA-256, chunk index, 원래 token 시작과 끝 위치가 저장된다. 상세 계약은 `docs/DATA_CONTRACT.md`에 있다.

원문 길이가 최대 context의 정확한 배수이면 EOS 단독 조각은 causal target이 없어 제외된다. 서로 다른 parent는 별도 forward이므로 이 경계 동작이 문서 사이의 model state를 연결하지는 않는다.

## Model과 precision

`model_backend`는 `huggingface`와 `reference`다. 두 구현은 같은 canonical weight key를 쓴다. LM head와 cross entropy는 sequence chunk 단위로 계산한다.

`attention_backend=automatic`은 로컬에 설치된 kernel만 검사한다. 각 후보의 forward, backward, finite 값, eager 오차, 오른쪽 padding, level별 시간을 실제로 시험한다. 모든 rank에서 가능한 후보 중 가장 빠른 것을 고른다. 3% 이내면 PyTorch backend를 우선한다.

Precision은 다음 네 가지다.

- `float32`: autocast와 scaler 없음
- `float16`: FP16 autocast와 dynamic GradScaler
- `bfloat16`: BF16 autocast와 FP32 loss reduction
- `float8`: TorchAO FP8 Linear와 BF16 attention

FP8이 불가능한 GPU나 TorchAO 환경에서는 즉시 종료한다.

## Optimizer와 scheduler

Optimizer는 `adamw`, `adam`, `sgd`, `sgdm`, `muon`, `adamw_8bit`다. AdamW 8-bit는 TorchAO를 사용한다.

Muon은 attention과 MLP의 2차원 hidden weight에만 적용한다. Embedding, LM head, normalization, bias는 AdamW가 맡는다. tied weight는 한 번만 분류한다.

Scheduler는 `cosine`과 `linear`다.

Multi-GPU 조합의 코드는 구현되어 있다. 현재 저장소의 공식 외부 검증 상태는 `pending`이다. `docs/HANDOFF_TO_EXPERIMENT_SERVER.md`의 전체 검증을 통과하기 전에는 지원 완료로 간주하지 않는다.

## Resume와 backend 전환

Native resume는 같은 backend에서만 가능하다.

```bash
--continue_from checkpoint/umcg_350m_rr/checkpoint-00020000
```

Model, backend, world size, workers, batch, optimizer, scheduler, precision, tokenizer, C4 revision, estimator, attention package가 달라지면 종료한다.

Backend를 바꾸려면 먼저 공통 FP32 weight를 만든다.

```bash
python export_weights_main.py \
  --checkpoint checkpoint/umcg_350m_rr/checkpoint-00020000 \
  --output exported/umcg_350m_step20000.safetensors
```

새 run에서 다음 값을 준다.

```bash
--initial_weights exported/umcg_350m_step20000.safetensors
```

이 경우 optimizer, scheduler, data iterator는 새로 시작한다.

## 보조 작업

CPU smoke test:

```bash
python smoke_main.py \
  --device cpu \
  --model_backend reference \
  --model_config configs/model/llama_tiny_smoke_1024.json \
  --precision float32 \
  --attention_backend eager \
  --estimator_config configs/estimator/russian_roulette_safe_1024.json \
  --gradient_estimator russian_roulette \
  --optimizer adamw \
  --scheduler cosine \
  --batch_size 1 \
  --num_training_steps 1 \
  --save_dir /tmp/umcg-smoke
```

Calibration, local C4, VRAM 검사는 각 문서를 참고한다.

- `docs/CONFIG_REFERENCE.md`
- `docs/DATA_CONTRACT.md`
- `docs/SCIENTIFIC_CONTRACT.md`
- `docs/SMOKE_TEST_PROTOCOL.md`
- `docs/HANDOFF_TO_EXPERIMENT_SERVER.md`
