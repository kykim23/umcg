
# Codex 실행 지시서: UMCG 코드 수정, 기존 Checkpoint 전수 Audit, 160M C4-long 실험

상태: **최종 실행 지시서**

최종 갱신: 2026-07-24 (Asia/Seoul)

---

# 0. 최상위 목표

이 작업의 목표는 다음 세 가지다.

1. UMCG의 사용자-facing 확률 설정과 sampling 구현을 더 명확하고 안정적으로 수정한다.
2. 기존 350M Full/RR 실험의 모든 저장 checkpoint에서 optimizer 및 spectral mechanism을 정밀 분석한다.
3. Padding이 없는 C4-long에서 160M Full, IID Russian Roulette, coordinated Russian Roulette을 공정하게 비교한다.

최종 목표는 단순한 engineering speedup이 아니다.

다음을 논문 수준으로 판별해야 한다.

- RR의 raw gradient unbiasedness가 AdamW update 품질로 이어지는가
- Microbatch별 독립 context sampling이 추가적인 optimizer noise를 만드는가
- Coordinated sampling이 그 문제를 줄이는가
- Padding 없는 실제 4K document window에서도 계산 이득이 남는가
- Context 길이별 full gradient와 correction gradient의 spectral structure가 학습 중 어떻게 달라지는가
- 향후 Dataset Decomposition과 결합할 가치가 있는가

---

# 1. 절대 준수사항

## 1.1 GitHub 권한

Codex는 로컬 코드만 수정한다.

다음을 수행하지 않는다.

- Git branch 생성
- commit
- push
- pull request
- issue
- tag
- remote 변경
- GitHub API write
- `gh` write 명령

GitHub 작업 권한은 사용자에게만 있다.

## 1.2 Conda 환경

오직 다음 환경만 사용한다.

```bash
conda activate umcg
```

또는 비대화형 명령에서는 다음을 사용한다.

```bash
conda run -n umcg ...
```

금지:

- base 환경 수정
- 다른 Conda 환경 수정
- 새 Conda 환경 생성
- bare `pip`
- `sudo pip`
- system Python 변경

새 package가 필요하면 `umcg` 환경에만 설치한다.

설치 전후 package와 version, 설치 명령, 설치 이유를 작업 로그에 기록한다.

## 1.3 작업 로그

매 세션 시작 전에 다음 파일을 읽는다.

```text
codex_log/user_instructions.md
codex_log/conversation_history.md
codex_log/work_log.md
```

작업 중 다음을 계속 갱신한다.

```text
codex_log/work_log.md
```

매 phase 종료 시 반드시 기록한다.

- 완료한 항목
- 변경 파일
- 실행 명령
- 테스트 결과
- 출력 경로
- 실패와 수정
- 남은 blocker
- 다음 시작점

장시간 분석 전후에는 이 지시서와 `codex_log`를 다시 읽는다.

## 1.4 기존 결과 불변성

기존 350M 결과와 checkpoint를 수정하거나 덮어쓰지 않는다.

권위 원본:

```text
/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32
```

분석용 저장소 복사본:

```text
/home/ubuntu/keunyoung/workspace/umcg/results/full_gradient_3000_20260721T232425_KST_f57ee32
```

새 분석은 새로운 experiment ID 아래에 저장한다.

## 1.5 과학적 비교 계약

Primary Full/RR 비교에서는 다음을 동일하게 유지한다.

- Model architecture
- Initial weight
- Data order
- Tokenizer
- Optimizer
- Learning rate
- Scheduler
- Warmup
- Weight decay
- Beta
- Precision
- Attention backend
- Hardware
- Distributed launcher
- Global batch
- Validation set

RR만 learning rate를 별도로 튜닝하지 않는다.

Correction coefficient를 임의로 조절하지 않는다.

Control variate와 dynamic probability schedule은 이번 작업에서 구현하지 않는다.

Multi-seed는 수행하지 않는다.

---

# 2. 로컬 입력 경로

## 2.1 UMCG repository

```text
/home/ubuntu/keunyoung/workspace/umcg
```

## 2.2 160M model config

```text
/home/ubuntu/keunyoung/workspace/umcg/configs/model/llama_160m_t5_4096.json
```

이 파일은 현재 remote default branch에서 확인되지 않았으므로 로컬 파일이 유일한 기준이다.

## 2.3 Dataset Decomposition

```text
/home/ubuntu/keunyoung/workspace/ml-dataset-decomposition
```

현재는 baseline 대기 작업이다.

즉시 수정하거나 실행하지 않는다.

## 2.4 C4-long

사용자가 이미 C4-long train/validation을 생성했다.

Codex는 `codex_log`, 기존 config, 사용자 생성 manifest에서 실제 경로를 확인한다.

경로가 하나로 명확하지 않으면 임의로 추측하지 않는다.

다음 변수명을 작업 로그에 확정해 기록한다.

```text
C4_LONG_TRAIN_PATH
C4_LONG_VALID_PATH
C4_LONG_MANIFEST_PATH
```

원본 C4에서 다시 추출하지 않는다.

---

# 3. 용어 및 수학적 표기 계약

## 3.1 Document

앞으로 새 코드와 문서에서는 다음을 사용한다.

```text
source_document
document_window
full_document_window
partial_document_window
```

새 코드에 `parent`라는 명칭을 추가하지 않는다.

기존 checkpoint/schema의 `parent` field는 재현성을 위해 변경하지 않는다.

## 3.2 Context-level gradient

Document batch \(b\), context level \(L_k\)의 gradient를 다음처럼 쓴다.

\[
G_{b,k}
=
\nabla_\theta L_{b,L_k}
\]

코드 이름:

```text
level_gradient
```

## 3.3 Correction gradient

\[
\Delta G_{b,0}=G_{b,0}
\]

\[
\Delta G_{b,k}
=
G_{b,k}-G_{b,k-1}
=
\nabla_\theta
\left(
L_{b,L_k}-L_{b,L_{k-1}}
\right)
\]

코드 이름:

```text
correction_gradient
```

Correction은 full causal-LM gradient와 역할이 다르지만 signed correction objective의 gradient다.

## 3.4 Second moment와 variance

기존 generic `V_k` 이름을 새 output에 사용하지 않는다.

Correction gradient의 uncentered second moment:

\[
M_{2,k}
=
\mathbb E_b
\left[
\|\Delta G_{b,k}\|_2^2
\right]
\]

필드:

```text
correction_gradient_second_moment_m2
```

Centered variance trace:

\[
\operatorname{VarTrace}_k
=
\mathbb E_b
\left[
\|\Delta G_{b,k}-\mathbb E_b\Delta G_{b,k}\|_2^2
\right]
\]

\[
=
M_{2,k}
-
\|\mathbb E_b\Delta G_{b,k}\|_2^2
\]

필드:

```text
correction_gradient_variance_trace
```

후보 estimator 전체의 centered gradient variance:

```text
estimator_gradient_variance
```

기존 result JSON은 수정하지 않는다.

Legacy reader만 과거 `V_k`를 읽고 새 명칭으로 해석한다.

---

# 4. Phase 0 — 시작 감사

코드를 수정하기 전에 다음을 수행한다.

1. `codex_log` 3개 문서 읽기
2. 현재 source tree 상태 확인
3. 현재 CPU test 목록 확인
4. 현재 GPU process 확인
5. 기존 350M result index 확인
6. 저장 checkpoint 10개 존재 확인
7. 160M local config 존재 확인
8. C4-long 경로 확인
9. Dataset Decomposition clone 존재 확인

어떤 파일도 변경하기 전에 `codex_log/work_log.md`에 시작 상태를 기록한다.

---

# 5. Phase 1 — 160M model config 검증

GPU 작업보다 먼저 수행한다.

대상:

```text
configs/model/llama_160m_t5_4096.json
```

## 5.1 JSON 및 architecture validation

기존 `validate_model_config()`를 통과해야 한다.

추가로 다음을 검증한다.

- `architectures=["LlamaForCausalLM"]`
- `model_type="llama"`
- `vocab_size=32100`
- `max_position_embeddings>=4096`
- `use_cache=false`
- `hidden_act="silu"`
- `attention_dropout=0`
- `pad_token_id=0`
- `eos_token_id=1`
- `hidden_size % num_attention_heads == 0`
- Head dimension이 짝수
- `num_attention_heads % num_key_value_heads == 0`
- 모든 dimension이 positive integer
- `lm_head_chunk_size`가 positive
- Hugging Face와 reference backend 양쪽에서 model 생성 가능

## 5.2 Parameter count

Model을 CPU 또는 meta device에서 생성해 다음을 출력한다.

```text
total_parameter_count
trainable_parameter_count
embedding_parameter_count
attention_parameter_count
mlp_parameter_count
normalization_parameter_count
lm_head_parameter_count
```

허용 범위:

```text
150,000,000 <= trainable_parameter_count <= 170,000,000
```

범위를 벗어나면 자동 수정하지 않는다.

실험을 중단하고 config audit report만 생성한다.

## 5.3 출력

```text
results/config_audit_160m_<timestamp>/ANALYSIS_INDEX.md
results/config_audit_160m_<timestamp>/essential/model_config_audit.json
results/config_audit_160m_<timestamp>/essential/model_config_audit.md
```

이 gate가 통과하기 전에는 160M GPU 실행을 금지한다.

---

# 6. Phase 2 — Probability config schema v2

현재 user-facing tail probability \(Q\)를 actual maximum-context probability \(P\)로 교체한다.

## 6.1 새 canonical schema

```json
{
  "schema_version": 2,
  "context_levels": [512, 1024, 2048, 4096],
  "maximum_context_probabilities": [0.0, 0.25, 0.50, 0.25],
  "sampling": "iid_global_microbatch",
  "source": {
    "type": "example"
  }
}
```

지원 sampling:

```text
iid_global_microbatch
coordinated_global_update
```

## 6.2 내부 inclusion probability

\[
Q_k
=
P(N\ge k)
=
\sum_{j=k}^{K}P(N=j)
\]

예:

```text
maximum-context P:         [0.00, 0.25, 0.50, 0.25]
correction-inclusion Q:    [1.00, 1.00, 0.75, 0.25]
inverse correction weight: [1.00, 1.00, 1.333333, 4.00]
```

`P_512=0`이어도 512 base gradient는 모든 outcome에 들어간다.

## 6.3 Validation

- Probability 수와 context level 수가 같음
- 모든 probability가 finite
- 모든 probability가 0 이상
- 합이 tolerance 내 1
- Longest level probability > 0
- Derived inclusion probability가 모두 > 0
- Context level strictly increasing
- Model maximum position 이하
- v1/v2 field 동시 입력 금지

## 6.4 Legacy

Schema v1:

```text
tail_probabilities
sampling=shared_global_microbatch
```

는 기존 run 재현용 read-only 지원을 유지한다.

새 run writer는 schema v2만 출력한다.

기존 checkpoint와 result 파일은 수정하지 않는다.

## 6.5 Resolved config와 metrics

다음을 모두 기록한다.

```text
maximum_context_probabilities
correction_inclusion_probabilities
correction_inverse_weights
sampling_strategy
```

## 6.6 Calibration output

Calibration의 내부 후보 검색은 inclusion probability를 사용해도 된다.

그러나 최종 estimator config의 canonical field는 반드시 `maximum_context_probabilities`다.

Calibration report에는 \(P\), derived \(Q\), inverse weight를 모두 적는다.

---

# 7. Phase 3 — Generalized coordinated RR sampler

## 7.1 현재 문제

현재 code는 optimizer update에 포함된 각 microbatch의 maximum context를 독립적으로 sample한다.

이는 장기 marginal probability는 정확하지만 optimizer update마다 context composition이 크게 달라질 수 있다.

## 7.2 새 알고리즘

Arbitrary accumulation count를 지원하는 systematic coordinated block sampling을 구현한다.

Maximum-context probability를 \(p\), microbatch 수를 \(m\)이라고 한다.

권장 구현:

1. CDF 계산
2. \(u\sim U(0,1/m)\)
3. 다음 stratified point 생성

\[
z_i=u+\frac{i}{m},
\qquad
i=0,\ldots,m-1
\]

4. 각 \(z_i\)를 CDF에 매핑해 maximum context level 선택
5. 생성된 level list를 별도 RNG permutation으로 섞음
6. Rank 0이 block 생성
7. 모든 rank에 동일 block broadcast

이 방식은 다음을 만족해야 한다.

- 임의의 \(m\) 지원
- 각 microbatch position의 장기 marginal probability가 \(p\)
- Update별 count가 \(mp\)에 가깝게 유지
- Power-of-two 가정 없음
- Sequence order randomization
- Exact resume

## 7.3 API

```python
sample_block(microbatch_count: int) -> tuple[int, ...]
```

Production runner에서 update마다 한 번만 호출한다.

IID mode는 기존 방식으로 유지한다.

## 7.4 State

Checkpoint sampler state:

```text
schema_version
sampling_strategy
maximum_context_probabilities
generator_state
permutation_generator_state
block_count
sample_count
maximum_level_counts
```

## 7.5 Metrics

Update마다 다음을 기록한다.

```text
selected_maximum_contexts
selected_maximum_context_counts
cumulative_maximum_context_counts
empirical_maximum_context_probabilities
block_expected_counts
block_count_error_l1
block_count_error_l2
```

## 7.6 필수 테스트

Accumulation:

```text
1, 2, 3, 4, 5, 7, 8
```

Probability 사례:

```text
[0.0, 0.25, 0.50, 0.25]
[0.10, 0.20, 0.30, 0.40]
[0.0, 0.0, 0.5, 0.5]
```

검증:

- Count 합이 항상 \(m\)
- Invalid index 없음
- Empirical marginal probability 수렴
- IID보다 block count variance가 작거나 같음
- Expected correction coefficient가 모두 1
- Rank broadcast consistency
- Exact resume replay
- Zero-probability category 처리
- v1/v2 equivalent schedule
- Full estimator 결과 불변

---

# 8. Phase 4 — Muon config fail-fast 분리

이번 작업에서는 Muon 장기학습을 하지 않는다.

Config와 test만 수정한다.

## 8.1 Muon 전용 인자

```text
muon_learning_rate
muon_momentum
muon_weight_decay

aux_adamw_learning_rate
aux_adamw_beta1
aux_adamw_beta2
aux_adamw_epsilon
aux_adamw_weight_decay
```

## 8.2 `optimizer=muon`

- Hidden matrix에 Muon
- 나머지 parameter에 auxiliary AdamW
- 두 base learning rate를 분리
- Scheduler multiplier는 같아도 base LR 비율은 유지
- Resolved config에 두 optimizer를 명확히 분리

## 8.3 `optimizer!=muon`

Muon 전용 인자가 명시되면 즉시 오류를 낸다.

Silent ignore 금지.

## 8.4 Generic field ambiguity

`optimizer=muon`에서 generic `learning_rate`, `beta1`, `beta2`, `epsilon`, `weight_decay`, `momentum`을 Muon-specific/aux-specific 값과 동시에 모호하게 사용하지 않는다.

하나의 명확한 migration policy를 문서화한다.

권장:

- 새 Muon run에서는 전용 필드를 필수로 사용
- Generic optimizer field는 non-Muon optimizer용
- Legacy Muon checkpoint resume만 compatibility path 허용
- 새 Muon run에서 generic field와 전용 field가 충돌하면 오류

## 8.5 테스트

- AdamW + Muon arg 실패
- SGD + Muon arg 실패
- Muon 필수 arg 누락 실패
- Muon/aux AdamW LR 분리
- Parameter 분류 누락/중복 없음
- Scheduler multiplier 적용 후 LR 비율 유지
- Save/resume state 동일
- Resolved config 명확성

---

# 9. Phase 5 — 진단 코드 통합

기존 checkpoint를 여러 번 불필요하게 load하거나 동일 gradient를 중복 계산하지 않도록 unified diagnostic runner를 구현한다.

권장 entrypoint:

```text
audit_checkpoint_main.py
```

한 checkpoint session에서 가능한 한 다음을 공유한다.

- Model load
- Optimizer state load
- Validation model
- Calibration document-window cache
- Context-level forward/backward
- Layer gradient capture

Subcommand 또는 flag:

```text
--position_bins
--context_cosines
--weight_metrics
--adamw_virtual_update
--level_spectrum
--correction_spectrum
```

분석 모듈은 training runner와 분리한다.

실제 checkpoint parameter와 optimizer state를 변경하지 않는다.

---

# 10. Phase 6 — Existing 350M checkpoint 전수 audit

## 10.1 대상

권위 원본:

```text
/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32
```

Full native checkpoint:

```text
full_3000_no_ddp_split/checkpoint-00000030
full_3000_no_ddp_split/checkpoint-00000300
full_3000_no_ddp_split/checkpoint-00001500
full_3000_no_ddp_split/checkpoint-00002700
full_3000_no_ddp_split/checkpoint-00003000
```

RR native checkpoint:

```text
rr_3000/checkpoint-00000030
rr_3000/checkpoint-00000300
rr_3000/checkpoint-00001500
rr_3000/checkpoint-00002700
rr_3000/checkpoint-00003000
```

실제 디렉터리 이름은 manifest로 검증한다.

추측으로 경로를 고치지 않는다.

## 10.2 실행 순서

Checkpoint별 순차 처리:

1. Manifest 및 optimizer state 검증
2. Position-bin validation
3. Context-level gradient 계산
4. 모든 pair cosine 계산
5. Full gradient spectrum
6. Correction-gradient spectrum
7. Virtual AdamW audit
8. Streaming output flush
9. GPU memory 정리
10. 다음 checkpoint

Full/RR의 같은 step은 연속으로 처리해 비교 report를 즉시 생성한다.

권장 순서:

```text
Full 30 → RR 30
Full 300 → RR 300
Full 1500 → RR 1500
Full 2700 → RR 2700
Full 3000 → RR 3000
```

---

# 11. Position-bin validation

## 11.1 Bin

```text
1–512
513–1024
1025–2048
2049–4096
```

Causal target index의 off-by-one을 unit test로 고정한다.

## 11.2 지표

각 checkpoint/bin:

```text
nll
ppl
valid_target_count
document_window_count
batch_mean
batch_standard_deviation
standard_error
confidence_interval
```

Support가 부족하면 숨기지 않는다.

```text
status=insufficient_support
```

## 11.3 Validation data

기존 350M 비교와 동일한 고정 validation stream을 사용한다.

Full과 RR에서 동일 document window와 동일 target mask를 사용한다.

---

# 12. 모든 context-level pair cosine

## 12.1 대상

\[
G_{512},G_{1024},G_{2048},G_{4096}
\]

## 12.2 Pair

```text
512–1024
512–2048
512–4096
1024–2048
1024–4096
2048–4096
```

## 12.3 출력

각 pair:

```text
mean_gradient_cosine
per_batch_mean
per_batch_median
per_batch_standard_deviation
per_batch_p05
per_batch_p95
sample_count
```

산출물:

```text
context_level_pairwise_cosines.json
context_level_pairwise_cosines.csv
context_level_cosine_heatmap_<arm>_<step>.svg
context_level_pairwise_trajectory.svg
```

기존 minimum cosine은 compatibility summary로만 유지한다.

---

# 13. Full gradient 및 correction-gradient singular spectrum

## 13.1 Full context gradient

\[
G_{512},G_{1024},G_{2048},G_{4096}
\]

## 13.2 Correction gradient

\[
\Delta G_{1024}=G_{1024}-G_{512}
\]

\[
\Delta G_{2048}=G_{2048}-G_{1024}
\]

\[
\Delta G_{4096}=G_{4096}-G_{2048}
\]

둘 다 분석한다.

## 13.3 Layer

모든 Transformer block에서 다음 matrix를 분석한다.

```text
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
self_attn.o_proj.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

LM head와 embedding은 별도 appendix로 분석할 수 있다.

## 13.4 SVD

Matrix가 작으면 exact SVD를 사용한다.

비용이 크면 randomized SVD를 사용하되 다음을 기록한다.

```text
algorithm
target_rank
oversampling
power_iterations
seed
residual_error
```

## 13.5 지표

```text
spectral_norm
frobenius_norm
stable_rank
effective_rank
spectral_entropy
top_1_energy
top_4_energy
top_8_energy
top_16_energy
top_32_energy
top_64_energy
tail_energy_after_32
tail_energy_after_64
```

Context 간:

```text
top_r_principal_angles
top_r_subspace_overlap
```

Checkpoint 간:

```text
adjacent_checkpoint_subspace_overlap
```

## 13.6 Aggregation

- Layer별 raw 결과
- Depth group: early/middle/late
- Projection role별 평균
- Full gradient 대 correction gradient 비교
- Full arm 대 RR arm 비교
- Step trajectory

## 13.7 주요 질문

분석 report는 다음을 직접 답해야 한다.

1. Full gradient의 effective rank가 학습 중 증가하는가, 감소하는가
2. Context가 길수록 effective rank가 달라지는가
3. Correction gradient는 base/full gradient보다 high-rank인가
4. 후기 long-context correction의 tail energy가 증가하는가
5. Spectrum 변화가 context cosine 감소와 연관되는가
6. Spectrum 변화가 AdamW update distortion과 연관되는가

---

# 14. Weight comparison metric 수정

## 14.1 Legacy metric 보존

기존 cumulative displacement cosine:

\[
\cos(
\theta_F(t)-\theta_0,
\theta_R(t)-\theta_0
)
\]

은 정의상 \(\theta_0\)가 필요하다.

기존 result에서 삭제하지 않는다.

새 report에서는 다음 이름으로 appendix에 보존한다.

```text
legacy_cumulative_displacement_cosine_from_initial
```

## 14.2 새 primary metric: \(\theta_0\) 미사용

Same-step raw weight cosine:

\[
\cos(\theta_F(t),\theta_R(t))
\]

Relative direct distance:

\[
\frac{
\|\theta_F(t)-\theta_R(t)\|_2
}{
\|\theta_F(t)\|_2
}
\]

Symmetric relative distance:

\[
\frac{
2\|\theta_F(t)-\theta_R(t)\|_2
}{
\|\theta_F(t)\|_2+\|\theta_R(t)\|_2
}
\]

각 tensor와 전체 model에서 계산한다.

## 14.3 Adjacent-checkpoint interval metric

다음 구간을 사용한다.

```text
30→300
300→1500
1500→2700
2700→3000
```

Full interval update:

\[
d_F(t_1,t_2)
=
\theta_F(t_2)-\theta_F(t_1)
\]

RR interval update:

\[
d_R(t_1,t_2)
=
\theta_R(t_2)-\theta_R(t_1)
\]

지표:

```text
interval_update_cosine
interval_update_norm_full
interval_update_norm_rr
interval_update_norm_ratio_rr_over_full
interval_update_difference_l2
relative_interval_update_difference
```

이 계산에는 \(\theta_0\)가 들어가지 않는다.

## 14.4 Virtual optimizer-step metric

Virtual AdamW audit에서 다음도 primary로 출력한다.

```text
expected_rr_vs_full_optimizer_update_cosine
outcome_optimizer_update_cosines
```

---

# 15. Virtual AdamW audit

## 15.1 목적

Raw RR gradient가 unbiased해도 AdamW의 nonlinear first/second moment update는 Full과 다를 수 있다.

이를 실제 checkpoint optimizer state에서 측정한다.

## 15.2 대상

모든 10개 checkpoint.

## 15.3 Audit data

기존 calibration audit split의 동일 document-window cache를 사용한다.

Full/RR checkpoint에서 동일 batch grouping을 사용한다.

## 15.4 Optimizer update 단위

기존 350M run의 gradient accumulation은 4다.

AdamW step은 4 microbatch gradient가 누적된 뒤 한 번 발생한다.

따라서 single microbatch가 아니라 4-microbatch assignment 전체를 분석한다.

## 15.5 IID RR

Maximum-context category 수가 3이고 accumulation이 4이면 assignment 수는 다음이다.

\[
3^4=81
\]

가능한 81 assignment를 exact enumeration한다.

각 assignment에서 어떤 document microbatch에 어떤 context가 배정되는지가 다르므로 단순 count vector로 축약하지 않는다.

## 15.6 Coordinated RR

Systematic coordinated sampler가 만들 수 있는 block과 random permutation을 exact 또는 finite enumeration한다.

가능한 block 수가 작으면 exact enumeration한다.

너무 크면 common-random-number Monte Carlo를 사용한다.

Monte Carlo 사용 시:

```text
sample_count
seed
standard_error
confidence_interval
```

를 기록한다.

## 15.7 AdamW functional reference

PyTorch AdamW와 동일하게 처리한다.

- `exp_avg`
- `exp_avg_sq`
- Bias correction
- Parameter-group step
- Epsilon
- Learning rate
- Decoupled weight decay

Tiny tensor test에서 실제 `torch.optim.AdamW.step()`과 numerical agreement를 확인한다.

## 15.8 지표

Full optimizer update:

\[
u_F
\]

RR outcome update:

\[
u_r
\]

Expected RR update:

\[
\bar u_R
=
\mathbb E_r[u_r]
\]

Expected bias:

\[
B
=
\|\bar u_R-u_F\|_2
\]

Update variance:

\[
S
=
\mathbb E_r
\|u_r-\bar u_R\|_2^2
\]

Update MSE:

\[
M
=
\mathbb E_r
\|u_r-u_F\|_2^2
\]

검산:

\[
M\approx B^2+S
\]

출력:

```text
full_update_norm
expected_rr_update_norm
expected_rr_vs_full_optimizer_update_cosine
expected_update_bias_l2
relative_expected_update_bias
rr_update_variance
rr_update_mse_to_full
mse_bias_variance_residual
expected_exp_avg_sq_inflation
exp_avg_sq_inflation_p50
exp_avg_sq_inflation_p90
exp_avg_sq_inflation_p99
outcome_update_norms
outcome_update_cosines_to_full
layerwise_metrics
parameter_role_metrics
```

## 15.9 Streaming 구현

Checkpoint를 변경하지 않는다.

가능한 구현:

1. 한 global optimizer-update group의 4 logical microbatch에서 각 context-level gradient 계산
2. Gradient를 CPU FP32 tensor 또는 chunked temporary storage에 보존
3. Parameter tensor 단위로 IID/coordinated outcome aggregate gradient 구성
4. Virtual AdamW update 계산
5. Dot product와 norm을 FP64로 누적
6. Tensor 결과 폐기
7. 다음 parameter tensor 진행

Full 368M parameter outcome을 한꺼번에 GPU에 올리지 않는다.

## 15.10 Runtime gate

먼저 Full step 30 checkpoint 한 개에서 profile한다.

다음이 하나라도 발생하면 전체 10개 audit 전에 구현을 최적화한다.

- Peak GPU memory > 80%
- Host memory 비정상 증가
- Per-checkpoint 예상 시간 > 90분
- Temporary storage > 500GB
- Numerical mismatch
- AdamW reference test 실패

Scientific scope를 줄이지 말고 streaming/chunking을 개선한다.

---

# 16. Phase 6 결과 통합

새 experiment ID 예:

```text
checkpoint_mechanism_audit_350m_<timestamp>
```

구조:

```text
results/<experiment_id>/
  ANALYSIS_INDEX.md
  00_overview/
    essential/
  01_weight_metrics/
    essential/
    appendix/
  02_position_bins/
    essential/
    appendix/
  03_context_cosines/
    essential/
    appendix/
  04_spectrum/
    essential/
    appendix/
  05_adamw_virtual_audit/
    essential/
    appendix/
  90_logs/
```

Primary summary는 다음을 답한다.

- IID RR의 AdamW expected-update bias는 학습 중 커지는가
- Coordinated RR이 IID보다 update variance/MSE를 줄이는가
- Long-position PPL gap이 존재하는가
- Context gradient pair 중 어떤 pair가 가장 먼저 분리되는가
- Full/correction spectrum은 학습 중 어떻게 변하는가
- Weight trajectory 차이는 어느 interval에서 가장 크게 생기는가

---

# 17. Phase 7 — C4-long 최소 preflight

C4-long을 다시 생성하지 않는다.

학습 전에 다음만 확인한다.

- Train/validation path 존재
- Format parse 가능
- Tokenizer/vocab 일치
- 모든 document window에 4,096 active token
- 모든 window에 4,095 valid causal target
- Padding target 0
- Train/validation source-document ID overlap 0
- Duplicate record/window ID 없음
- 설정한 총 update에 필요한 window 수 공급 가능
- Exact resume용 iterator state 가능
- Window order 결정적

결과:

```text
results/c4_long_preflight_<timestamp>/essential/preflight.json
results/c4_long_preflight_<timestamp>/essential/preflight.md
```

광범위한 corpus 분석은 하지 않는다.

---

# 18. Phase 8 — 160M C4-long initial calibration

## 18.1 Model

검증을 통과한 다음 config 사용:

```text
configs/model/llama_160m_t5_4096.json
```

## 18.2 Context

```text
[512, 1024, 2048, 4096]
```

## 18.3 Calibration split

기존 과학적 계약과 동일한 수준을 우선 사용한다.

```text
measurement logical batches: 64
selection logical batches:   32
audit logical batches:       32
logical document windows per batch: 128
```

160M/C4-long에서 runtime이 충분히 낮다면 줄이지 않는다.

Train/selection/audit source document가 겹치지 않아야 한다.

## 18.4 Candidate search

후보를 내부 inclusion \(Q\) 또는 categorical \(P\)로 평가할 수 있다.

최종 output은 schema v2 `maximum_context_probabilities`다.

후보마다 기록:

```text
maximum_context_probabilities
correction_inclusion_probabilities
correction_inverse_weights
estimator_gradient_variance
expected_cuda_cost_ms
variance_cost_objective
independent_audit_confidence_interval
```

## 18.5 Selection

현재 phase에서는 기존 raw gradient variance × cost criterion을 사용한다.

Virtual AdamW audit은 진단 gate로만 사용한다.

AdamW audit 결과를 보고 사후적으로 다른 후보를 선택하지 않는다.

후보 선택 규칙은 calibration 전에 고정한다.

## 18.6 Same P

선택된 동일 \(P\)를 다음 두 RR arm에 사용한다.

```text
IID RR
Coordinated RR
```

Sampling strategy만 다르게 한다.

## 18.7 Initial virtual AdamW gate

Training 전 zero/initialized optimizer state에서 IID/coordinated virtual AdamW audit을 수행한다.

기록만 하되 다음 engineering failure는 중단한다.

- Non-finite
- Expected coefficient mismatch
- Functional AdamW mismatch
- Coordinated outcome distribution mismatch

---

# 19. Phase 9 — 160M Stage D: 300-update triage

## 19.1 Hardware

기존 350M 비교와 같은 두 H100 BF16 환경을 사용한다.

Full과 RR 모두 동일 launcher 및 DDP graph-splitting policy를 사용한다.

## 19.2 Batch

초기 기준:

```text
global total batch size = 512 document windows
```

Physical batch는 Full 4K VRAM probe로 먼저 결정한다.

선택한 physical batch와 accumulation을 모든 arm에서 동일하게 고정한다.

RR에만 더 큰 physical batch를 사용하지 않는다.

## 19.3 Scheduler

모든 run의 최종 horizon은 처음부터 다음으로 설정한다.

```text
num_training_steps = 3000
warmup_steps = 300
```

300-update triage를 위해 scheduler horizon을 300으로 줄이지 않는다.

필요하면 다음 기능을 추가한다.

```text
--stop_at_step
```

조건:

- `num_training_steps`와 scheduler horizon은 3000으로 유지
- `stop_at_step=300`에서 완전한 checkpoint 저장
- 정상 종료
- Resume 시 같은 scheduler/data/sampler state로 301부터 진행

이 기능은 scientific algorithm을 변경하지 않는 experiment-control 기능이다.

## 19.4 Arms

### D1 Full

```text
gradient_estimator=full
maximum context=4096
```

### D2 IID RR

```text
gradient_estimator=russian_roulette
sampling=iid_global_microbatch
P=<160M C4-long calibration result>
```

### D3 Coordinated RR

```text
gradient_estimator=russian_roulette
sampling=coordinated_global_update
P=<same calibration result>
```

## 19.5 동일 조건

- Initial FP32 weight export
- Data iterator order
- Tokenizer
- Global batch
- Physical batch
- Accumulation
- LR
- Warmup
- Scheduler
- AdamW betas
- Weight decay
- Precision
- Attention backend
- Compile/DDP policy
- Evaluation data

Sampling strategy와 estimator 이외의 차이를 report한다.

차이가 존재하면 실험을 시작하지 않는다.

## 19.6 Checkpoint와 evaluation

```text
save: 30, 300
eval: every 100
```

300에서 정상 종료한다.

## 19.7 300-update report

다음 결과를 하나의 표로 제공한다.

```text
PPL at 100/200/300
NLL at 100/200/300
cumulative core update time
cumulative total wall-clock
median steady update time
actual valid tokens
full-equivalent tokens
GPU-seconds
selected maximum-context frequency
block count error
gradient norm
non-finite count
peak VRAM
```

또한 D3 coordinated가 D2 IID 대비 다음을 보이는지 보고한다.

- Update time overhead
- PPL gap
- Virtual Adam update MSE
- Level-count variance
- Outcome frequency

300 이후 자동으로 계속하지 않는다.

보고서를 남기고 다음 milestone 진행 여부를 사용자가 판단할 수 있게 한다.

---

# 20. Phase 10 — Stage E: 3,000-update confirmation

사용자 승인 후 300 checkpoint에서 resume한다.

## 20.1 Milestone

```text
1500
2700
3000
```

## 20.2 Arms

원칙적으로 다음 세 arm을 유지한다.

```text
Full
IID RR
Coordinated RR
```

다만 300-update에서 한 RR 방식이 명확한 engineering failure를 보이면 해당 arm은 중단하고 실패 근거를 보존한다.

## 20.3 Primary metric

PPL을 primary로 사용한다.

NLL은 항상 병기한다.

Primary curves:

```text
PPL vs cumulative update time
PPL vs end-to-end wall-clock
PPL vs actual valid tokens
PPL vs GPU-seconds
```

Secondary:

```text
NLL vs same four axes
```

## 20.4 Saved diagnostics

각 arm:

```text
30, 300, 1500, 2700, 3000
```

에서 다음을 기록한다.

- Position-bin PPL
- Context pair cosine
- Selected-level frequencies
- Gradient norm
- Optimizer state summary
- Weight snapshot

전체 full-coordinate audit는 매 checkpoint training 중 inline으로 실행하지 않는다.

Training 완료 후 별도 diagnostic run으로 수행한다.

---

# 21. Phase 11 — Stage F: Batch reinvestment

Stage E가 완료되고 best RR sampling strategy가 결정된 뒤 수행한다.

## 21.1 새 variance/cost 측정

160M C4-long checkpoint에서 estimator variance와 actual cost를 재측정한다.

최소:

```text
step 300
step 1500
step 3000
```

초기 350M C4-all 수치를 재사용하지 않는다.

## 21.2 Variance-matched batch

\[
B_{\mathrm{RR}}
=
B_{\mathrm{Full}}
\frac{
V_{\mathrm{RR}}
}{
V_{\mathrm{Full}}
}
\]

여기서 \(V\)는 동일 batch definition의 centered estimator gradient variance다.

여러 checkpoint의 비율이 다르면 보수적인 held-out estimate를 사용한다.

계산 결과를 hardware-compatible divisor로 반올림한다.

반올림 규칙을 사전에 고정한다.

## 21.3 Wall-clock-matched batch

Measured update cost를 기반으로 RR global batch를 늘려 Full update wall-clock과 맞춘다.

OOM 없이 가능한 최대 batch를 임의로 고르지 않는다.

다음 objective를 따른다.

```text
RR expected update time ≈ Full expected update time
```

## 21.4 Arms

```text
Full baseline batch
Best RR same batch
Best RR variance-matched batch
Best RR wall-clock-matched batch
```

모든 arm에서 learning rate와 scheduler는 동일하다.

## 21.5 결과

- Same-update
- Same-actual-token
- Same-wall-clock
- Time-to-target PPL

네 축을 분리해 보고한다.

---

# 22. Phase 12 — Dataset Decomposition baseline

우선순위는 UMCG core experiment 이후다.

Local official repository:

```text
/home/ubuntu/keunyoung/workspace/ml-dataset-decomposition
```

공식 implementation은 OpenLM patch와 bucketization script를 사용한다.

## 22.1 즉시 수행하지 않을 것

- UMCG repository 내부 scratch reimplementation
- 공식 코드를 임의 변형해 먼저 돌리기
- DD+UMCG 즉시 통합
- Git 작업

## 22.2 향후 순서

1. Official repository 및 pinned OpenLM commit 검증
2. `umcg` Conda 환경에서 dependency compatibility 확인
3. 충돌 시 다른 환경을 만들지 않고 중단·보고
4. Official small reproduction
5. Official 160M baseline reproduction
6. UMCG 160M과 공정한 dataset/compute protocol 설계
7. 별도 승인 후 DD + UMCG prototype

Dataset Decomposition은 중요한 baseline이지만 Stage D–F보다 뒤에 둔다.

---

# 23. 명시적 보류 항목

이번 지시서에서 실행하지 않는다.

- Dynamic/piecewise \(P\) schedule
- Online calibration pause
- Correction predictor/control variate
- RR 전용 learning-rate tuning
- Correction coefficient shrinkage
- Multi-seed
- 350M 신규 학습
- 1B/8K 학습
- Muon 장기학습
- DD + UMCG 구현
- C4-long 재추출
- 전체 C4 광범위 length audit
- 기존 result 파일 수정
- Git/GitHub 작업

---

# 24. 자동 중단 조건

다음은 즉시 중단한다.

## 코드 및 config

- 160M parameter count가 150M–170M 밖
- Model config validation 실패
- Schema v1/v2 equivalence 실패
- Expected correction coefficient 불일치
- Coordinated sampler marginal probability 실패
- Resume replay 실패
- Muon config fail-fast test 실패

## Audit

- Checkpoint optimizer state 누락
- AdamW functional test 불일치
- Non-finite virtual update
- Spectrum residual error 초과
- Position-bin target indexing 실패
- Full/RR validation sample 불일치
- Existing checkpoint 변경 감지

## Training

- Non-finite loss/gradient
- Checkpoint 불완전
- Full/RR resolved config 불일치
- Data window에 padding 발견
- Train/validation document overlap
- Observed sampling distribution이 설정과 통계적으로 양립하지 않음
- OOM 또는 backend mismatch
- Resume가 다음 step을 정확히 재현하지 못함

Scientific 성능이 기대보다 낮다는 이유로 결과를 삭제하지 않는다.

---

# 25. 결과 보존 구조

모든 새 작업은 다음 규칙을 따른다.

```text
/home/ubuntu/checkpoint/keunyoung/umcg/<experiment_id>
```

권위 원본.

분석용 복사본:

```text
/home/ubuntu/keunyoung/workspace/umcg/results/<experiment_id>
```

구조:

```text
ANALYSIS_INDEX.md
00_overview/
  essential/
  appendix/
01_config/
02_code_validation/
03_full/
04_iid_rr/
05_coordinated_rr/
06_joint_analysis/
90_logs/
```

`ANALYSIS_INDEX.md`는 다음을 명시한다.

- 공식 비교 arm
- 읽기 순서
- Essential/appendix 구분
- 실패 run
- 보조 run
- Canonical config
- Primary metric
- 남은 한계

---

# 26. Codex 최종 보고 형식

각 phase 종료 시 다음을 보고한다.

## 완료 작업

```text
Phase
변경 파일
새 파일
삭제 파일
```

## 테스트

```text
명령
통과 수
실패 수
실행시간
```

## GPU 실행

```text
hardware
precision
backend
peak VRAM
wall-clock
output path
```

## 과학적 결과

```text
PPL
NLL
actual valid tokens
GPU-seconds
update time
sampling distribution
AdamW update bias/variance/MSE
spectrum summary
```

## 판단

```text
PASS
FAIL
BLOCKED
NEEDS_USER_REVIEW
```

## 다음 시작점

정확한 파일과 명령을 남긴다.

---

# 27. 최종 실행 순서 요약

1. `codex_log` 및 현재 상태 감사
2. 로컬 160M config 검증
3. Probability schema v2 구현
4. Coordinated sampler 구현
5. Muon config fail-fast 구현
6. Pairwise cosine/weight metric/position-bin/spectrum/AdamW audit 구현
7. CPU unit/integration test
8. Tiny/small GPU smoke
9. 기존 350M checkpoint 10개 전수 audit
10. Audit 종합 보고
11. C4-long 최소 preflight
12. 160M C4-long initial calibration
13. Full/IID/coordinated 300-update triage
14. 사용자 검토용 300-update 보고
15. 3,000-update confirmation
16. 160M batch reinvestment
17. Dataset Decomposition 공식 baseline
18. 향후 별도 승인 후 DD + UMCG

이 순서를 건너뛰지 않는다.
