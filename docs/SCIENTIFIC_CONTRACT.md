# UMCG scientific contract

## Estimator

Context level loss를 `L_1, ..., L_K`라고 한다.

```text
L_K = L_1 + sum_{k=2..K} (L_k - L_{k-1})
```

Russian Roulette estimator는 다음과 같다.

```text
L_hat = L_1 + sum_{k=2..M} (L_k - L_{k-1}) / Q_k
Q_k = P(M >= k)
```

최대 level 자체의 확률은 `k < K`에서 `P(M=k)=Q_k-Q_(k+1)`, 마지막 level에서 `P(M=K)=Q_K`다. 예를 들어 `[1, 0.5, 0.25, 0.125]`는 `[0.5, 0.25, 0.125, 0.125]`의 최대-level 확률을 만든다.

2048이 선택되면 gradient는 다음과 같다.

```text
G_512 + 2 * (G_1024 - G_512) + 4 * (G_2048 - G_1024)
```

4096이 선택되면 여기에 `8 * (G_4096 - G_2048)`을 더한다. 선택된 최대 context에서 token loss를 한 번 계산하고 level별 normalized mask 차이를 signed coefficient로 합친다. 별도의 512·1024·2048 forward를 다시 실행하지 않는다.

`gradient_estimator=full`은 estimator JSON의 `Q_k`를 무시하고 항상 `L_K`를 계산한다. 같은 calibration JSON을 full과 RR 비교에 사용해도 full gradient는 바뀌지 않는다.

Train scalar는 음수가 될 수 있다. 따라서 train perplexity는 기록하지 않는다. Perplexity는 최대 context의 full validation loss에서만 계산한다.

## Global token 평균

한 optimizer update에서 다음 순서를 지킨다.

1. 필요한 parent microbatch를 CPU에 모두 준비한다.
2. Rank 0이 microbatch별 최대 level을 표본 추출한다.
3. 선택 level을 모든 rank에 broadcast한다.
4. 모든 rank가 같은 sequence shape를 실행한다.
5. 모든 microbatch와 level의 valid target 수를 센다.
6. Count vector를 모든 rank에서 합친다.
7. Global denominator로 local token loss를 정규화한다.
8. 마지막 microbatch에서 gradient 통신을 끝낸다.
9. Optimizer update를 한 번 수행한다.

DDP와 FSDP2의 gradient 평균을 보정하기 위해 local objective에 world size를 곱한다. DeepSpeed가 accumulation loss를 나누는 양도 되돌린다. 최종 gradient는 backend와 무관하게 global valid-token 평균이다.

Local valid target이 0인 rank는 정상이다. 모든 rank를 합친 target 수가 0일 때만 함께 종료한다.

## 고정된 자동 시험

- `Q=1` RR gradient와 full gradient
- Padding이 다른 microbatch의 global 평균
- Split accumulation과 한 번에 처리한 batch
- Local zero와 global positive target
- Global zero target 종료
- RR Monte Carlo unbiasedness
- 65,536표본의 4096 parameter-gradient Monte Carlo 수렴
- Hugging Face와 reference logits, token loss, gradient
- Causal prefix invariance

## Calibration

Calibration은 optimizer update를 하지 않는다. Document hash로 측정 64, 선택 32, 독립 감사 32 **논리 parent batch**를 분리한다. 논리 batch 하나는 두 rank를 합쳐 parent 128개이므로 세 split은 각각 8,192개, 4,096개, 4,096개 parent를 포함한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 \
torchrun --standalone --nproc_per_node 2 calibrate_main.py \
  --model_backend huggingface \
  --model_config configs/model/llama_350m_t5_4096.json \
  --tokenizer t5-base \
  --precision bfloat16 \
  --attention_backend automatic \
  --context_preset 4096 \
  --c4_source local_raw \
  --c4_local_path /home/ubuntu/data/c4_en/en \
  --c4_revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2 \
  --measurement_parent_batches 64 \
  --selection_parent_batches 32 \
  --audit_parent_batches 32 \
  --logical_parent_batch_size 128 \
  --max_parent_batch_size_per_gpu 64 \
  --memory_limit_fraction 0.85 \
  --activation_checkpointing \
  --timing_parent_batches 8 \
  --timing_repeats 1 \
  --output /home/ubuntu/checkpoint/keunyoung/umcg/calibration/estimator_4096.json
```

각 논리 batch에서 full parameter gradient 네 개를 잠시 보유한다. 원본 좌표의 level Gram과 직접 차분 correction Gram을 FP64로 누적한 뒤 그 batch gradient는 폐기한다. 따라서 전체 gradient 이력을 저장하지 않으면서도 CountSketch나 다른 압축 없이 correction의 평균 제곱 크기 `V_k`, level·correction cosine, 최대 level별 평균 GPU 시간 `C_k`와 추가 비용을 구한다.

`Q_512=1`과 `{1,.75,.5,.25,.125}` 격자에서 tail probability 단조감소를 만족하는 35개 일정만 평가한다. 각 일정의 네 최대-level 결과를 해석적으로 전부 합산해 `gradient variance × expected distributed CUDA cost`를 계산하므로 일정 선택에는 Monte Carlo 오차가 없다.

일정을 고정한 뒤 감사 자료에서 다음을 확인한다.

- 해석적 기대 gradient와 full gradient의 상대 L2 오차와 cosine similarity
- Q=1 대비 효율 목적함수 비율과 95% 신뢰구간
- 측정·선택·감사 document hash의 무교집합
- Level Gram에서 변환한 correction Gram과 직접 차분해 계산한 Gram의 상대 잔차

Unbiasedness 또는 효율성 감사가 실패하면 진단 report만 남기고 estimator JSON을 만들지 않는다. 감사 결과를 보고 차선 후보를 고르지 않는다.

생성된 estimator JSON은 학습 중 읽기 전용이다.

중간 checkpoint 진단은 같은 parent cache와 고정 estimator를 사용한다. 그 시점에서 더 좋아 보이는 일정이 나타나더라도 3,000-update 비교 중에는 일정을 교체하지 않는다.
