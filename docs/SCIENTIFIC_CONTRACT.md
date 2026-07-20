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
- Hugging Face와 reference logits, token loss, gradient
- Causal prefix invariance

## Calibration

Calibration은 optimizer update를 하지 않는다. 기본값은 64 parent batch다.

```bash
CUDA_VISIBLE_DEVICES=0 python calibrate_main.py \
  --model_backend huggingface \
  --model_config configs/model/llama_350m_t5_4096.json \
  --tokenizer t5-base \
  --precision bfloat16 \
  --attention_backend automatic \
  --context_preset 4096 \
  --c4_source streaming \
  --c4_repo allenai/c4 \
  --parent_batches 64 \
  --batch_size 1 \
  --output configs/estimator/russian_roulette_calibrated_4096.json
```

Level별 correction gradient CountSketch, CUDA 시간, peak memory를 모은다. 후보마다 RR gradient variance와 expected CUDA cost를 계산한다. 추천 기준은 두 값의 곱이다.

생성된 estimator JSON은 학습 중 읽기 전용이다.
