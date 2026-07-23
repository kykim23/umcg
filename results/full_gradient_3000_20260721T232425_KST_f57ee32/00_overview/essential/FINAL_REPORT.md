# UMCG Full-Gradient 검증 및 3,000-update Full/RR 비교 최종 보고서

- 실험 기준 커밋: `f57ee32c942f0c507becbd4f21955f9ac53b1c8a`
- 실험 source tree SHA-256: `952ae8bbcc29b075017527c68052eec748e228278edde7b0da666a3351805163`
- 자료: 로컬 C4 English `local_raw`
- 모델: Llama 계열 368,174,080 parameters, 관례상 350M 모델
- GPU: NVIDIA H100 80GB HBM3 두 장
- 정밀도: bfloat16(BF16), FP16 실험 없음
- 최종 갱신: 2026-07-23, Asia/Seoul

## 1. 결론부터

이번 계획에 포함된 구현, CPU 검증, 두 H100의 full-size calibration, Full 3,000 update,
Russian Roulette 3,000 update, 열 개 checkpoint의 full-gradient 진단을 모두 완료했다.

가장 중요한 결론은 세 가지다.

1. **256차원 CountSketch 없이도 calibration이 가능하다.**
   368,174,080개 전체 parameter 좌표에서 Gram matrix를 계산했다. 각 논리 batch의
   gradient만 잠시 보유하고 FP64로 내적을 누적한 뒤 즉시 버렸기 때문에 수백 GB의
   gradient 이력을 저장하지 않았다.
2. **고정 Russian Roulette 일정은 계산 효율 지표상 일관되게 유효했다.**
   초기 독립 감사와 10개 checkpoint 감사 모두 통과했다. Q=1 기준 대비
   `gradient variance × expected GPU cost` 비율은 0.6529–0.6825였고, 가장 불리한
   95% 신뢰구간 상한도 0.7358로 1보다 낮았다.
3. **하지만 같은 3,000 update에서 학습 품질이 Full과 같지는 않았다.**
   최종 validation loss는 Full 3.419851, Russian Roulette 3.472711로 Russian
   Roulette이 0.052860 높았다. 반면 안정 update 시간 중앙값은 5.940초로 Full의
   12.545초보다 52.65% 짧았다.

따라서 이번 결과는 다음을 지지한다.

> Russian Roulette(RR)은 unbiased한 full-objective gradient estimator이며,
> 이번 일정은 variance와 GPU 비용의 곱을 줄였다. 그러나 단일 seed의 동일-update
> 비교만으로 Full과 같은 최종 학습 품질을 달성했다고 말할 수는 없다.

동일 wall-clock 시간 비교와 다중 seed 반복이 다음 과학적 관문이다.

## 2. 용어와 비교 대상

- **Full**: 매 microbatch에서 항상 최대 context 4,096까지 계산하는 기준선이다.
- **Russian Roulette(RR)**: 최대 context를 확률적으로 고르고, 생략된 correction을
  inverse tail probability로 보정하는 unbiased estimator다.
- **Tail probability** `Q_k`: `P(N >= k)`, 즉 선택된 최대 level이 level `k` 이상일
  확률이다.
- **Parent**: 같은 원문에서 만든 최대 4,096-token 학습 표본이다. 하나의 parent에서
  512, 1,024, 2,048, 4,096 prefix gradient를 비교한다.
- **Gram matrix**: 여러 gradient의 모든 좌표 내적을 모은 작은 행렬이다. Norm뿐 아니라
  gradient 사이의 방향과 cross term도 보존한다.
- **95% CI**: 95% confidence interval, 즉 95% 신뢰구간이다.

고정 tail schedule은 다음과 같다.

```text
context levels:       [512, 1024, 2048, 4096]
tail probabilities Q: [1.0, 1.0, 0.75, 0.25]
maximum-level P(N=k): [0.0, 0.25, 0.50, 0.25]
```

따라서 512에서 바로 끝날 확률은 0%, 1,024에서 끝날 확률은 25%, 2,048에서 끝날
확률은 50%, 4,096까지 갈 확률은 25%다. 기대 최대 context는 2,304다.

## 3. 사용자가 확정한 실험 조건

| 항목 | 실제 적용값 |
|---|---:|
| Calibration 논리 parent batch size | 128 |
| GPU당 physical parent batch size | 64 |
| GPU 수 | 2 |
| Exact-path memory limit | 0.85 |
| Timing repeats | 1 |
| 측정/선택/감사 논리 batch | 64 / 32 / 32 |
| 측정/선택/감사 parent sample | 8,192 / 4,096 / 4,096 |
| Tail schedule 후보 | 단조감소 35개 |
| Training update | Full 3,000 + RR 3,000 |
| Training total batch size | 512 |
| Gradient accumulation | 4 |
| Warm-up | 300 update |
| Learning rate | peak `3e-4`, cosine decay |
| Validation | 100 update마다 full 4,096 context |
| Milestone checkpoint | 30, 300, 1,500, 2,700, 3,000 |
| Precision | BF16 only |
| Attention | PyTorch SDPA cuDNN |

Scaled Dot Product Attention(SDPA)은 attention 계산의 표준 kernel interface다. 현재
입력과 mask 조합에서 Flash 전용 probe는 `No available kernel`로 거부됐고 cuDNN
backend가 자동 선택됐다. 두 학습 arm과 모든 진단이 같은 backend를 사용했으므로
비교 조건은 일치한다.

## 4. Full-size gradient를 어떻게 측정했나

각 parent batch `b`에서 다음 네 gradient를 계산했다.

```text
G_b,512, G_b,1024, G_b,2048, G_b,4096
```

그리고 모든 368,174,080개 좌표의 내적을 FP64로 누적했다.

```text
H_b[i,j] = <G_b,C_i, G_b,C_j>
```

Correction은 다음과 같다.

```text
Delta G_b,1 = G_b,512
Delta G_b,k = G_b,C_k - G_b,C_(k-1)
```

Level Gram에서 선형변환한 correction Gram과 correction을 직접 빼서 계산한 Gram을
둘 다 구했다. 열 개 checkpoint 전체에서 두 경로의 최대 상대 잔차는 다음과 같다.

- Mean-gradient Gram: `2.15e-14`
- Per-batch Gram: `1.23e-15`

즉, CountSketch, SVD, random projection을 사용하지 않았다. 모델 forward/backward는
BF16 혼합 정밀도 조건이고, 전체 좌표 내적의 누적기는 FP64다.

`V_k`는 correction의 평균 제곱 L2 크기다.

```text
V_k = mean_b ||Delta G_b,k||_2^2
```

Parameter가 matrix이면 각 matrix의 Frobenius norm 제곱을 모두 더한 것과 같다. 그러나
일정 선택은 `V_k` 대각값만 보지 않는다. Full correction Gram의 off-diagonal cross term과
실측 GPU 비용도 함께 사용해 estimator variance를 계산한다. 따라서 correction 방향이
다를 수 있다는 문제를 계산에서 버리지 않았다.

## 5. 초기 full-coordinate calibration

초기 가중치에서 실행한 calibration은 1,678.63초, 약 27분 59초가 걸렸다.

- Model parameters: 368,174,080
- Projection: 없음
- Candidate schedules: 35개
- 최대 PyTorch memory fraction: 0.64078
- Parent cache SHA-256:
  `eb7544d32952f95fd9cb4b8344a13e38bf31f51c368e604814051af143466819`
- 측정/선택/감사 document hash 교집합: 모두 0
- 고유 document: 8,088 / 4,042 / 4,052

### 5.1 측정된 correction과 비용

| Context | `V_k` | `C_k`, ms | Incremental cost, ms |
|---:|---:|---:|---:|
| 512 | 9.536948 | 518.712 | 518.712 |
| 1,024 | 1.143079 | 1,048.112 | 529.400 |
| 2,048 | 0.976683 | 2,276.846 | 1,228.735 |
| 4,096 | 1.005489 | 5,451.430 | 3,174.584 |

### 5.2 선택 결과

35개 중 `[1, 1, 0.75, 0.25]`가 선택됐다.

| 감사 지표 | 결과 |
|---|---:|
| Expected correction coefficient | `[1, 1, 1, 1]` |
| Analytic relative L2 error | 0 |
| Analytic cosine similarity | 1 |
| Locked gradient variance | 9.966028 |
| Q=1 gradient variance | 6.956832 |
| Locked expected GPU cost | 2,763.309 ms |
| Q=1 expected GPU cost | 5,451.430 ms |
| Efficiency ratio vs Q=1 | **0.726155** |
| 95% CI | **[0.699319, 0.768031]** |

RR 자체의 추가 variance는 생기지만 예상 비용이 약 절반이어서 둘의 곱은 Q=1보다
낮았다. 독립 감사 신뢰구간 전체가 1 미만이므로 초기 gate를 통과했다.

## 6. 3,000-update 장기학습 결과

PyTorch 2.11의 compiled Distributed Data Parallel(DDP) graph splitting에서 RR의 여러
context shape가 AOT/Inductor 오류를 일으켰다. PyTorch traceback이 안내한
`torch._dynamo.config.optimize_ddp=False`로 DDP graph splitting만 껐다.
`torch.compile` 자체는 유지했다. 공정한 비교를 위해 Full도 동일 launcher로 3,000
update를 다시 실행했고, 이 matched pair만 primary 결과로 사용했다.

### 6.1 구조와 안정성

- 두 arm 모두 정확히 1–3,000의 연속 metrics 3,000행
- Validation 30회씩
- Non-finite train/validation metric 0건
- 다섯 milestone checkpoint 모두 `COMPLETE`
- 두 실행의 설정은 estimator 종류, 출력 이름/경로 외에 일치
- RR 관측 최대-level count: `[0, 3003, 6040, 2957]`, 총 12,000회
- RR 관측 tail: `[1.0, 1.0, 0.74975, 0.246417]`

관측 확률은 목표 `[0, 0.25, 0.50, 0.25]`와 잘 맞는다.

### 6.2 핵심 비교

| 지표 | Full | RR | 해석 |
|---|---:|---:|---|
| 최종 validation loss | **3.419851** | **3.472711** | RR +0.052860 |
| 최종 perplexity | **30.5649** | **32.2240** | RR 5.43% 높음 |
| 안정 update 중앙값 | 12.5449 s | 5.9397 s | RR 52.65% 짧음 |
| Update time 합계 | 37,677.7 s | 17,869.2 s | RR core update 약 2.11배 빠름 |
| 실제 valid tokens | 766,069,681 | 691,592,591 | RR은 Full의 90.28% |
| Full-equivalent tokens | 766,069,681 | 766,069,681 | 동일 |
| Valid tokens/s 중앙값 | 20,264 | 39,309 | RR 1.94배 |
| Rank별 peak allocation | 50.32 GiB | 48.95 GiB | 둘 다 H100 80GB 내 |

Update time 합계는 compile 이후의 각 학습 update 시간을 더한 값이며, 공통 validation과
checkpoint 저장 시간은 포함하지 않는다. Full은 약 10시간 28분, RR은 약 4시간 58분의
core update 시간이 들었다.

30번의 모든 validation에서 RR loss가 Full보다 높았다. 차이는 update 600에서
0.10405로 가장 컸고, 이후 줄어 최종 0.05286이 됐다.

![Full과 RR validation loss](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/plots/validation_loss.svg)

이 비교는 **동일 update 및 동일 full-equivalent parent budget 비교**다. RR이 실제로
읽고 미분한 target token은 더 적다. RR을 Full과 같은 wall-clock 시간만큼 더 학습하는
compute-matched 비교는 이번 실험에 포함되지 않았다.

## 7. 열 개 checkpoint의 exact-gradient drift 진단

두 arm의 update 30, 300, 1,500, 2,700, 3,000에서 같은 parent cache로 calibration을
다시 측정했다. 일정을 다시 선택해 학습에 적용하지 않고 초기 일정을 고정한 채
diagnostic-only로 감사했다.

- 총 보고서: 10개
- 총 진단 wall time: 20,514.31초, 약 5시간 42분
- 총 논리 parent batch: 1,280
- 총 distributed full-coordinate level-gradient 계산: 5,120회
- 모든 report의 source tree와 parent cache hash 일치
- 모든 independent audit 및 unbiasedness gate 통과
- 최대 memory fraction: 0.64078
- 자동 구조 검증 실패: 0건

### 7.1 고정 일정의 효율과 문맥 gradient 정렬

아래 cosine은 measurement 자료에서 구한 네 mean context-level gradient 사이의
off-diagonal 최소 cosine이다.

| Arm | Update | Efficiency ratio | 95% CI | 최소 level cosine |
|---|---:|---:|---:|---:|
| Full | 30 | 0.652890 | [0.618820, 0.700595] | 0.998941 |
| Full | 300 | 0.677057 | [0.644103, 0.730149] | 0.985569 |
| Full | 1,500 | 0.675563 | [0.651840, 0.712168] | 0.940898 |
| Full | 2,700 | 0.677668 | [0.654352, 0.714811] | 0.782000 |
| Full | 3,000 | 0.677847 | [0.654446, 0.715034] | 0.734683 |
| RR | 30 | 0.652899 | [0.618827, 0.700636] | 0.998907 |
| RR | 300 | 0.682475 | [0.650080, 0.735779] | 0.986938 |
| RR | 1,500 | 0.668755 | [0.646874, 0.703837] | 0.936764 |
| RR | 2,700 | 0.668532 | [0.646531, 0.703589] | 0.793021 |
| RR | 3,000 | 0.669346 | [0.647400, 0.704416] | 0.742705 |

![고정 일정의 효율비](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/plots/efficiency_ratio.svg)

![문맥별 평균 gradient 정렬](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/plots/level_alignment.svg)

두 arm 모두 학습 후기로 갈수록 context-level gradient 방향이 덜 정렬됐다. 이는
초기 한 시점의 gradient 구조가 장기학습 전체를 대표하지 않는다는 사용자의 우려를
실제로 확인한 결과다. 그럼에도 고정 일정의 efficiency CI는 모든 시점에서 1보다
충분히 낮아 계산 효율성은 유지됐다.

### 7.2 Correction second moment `V_k`

| Arm | Update | V512 | V1024 corr. | V2048 corr. | V4096 corr. |
|---|---:|---:|---:|---:|---:|
| Full | 30 | 3.332064 | 0.019032 | 0.013921 | 0.014012 |
| Full | 300 | 0.412814 | 0.038615 | 0.024347 | 0.021409 |
| Full | 1,500 | 0.251749 | 0.053439 | 0.027712 | 0.019997 |
| Full | 2,700 | 0.269192 | 0.063123 | 0.033060 | 0.020784 |
| Full | 3,000 | 0.270646 | 0.063847 | 0.033436 | 0.020956 |
| RR | 30 | 3.298969 | 0.019090 | 0.013953 | 0.014058 |
| RR | 300 | 0.470845 | 0.038642 | 0.024967 | 0.023009 |
| RR | 1,500 | 0.213774 | 0.045628 | 0.023333 | 0.015464 |
| RR | 2,700 | 0.240035 | 0.055771 | 0.027535 | 0.016855 |
| RR | 3,000 | 0.239862 | 0.056151 | 0.027796 | 0.016985 |

![Full trajectory V_k](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/plots/correction_second_moment_full.svg)

![RR trajectory V_k](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/plots/correction_second_moment_rr.svg)

### 7.3 Post-hoc schedule drift

초기 calibration은 `[1, 1, 0.75, 0.25]`를 선택했다. 그러나 update 30 이후 열 개
checkpoint의 selection split에서는 모두 `[1, 1, 0.5, 0.25]`가 35개 grid 중 최적이었다.

```text
post-hoc tail Q:       [1.0, 1.0, 0.5, 0.25]
maximum-level P(N=k): [0.0, 0.5, 0.25, 0.25]
expected max context: 2048
```

이 post-hoc 일정은 selection objective를 고정 일정 대비 약 2.19–2.68% 더 낮췄고,
각 checkpoint의 독립 감사도 통과했다. 하지만 이번 비교에서는 사전에 확정한 일정을
중간에 바꾸지 않는 것이 원칙이므로 실제 학습에 적용하지 않았다. 이는 다음 실험의
유력한 고정-arm 후보이지, 이번 결과를 소급해 바꾸는 근거가 아니다.

## 8. Parameter trajectory 보조 비교

각 checkpoint의 219개 tensor, 368,174,080개 전체 좌표에서 초기 weight 대비 parameter
update vector를 스트리밍 비교했다. 이것은 gradient 진단을 대체하지 않는 보조 지표다.

| Update | Full delta/init | RR delta/init | Full/RR delta cosine | Full-RR diff / Full delta |
|---:|---:|---:|---:|---:|
| 30 | 0.008551 | 0.008480 | 0.993850 | 0.110752 |
| 300 | 0.169106 | 0.168001 | 0.944935 | 0.330837 |
| 1,500 | 0.485595 | 0.485117 | 0.743393 | 0.716037 |
| 2,700 | 0.522421 | 0.521641 | 0.723555 | 0.743011 |
| 3,000 | 0.522613 | 0.521820 | 0.723504 | 0.743072 |

두 arm은 초기값에서 이동한 크기는 거의 같지만 방향은 점차 갈라졌다. RR estimator의
추가 sampling noise가 한 원인일 수 있으나, 단일 seed의 두 stochastic trajectory만으로
그 차이를 RR에만 인과 귀속할 수는 없다. Full 대 Full 반복 seed가 비교 기준으로 필요하다.

## 9. 구현 및 검증 결과

### 9.1 주요 구현

- 학습 CLI help를 9개 기능 그룹으로 정리하고 runner 시작부에 지원 인자 요약 추가
- Parser, `RuntimeConfig`, runner 지원 인자의 일치 단위 테스트 추가
- CountSketch 없는 `full_coordinate_streaming_gram` calibration 구현
- 35개 단조 tail schedule을 네 outcome의 해석적 열거로 평가
- Measurement/selection/audit document-hash 분리 및 overlap 검사
- 고정 schedule diagnostic-only 독립 감사 구현
- Update 30/300/1500/2700/3000 milestone checkpoint 저장 지원
- EOS `C-1`, `C`, `C+1`, `2C` 경계 테스트 보강
- PyTorch 2.11 compiled DDP evaluation과 RR multi-shape 경로의 안정화

### 9.2 자동 검사

| 검사 | 결과 |
|---|---:|
| 전체 CPU pytest suite | **93 passed**, 97.17초 |
| Ruff 정적 검사 | **0 findings** |
| 4,096 parameter-gradient Monte Carlo sample | 65,536 |
| Monte Carlo relative L2 error | 0.00346481 |
| Monte Carlo cosine similarity | 0.9999962823 |
| Monte Carlo L2 standard error | `2.53167e-5` |
| Error / standard error | 0.303752 |
| 열 개 checkpoint 자동 검증 | **passed, failure 0** |

## 10. 실패 관문과 처리

실패 시 CPU 처음부터 돌아가지 않고 해당 gate부터 재개했다.

1. **350M physical batch 64 OOM pilot**
   - 원인: activation checkpointing이 꺼져 있었다.
   - 처리: 장기학습과 동일하게 activation checkpointing을 켜고 같은 VRAM gate부터 재개.
   - 결과: 최대 memory fraction 0.64078로 0.85 제한 통과.
2. **PyTorch 2.11 RR compile 오류**
   - 원인: compiled DDP graph splitting과 여러 context shape의 AOT/Inductor 조합.
   - 처리: graph splitting만 끄고 compile 유지. Full도 같은 조건으로 다시 실행.
   - 결과: matched Full/RR 3,000 update 모두 완주.
3. **마지막 RR 3,000 diagnostic audit의 느린 tail**
   - 한 rank가 CPU full-coordinate 통계를 누적하는 동안 다른 rank가 분산 대기해 일부
     batch가 느려졌다. 두 process가 계속 전진했고 오류가 없어 재시작하지 않았다.
4. **`torchao` optional extension warning**
   - 설치 wheel과 맞지 않는 선택적 MXFP8/CUTLASS shared library warning이다.
   - 이번 BF16 경로는 해당 extension을 사용하지 않아 모든 gate와 결과에 영향이 없었다.

실제 GPU 실험 직전에만 `galore` Conda 환경의 instance-saver를 확인해 종료했다. 실험
종료 시 두 H100은 각각 1MiB, utilization 0%이며 saver를 임의로 다시 시작하지 않았다.

## 11. 과학적 한계와 다음 권장 실험

### 현재 말할 수 있는 것

- Estimator의 기대 gradient는 full gradient와 일치한다.
- `[1,1,0.75,0.25]`는 초기와 3,000 update까지 모든 측정 시점에서 Q=1보다 낮은
  variance-cost product를 보였다.
- RR은 동일 update에서 core update 시간을 약 절반으로 줄였다.
- Context gradient 구조는 학습 중 크게 변하므로 여러 checkpoint 재측정은 필요했다.

### 아직 말할 수 없는 것

- RR이 같은 wall-clock 시간에서 Full보다 낫거나 같은가
- 여러 seed에서 최종 validation 차이가 재현되는가
- `[1,1,0.5,0.25]`가 실제 장기학습 품질까지 더 좋은가
- 3,000 update 이후의 더 긴 학습에서도 같은 관계가 유지되는가

### 다음 순서

1. **Compute-matched 비교**
   - RR의 core update 속도는 약 2.11배였다. 같은 core-update 시간이라면 대략 6,300
     update 규모가 가능하지만, 현재 cosine scheduler가 3,000에서 거의 0으로 끝났으므로
     단순 resume하면 안 된다. 총 step과 learning-rate schedule을 먼저 확정해야 한다.
2. **다중 seed**
   - 최소 3 seed의 Full/고정 RR을 비교해 validation 차이와 parameter trajectory의
     자연 변동 범위를 추정한다.
3. **새 고정 일정 arm**
   - 열 개 checkpoint에서 일관되게 추천된 `[1,1,0.5,0.25]`를 사전 등록한 별도 arm으로
     시험한다. 기존 실행을 사후 변경하지 않는다.
4. **Attention 최적화는 별도 gate**
   - 현재 cuDNN SDPA는 안정적이고 충분히 빠르게 완주했다. Flash backend가 필요하면
     padded dense boolean mask/layout을 분리 probe하고 모델 입력 계약을 바꾸는 별도
     성능 실험으로 다룬다.

## 12. 결과 파일 이정표

### 사람이 먼저 읽을 파일

- 이 보고서: `FINAL_REPORT.md`
- [초기 estimator 설정](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/calibration_350m_exact_64_32_32.json)
- [초기 full-coordinate calibration 상세 보고서](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/calibration_350m_exact_64_32_32.json.report.json)
- [장기학습 비교 요약](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/long_run_comparison.json)
- [Checkpoint 진단 통합 요약](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/checkpoint_diagnostics_summary.json)
- [Checkpoint 진단 표](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/checkpoint_diagnostics_summary.csv)
- [Post-hoc schedule drift 요약](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/schedule_drift_summary.json)
- [Parameter trajectory 비교](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/weight_trajectory_comparison.json)

### 원본 자료

- [Matched Full metrics](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/full_3000_no_ddp_split/metrics.jsonl)
- [RR metrics](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/rr_3000/metrics.jsonl)
- [열 개 checkpoint 진단 디렉터리](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/diagnostics)
- [진단 원본 로그 디렉터리](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/logs/diagnostics)
- [Canonical FP32 checkpoint exports](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/exports)
- [CPU JUnit 결과](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/cpu_tests.xml)
- [Ruff 결과](/home/ubuntu/checkpoint/keunyoung/umcg/full_gradient_3000_20260721T232425_KST_f57ee32/ruff.json)

Primary Full은 `full_3000_no_ddp_split`이다. 먼저 실행한 `full_3000`은 DDP graph-split
조건이 RR과 달라 보조 결과로만 보존했다.
