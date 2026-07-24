# UMCG 연구 대화 이력 및 결정 로그

최종 갱신: 2026-07-24 (Asia/Seoul)

## 문서 목적

이 문서는 UMCG(Unbiased Multi-Resolution Context Gradients) 연구에 관한 사용자와 ChatGPT의 누적 대화 이력, 실험 해석, 합의된 결정, 남은 질문, 다음 실험 순서를 기록한다.

앞으로 UMCG 관련 대화가 이어질 때 이 파일을 최신 상태로 갱신한다.

## 용어

- **원문 document**: C4 row처럼 데이터셋에 존재하는 원래 문서.
- **document window**: 원문 document에서 잘라 학습에 투입하는 최대 문맥 길이 이하의 연속 token 구간.
- 과거 대화에서 사용한 `parent`는 앞으로 원칙적으로 `document window`로 부른다.
- 기술적으로 원문 document와 document window가 같지 않을 수 있으므로, 둘을 구분한다.

---

# 1. 연구 주제 형성

## 1.1 핵심 아이디어

최대 문맥 gradient를 다음 telescoping sum으로 표현한다.

\[
G_{L_K}
=
G_{L_0}
+
\sum_{k=1}^{K}
\left(
G_{L_k}-G_{L_{k-1}}
\right)
\]

짧은 문맥의 base gradient는 자주 계산하고, 비싼 긴 문맥 correction은 드물게 계산한다.

Russian Roulette estimator에서는 correction \(k\)가 포함될 tail probability를 \(Q_k\)로 두고 다음처럼 계산한다.

\[
\widehat G
=
\Delta_0+
\sum_k
\frac{I_k}{Q_k}\Delta_k,
\qquad
I_k=\mathbf 1[N\ge k]
\]

조건이 정확하면 기대 gradient는 최대 문맥의 full gradient와 같다.

## 1.2 설계 A와 설계 B

- **설계 A**: 문맥 길이가 길어지면서 target token 집합도 늘어난다. 한 번의 긴 forward에서 signed token-loss coefficient로 correction을 계산할 수 있다.
- **설계 B**: 동일한 target token을 고정하고 볼 수 있는 과거 context만 바꾼다. 장거리 context의 순수한 효과를 더 깨끗하게 측정하지만 별도 forward가 필요하다.
- 주 학습 알고리즘은 설계 A, 해석용 진단은 설계 B로 두는 방향을 채택했다.

---

# 2. 주요 선행연구 검토

다음 연구들을 검토했다.

- Randomized Telescoping, ICML 2019
- ARTBP
- MLMC gradient estimators
- CNN Multiscale Gradient Estimation, TMLR 2026
- SpaCO, Findings of ACL 2025
- SUS Backprop
- Sequence Length Warmup, NeurIPS 2022
- Dataset Decomposition, NeurIPS 2024
- GaLore, ICML 2024
- AdaRankGrad, ICLR 2025
- SubTrack++, NeurIPS 2025
- Gradient subspace 및 spectral dynamics 관련 후속연구

현재 방어 가능한 연구 공백은 다음과 같이 정리했다.

> 표준 dense decoder-only causal LM의 from-scratch pretraining에서 context length를 직접 multilevel fidelity로 사용하고, full long-context objective의 gradient를 기대값에서 보존하면서 실제 time-to-quality를 줄이는 연구.

Randomized telescoping과 MLMC 원리는 기존 연구이므로, UMCG는 새로운 estimator 원리 자체보다 다음을 기여해야 한다.

- Causal LM context hierarchy에서 correction이 저분산이 되는 조건
- document-aware data construction
- one-pass signed correction
- nonlinear optimizer와의 상호작용
- 실제 wall-clock time-to-quality 개선

---

# 3. 구현 및 실험 지시서

Codex용 지시서를 두 문서로 분리했다.

- 코드 구현 지시서
- 중단 실험 지시서

주요 구현 결정은 다음과 같다.

- 기존 GaLore/DoRA 코드에 직접 덧붙이지 않고 독립 `umcg_pretraining` 구조 사용
- Hugging Face 호환 JSON model config 사용
- `conda umcg` 환경만 수정
- hash 및 보안 인프라는 가설검증 단계에서 제거
- 구현 진행상황을 별도 Markdown에 기록
- 구형 TITAN RTX 서버에서는 60M smoke test만 수행
- 실제 중단 실험은 별도 H100 서버에서 수행
- GitHub branch, commit, push, PR 권한은 Codex에 부여하지 않음

---

# 4. UMCG 저장소 구현 결과

저장소 `kykim23/umcg`에 다음이 구현됐다.

- Full gradient와 Russian Roulette gradient estimator
- Tail probability에서 maximum-level categorical probability 변환
- 선택된 최대 level까지 correction 누적
- 선택된 최대 context 한 번의 forward와 signed token coefficient
- Global valid-token normalization
- DDP/FSDP/ZeRO 경로
- AdamW, SGD, SGDM, Muon 등 optimizer factory
- C4 single-document chunking
- exact full-coordinate calibration
- checkpoint 및 학습 지표 기록
- 65,536회 Monte Carlo gradient 수렴 테스트
- 350M full-coordinate gradient Gram 진단

---

# 5. 350M, 3,000-update 실험

## 5.1 설정

- 모델: Llama 계열 약 368M parameter
- 데이터: C4 English
- 최대 문맥: 4,096
- context level: `[512, 1024, 2048, 4096]`
- RR tail probability: `[1, 1, 0.75, 0.25]`
- 최대 level 확률: `[0, 0.25, 0.50, 0.25]`
- optimizer: AdamW
- Full/RR 모두 3,000 optimizer update
- 동일 초기 weight, 동일 data stream
- 두 H100 BF16

## 5.2 주요 결과

- Full validation loss: 약 3.419851
- RR validation loss: 약 3.472711
- Full PPL: 약 30.5649
- RR PPL: 약 32.2240
- RR PPL은 약 5.43% 높음
- RR core update 시간은 Full보다 약 52.65% 짧음
- RR은 약 2.11배 빠른 update
- Full actual valid targets: 766,069,681
- RR actual valid targets: 691,592,591
- RR은 Full의 약 90.28% actual valid target을 사용

## 5.3 데이터 occupancy 해석

Full은 3,000 update × global batch 512 = 1,536,000개의 document window를 처리했다.

Document window당 평균 valid target은 약 498.7개다.

최대 4,095 target 중 약 12.2%만 valid했다.

따라서 현재 Full 기준선의 4K dense shape에는 padding이 매우 많았고, RR 속도 이득의 일부는 이 padding compute를 피한 효과일 가능성이 높다.

---

# 6. 현재 사용자 메시지의 21개 논점과 답변 결론

## 6.1 PPL과 loss

- 논문 headline과 주요 표에는 PPL을 사용하는 것이 적절하다.
- 다만 PPL은 loss의 지수이므로 분석에는 validation NLL/loss도 함께 제시한다.
- Primary plot은 time-to-target PPL과 time-to-target loss를 함께 제공한다.

## 6.2 `parent` 용어

- 일상 대화에서는 `document`보다 `document window`가 더 정확하다.
- 원문 document 하나가 여러 4K window를 만들 수 있으므로 둘을 구분한다.

## 6.3 padding 없는 baseline

다음 세 기준선을 분리한다.

1. C4-long의 완전한 4K window만 사용하는 Full
2. 전체 C4에서 실제 길이에 맞춘 exact length-bucketed Full
3. Dataset Decomposition

C4-long Full은 long-context mechanism을 검증한다.

Length-bucketed Full은 padding 제거만으로 얻을 수 있는 속도를 측정한다.

Dataset Decomposition은 강한 variable-length curriculum baseline이다.

## 6.4 coefficient

Strict unbiasedness를 유지하려면 correction coefficient는 반드시 \(1/Q_k\)다.

임의 조절은 biased estimator가 되므로 현재 주 방법에서는 사용하지 않는다.

## 6.5 gradient variance와 variance × cost

Calibration은 각 논리 batch에서 네 context gradient를 전체 parameter 좌표로 계산하고 correction Gram을 만든다.

후보 schedule의 모든 maximum-level outcome을 해석적으로 열거해 다음을 계산한다.

\[
V(Q)=
\mathbb E\|\widehat G\|^2
-
\|\mathbb E\widehat G\|^2
\]

Q=1은 data mini-batch variance만 포함한다.

RR은 data variance와 level-sampling variance를 함께 포함한다.

고정 wall-clock \(T\)에서 sample 수가 \(T/C\)이므로 평균 gradient variance는 대략 \(VC/T\)다.

그래서 \(V\times C\)가 작을수록 같은 시간에 더 정확한 평균 gradient를 얻을 수 있다.

다만 이 기준은 SGD식 평균에는 자연스럽지만 AdamW의 nonlinear moment update를 완전히 설명하지 못한다.

## 6.6 variance-matched batch

평균 gradient variance가 \(V/B\)에 비례한다고 근사하면 다음이 된다.

\[
B_{\mathrm{RR}}
=
B_{\mathrm{Full}}
\frac{V_{\mathrm{RR}}}{V_{\mathrm{Full}}}
\]

현재 비율을 그대로 사용하면 약 734이고, 실험 후보는 768이다.

하지만 이 계산은 근사이므로 실제 batch별 재측정이 필요하다.

C4-long 및 강한 Full baseline 확인 뒤 수행한다.

## 6.7 balanced sampling

Microbatch별 독립 sampling은 optimizer update마다 context 구성의 변동을 크게 만든다.

모든 microbatch에 하나의 동일 level을 공유하는 방법도 update 간 변동이 커질 수 있다.

권장 방식은 optimizer update 전체에 대한 joint sampling이다.

Accumulation 수가 임의의 정수여도 다음 방식으로 처리할 수 있다.

1. 원하는 기대 count \(mp_j\) 계산
2. floor count 배정
3. 남은 count를 fractional part에 따라 dependent rounding
4. microbatch 순서를 무작위 permutation

Power-of-two accumulation이 필요하지 않다.

## 6.8 learning rate

Primary comparison에서는 Full과 RR의 learning rate와 scheduler를 동일하게 유지한다.

RR gap을 learning-rate tuning으로 고치는 제안은 기각한다.

향후 appendix의 robustness 분석에서만 동일한 tuning budget으로 각 방법을 따로 조정할 수 있다.

## 6.9 control variate

Correction \(\Delta_k\)의 값싼 predictor \(h_k\)가 있을 때 다음 estimator를 쓸 수 있다.

\[
\widehat{\Delta}_k
=
h_k+
\frac{I_k}{Q_k}(\Delta_k-h_k)
\]

기대값은 정확히 \(\Delta_k\)다.

큰 correction 전체가 아니라 예측하지 못한 residual만 확률적으로 증폭하므로 variance를 줄일 수 있다.

가능한 predictor는 과거 correction EMA, lower-level gradient 기반 linear predictor, domain/length별 correction mean이다.

## 6.10 공정한 비교

다음 축을 모두 분리한다.

- 같은 update
- 같은 actual valid token
- 같은 wall-clock
- 같은 GPU-seconds/FLOPs
- time-to-target PPL

강한 baseline 정리 뒤 진행한다.

## 6.11 singular spectrum 가설

Checkpoint spectrum 분석은 유의미하다.

하지만 일반적인 “초기 dominant mode, 후기 flatter spectrum” 자체는 이미 인접 연구가 존재한다.

특히 다음 결과들이 관련된다.

- GaLore: gradient low-rank structure
- AdaRankGrad: training이 진행될수록 estimated gradient rank가 감소한다는 주장
- Randomized Gradient Subspaces: core subspace 영향이 시간에 따라 줄고 residual bulk 중요성이 커진다는 관측
- SubTrack++: evolving gradient subspace와 optimizer-state transport
- Q-GaLore: layer별 gradient subspace convergence 차이

따라서 단순 확인만으로는 탑티어 contribution이 약하다.

다음처럼 context-conditioned spectral law와 알고리즘으로 연결해야 강하다.

- Base와 각 context correction의 spectrum 변화
- Context 길이별 effective rank
- Spectrum이 RR quality gap 또는 Adam distortion을 예측
- Spectrum 기반 dynamic schedule 또는 control variate

## 6.12 두 cosine의 의미

### Full/RR cumulative parameter-update cosine

\[
\Delta\theta_F(t)=\theta_F(t)-\theta_0
\]

\[
\Delta\theta_R(t)=\theta_R(t)-\theta_0
\]

두 벡터의 cosine이다.

두 모델이 초기 weight에서 어느 방향으로 이동했는지 비교한다.

### Context-level gradient minimum cosine

한 checkpoint에서 동일한 calibration data로 계산한

\[
G_{512},G_{1024},G_{2048},G_{4096}
\]

사이의 여섯 pairwise cosine 중 최솟값이다.

Full/RR 모델끼리의 비교가 아니라, 한 모델 내부에서 context 길이에 따라 gradient 방향이 얼마나 달라지는지를 측정한다.

## 6.13 trajectory divergence

Weight cosine 하락은 실제로 다른 trajectory를 배웠다는 증거다.

하지만 원인이 objective bias라는 뜻은 아니다.

RR level sampling과 AdamW nonlinearity가 다른 realized update를 만들 수 있다.

비용이 큰 Full-vs-Full 반복보다 기존 checkpoint의 local optimizer audit를 먼저 수행한다.

## 6.14 AdamW 문제

Unbiased raw gradient라도 AdamW update는 unbiased하지 않을 수 있다.

AdamW는 squared gradient의 EMA를 사용한다.

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

노이즈가 있으면

\[
\mathbb E[\widehat G^2]
=
G^2+\operatorname{Var}(\widehat G)
\]

가 된다.

따라서 RR noise가 second moment를 키우고 특정 방향의 effective learning rate를 낮출 수 있다.

기존 checkpoint의 model 및 Adam state를 고정하고, 가능한 RR outcome을 모두 열거해 virtual Adam update를 계산한다.

측정값은 다음이다.

- Expected Adam update bias
- Adam update variance
- Full update와의 cosine
- Layer별 second-moment inflation

## 6.15 Muon config

Muon 전용 config를 분리한다.

- `muon_learning_rate`
- `aux_adamw_learning_rate`
- `muon_momentum`
- 각 optimizer의 weight decay 및 beta

`optimizer=muon`일 때만 활성화한다.

다른 optimizer에서 Muon 전용 인자를 명시하면 silent ignore보다 즉시 오류를 내는 정책을 권장한다.

## 6.16 Muon 위험

Muon이 RR에 불리할 가능성은 실험을 중단할 정도의 위험은 아니다.

다만 Muon은 singular values를 flatten하므로 작은 singular direction이 signal이면 유리하고 noise이면 불리할 수 있다.

Raw gradient, momentum, Muon update의 spectrum을 비교해 판정한다.

## 6.17 데이터셋 순서

C4-long 310만 document를 첫 positive-control로 사용하는 데 동의한다.

조건은 다음과 같다.

- exact tokenizer 기준 4K 이상
- source-document 단위 train/validation 분리
- random offset 또는 non-overlapping full window
- partial padded window 제외
- global document-window shuffle
- 중복 및 URL domain 통계

그 다음은 다음 순서가 적절하다.

1. DCLM-Baseline 또는 FineWeb의 long subset
2. peS2o V2 `s2orc`
3. 필요 시 FinePDFs

SlimPajama는 널리 사용됐지만 오래된 multi-source corpus다.

RedPajama는 더 오래됐고 일부 book source의 저작권 문제가 있어 primary dataset 우선순위가 낮다.

FineWeb은 현대적인 general-web replication에 유리하다.

## 6.18 dynamic tail probability

Step 또는 block \(t\)마다 \(Q_{t,k}\)를 바꿀 수 있다.

그 step에서 coefficient를 \(1/Q_{t,k}\)로 사용하면 raw gradient unbiasedness가 유지된다.

단, \(Q_{t,k}\)는 현재 correction을 계산하기 전에 과거 정보와 held-out calibration으로 결정해야 한다.

권장 방식은 piecewise constant schedule이다.

예:

- update 0–300
- 300–1,500
- 1,500–3,000

각 boundary에서 optimizer-aware held-out audit를 수행하고, 충분한 개선과 신뢰구간이 있을 때만 다음 block의 schedule을 변경한다.

## 6.19 일곱 baseline

다음 baseline을 차례로 실험한다.

1. Padded Full
2. Exact length-bucketed Full
3. Document-packed Full
4. Dataset Decomposition
5. Sequence Length Warmup
6. UMCG
7. Dataset Decomposition + UMCG

## 6.20 Length-Stratified UMCG

Dataset Decomposition의 bucket 안에서만 UMCG를 적용한다.

예:

- 512 bucket: Full
- 1K bucket: 512→1K UMCG
- 2K bucket: 512→1K→2K UMCG
- 4K bucket: 512→1K→2K→4K UMCG

길이 stratum \(s\)의 target mixture weight를 \(w_s\), sampling probability를 \(p_s\)라 하면 다음 형태가 가능하다.

\[
\widehat G
=
\frac{w_s}{p_s}
\widehat G_s^{\mathrm{UMCG}}
\]

Expectation은 전체 target mixture gradient와 같다.

공개 문헌 검색에서는 Dataset Decomposition과 unbiased context-level multilevel estimator를 이 형태로 결합한 연구를 확인하지 못했다.

성공하면 강한 contribution 후보다.

## 6.21 현재 실행 순서

### 단계 A: 새 장기학습 전

CPU 중심:

1. C4 전체와 C4-long length audit
2. current metrics에서 cumulative time-to-target PPL 계산
3. exact length-bucketed Full 및 Dataset Decomposition 구현
4. generalized balanced RR sampler 구현 및 unit test
5. Muon conditional config validation 구현

GPU 진단이 필요한 작업:

6. 기존 checkpoint position-bin validation NLL
7. exact Adam update audit
8. layer/context correction singular-spectrum audit

### 단계 B: 첫 새 학습

C4-long, 60M 또는 130M에서 다음을 같은 learning rate로 비교한다.

1. Full 4K
2. 기존 i.i.d. RR
3. generalized balanced RR

첫 단일 seed로 경향을 본 뒤 유력 arm만 3 seed로 반복한다.

### 단계 C

C4 전체에서 다음을 비교한다.

1. Exact length-bucketed Full
2. Dataset Decomposition
3. Length-Stratified UMCG

### 단계 D

유력한 방법만 350M, 수십억 token 규모로 확장한다.

### 단계 E

Adam optimizer distortion이 확인된 뒤 Muon을 별도 optimizer study로 추가한다.

---

# 7. 현재 핵심 결정

- PPL을 논문의 직관적 주요 지표로 사용하되 loss도 병기한다.
- `parent` 대신 `document window`를 사용한다.
- coefficient 임의 조절은 하지 않는다.
- learning-rate tuning으로 RR gap을 메우는 방안은 primary method에서 기각한다.
- C4-long을 다음 mechanism dataset으로 사용한다.
- padding-free Full과 Dataset Decomposition을 먼저 구축한다.
- microbatch i.i.d. sampling을 generalized balanced joint sampling으로 개선한다.
- dynamic Q는 optimizer-aware, held-out, piecewise-constant 방식으로 검토한다.
- 기존 checkpoint의 Adam update 및 spectrum 분석을 새 장기학습 전에 수행한다.
- Dataset Decomposition + UMCG 통합을 장기적인 주요 contribution 후보로 둔다.

---

# 8. 2026-07-24 후속 논의: 표기, 확률 설정, 진단 및 130M 실험

## 8.1 Correction gradient와 통계량 표기

향후 다음을 분명히 구분한다.

\[
\Delta G_{b,k}=G_{b,L_k}-G_{b,L_{k-1}}
\]

\(\Delta G_{b,k}\)는 두 gradient의 차이지만 다음과 같이 signed correction objective의 gradient다.

\[
\Delta G_{b,k}
=
\nabla_\theta(L_{b,L_k}-L_{b,L_{k-1}})
\]

기존 결과의 `V_k`는 correction 자체가 아니라 다음 uncentered second moment였다.

\[
M_{2,k}=\mathbb E_b\|\Delta G_{b,k}\|_2^2
\]

Centered variance trace는 다음이다.

\[
\operatorname{VarTrace}_k
=
\mathbb E_b
\|\Delta G_{b,k}-\mathbb E_b\Delta G_{b,k}\|_2^2
=
M_{2,k}-\|\mathbb E_b\Delta G_{b,k}\|_2^2
\]

새 코드와 결과에서는 generic `V_k`를 피하고 다음 이름을 사용한다.

- `level_gradient`
- `correction_gradient`
- `correction_gradient_second_moment_m2`
- `correction_gradient_variance_trace`
- `estimator_gradient_variance`

## 8.2 Maximum-context probability를 canonical config로 사용

사용자-facing config에는 tail probability \(Q\) 대신 실제 maximum context 선택 확률 \(P\)를 쓴다.

```json
{
  "context_levels": [512, 1024, 2048, 4096],
  "maximum_context_probabilities": [0.0, 0.25, 0.50, 0.25]
}
```

내부에서 다음을 유도한다.

\[
Q_k=P(N\ge k)=\sum_{j\ge k}P(N=j)
\]

```text
P = [0.00, 0.25, 0.50, 0.25]
Q = [1.00, 1.00, 0.75, 0.25]
```

Correction coefficient는 여전히 \(1/Q_k\)다.

`P_512=0`은 512 base contribution이 사라진다는 뜻이 아니다. 최대 context가 512에서 끝나는 update가 없다는 뜻이다.

## 8.3 Generalized coordinated sampling

Microbatch별 independent sampling을 대신할 일반화된 block sampler를 승인했다.

Optimizer update에 \(m\)개 microbatch가 있고 maximum-context probability가 \(p_j\)라면:

1. \(mp_j\) 계산
2. floor count 배정
3. 남은 slot을 fractional residual에 따라 unbiased rounding
4. 최종 level list를 무작위 permutation
5. rank 0이 생성하고 모든 rank에 broadcast

임의의 accumulation 수를 지원하며 power-of-two 가정을 두지 않는다.

## 8.4 Spectrum 분석 범위

사용자의 학습-dynamics 가설을 검증하기 위한 primary 대상은 다음 full context gradient다.

\[
G_{512},G_{1024},G_{2048},G_{4096}
\]

ChatGPT는 correction도 수학적으로 gradient이며 UMCG가 실제 sampling하는 대상이므로 correction spectrum도 중요하다고 판단한다.

다만 이 부분에는 이견이 남아 있으므로 다음 correction-spectrum 실행은 사용자 추가 승인 전 보류한다.

\[
G_{1024}-G_{512},\;
G_{2048}-G_{1024},\;
G_{4096}-G_{2048}
\]

## 8.5 Parameter cosine

기존 cumulative displacement cosine은 다음을 비교한다.

\[
\Delta\theta_F(t)=\theta_F(t)-\theta_0
\]

\[
\Delta\theta_R(t)=\theta_R(t)-\theta_0
\]

공유 random initialization을 제거해 실제 학습 이동 방향을 보는 지표다.

사용자는 raw weight를 직접 비교해야 한다고 제안했다.

ChatGPT는 raw weight cosine이 공유 initialization에 지배될 수 있으므로 기존 지표를 유지해야 한다고 판단한다.

대신 다음 보조 지표를 추가할 수 있다.

- raw same-time weight cosine
- normalized same-time weight distance
- checkpoint interval update cosine
- virtual optimizer-step update cosine

기존 지표 삭제 여부는 보류한다.

## 8.6 Context-level cosine

Essential summary에서 최소 cosine 하나만 쓰지 않는다.

다음 여섯 조합을 모두 기록한다.

- 512–1024
- 512–2048
- 512–4096
- 1024–2048
- 1024–4096
- 2048–4096

Mean-gradient cosine과 per-batch 분포를 모두 제공한다.

## 8.7 Virtual AdamW audit

기존 checkpoint의 model parameter와 AdamW state를 고정하고 실제 weight를 변경하지 않은 채 다음을 계산한다.

- Full optimizer-update gradient
- IID RR optimizer-update gradient distribution
- Coordinated RR optimizer-update gradient distribution
- Expected AdamW update bias
- Update variance
- Update MSE
- Layer별 second-moment inflation
- Full update와의 cosine

Gradient accumulation 전체를 하나의 optimizer update 단위로 분석한다.

## 8.8 Muon config

Muon 전용 learning rate, momentum, weight decay와 auxiliary AdamW 설정을 분리한다.

`optimizer=muon`일 때만 Muon 전용 인자를 허용한다.

다른 optimizer에서 해당 인자를 명시하면 silent ignore하지 않고 오류로 종료한다.

## 8.9 Dynamic probability schedule

현재 exact full-coordinate checkpoint audit는 350M에서 한 번에 약 30–40분이 소요됐다.

따라서 block 사이마다 exact audit을 삽입하는 dynamic schedule은 즉시 구현하지 않는다.

향후 후보는 다음이다.

- 별도 pilot에서 piecewise schedule 사전 결정
- 값싼 proxy를 exact audit과 먼저 검증
- adaptation overhead를 최종 wall-clock에 포함

## 8.10 SkyLadder 정정

SkyLadder는 NeurIPS 2025 Main Conference Track 논문이다.

이전 답변에서 preprint로만 설명한 것은 오류였다.

## 8.11 Dataset Decomposition 공식 코드

Apple의 공식 `ml-dataset-decomposition` repository가 존재한다.

OpenLM 특정 commit에 patch를 적용하고, tokenization/bucketization 및 variable-length training script를 제공한다.

즉시 UMCG 안에 scratch implementation을 만들지 않는다.

사용자가 local checkout 경로를 제공한 뒤 별도 baseline workstream에서 다룬다.

## 8.12 최신 실행 결정

- C4-long train/validation은 이미 준비됐으므로 광범위한 재추출과 A1–A3 audit은 수행하지 않는다.
- Exact length-bucketed Full도 이번 즉시 작업에서는 보류한다.
- Dataset Decomposition은 공식 코드 기반 별도 workstream이다.
- B1 generalized coordinated sampler 승인
- B2 Muon conditional config 승인
- C1 position-bin NLL 승인
- C2 virtual AdamW audit 승인
- C3 full-gradient spectrum 승인
- 신규 학습 D/E/F/G는 모두 130M
- Multi-seed는 단일 seed 결과가 유망하고 방법론이 확정된 뒤로 연기
- Primary Full/RR 비교에서는 learning rate와 scheduler를 동일하게 유지
- Coefficient 임의 조절 금지
- Control variate와 dynamic probability는 후순위

---

# 9. 2026-07-24 최종 승인사항 및 실행 순서 확정

## 9.1 승인된 항목

사용자는 다음을 승인했다.

1. Correction gradient 및 second-moment/variance 표기 정정
2. 사용자-facing estimator config를 maximum-context categorical probability \(P\) 중심으로 변경
3. `variance × cost`와 variance-matched batch의 수학적 설명
4. Generalized coordinated RR sampling
5. Correction control variate는 후순위로 보류
6. Full context gradient spectrum과 correction-gradient spectrum을 모두 측정
7. 기존 cumulative displacement cosine을 삭제하지 않되, 새 primary weight comparison에서는 \(\theta_0\)를 사용하지 않는 직접 지표들을 추가
8. 모든 context-level gradient pair cosine 공개
9. 저장된 모든 Full/RR checkpoint에서 virtual AdamW audit 수행
10. Muon 전용 config의 조건부 활성화와 fail-fast 검증
11. Online/piecewise dynamic probability schedule은 현재 단계에서 보류
12. SkyLadder가 NeurIPS 2025 main-track임을 확인
13. Generalized coordinated sampler와 Muon config 수정
14. Position-bin validation, virtual AdamW audit, spectrum audit
15. 신규 학습은 60M/130M이 아니라 로컬 160M config 사용
16. Multi-seed는 방법론 확정 및 단일-seed 성공 이후로 연기

## 9.2 Weight 비교 지표의 최종 해석

`cumulative displacement cosine`은 정의상 다음과 같이 \(\theta_0\)가 필요하다.

\[
\cos(
\theta_F(t)-\theta_0,
\theta_R(t)-\theta_0
)
\]

따라서 \(\theta_0\) 없이 같은 이름의 지표를 계산할 수는 없다.

최종 정책은 다음이다.

- 기존 cumulative displacement cosine은 과거 결과와 호환되는 **legacy/appendix 지표**로 보존
- 새 primary weight comparison은 \(\theta_0\)를 사용하지 않음
- 추가할 primary 지표:
  - same-step raw weight cosine
  - relative same-step weight distance
  - symmetric relative weight distance
  - adjacent-checkpoint interval update cosine
  - adjacent-checkpoint interval update norm ratio
  - virtual AdamW update cosine

인접 checkpoint interval은 다음이다.

```text
30→300
300→1500
1500→2700
2700→3000
```

이 지표들은 \(\theta_0\)를 사용하지 않는다.

## 9.3 기존 checkpoint 전체 audit

다음 10개 checkpoint를 모두 대상으로 한다.

```text
Full: 30, 300, 1500, 2700, 3000
RR:   30, 300, 1500, 2700, 3000
```

각 checkpoint에서 다음을 수행한다.

- position-bin NLL/PPL
- 모든 context-level gradient pair cosine
- full context gradient singular spectrum
- correction-gradient singular spectrum
- virtual AdamW update audit
- same-step Full/RR weight comparison
- adjacent-checkpoint interval update comparison

## 9.4 160M model config

사용자는 다음 로컬 config를 준비했다.

```text
/home/ubuntu/keunyoung/workspace/umcg/configs/model/llama_160m_t5_4096.json
```

이 파일은 현재 GitHub default branch에서는 확인되지 않았다.

따라서 Codex는 GPU 실행 전에 반드시 로컬 파일을 직접 읽고 다음을 검증한다.

- canonical JSON schema
- exact trainable parameter count
- 150M–170M 범위
- 4,096 position 지원
- tokenizer vocab 32,100과 일치
- attention head divisibility
- even head dimension
- KV head divisibility
- `use_cache=false`
- token ID 계약
- LLaMA 계열 architecture 생성 성공

자동으로 config를 수정하지 않는다. 실패하면 중단하고 보고한다.

## 9.5 Dataset Decomposition

공식 repository는 다음 위치에 clone돼 있다.

```text
/home/ubuntu/keunyoung/workspace/ml-dataset-decomposition
```

이는 중요한 baseline이지만 현재 최우선 실험은 아니다.

우선순위는 다음이다.

1. UMCG 코드와 진단 수정
2. 기존 350M checkpoint 전체 audit
3. 160M C4-long Full/IID RR/coordinated RR
4. Batch reinvestment
5. Dataset Decomposition 공식 구현 재현
6. DD + UMCG 통합

Dataset Decomposition은 UMCG repository 내부에 즉시 재구현하지 않는다.

## 9.6 신규 160M 실험의 순서

### Stage D0: 코드 및 데이터 preflight

- 160M config 검증
- C4-long train/validation 최소 계약 검증
- schema v2 및 coordinated sampler smoke
- 2×H100 BF16 1–2 update smoke

### Stage D1: 초기 160M C4-long calibration

- Full-coordinate level/correction statistics
- 측정/선택/감사 split
- 사용자-facing output은 maximum-context probability \(P\)
- 선택된 동일 \(P\)를 IID RR과 coordinated RR에 사용
- initial-state virtual AdamW diagnostic

### Stage D2: 300-update triage

동일 조건으로 다음 세 arm 실행.

```text
Full 4096
IID RR
Coordinated RR
```

총 scheduler horizon은 처음부터 3,000 update로 고정한다.

각 arm은 300에서 정상 checkpoint를 남기고 일시 정지한다.

### Stage E: 3,000-update confirmation

300-update 보고를 검토한 뒤 생존 arm을 1,500, 2,700, 3,000까지 이어서 실행한다.

최종적으로 논문 ablation에 필요한 경우 Full/IID/coordinated 세 arm을 모두 3,000까지 완료한다.

### Stage F: Batch reinvestment

C4-long 160M에서 새로 측정한 variance와 실제 cost를 이용한다.

- same-batch coordinated RR
- variance-matched coordinated RR
- wall-clock-matched coordinated RR

모든 arm은 동일 learning rate를 사용한다.

### Stage G: Dataset Decomposition

UMCG core mechanism이 C4-long에서 성공한 뒤에만 공식 코드 재현을 진행한다.


