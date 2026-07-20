# C4 단일-document 계약

## 경계

C4 English row 하나를 하나의 document 경계로 사용한다. Row는 정제된 웹페이지다. 완전한 기사 단위라는 뜻은 아니다.

필수 field는 `text`다. `url`과 `timestamp`는 metadata로 보존한다.

## 변환 순서

1. `text`를 special token 없이 tokenize한다.
2. 실제 document 끝에 EOS를 한 번 붙인다.
3. 가장 긴 context 길이로 겹치지 않게 자른다.
4. 중간 parent에는 EOS를 만들지 않는다.
5. 마지막 parent를 다른 document로 채우지 않는다.
6. 마지막 parent 오른쪽만 PAD로 채운다.
7. causal target이 없는 parent는 제외한다.

각 parent는 다음 값을 갖는다.

- `input_ids`: `[maximum_context]`
- `attention_mask`: 실제 token 위치의 bool mask
- `causal_target_mask`: `[maximum_context - 1]`의 bool mask
- `position_ids`: parent 안에서 `0`부터 시작
- `document_hash`: 원본 text의 SHA-256
- `chunk_index`: document 안의 parent 순서
- `token_start`, `token_end`: EOS를 붙이기 전 원문 token의 반열린 구간 좌표
- `url`, `timestamp`

Attention mask와 causal target mask는 서로 다른 tensor다. PAD 뒤의 target은 loss denominator에 들어가지 않는다.

## Prefix

모든 level prefix는 같은 parent의 시작점에서 자른다. 두 번째 parent가 원본 token 4096에서 시작하면 512, 1024, 2048, 4096 prefix도 모두 원본 token 4096에서 시작한다.

## Streaming

Train은 shard 순서만 seed로 섞는다. Row shuffle buffer는 쓰지 않는다. Validation은 섞지 않는다.

Checkpoint는 rank와 worker마다 다음 상태를 저장한다.

- Hugging Face IterableDataset state
- 처리한 row 수와 parent 수
- 현재 row에서 아직 내보내지 않은 parent
- 다음 logical worker
- worker와 rank topology

Resume 시 world size, worker 수, batch와 accumulation이 같아야 한다.

## Local 전처리

입력은 C4와 같은 JSONL이다.

```bash
python prepare_c4_main.py \
  --input_jsonl data/c4_rows.jsonl \
  --output_dir data/c4_parents_4096 \
  --tokenizer t5-base \
  --tokenizer_revision main \
  --estimator_config configs/estimator/russian_roulette_safe_4096.json \
  --parents_per_shard 10000
```

Manifest에는 tokenizer commit, vocabulary, EOS, PAD, 최대 context와 shard 목록이 들어간다. 실제 학습 tokenizer와 하나라도 다르면 local dataset을 열지 않는다.

Train에서 local shard 순서는 seed로 정한다. 그 순서도 checkpoint topology state에 포함된다.

참고: [C4 dataset card](https://huggingface.co/datasets/allenai/c4), [Hugging Face IterableDataset state](https://huggingface.co/docs/datasets/main/package_reference/main_classes)
