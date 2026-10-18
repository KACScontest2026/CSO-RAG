# 근거 포함 RAG 기반 C/S/O 문단 분류

Qwen3-Embedding-0.6B, Chroma 벡터 검색(top-3), Qwen3-8B를 사용하여 대상 문단의 후보 등급과 근거를 생성하는 **로컬 연구용 구현**입니다. 검색 사례의 본문·문단 등급·근거 구절·적용 기준 ID·충족 사유 및 연결 문맥을 제공합니다. 추가 파인튜닝을 수행하지 않습니다.

**근거 포함 RAG만 실행합니다.** A/B 비교, 기존 실험 실행기, 실험 정답·예측·실제 데이터셋·벡터·모델 가중치는 포함하지 않았습니다. 기존 연구 파일과 결과를 수정하지 않은 별도 공개용 리팩터링입니다. 기존 최종 실험을 이 새 패키지로 다시 실행하여 성능을 재현한 것은 아닙니다. 이 수정 배포본은 기존 가중치를 재사용하여 가상 검색 사례 3개와 가상 문서 1개·2문단에서 실제 GPU 임베딩·검색·생성을 확인했습니다. 실제 연구 데이터의 재실험이나 모델 다운로드·기존 인덱스 변경은 수행하지 않았습니다.

## 처리 흐름

미리 구분된 문단 → 질의 임베딩 → 유사 사례 3개 → 공통 기준/선택 상세항목 + 원문/동일 문서 문맥 → Qwen JSON 생성 → 근거 원문 복원·검증 r1 → 문단 결과 및 문서 집계.

- 문서 전체를 자동 청킹하지 않습니다. 문단 경계와 Python Unicode 문자 기준, **0부터 시작·끝 위치 포함** 위치를 입력해야 합니다.
- 질의에는 대상 문단만, 생성모델에는 대상 문단과 같은 문서의 나머지 모든 문단을 제공합니다.
- 검색용 문단 원문만 벡터화합니다. 라벨·근거·기준·사유는 벡터화하지 않습니다.
- 검색 사례는 참고자료이며, 평가 원문 근거로 선택할 수 없습니다. 실제 제공한 대상/문맥 E ID만 선택합니다.
- 기준 상세항목은 대상·문맥의 고정 문자열 규칙으로 선택합니다. 정답·검색 사례·모델 예측은 선택에 사용하지 않습니다. 미선택은 법적 비해당 판정이 아닙니다.

## 고정 버전과 설정

| 항목 | 값 |
|---|---|
| 분류 기준 | `v3-confirmed-20261011-02` |
| 입력 프로토콜 | `v3.2-selective-criteria-20261012-03` |
| 자동 검증 | `v3.2-validation-r1-20261015-01` |
| 실행 설정 | `exec-qwen3-8b-ctx45056-yarn4-orig32768-20261015-01` |
| 임베딩 | `Qwen/Qwen3-Embedding-0.6B`, 리비전 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| 임베딩 처리 | 마지막 토큰 풀링, L2 정규화, 1,024차원, float32, left padding, SDPA, batch 1, TF32 비활성 |
| 검색 | Chroma, HNSW cosine, top-3, 별도 재정렬/키워드 융합 없음 |
| 생성 | Qwen3-8B Q4_K_M GGUF, 네이티브 템플릿, thinking 비활성 |
| 문맥 / 출력 / 안전 여유 | 45,056 / 4,096 / 128 토큰 |
| YaRN | rope_scaling_type=2, rope_freq_scale=0.25, yarn_orig_ctx=32768 |
| 기타 | temperature 0, seed 42, Flash Attention, batch/ubatch 128, threads 6 |
| 중단 | 300초, 기존 반복 감지; 환경 오류 시 저장 후 중단, 자동 설정 변경/선택적 재시도 없음 |

전체 생성 파라미터, 가중치·템플릿 SHA-256은 `config.example.json`에 있습니다. config의 YaRN orig_ctx 값은 학습 시 기본 문맥 길이를 입증하는 자료가 아닙니다. GPU 메모리 사용은 하드웨어·빌드에 따라 달라집니다. 연구 환경은 Windows 11 / RTX 3060 12 GB / RAM 32 GB / Python 3.12.14였으며, 새 환경에서의 GPU 적재 성공은 별도 확인해야 합니다. 시간 제한은 토큰 생성 콜백에서 확인하므로 네이티브 prefill 자체가 오래 걸리면 즉시 선점 중단하는 OS watchdog은 아닙니다.

## 설치

저장소 루트에서 새 가상환경을 사용하세요. 기존 실험 환경에 덮어 설치하지 마세요.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[retrieval,generation,test]"
Copy-Item config.example.json config.local.json
```

PyTorch는 연구 환경에서 `2.11.0+cu128`을 사용했습니다. 설치 환경에 맞는 CUDA 지원 PyTorch를 먼저 준비하세요. `llama-cpp-python` 역시 **CUDA 지원 빌드**가 필요하며, 위 pip 명령만으로 CUDA 빌드를 보장하지 않습니다. 필요하면 CUDA 도구 체인을 준비하고 `$env:CMAKE_ARGS='-DGGML_CUDA=on'`을 지정하여 `llama-cpp-python==0.3.36`을 별도로 빌드하세요. 드라이버·휠·빌드 조합이 다르면 설치 방법도 달라집니다. CPU 자동 대체는 하지 않습니다.

`config.local.json`의 `model_path`를 사용자가 준비한 GGUF 경로로 바꾸세요. 임베딩은 지정 리비전의 Hugging Face 캐시를 재사용하며 캐시가 없으면 해당 라이브러리가 다운로드할 수 있습니다. 오프라인 실행 시 미리 모델을 준비하고 `HF_HUB_OFFLINE=1`을 설정하세요. GGUF를 자동 다운로드하지 않습니다. 모델 원본·커뮤니티 변환본의 라이선스는 별도 적용됩니다.

Windows DLL 탐색이 필요한 환경에서는 `runtime_dll_dirs` 배열에 해당 환경의 llama_cpp/lib 및 CUDA DLL 디렉터리를 명시할 수 있습니다. 이 설정은 해당 프로세스에만 적용됩니다. 개인 PC의 절대경로를 공개 설정에 커밋하지 마세요.

## 입력 형식

### 평가/신규 입력: 정답 없는 문서 배열

`examples/documents.json` 형식을 사용합니다. 허용 필드는 문서의 `doc_id, group_id, text, paragraphs`, 문단의 `paragraph_id, text, start, end`뿐입니다. **정답 등급·근거·적용 기준·사유를 넣으면 검사에서 거부합니다.** ID에 등급을 인코딩하지 않는 것을 권장합니다. group_id는 번역/파생본까지 같은 원본 그룹으로 사전 정리해야 합니다.

### 검색 사례: 라벨·근거 메타데이터 포함

`examples/retrieval_cases.json`은 모두 직접 작성한 가상 예시이며 성능 평가 자료가 아닙니다. 사례 필드는 `case_id, paragraph_id, doc_id, group_id, text, label, evidence_spans, criterion_ids, reason, case_context`입니다. evidence_spans는 **출처 문단 내부** 위치이며, 신규 입력의 start/end는 **문서 전체 원문** 위치입니다. 연결 문맥은 `paragraph_id, text`를 보관합니다.

원래 기준 ID는 별도 변경 이력에 보존하고, 확인·승인된 별칭만 입력용 canonical ID로 제공하세요. 이 패키지는 `O-FOI-9-NONE` 같은 미확인 별칭을 이름만 보고 자동 매핑하지 않습니다. C/S 비해당 O와 정보 부족은 다르며, 기간 경과에 따른 공개 전환 경로는 연구 지침상 사용하지 않습니다.

검사는 제공된 그룹 ID와 정확한 문단 문자열 중복을 확인합니다. 번역·유사한 파생본·기관/인물명만 바꾼 원본 관계를 자동으로 모두 탐지하는 기능은 아닙니다. 원본 관계를 사전 검토해야 합니다. 실제 데이터의 한글 키 형식은 이 공개용 명시 스키마로 별도 변환해야 하며, 원본을 자동 수정하지 않습니다.

## 실행 순서

처음에는 가상 예시와 새 폴더를 사용하세요. 아래 임베딩/생성 명령은 사용자가 직접 실행할 때만 모델을 사용합니다.

```powershell
# 모델 없이 필수 필드·위치·분할 중복 검사
cso-rag --config config.local.json check-input --input examples/documents.json --cases examples/retrieval_cases.json

# 검색 사례만 임베딩하여 새로운 전용 인덱스 생성 (기존 경로면 거부)
cso-rag --config config.local.json build-index --cases examples/retrieval_cases.json --index indices/example

# 질의 임베딩·top-3 검색·기준 선택·E ID·스키마·문법 준비
cso-rag --config config.local.json prepare --input examples/documents.json --index indices/example --output runs/example_prepared

# 실제 네이티브 토크나이저로 전체 입력 수용량 검사 (답변 생성 없음)
cso-rag --config config.local.json capacity --prepared runs/example_prepared --report runs/example_capacity.json

# 별도 프로세스이므로 임베딩 모델이 해제된 후 생성 모델 적재
cso-rag --config config.local.json run --prepared runs/example_prepared --capacity runs/example_capacity.json --output runs/example_results 2> runs/example_native.log
```

`capacity` 성공은 토큰 수용량 확인이며 GPU 메모리/실제 생성 성공 확인이 아닙니다. run의 llama.cpp stderr 로그에서 실제 GPU 적재 레이어 수·Flash Attention을 확인하세요. 오류가 발생하면 완료 결과를 보존하고 중단합니다. GPU 여유 부족 시 문맥·사례 수·원문을 자동 축소하지 않습니다.

## 결과와 재개

- `items/*.json`: 실제 입력 구성, 기준 선택/제외 이유, 검색 사례 순위·거리·제공 수, E registry, 스키마와 문법.
- `paragraphs/*/input.json, schema.json, grammar.gbnf, chunks.jsonl, partial.txt, result.json`: 모델 입력과 부분 출력·완성 출력·검증·토큰·종료·시간.
- `documents.json`: 모든 문단 등급 완성 시 C>S>O 최고 등급, 일부 미분류 시 불완전 및 부분 제안.
- `runtime.json, gpu_samples.jsonl, completed.json, identity.json`: 실제 설정, GPU 메모리/활용도 표본, 완료 목록과 재개 동일성.
- 동일한 코드·설정·입력·기준으로 같은 run 명령을 실행하면 저장된 결과를 재생성하지 않습니다. 실패·미분류도 재시도하지 않습니다. 현재 시도가 강제 중단되어 partial만 남은 경우 자동 재시도는 막으며, 명시적인 새 실행 계획이 필요합니다.

결과는 JSON입니다. 별도 업로드 웹 앱이나 HTML 사용자 UI는 이 최소 패키지에 포함하지 않았습니다. 결과의 `evidence_spans`와 `criterion_groundings`에서 실제 구절·출처·위치를 확인할 수 있습니다.

## 자동 검증과 한계

원문 ID·출처·인용·위치, 기준별 대상 문단 근거 포함, 중복·직렬화 순서, 스키마 및 기준/등급 형식적 일관성을 검사합니다. r1은 `context_relation_IDs_not_explicit`를 진단으로만 남깁니다. 이 진단만으로 실패나 검토 필요가 되지 않습니다. 기준별 evidence_ids와 satisfaction_reason 및 문맥 관계 설명 요구는 유지됩니다. 생성 문법은 등급별 분기·허용 ID·순서·중복 등을 강제하지만 자연어 설명의 사실성·위해의 인과관계를 증명하지 않습니다.

검증 통과 ≠ 의미상 타당성 검증. 모델 자체 검토 요청과 자동 검증 실패는 별도입니다. 완성된 후보 등급은 검증 실패와 관계없이 보존하며, 미완성/반복/시간 제한/명시적 보류는 미분류입니다. 이 배포본은 정답 기반 성능 계산기를 포함하지 않으며, 원래 논문 지표를 새 실행 결과에 자동 부여하지 않습니다.

최종 연구의 검증 r1은 최종 테스트 A의 초기 완료 응답 10건을 확인한 뒤 개정되었고, 해당 응답은 생성 없이 재검증되었습니다. 이 이력을 숨기거나 원래 후보 선정의 엄격 검증 점수와 혼합하지 마세요. `provenance/`는 원본 파일 해시와 공개용 변경 범위를 기록합니다.

## 코드 안내

| 파일 | 주요 함수 / 역할 |
|---|---|
| `src/cso_rag/core.py` | documents/cases/check_separation, prepare_one, aggregate |
| `src/cso_rag/retrieval.py` | encoder/vectorize, build_index, prepare |
| `src/cso_rag/criteria.py` | criteria_engine: 기존 source-only 선택 규칙 추출 |
| `src/cso_rag/generation.py` | load_model, preflight, run: 독립 요청·생성·저장·재개 |
| `src/cso_rag/validation/validator.py` | schema_for, make_grammar, validate: 실험 코드 그대로 |
| `src/cso_rag/validation/validator_base.py` | segments/annotate, 원문 복원·r1 검증: 그대로 |
| `src/cso_rag/case_support.py` | deduplicate_payload: 동일 ID 본문/연결 문맥 중복 제거 |
| `assets/common_system_prompt.txt` | 실제 공통 생성 프롬프트 |
| `assets/criteria_v3_complete_reference.json` | 자동 검증용 canonical 기준 목록 |
| `assets/criteria_reference_v3/` | 네 영역의 전체 출처 확인용 참조본 |
| `assets/criteria_input/` | 항상 제공할 핵심 기준·고정 선택 규칙·출처표 |
| `tests/test_contract.py` | 모델 없는 회귀 검사; 실행 성능 평가 아님 |

## 공개 전 확인

코드 라이선스는 저작권자의 선택 없이 임의로 지정하지 않았습니다. 공개자는 프로젝트 코드의 배포 라이선스를 확정하여 LICENSE를 추가하세요. 기준 참조본에는 공식 원문 발췌와 프로젝트 해석이 함께 출처별로 구분되어 있습니다. 법령·기관 자료·N2SF 발췌의 재배포 조건을 확인해야 하며 코드 라이선스로 이들의 권리를 일괄 부여하지 않습니다. 모델 라이선스도 별도입니다. 실행 로그/입력/인덱스에는 원문이 저장되므로 Git 추적 대상에서 제외했습니다. 이 패키지는 GitHub에 업로드하거나 원격 저장소를 생성하지 않았습니다.

## 자체 검사

```powershell
python -m unittest discover -s tests -v
```

`provenance/verification.json`에는 이번 준비에서 실제 수행한 모델 없는 검사와 미수행 사항을 기록합니다.


## 공개용 GPU 점검 (수정 배포본 02)

직접 작성한 가상 예시에서 GPU 벡터 생성(1,024차원, lasttoken, L2 정규화), 신규 Chroma top-3 검색, 입력 수용량, Qwen GPU 37/37 레이어 적재, JSON 생성과 검증 r1 통과를 확인했습니다. 두 응답의 상세 기록은 `provenance/gpu_smoke_summary.json`을 참조하세요. ST 5.7 pooling_mode 속성 검사와 Windows CUDA DLL의 프로세스 한정 탐색 경로를 보완했으며, 모델·기준·프롬프트·생성 설정·검증은 바꾸지 않았습니다. 기존 실험 코드·결과는 보존했습니다. 새로운 환경에서의 설치 검증이나 의미상 타당성·성능 평가로 해석하지 마세요. GPU 메모리 수치는 1초 간격 표본입니다.
