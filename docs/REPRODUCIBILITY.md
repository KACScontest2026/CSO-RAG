# CSO-RAG 재현 안내

이 문서는 README의 논문 보충 내용을 유지하면서, 공개 실행 코드의 설치 조건과 입력 계약 및 실행 순서를 별도로 설명한다. 대상은 **근거 포함 RAG 한 경로**다. A/B/C 비교 실행기나 논문의 정답 기반 평가기는 포함하지 않는다.

공개 관리 보완의 기준 커밋은 [`c33c8f920ee28e0bc0f6bb8fcce197bab067dccb`](https://github.com/KACScontest2026/CSO-RAG/tree/c33c8f920ee28e0bc0f6bb8fcce197bab067dccb)다. 이 보완본에서는 실행 코드와 기준, 생성 프롬프트 및 모델 설정을 변경하지 않았다. 아래 모델 다운로드와 실행 명령은 독자가 선택하여 실행하는 안내이며, **이번 보완 작업에서는 모델 다운로드·추론·전체 실험을 수행하지 않았다.**

## 1. 재현 범위와 확인 상태

- CSO-RAG 판정 기준의 내부 버전 ID: `v3-confirmed-20261011-02`
- RAG 기반 문단별 등급 판정의 내부 버전 ID: `v3.2-selective-criteria-20261012-03`
- 자동 검증: `v3.2-validation-r1-20261015-01`
- 실행 설정: `exec-qwen3-8b-ctx45056-yarn4-orig32768-20261015-01`
- 모델: Qwen3-Embedding-0.6B + Qwen3-8B Q4_K_M
- 검색: Chroma HNSW cosine top-3. 키워드 융합·재정렬 없음.

표시 명칭은 논문에 맞추었다. 기존 문서에서 통합 분류기준 v3 또는 프로토콜 v3.2라고 부른 대상의 이름을 정리한 것이며, 실제 판정 조건·프롬프트·입력 구성·검증·생성 설정을 바꾸지 않았다. 버전 ID와 코드 필드 이름은 그대로 유지한다. [논문 용어 대응표](TERMINOLOGY.md)를 참고한다.

기존 공개 코드의 가상 예시에서 GPU 작동을 확인한 기록은 [gpu_smoke_summary.json](../provenance/gpu_smoke_summary.json)에 있다. 그때의 생성 결과에는 형식 검증을 통과했지만 의미상 부적절한 기준을 적용한 사례도 포함된다. 작동 확인을 의미상 정확성이나 분류 성능의 검증으로 해석하지 않는다. 공개 코드를 이용해 논문의 전체 응답을 새로 생성하여 동일 점수까지 확인한 것은 아니다.

이번 보완본에서 새 가상환경을 만들고 패키지를 내려받는 설치 검증은 하지 않았다. 저장된 실험 환경과 의존성 버전을 명시했지만 다른 CUDA·컴파일러·GPU 조합에서 설치와 결과 동일성을 보장하지 않는다.

## 2. 실행 환경과 설치 조건

기존 실제 실행 환경은 Windows 11, Intel Core i5-10400, RAM 32 GB, NVIDIA RTX 3060 12 GB, Python 3.12.14였다. 임베딩에는 CUDA 지원 PyTorch 2.11.0+cu128과 Sentence-Transformers 5.7.0을 사용했고, 생성에는 CUDA 지원 llama-cpp-python 0.3.36을 사용했다. 관련 Python 패키지 버전은 [pyproject.toml](../pyproject.toml)을 확인한다.

1. 저장소를 로컬에 준비하고 저장소 루트에서 새 가상환경을 만든다.
2. 해당 GPU와 드라이버에 맞는 CUDA 지원 PyTorch를 먼저 준비한다. [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)를 참고한다. 기록된 cu128 빌드가 다른 환경에서도 자동 선택된다고 가정하지 않는다.
3. CUDA 지원 llama-cpp-python을 준비한다. 일반 pip 설치가 CUDA 빌드를 보장하는 것은 아니다. [공식 설치 안내](https://github.com/abetlen/llama-cpp-python#installation)를 참고하고, 소스 빌드가 필요하면 적합한 CUDA 도구 체인과 C/C++ 컴파일러를 준비한다.

PowerShell의 기본 가상환경과 패키지 설치 예시는 다음과 같다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[retrieval,generation,test]"
Copy-Item config.example.json config.local.json
```

CUDA 소스 빌드가 필요한 경우의 llama-cpp-python 명령 형태는 다음과 같다. 필요한 도구 체인 설치가 선행되어야 하며 이 명령 자체가 모든 환경에서 성공한다는 확인은 아니다.

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
python -m pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.36
Remove-Item Env:CMAKE_ARGS
```

이미 검증한 CUDA 환경이 있다면 이를 덮어 설치하지 말고 별도 가상환경을 사용한다. `config.local.json`에는 각 사용자 환경의 경로만 넣고 커밋하지 않는다. Windows에서 native DLL 탐색 문제가 발생하면 `runtime_dll_dirs` 배열에 실제 설치된 CUDA 및 llama_cpp DLL 폴더를 지정할 수 있다. 코드는 해당 프로세스에만 DLL/PATH 설정을 적용한다.

임베딩의 `torch.cuda.is_available()`와 생성의 llama.cpp GPU offload 지원이 필요하다. GPU가 없거나 적재 중 환경 오류가 발생하면 CPU로 자동 전환하거나 문맥을 자동 축소하지 않는다. 실제 적재 레이어 수와 Flash Attention 적용은 생성 시 native stderr 로그를 확인한다.

## 3. 확인된 모델 출처와 가중치 식별

아래 값은 논문 준비 때 보존한 `preparation_20261007/manifest.json`의 저장소·리비전·파일·가중치 해시와 실행 설정을 대조하여 확인했다. 현재 모델 카드의 최신 상태나 파일 이름만으로 리비전을 추정한 것이 아니다. 공개용 출처 기록은 [confirmed_models.json](../provenance/confirmed_models.json)에 있다. 모델 호스팅 접근 가능 여부를 이번 보완 과정에서 새로 다운로드하여 확인한 것은 아니다.

| 용도 | 저장소 | 고정 리비전 | 가중치 파일 |
|---|---|---|---|
| 임베딩 | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | `model.safetensors`와 해당 리비전의 토크나이저·모듈 설정 |
| 생성 | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/7c41481f57cb95916b40956ab2f0b139b296d974) | `7c41481f57cb95916b40956ab2f0b139b296d974` | `Qwen3-8B-Q4_K_M.gguf` |

가중치 SHA-256:

```text
Embedding model.safetensors:
0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd

Qwen3-8B-Q4_K_M.gguf:
d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785

GGUF native chat template:
57f1fd00f0013a2be96aa79b857391f27e23df5b5f847072b524c897e24d0361
```

이미 준비된 가중치와 캐시는 재사용한다. 새로 받아야 하는 경우의 선택적 명령은 다음과 같다. 임베딩은 지정 리비전의 기본 Hugging Face 캐시에 준비하고, 생성모델은 GGUF 하나만 받는다. 원본 8B 전체 정밀도 가중치를 함께 받을 필요는 없다.

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-Embedding-0.6B', revision='97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen3-8B-GGUF', revision='7c41481f57cb95916b40956ab2f0b139b296d974', filename='Qwen3-8B-Q4_K_M.gguf', local_dir='models')"
Get-FileHash -Algorithm SHA256 -LiteralPath 'models/Qwen3-8B-Q4_K_M.gguf'
```

CLI 실행 전에 가중치 해시를 위 기록과 대조한다. 생성 코드는 GGUF와 채팅 템플릿 해시를 검사한다. 임베딩 코드는 지정 리비전과 차원·풀링·벡터 유한값 및 정규화를 확인하지만 매번 임베딩 가중치 파일 전체의 SHA-256을 다시 계산하지 않으므로 파일 식별은 사용자가 준비 단계에서 확인한다.

`config.local.json`의 `model_path`는 받은 GGUF 경로로 설정한다. 임베딩은 `config.example.json`의 저장소 ID와 `embedding_revision`으로 기본 캐시를 사용한다. 캐시를 다른 위치에 뒀다면 Hugging Face의 로컬 환경 설정으로 경로를 관리한다. 오프라인 실행 시 필요한 전체 임베딩 설정·토크나이저까지 미리 준비하고 `HF_HUB_OFFLINE=1`을 사용한다. 인증이 필요하더라도 토큰 값은 로컬 환경에서만 관리하고 설정 파일이나 로그를 저장소에 올리지 않는다.

두 모델은 기존 다운로드 기록에서 Apache-2.0으로 확인되었지만 이는 모델의 표시 조건이며 **이 프로젝트 코드에 라이선스를 부여한 것이 아니다.** 각 고정 리비전의 LICENSE/NOTICE와 외부 기준 발췌의 이용 조건은 따로 확인한다. 프로젝트 코드 라이선스는 아직 미정이고 이 보완본도 임의로 선택하지 않았다.

## 4. 입력 데이터 계약

입력 파일은 UTF-8 JSON 배열이다. 원래 논문 데이터의 한글 키를 자동 변환하는 기능은 없으므로 아래 명시 계약으로 별도 변환해야 한다. 원본 라벨과 문단 경계는 보존하고 자동으로 수정하지 않는다.

### 4.1 신규 판정 문서 — 정답 제외

[examples/documents.json](../examples/documents.json)을 참고한다.

| 범위 | 허용 필드 | 의미 |
|---|---|---|
| 문서 | `doc_id`, `group_id`, `text`, `paragraphs` | 문서·원본 그룹 ID, 문서 전체 원문, 제공된 문단 목록 |
| 문단 | `paragraph_id`, `text`, `start`, `end` | 문단 ID와 본문, 문서 전체 원문에서의 위치 |

위치는 Python Unicode 문자 기준 **0-based, 끝 위치 포함**이며 `document_text[start:end+1] == paragraph_text`여야 한다. 문단은 원문에서 순서대로 겹치지 않아야 하며 문단 간 구분용 공백·줄바꿈이 남을 수 있다. 별도 자동 청킹 기능은 없다.

평가 문서의 정답 등급, 정답 근거, 정답 기준과 사유는 이 입력에 넣지 않는다. 허용 필드 이외의 정답 메타데이터가 들어오면 검사에서 거부한다. ID에 정답 등급을 암호처럼 넣지 않는다. `group_id`에는 알려진 동일 원본·번역·파생 관계를 미리 반영한다.

### 4.2 검색 사례 — 근거 메타데이터 포함

[examples/retrieval_cases.json](../examples/retrieval_cases.json)은 직접 작성한 가상 사례 3개다.

| 필드 | 의미 |
|---|---|
| `case_id`, `paragraph_id`, `doc_id`, `group_id` | 사례·문단·문서·원본 그룹 식별자 |
| `text` | 검색용 문단 본문. 벡터화 대상은 이 본문뿐 |
| `label` | 검색 사례 문단의 C/S/O 등급 |
| `evidence_spans` | `source_paragraph_id`, `source_scope`, `start`, `end`, `text`를 가진 근거 목록 |
| `criterion_ids` | 확인된 canonical 기준 ID 목록 |
| `reason` | 기존 라벨의 기준 충족 사유 |
| `case_context` | 연결 문맥 목록. 각 항목은 `paragraph_id`, `text` |

검색 사례의 근거 위치는 **근거 출처 문단 내부**의 inclusive 위치다. 대상 문단과 연결 문맥에 걸친 근거는 각 출처 문단 ID로 구분한다. 본문·기준·사유를 임의로 만들어 보정하거나 미확정 별칭을 자동 매핑하지 않는다. 최소 3개의 검색 사례가 필요하다.

신규 문서는 검색 사례와 문서·원본 그룹이 겹치면 안 된다. 코드는 제공된 ID와 정확한 문단 본문 중복을 검사하지만, 모든 번역·유사 파생본 관계를 자동 탐지하는 기능은 아니다. 검색 사례의 연결 문맥과 신규 문단 간 정확 중복도 검사한다. 원본 관계에 대한 사전 검토가 필요하다.

## 5. 실행 순서

다음 명령은 저장소 루트에서, 모델 경로를 설정한 로컬 환경을 기준으로 한다. 새 인덱스와 결과 폴더를 사용하고 기존 실험 경로를 지정하지 않는다. 아래 예제는 작동 확인용이며 성능 평가 자료가 아니다.

```powershell
# 1. 모델 없이 입력 필드·원문 위치·검색/신규 입력 분리 검사
cso-rag --config config.local.json --assets assets check-input --input examples/documents.json --cases examples/retrieval_cases.json

# 2. 검색용 문단만 임베딩하고 새로운 전용 Chroma 인덱스 구축
cso-rag --config config.local.json --assets assets build-index --cases examples/retrieval_cases.json --index indices/example

# 3. 대상 문단 질의 임베딩 → 검색 3개 → 메타데이터 조회 → 기준 선택 → 입력·스키마·문법 저장
cso-rag --config config.local.json --assets assets prepare --input examples/documents.json --index indices/example --output runs/example_prepared

# 4. 실제 네이티브 토크나이저로 모든 준비 입력의 수용량 확인 (답변 생성 없음)
cso-rag --config config.local.json --assets assets capacity --prepared runs/example_prepared --report runs/example_capacity.json

# 5. 위 검사를 통과한 동일 입력으로 근거 포함 RAG만 실행
cso-rag --config config.local.json --assets assets run --prepared runs/example_prepared --capacity runs/example_capacity.json --output runs/example_results 2> runs/example_native.log
```

각 CLI 명령은 별도 프로세스다. 임베딩 단계가 종료되어 해당 모델이 해제된 뒤 생성모델을 적재한다. `capacity`는 vocabulary-only 적재로 토큰 수용량을 확인하는 것이며 GPU 전체 적재·실제 생성 성공 검사와는 다르다. 생성 로그에서 실제 GPU 적재와 명시적 YaRN 적용을 확인해야 한다.

실행 설정은 전체 문맥 45,056, 출력 예약 4,096, 안전 여유 128토큰이다. 입력+출력 예약+여유가 문맥 한도를 넘으면 자동 축약하지 않는다. Flash Attention, 배치 128, seed 42, temperature 0, thinking 비활성, 명시적 YaRN(type 2 / scale .25 / orig_ctx 32,768)을 설정에서 유지한다. YaRN orig_ctx가 학습 시 문맥 길이를 자체 입증하는 자료는 아니다.

반복 감지와 300초 중단 조건은 생성 콜백에서 확인한다. native prefill을 OS watchdog처럼 즉시 선점하는 제한은 아니다. 환경 오류 시 완료·부분 결과를 보존하고 중단하며, 문맥·사례 수·모델 설정을 자동 변경하지 않는다. 실패만 선택하여 반복 실행하는 기능도 제공하지 않는다.

## 6. 결과 확인과 재개

| 위치 | 내용 |
|---|---|
| 준비 폴더 `items/*.json` | 대상·제공 문맥, 실제 검색 사례, 기준 선택·제외 이유, E registry, 출력 schema와 grammar |
| 준비 폴더 `manifest.json` | 입력 구성·설정·기준 자산의 식별 해시 및 문서·문단 수 |
| 결과 폴더 `paragraphs/*/input.json` | 실제 system/user 메시지 |
| `partial.txt`, `chunks.jsonl` | 부분 출력과 스트리밍 기록 |
| `result.json` | 원시 출력, 제안 등급, 복원 근거, 검증, 종료 사유, 토큰 수와 생성 시간 |
| `documents.json` | 모든 문단 완성 시 C>S>O 집계, 미분류가 있으면 불완전 및 부분 제안 |
| `runtime.json`, `gpu_samples.jsonl` | 적용 설정과 GPU 표본. native 적재 레이어는 stderr 로그와 함께 확인 |
| `completed.json`, `identity.json` | 완료 목록과 동일 코드·입력·설정 여부 확인 |

완성된 제안 등급은 검증 실패와 관계없이 보존한다. 모델 자체 검토 요청은 자동 검증 실패와 구분한다. r1에서는 사유 속 ID 재표기 누락을 진단으로만 남기며 그 항목만으로 검토 필요로 만들지 않는다. ID·출처·인용·위치·중복·대상 근거 포함 및 기준/등급 형식 검사는 유지한다. 의미상 타당성은 자동 검증되지 않는다.

동일한 `run` 명령과 동일한 결과 폴더를 사용하면 완료된 결과를 재생성하지 않는다. 코드·설정·입력·기준이 다르면 재개를 거부한다. 실패·미분류도 완료 기록으로 보존하고 점수를 높이기 위한 자동 재시도를 하지 않는다. 강제 중단되어 `attempt_started.json`만 남은 경우 자동 재시도를 차단한다. 환경 오류가 기록된 실행도 자동 재개하지 않는다. 이 경우 부분 출력을 보존하고 별도 복구 계획을 정해야 한다.

현재 파일에는 코드 해시가 포함되므로 주석 변경도 기존 공개 실행 결과의 자동 재개를 차단할 수 있다. 이 관리 보완본은 검토 커밋 대비 실행 파일 바이트를 바꾸지 않았지만, 더 오래된 배포본에서 생성한 결과와 동일 코드라고 임의로 간주해서는 안 된다.

사람이 읽는 HTML을 만드는 서비스나 정답 기반 성능 평가 계산기는 배포 범위에 없다. README의 결과표는 보존된 논문 실험 결과를 소개한 것이며 새 예제 실행에 동일 점수를 부여한 것이 아니다.

## 7. 모델 없는 연결 검사

```powershell
python -m unittest discover -s tests -v
```

기준 커밋의 테스트는 12개다. 이 검사는 입력 분리, 위치, ID·스키마·문법·r1 검증과 가짜 생성기의 저장·재개 연결을 확인한다. 실제 GPU 실행이나 의미상 평가를 새로 수행하지 않는다. `test_runner.py`는 임시 데이터를 제외된 `runs/` 아래에 만들고 지운다.

## 8. 원본 해시와 공개본 해시

- [original_execution_code_hashes.json](../provenance/original_execution_code_hashes.json): 원래 실험 코드 해시 기록. 관리 보완으로 덮어쓰지 않았다.
- [source_files.json](../provenance/source_files.json): 원본 `original_sha256`을 유지하고 현재 공개 파일 `export_sha256`과 `verbatim`을 구분했다. `verbatim`은 바이트 동일일 때만 true다.
- [public_code_change_history.json](../provenance/public_code_change_history.json): 이전 GPU 확인 배포본과 검토 커밋 사이의 주석·docstring·줄바꿈 차이 및 실행 AST 비교를 기록한다. docstring 삭제는 `__doc__` 메타데이터의 차이도 만든다.
- [history](../provenance/history/): 검토 커밋 당시의 해시·출처·검사 기록을 원래 바이트로 보존한다. 과거 기록을 현재 파일의 해시 목록으로 오인하지 않는다.
- [export_hashes.json](../provenance/export_hashes.json): 이 관리 보완본의 전체 공개 파일 SHA-256. 자기 자신은 순환 참조 방지를 위해 제외한다.
- [publication_management_checks.json](../provenance/publication_management_checks.json): 이번 관리 보완의 파일·링크·제외 규칙 및 보존 점검 결과다. 모델 작동 확인 기록과는 별개다.

Git 체크아웃 시 CRLF/LF 자동 변환이 있으면 작업 폴더 바이트와 릴리스 ZIP 해시가 다를 수 있다. 이 배포본은 기준 커밋의 Git blob을 직접 복사했으며, 해시 대조는 ZIP의 실제 바이트를 기준으로 한다. 주석·설명문 삭제 이후에도 실행 구문이 같다는 것과 원본에 바이트 단위로 동일하다는 것은 구분해야 한다.

## 9. 공개 범위와 라이선스 미정

README는 논문 보충 자료이므로 표와 화면 설명을 중심으로 유지한다. 화면 이미지의 설명은 “합성 문서의 실제 저장 응답을 표시한 HTML 결과 패널의 내보내기 이미지”이며, 원본 HTML 코드나 브라우저 캡처를 배포했다는 뜻은 아니다. 실제 원문 예시는 합성 문서의 저장 결과이며 실존 인물의 정보임을 확인했다는 뜻이 아니다.

프로젝트 코드 라이선스는 **미정**이다. 이 보완 작업에서는 LICENSE를 추가하거나 특정 라이선스를 선택하지 않았다. 법령과 행정안전부 자료, N2SF 발췌, 모델 가중치 및 프로젝트 코드의 이용 조건은 각각 별개다. 외부 기준의 출처와 확인 상태는 `assets/criteria_reference_v3/` 및 `assets/criteria_input/source_catalog.json`에 보존되어 있으며, 모든 자료의 재배포 허락을 확인한 것으로 표현하지 않는다.

로컬 모델·인덱스·입력 데이터·실행 결과·캐시·인증정보는 `.gitignore`에서 제외한다. `.gitignore`는 이미 Git이 추적하는 파일을 자동으로 추적 해제하지 않으므로 업로드 전 파일 목록을 확인해야 한다. 필요한 가상 예시와 기준 자산만 의도적으로 배포한다.
