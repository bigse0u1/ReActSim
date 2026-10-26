# ReActSim — Meta Global AI Developer Hackathon 개발 계획

> **프로젝트:** ReActSim — A Physics-Grounded, Self-Correcting Robot Agent  
> **트랙:** Push the Frontier (로봇/물리 시스템)  
> **환경:** Ubuntu 22.04 + NVIDIA RTX 4060 Ti 8GB + VS Code  
> **시뮬레이터:** ManiSkill 3 (SAPIEN 기반), Franka Panda  
> **Meta 모델:** SAM 3.1 + Muse Spark (Meta Model API)  
> **제출 마감:** 2026-11-05 11:00 KST (2026-11-04 21:00 EST)  
> **원칙:** 해커톤 기간에 개발한 신규 기능과 코드를 제출하고 공개 저장소에 변경 내역을 남긴다.

## 1. 프로젝트 핵심

사용자가 자연어로 목표를 제시하면 로봇이 시뮬레이션에서 목표 객체를 찾고 행동을 수행한다. 도중에 물체가 이동하거나 떨어지는 등 예상과 다른 상황이 발생하면 **다시 관측 → 상태/실패 판정 → 행동 재계획 → 검증 → 재실행**을 반복한다.

**한 줄 소개:** From Following Instructions to Understanding Failures.

### 반드시 보여줄 장면

1. 사용자가 `Put the red cube on the target plate` 명령 입력.
2. SAM 3.1이 물체를 분할하고 목표를 식별.
3. Muse Spark가 고수준 행동(`pick`, `place`, `reobserve` 등)을 구조화하여 선택.
4. 시뮬레이터에서 Panda 로봇이 기본 동작 실행.
5. 목표 물체 위치 변경 또는 실패 발생.
6. 관측 업데이트 후 재계획하여 작업 완료.

**중요한 설계 구분:** SAM 3.1의 마스크는 직접적인 3D grasp pose가 아니다. RGB-D/카메라 행렬과 기하학적 변환 또는 초기에는 시뮬레이터의 ground-truth 물체 pose를 활용한다. 또 Muse Spark는 저수준 관절 토크를 직접 제어하지 않는다. IK·기존 motion planning·그리퍼 제어가 실제 동작을 담당한다.

## 2. 왜 이 프로젝트인가?

- **창의성:** 단순 언어→집기 파이프라인이 아니라 물리적 실패 후 자동 복구.
- **기술적 도전:** 시각 인식, 공간적 grounding, 고수준 계획, 물리 기반 성공 판정, closed-loop recovery의 통합.
- **실제 작동성:** 로컬 물리 시뮬레이션에서 반복 테스트와 3분 데모 가능.
- **Meta 활용:** SAM 3.1이 시각 인식, Muse Spark가 계획 및 복구 선택을 실제로 담당.
- **신규성 명확화:** 기존 Vesalius 데이터를 학습하거나 기존 로봇 제어 코드를 그대로 신규 결과물로 주장하지 않음.

## 3. 전체 아키텍처

```text
Natural-language instruction
              |
              v
   ManiSkill RGB (+ depth)
              |
              v
      Meta SAM 3.1 API
      masks / bounding boxes
              |
              v
 Perception grounding & state representation
 (object ID, estimated/GT pose, goal, gripper state)
              |
              v
       Meta Muse Spark API
      structured high-level plan
              |
              v
       Action validator
 (allowed tools, geometry, feasibility checks)
              |
              v
      Low-level controllers
  pick / place / move / open / close
              |
              v
      ManiSkill physics step
              |
              v
    Objective success checker
          |           |
        success     failure
          |           |
          v           +---> reobserve / replan (bounded retries)
        done
```

### 모델/컴포넌트의 역할

| 컴포넌트 | 담당 | MVP에서 주의할 점 |
|---|---|---|
| SAM 3.1 | 자연어 기반 객체 분할 | 마스크→물체 위치 매핑 정확도 검증 |
| Muse Spark | 목표를 제한된 행동 목록으로 분해, 복구 행동 선택 | JSON 스키마 검증, 잘못된 출력 거부 |
| ManiSkill 3 | Panda 물리 시뮬레이션, 관측, 성공 판정 | 먼저 단일 환경 CPU/state 모드에서 검증 |
| 제어 모듈 | IK, 이동, 파지, 배치 | 제로부터 VLA 재학습하지 않기 |
| 평가 모듈 | 성공률, 복구 성공률, 단계별 로그 | 성공 여부는 LLM 아닌 환경의 물리 상태로 판정 |

## 4. 현지 Ubuntu에서 환경 점검

기존 Isaac Sim, ROS2, LeRobot 환경을 건드리지 않고 **별도 Python 가상환경**을 만든다.

### 4.1 시스템 확인

```bash
lsb_release -a
uname -r
nvidia-smi
python3 --version
free -h
df -h
```

드라이버와 GPU 인식에 문제가 있다면, 당장 드라이버나 CUDA를 재설치하지 말고 먼저 오류를 기록한다.

### 4.2 Vulkan 확인

```bash
sudo apt update
sudo apt install -y libvulkan1 vulkan-tools python3-venv python3-pip
vulkaninfo --summary
```

NVIDIA GPU가 Vulkan에서 식별되는지 확인한다. ManiSkill의 **상태 기반 시뮬레이션은 렌더링 없이 가능**하지만 RGB 렌더링에는 Vulkan이 필요하다.

### 4.3 프로젝트 폴더 및 가상환경

```bash
mkdir -p ~/projects/reactsim
cd ~/projects/reactsim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**참고:** 새 환경은 Ubuntu 22.04의 기본 Python 3.10을 사용할 수 있다. 라이브러리 호환 문제가 있다면 별도의 Python 버전을 선택하되 기존 시스템 Python은 변경하지 않는다.

### 4.4 ManiSkill 설치

```bash
python -m pip install --upgrade mani_skill torch
python -m pip show mani_skill torch
```

GPU별 PyTorch/CUDA 빌드가 필요하면 공식 PyTorch 설치 가이드를 먼저 확인한다. 설치가 충돌하면 무작정 시스템 CUDA를 변경하지 않는다.

### 4.5 공식 예제로 기본 실행

```bash
# 환경 및 물리 엔진이 정상 작동하는지 확인(무작위 행동, 작업 성공 테스트 아님)
python -m mani_skill.examples.demo_random_action -e PickCube-v1

# 데스크톱에서 GUI/렌더링 확인
python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode="human"
```

GUI에서 실패하더라도 위의 **headless 실행 결과**와 `vulkaninfo --summary` 결과를 구분하여 확인한다. GUI는 원격/디스플레이 구성에 따라 추가 설정이 필요할 수 있다.

### 4.6 프로젝트 기본 파일

```bash
mkdir -p src/reactsim tests configs scripts outputs docs
mkdir -p src/reactsim/{perception,planning,control,evaluation}
touch README.md .gitignore
printf '.venv/\n.env\n__pycache__/\noutputs/\n*.pyc\n' > .gitignore
git init
```

Meta API 키는 코드나 공개 Git 저장소에 절대로 올리지 않는다. `.env`는 무조건 `.gitignore`에 포함한다.

## 5. 개발 순서 / 완료 조건

### Phase 0. 환경 검증

- [ ] GPU가 `nvidia-smi`에서 확인된다.
- [ ] `vulkaninfo --summary`가 정상 작동한다.
- [ ] ManiSkill `PickCube-v1` 공식 예제가 정상 실행된다.
- [ ] 가능하면 GUI에서 Panda와 큐브가 보인다.
- **완료 기준:** 50-step 무작위 행동 예제 실행, 환경 오류 없이 종료.

### Phase 1. 로봇 조작 baseline

- [ ] 제어 방법(`pd_ee_delta_pose` 등)을 확인한다.
- [ ] 안정적인 단일 물체 pick-and-place 정책을 구현 또는 ManiSkill 제공 예제로 검증한다.
- [ ] 물리 상태를 이용해 목표 성공 여부를 판정한다.
- **완료 기준:** 동일 환경에서 여러 seed를 바꿔 기본 작업 성공률 측정.

### Phase 2. 시각 및 언어 인식

- [ ] 시뮬레이터에서 RGB 이미지와 카메라 메타데이터를 추출한다.
- [ ] SAM 3.1 API를 통해 `red cube` 등 객체 마스크를 얻는다.
- [ ] 마스크와 (초기에는 GT pose, 이후 깊이 기반) 물체 위치를 연결한다.
- [ ] Muse Spark에 제한된 행동 명세를 전달하고 JSON 결과를 검증한다.
- **완료 기준:** 언어 명령 → 대상 물체 ID → 제어 모듈 입력으로 변환 성공.

### Phase 3. 폐루프 복구 기능

- [ ] 작업 중 물체 위치를 바꾸는 테스트 조건을 만든다.
- [ ] 파지 실패 / 놓침 / 목표 미도달 여부를 물리 상태로 검출한다.
- [ ] 실패 시 목표 객체를 재관측한다.
- [ ] Muse Spark가 재계획하고 제어 모듈이 재실행한다.
- [ ] 최대 재시도 횟수, 타임아웃, 실패 기록을 구현한다.
- **완료 기준:** 예기치 않은 변화 조건에서 최소 한 종류의 자동 복구 시나리오가 반복 작동.

### Phase 4. 기술적 차별화

- [ ] 복수 후보 행동 생성(`top grasp`, `side grasp` 등).
- [ ] 시뮬레이터 복제/리셋 또는 미리 정의된 검증 시나리오에서 물리 타당성 평가.
- [ ] 충돌·관절 제한·목표 달성 가능성에 따른 행동 선택.
- **주의:** 물리적 미래 결과를 실제로 시뮬레이션하지 못한다면 `physics-based lookahead`라고 주장하지 않는다. 단순 feasibility check와 구분해야 한다.

### Phase 5. 평가 및 데모

- [ ] Baseline A: 재계획 없는 고정 정책.
- [ ] Baseline B: 규칙 기반 재시도.
- [ ] Proposed: Muse Spark + SAM 3.1 기반 복구.
- [ ] 같은 초기 상태/장애물 조건과 seed를 사용해 공정하게 비교.
- [ ] 성공률, 복구 성공률, 평균 지연시간, 충돌 횟수 측정.
- [ ] 실제 실행 영상과 로그를 근거로 3분 데모 제작.

## 6. 권장 실험 설계

| 환경 조건 | 설명 | 목적 |
|---|---|---|
| Normal | 물체 고정, 장애물 없음 | 기본 집기 성능 |
| Position shift | 실행 전/중 목표 물체 위치 변경 | 재인식 및 재계획 |
| Grasp failure | 안전하게 설정한 파지 실패 시나리오 | 복구 성공 여부 |
| Occlusion (확장) | 다른 물체가 목표를 가림 | 시각 재인식과 보조 행동 |
| Obstacle (확장) | 경로상 장애물 배치 | 충돌 회피 및 재계획 |

**측정 기준:**

- `task_success_rate = 성공한 에피소드 / 전체 에피소드`
- `recovery_success_rate = 초기 실패 이후 최종 성공한 에피소드 / 초기 실패가 발생한 에피소드`
- `mean_recovery_attempts`, `mean_latency_sec`, `collision_count`
- 실험마다 seed, 환경 설정, 모델, 실패 조건, 최종 결과를 JSON/CSV로 기록.
- 정량 결과가 없으면 임의의 성공률을 발표 자료에 기재하지 않는다.

## 7. 10일 일정표

| 날짜 | 목표 | 산출물 |
|---|---|---|
| 10/26–10/27 | 설치/기본 환경 및 제어 동작 검증 | 정상 작동하는 Panda 환경 |
| 10/28 | RGB 관측, SAM 연동 | 객체 마스크 및 매핑 |
| 10/29 | Muse Spark 계획 인터페이스 | 구조화된 동작 출력 |
| 10/30 | 언어→인식→행동 실행 | 첫 end-to-end 성공 |
| 10/31 | 실패 감지/재관측 | 상태 진단 모듈 |
| 11/01 | 복구 재계획 | 자동 복구 데모 |
| 11/02 | physics 검증 또는 장애물 보조 행동 | 차별화 기능 |
| 11/03 | 비교 실험/오류 수정 | 평가 로그/표/그래프 |
| 11/04 | 3분 영상/README/제출 | 공개 GitHub + 데모 + 제출 |

마감은 **2026-11-05 11:00 KST**. 최종 제출은 마감 직전보다 여유 있게 완료한다.

## 8. 해커톤 규정 체크리스트

- [ ] `Push the Frontier` 과제에 맞게 기술적 목적이 명확하다.
- [ ] 실제 Meta 모델을 프로젝트의 핵심 기능에 사용한다.
- [ ] 저장소는 공개(public)이다.
- [ ] 해커톤 기간의 **신규 코드/기능**을 커밋과 문서로 명확히 구분한다.
- [ ] 기존 Vesalius 코드/학습 결과물을 신규 해커톤 성과로 표시하지 않는다.
- [ ] Indie Developer에서 기존 코드베이스 사용 범위가 불명확하면 Discord 운영진에게 확인한다.
- [ ] 제3자 코드와 자산 라이선스를 준수한다.
- [ ] 데모는 실제 기능을 보여주고 실패 사례도 정직하게 설명한다.
- [ ] **3분 데모 영상**에 이번 해커톤에서 만든 부분을 보여준다.
- [ ] 모든 팀원과 공개 코드/영상 링크를 제출한다.

## 9. Meta API 크레딧 및 보안

- 공식 참가자 안내에 따라 개인별 **$150 Meta Model API 크레딧** 신청을 확인한다.
- 결제 카드 등록 요구사항과 Billing 페이지를 확인한다.
- 사용량 알림만으로 초과 결제가 차단된다고 가정하지 말고 실제 과금 상태를 점검한다.
- `MODEL_API_KEY`를 환경변수나 `.env`로 관리하며, 예제와 로그에 키를 출력하지 않는다.
- Meta API base URL: `https://api.meta.ai/v1`.
- 모델 ID는 공식 문서에서 현재 제공되는 항목을 확인해서 사용한다 (`sam-3.1`, Muse Spark 시리즈 등).
- API 호출 전후 추론 지연시간과 비용을 기록한다.

## 10. 3분 데모 구성

| 구간 | 내용 |
|---|---|
| 0:00–0:25 | 문제 소개: 일반 정책의 위치 변화/실패 취약성 |
| 0:25–1:00 | 구조 소개: SAM 3.1 + Muse Spark + ManiSkill |
| 1:00–2:00 | 실제 시뮬레이션: 성공 및 예상치 못한 실패 후 복구 |
| 2:00–2:35 | Baseline 비교와 정량 결과 |
| 2:35–3:00 | 신규 코드, 핵심 기여, 한계 및 확장 방향 |

## 11. 데스크톱에 도착하면 공유할 결과

다음 출력과 오류 메시지를 공유하면 설치 상태를 기준으로 다음 단계를 결정할 수 있다.

```bash
nvidia-smi
python3 --version
vulkaninfo --summary
python -m mani_skill.examples.demo_random_action -e PickCube-v1
```

마지막 명령은 위의 가상환경 활성화 및 설치를 마친 뒤 실행한다.

## 12. 참고 자료

- [해커톤 행사 페이지](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
- [프로젝트 제출 페이지](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit)
- [ManiSkill 설치](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [ManiSkill Quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html)
- [Meta Model API 개요](https://dev.meta.ai/docs/overview)
- [Meta Model API Quickstart](https://dev.meta.ai/docs/quickstart)
- [Meta API 모델 목록](https://dev.meta.ai/docs/models)

---

**최우선 작업:** Ubuntu에서 ManiSkill `PickCube-v1`을 실행하고, 무작위 제어 테스트와 실제 pick-and-place 성공 테스트를 분리한다. 첫 end-to-end 성공 전에는 월드모델 학습이나 VLA 재학습에 시간을 쓰지 않는다.
