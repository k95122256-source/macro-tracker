# Macro Regime & Risk Tracker Pro (GitHub Actions 365-Day Rolling DB)

하루 1회(매일 07:30 KST) GitHub Actions에서 FRED API(`3d08e04630fc05569922c338518764ed`) 및 글로벌 시장 데이터를 수집해 Z-Score와 자산배분 비중을 **사전 연산(Pre-compute)**하고, **365일 순연(Rolling Window) 구조**로 `macro_data.json`과 `latest_macro.json`을 자동 갱신하여 **웹페이지 접속 시 API 호출 및 연산 부하를 0으로 만든** 정적 매크로 대시보드입니다.

## 핵심 아키텍처 (Zero-Load Daily Rolling Pipeline)
1. **서버 사이드 일 1회 사전 연산 (`update_macro_data.py`)**
   - 브라우저가 접속할 때마다 외부 API(FRED, Yahoo Finance)를 호출하지 않습니다.
   - GitHub Actions가 매일 오전 07:30 KST에 데이터를 가져와 US/KR 5대 팩터 Z-Score, 종합 스코어($M_t$), 킬 스위치 여부, 시그모이드/구간형 권장 비중을 미리 계산해 JSON에 기록합니다.
2. **365일 순연 저장 (`MAX_ROLLING_DAYS = 365`) & 연도별 자동 아카이빙 (`archive/`)**
   - 매일 데이터가 쌓이더라도 `macro_data.json`은 최근 365일치만 유지하며 하루씩 순연됩니다.
   - 365일이 지난 과거 레코드는 삭제되지 않고 `archive/macro_data_YYYY.json`으로 자동 분리 저장되어, 10년 이상 운영해도 웹페이지 초기 로딩 파일 크기가 수십 KB 이내(`O(1)`)로 일정하게 유지됩니다.
3. **웹페이지 접속 시 무부하 자동 동기화 (`index.html`)**
   - 페이지 로드(`window.onload`) 시 `./macro_data.json`을 정적 캐시로 즉시 읽어와 대시보드와 누적 로그 테이블을 자동 갱신합니다.
