# Apache Superset

Apache Superset은 **오픈소스 데이터 시각화 및 데이터 탐색 도구**로, 다양한 데이터베이스와 연동하여 대시보드를 만들고 데이터를 분석이 가능.

---

## 1. 주요 특징

- **다양한 데이터 소스 지원**
  - MySQL, PostgreSQL, Oracle, SQL Server 등 RDBMS
  - Druid, Presto, ClickHouse 등 빅데이터 분석용 데이터베이스
  - CSV, Excel 등 파일 기반 데이터도 연결 가능
- **드래그 앤 드롭 기반 시각화**
  - 차트, 테이블, KPI, 지도 등 다양한 시각화 제공
  - SQL Lab을 통해 직접 쿼리 실행 가능
- **대시보드**
  - 여러 차트를 하나의 화면에 배치
  - 필터, 상호작용 기능 지원
- **사용자 및 권한 관리**
  - Role 기반 접근 제어(RBAC)
  - 사용자별 대시보드 접근 권한 설정 가능

---

## 2. 설치 방법

### Docker 사용
```bash
docker run -d -p 8088:8088 apache/superset
```

```python
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# Superset 설치
pip install apache-superset

# 초기화
superset db upgrade
superset fab create-admin
superset init

# 서버 실행
superset run -p 8088 --with-threads --reload --debugger
```

- 실제 해당 path에 존재하는 docker-compose yaml파일에서 해보면 kafka -> druid -> superset 연결 확인 가능