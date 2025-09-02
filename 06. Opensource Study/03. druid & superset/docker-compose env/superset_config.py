# superset_config.py

# -----------------------------
# CORS 설정 (임시 비활성화)
# -----------------------------
# ENABLE_CORS = True
# CORS_OPTIONS = {
#     "origins": ["*"],
#     "supports_credentials": True
# }

# -----------------------------
# Druid 설정
# -----------------------------
DRUID_IS_ACTIVE = True
DRUID_TZ = 'Asia/Seoul'

# Druid Broker URL (broker 컨테이너 이름과 포트 확인)
DRUID_BROKER_URL = 'http://broker:8082/druid/v2/sql/'

# Optional: Druid metadata DB (Superset가 자체적으로 저장)
SQLALCHEMY_DATABASE_URI = 'sqlite:////app/superset_home/superset.db'