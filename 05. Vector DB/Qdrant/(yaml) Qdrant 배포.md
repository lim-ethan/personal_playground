```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: always

  prometheus:
    image: prom/prometheus
    container_name: prometheus
    volumes:
      - "C:/Users/{user_pc_id}/Desktop/localtest/qdrant_test/prometheus.yml:/etc/prometheus/prometheus.yml" 
    # {user_pc_id}에는 본인 PC ID 입력 필요
    ports:
      - "9090:9090"
    restart: always

  grafana:
    image: grafana/grafana
    container_name: grafana
    ports:
      - "3000:3000"
    restart: always
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  qdrant_data:
  grafana_data:

```