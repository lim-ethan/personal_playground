from snowflake.snowpark.functions import col
from snowflake.ml.modeling.xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main(session):
    # 1. 데이터 로딩 및 전처리
    df = session.sql('''
        SELECT
          t.SD,
          t.SGG,
          t.EMD,
          t.BJD_CODE,
          t.YEAR,
          t.QUARTER,
          t.JEONSE_PRICE_PER_SUPPLY_PYEONG,
          a.AVERAGE_AREA,
          a.AVERAGE_MARKET_PRICE,
          a.AE_66_SQUARE_METER_BELOW_HOUSEHOLD_COUNT,
          a.AE_66_SQUARE_METER_HOUSEHOLD_COUNT,
          a.AE_99_SQUARE_METER_HOUSEHOLD_COUNT,
          a.AE_132_SQUARE_METER_HOUSEHOLD_COUNT,
          a.AE_165_SQUARE_METER_HOUSEHOLD_COUNT,
          a.ABOVE_600M_HOUSEHOLD_COUNT,
          p.TOT_WRC_POPLTN_CO,
          p.FML_WRC_POPLTN_CO,
          p.ML_WRC_POPLTN_CO,
          p.AGRDE_30_WRC_POPLTN_CO,
          p.AGRDE_40_WRC_POPLTN_CO,
          t.MEME_PRICE_PER_SUPPLY_PYEONG
        FROM (
            SELECT 
              SUBSTR(BJD_CODE, 1, 5) AS BJD_CODE,
              SD, SGG, EMD,
              SUBSTR(YYYYMMDD, 1, 4) AS YEAR,
              CASE 
                WHEN TO_NUMBER(SUBSTR(YYYYMMDD, 6, 2)) BETWEEN 1 AND 3 THEN '1'
                WHEN TO_NUMBER(SUBSTR(YYYYMMDD, 6, 2)) BETWEEN 4 AND 6 THEN '2'
                WHEN TO_NUMBER(SUBSTR(YYYYMMDD, 6, 2)) BETWEEN 7 AND 9 THEN '3'
                WHEN TO_NUMBER(SUBSTR(YYYYMMDD, 6, 2)) BETWEEN 10 AND 12 THEN '4'
              END AS QUARTER,
              MEME_PRICE_PER_SUPPLY_PYEONG,
              JEONSE_PRICE_PER_SUPPLY_PYEONG
            FROM PO3.POPULATION.DAILY_REAL_ESTATE_TRADES
        ) t
        LEFT JOIN (
            SELECT
              AVERAGE_AREA,
              AVERAGE_MARKET_PRICE,
              AE_66_SQUARE_METER_BELOW_HOUSEHOLD_COUNT,
              AE_66_SQUARE_METER_HOUSEHOLD_COUNT,
              AE_99_SQUARE_METER_HOUSEHOLD_COUNT,
              AE_132_SQUARE_METER_HOUSEHOLD_COUNT,
              AE_165_SQUARE_METER_HOUSEHOLD_COUNT,
              ABOVE_600M_HOUSEHOLD_COUNT,
              DISTRICT_CODE AS BJD_CODE,
              SUBSTR(STANDARD_YEAR_QUARTER_CODE, 1, 4) AS YEAR,
              SUBSTR(STANDARD_YEAR_QUARTER_CODE, 5, 1) AS QUARTER
            FROM PO3.POPULATION.DISTRICT_APARTMENT
        ) a ON t.BJD_CODE = a.BJD_CODE AND t.YEAR = a.YEAR AND t.QUARTER = a.QUARTER
        LEFT JOIN (
            SELECT 
              TOT_WRC_POPLTN_CO,
              FML_WRC_POPLTN_CO,
              ML_WRC_POPLTN_CO,
              AGRDE_30_WRC_POPLTN_CO,
              AGRDE_40_WRC_POPLTN_CO,
              MEGA_CD,
              SUBSTR(STDR_YYQU_CD, 1, 4) AS YEAR,
              SUBSTR(STDR_YYQU_CD, 5, 1) AS QUARTER
            FROM PO3.POPULATION.DISTRICT_WORK_POPULATION
        ) p ON SUBSTR(t.BJD_CODE, 1, 2) = p.MEGA_CD AND t.YEAR = p.YEAR AND t.QUARTER = p.QUARTER
        WHERE t.MEME_PRICE_PER_SUPPLY_PYEONG IS NOT NULL AND a.AVERAGE_AREA IS NOT NULL
    ''').na.drop()

    # 2. 형 변환
    cast_fields = [
        "MEME_PRICE_PER_SUPPLY_PYEONG",
        "AVERAGE_AREA",
        "AVERAGE_MARKET_PRICE",
        "ABOVE_600M_HOUSEHOLD_COUNT",
        "TOT_WRC_POPLTN_CO",
        "FML_WRC_POPLTN_CO",
        "ML_WRC_POPLTN_CO",
        "AGRDE_30_WRC_POPLTN_CO",
        "AGRDE_40_WRC_POPLTN_CO",
        "JEONSE_PRICE_PER_SUPPLY_PYEONG"
    ]
    for col_name in cast_fields:
        df = df.with_column(col_name, col(col_name).cast("FLOAT"))

    # 3. 피처/레이블/출력 정의
    FEATURE_COLS = cast_fields[:-1]  # 마지막이 label
    LABEL_COLS = ["JEONSE_PRICE_PER_SUPPLY_PYEONG"]
    OUTPUT_COLS = ["PREDICTED_JEONSE_PRICE"]

    # 4. 학습/테스트 분리
    train_df, test_df = df.random_split([0.8, 0.2], seed=42)

    # 5. 모델 생성 및 학습
    model = XGBRegressor(
        input_cols=FEATURE_COLS,
        label_cols=LABEL_COLS,
        output_cols=OUTPUT_COLS
    )
    model.fit(train_df)

    # 6. 예측
    predictions = model.predict(test_df)

    # 7. 예측 결과 디버깅용 컬럼 확인
    print("✅ 예측 결과 컬럼:")
    print(predictions.columns)

    # 8. 평가 지표 계산용 컬럼 alias 적용
    eval_df = predictions.select(
        col("JEONSE_PRICE_PER_SUPPLY_PYEONG").alias("actual"),
        col("PREDICTED_JEONSE_PRICE").alias("predicted")
    )

    # 9. pandas 변환 후 컬럼 확인
    pred_pd = eval_df.to_pandas()
    print("✅ pandas DataFrame 컬럼:")
    print(pred_pd.columns)

    # 10. 지표 계산
    rmse = mean_squared_error(pred_pd["actual"], pred_pd["predicted"], squared=False)
    mae = mean_absolute_error(pred_pd["actual"], pred_pd["predicted"])
    print(f"✅ RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # 8. 예측 결과 저장
    predictions.write.mode("overwrite").save_as_table("PO3.POPULATION.SNOWPARK_JEONSE_PRED_RESULT")

    # 9. 일부 출력
    predictions.select(
        col("MEME_PRICE_PER_SUPPLY_PYEONG"),
        col("JEONSE_PRICE_PER_SUPPLY_PYEONG"),
        col("PREDICTED_JEONSE_PRICE")
    ).show()

    return predictions
