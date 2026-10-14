"""Evaluate comparison model on 2023-2024, using existing features.csv."""
import argparse
import csv
import json
from pathlib import Path
from pyspark import StorageLevel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.window import Window
import common
from common import F, session, score_model

OUT = Path('output/task2/comparison')
parser = argparse.ArgumentParser()
parser.add_argument('--fraction', type=float, default=1.0, help='Fraction of test station-times; 1.0 is full test')
parser.add_argument('--auc', action='store_true', help='Calculate ROC-AUC and PR-AUC')
args = parser.parse_args()
if not 0 < args.fraction <= 1:
    parser.error('fraction must be in (0,1]')
meta = json.loads((OUT / 'training_metadata.json').read_text())
model = json.loads((OUT / 'lr_model.json').read_text())
common.THRESHOLDS = meta['thresholds_mm']
expected_thresholds = [0.0, 0.2, 0.4, 1.0, 3.0, 24.2]
if common.THRESHOLDS != expected_thresholds or meta.get('label_rule') != '>':
    raise ValueError('Retrain with 02_train_with_zero.py first: expected six thresholds and strict > labels')

spark = session()
pred = None
try:
    data = common.dataset(spark, [2023, 2024], args.fraction)
    # common.dataset supplies label = (future_rain_30m > threshold_mm).
    pred = score_model(data, model).persist(StorageLevel.MEMORY_AND_DISK)
    pclip = F.greatest(F.lit(1e-15), F.least(F.lit(1-1e-15), F.col('p')))
    summaries = pred.groupBy('threshold_mm').agg(
        F.count('*').alias('n'), F.avg('label').alias('positive_rate'),
        F.avg('p').alias('mean_predicted_probability'),
        F.avg((F.col('p') - F.col('label'))**2).alias('brier_score'),
        F.avg(-F.col('label')*F.log(pclip) - (1-F.col('label'))*F.log(1-pclip)).alias('log_loss')
    ).orderBy('threshold_mm').collect()
    rows = [r.asDict() for r in summaries]
    for row in rows:
        row['areaUnderROC'] = None
        row['areaUnderPR'] = None
        if args.auc and 0 < row['positive_rate'] < 1:
            subset = pred.filter(F.col('threshold_mm') == row['threshold_mm'])
            for metric in ['areaUnderROC', 'areaUnderPR']:
                evaluator = BinaryClassificationEvaluator(
                    labelCol='label', rawPredictionCol='score', metricName=metric)
                row[metric] = evaluator.evaluate(subset)
    print('\n=== PERFORMANCE ON 2023-2024 ===')
    print(f"{'Threshold':>10} {'Test rows':>12} {'Positive %':>12} {'ROC-AUC':>10} {'PR-AUC':>10} {'Brier':>12}")
    for r in rows:
        roc = 'N/A' if r['areaUnderROC'] is None else f"{r['areaUnderROC']:.4f}"
        pr = 'N/A' if r['areaUnderPR'] is None else f"{r['areaUnderPR']:.4f}"
        print(f"{r['threshold_mm']:>10.2f} {r['n']:>12,} {100*r['positive_rate']:>11.2f}% {roc:>10} {pr:>10} {r['brier_score']:>12.6f}")
    with (OUT / 'metrics_2023_2024.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cal = (pred.withColumn('probability_bin', F.least(F.floor(F.col('p')*20), F.lit(19)))
           .groupBy('threshold_mm', 'probability_bin')
           .agg(F.count('*').alias('n'), F.avg('p').alias('mean_probability'), F.avg('label').alias('event_rate')))
    cal.toPandas().to_csv(OUT / 'calibration_2023_2024.csv', index=False)
    w = Window.partitionBy('station_id', 'prediction_time').orderBy('threshold_mm')
    violations = (pred.withColumn('previous_p', F.lag('p').over(w))
                  .agg(F.sum(F.when(F.col('p') > F.col('previous_p') + 1e-10, 1).otherwise(0)).alias('n'))
                  .first()['n'])
    print('Threshold monotonicity violations:', violations)
    print('Results saved in:', OUT)
finally:
    if pred is not None: pred.unpersist()
    spark.stop()
