"""Train 2017-2021, validate 2022. Uses existing prepared features.csv."""
import argparse
import json
from pathlib import Path
from pyspark import StorageLevel
from pyspark.ml.functions import vector_to_array
import common
from common import F, session, pipeline, export_model, score_model

THRESHOLDS = [0.0, 0.2, 0.4, 1.0, 3.0, 24.2]
common.THRESHOLDS = THRESHOLDS  # dataset() looks up this module variable at call time
OUT = Path('output/task2/comparison')
OUT.mkdir(parents=True, exist_ok=True)
MODEL = OUT / 'lr_model.json'

parser = argparse.ArgumentParser()
parser.add_argument('--fraction', type=float, default=0.01)
parser.add_argument('--validation-fraction', type=float, default=0.02)
args = parser.parse_args()
if not (0 < args.fraction <= 1 and 0 < args.validation_fraction <= 1):
    parser.error('fractions must be in (0,1]')
if not Path(common.FEATURE_PATH).is_file():
    raise FileNotFoundError(f'Missing {common.FEATURE_PATH}; cannot skip feature preparation')

# common.dataset already implements the assignment's strict '>' label rule.

spark = session()
train = val = None
try:
    train = common.dataset(spark, list(range(2017, 2022)), args.fraction).persist(StorageLevel.MEMORY_AND_DISK)
    val = common.dataset(spark, [2022], args.validation_fraction).persist(StorageLevel.MEMORY_AND_DISK)
    counts = train.groupBy('threshold_mm').agg(F.count('*').alias('n'), F.avg('label').alias('positive_rate')).collect()
    if len(counts) != len(THRESHOLDS) or val.limit(1).count() == 0:
        raise ValueError('Missing thresholds or validation data')
    print('Training 2017-2021; validation 2022; test 2023-2024', flush=True)
    print('Training counts:', counts, flush=True)
    best_loss = float('inf')
    for reg in [0.001, 0.01]:
        print('Fitting regParam =', reg, flush=True)
        fitted = pipeline(reg).fit(train)
        pred = fitted.transform(val).withColumn('p', vector_to_array('probability')[1])
        p = F.greatest(F.lit(1e-15), F.least(F.lit(1-1e-15), F.col('p')))
        loss = pred.agg(F.avg(-F.col('label')*F.log(p)-(1-F.col('label'))*F.log(1-p)).alias('loss')).first()['loss']
        print('Validation log loss:', loss, flush=True)
        if loss is not None and loss < best_loss:
            # Verify before saving the selected model.
            check = fitted.transform(val.limit(1000)).withColumn('native_p', vector_to_array('probability')[1])
            # export_model writes the file; score_model validates its equivalence.
            payload = export_model(fitted, str(MODEL))
            difference = score_model(check, payload).agg(F.max(F.abs(F.col('native_p')-F.col('p'))).alias('difference')).first()['difference']
            if difference is None or difference > 1e-8:
                raise ValueError(f'Export verification failed: {difference}')
            best_loss = loss
            metadata = {
                'train_years': list(range(2017, 2022)), 'validation_years': [2022],
                'test_years': [2023, 2024], 'thresholds_mm': THRESHOLDS,
                'label_rule': '>', 'regParam': reg, 'validation_log_loss': loss,
                'sample_fraction': args.fraction, 'validation_fraction': args.validation_fraction,
                'training_positive_rates': {str(r['threshold_mm']): r['positive_rate'] for r in counts}
            }
            (OUT / 'training_metadata.json').write_text(json.dumps(metadata, indent=2))
            print('Selected new best model; verification error:', difference, flush=True)
    print('Finished. Model:', MODEL, flush=True)
finally:
    if train is not None: train.unpersist()
    if val is not None: val.unpersist()
    spark.stop()
