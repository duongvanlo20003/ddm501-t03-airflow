# Tutorial 03's image: stock Airflow plus the two libraries the DAG imports.
#
FROM apache/airflow:2.8.4-python3.11

USER airflow

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      "pandas==2.1.4" \
      "pyarrow==14.0.2" \
      "mlflow==2.19.0" \
      "scikit-learn==1.6.0"
