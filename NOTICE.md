# Происхождение

Задача, формат DataPoint, порядок признаков, данные и определение метрики происходят
из starter kit соревнования Alfa Connectome / Wunder Fund:
https://wundernn.io/connectome/docs/data_overview

`src/utils.py` включает callback/scoring utilities проекта; `src/train_*.py`,
`src/supervised_ae_mlp.py`, runtime и C-препроцессинг — использованный и разработанный
в ходе участия код. Источники и SHA256 исходных файлов/чекпойнтов указаны в
`reports/source_provenance.json`. Для переносимости изменено определение корня
проекта, а full-data training loop отделён от защитных проверок старых экспериментов.

Supervised autoencoder вдохновлён подходом Yirun Zhang к Jane Street supervised AE,
но не является его воспроизведением. Зависимости PyTorch, NumPy, SciPy, PyArrow,
ONNX и ONNX Runtime распространяются по собственным лицензиям.

Данные соревнования и сторонние статьи/книги не входят в этот проект. Этот пакет
не назначает новую лицензию материалам организаторов или сторонним компонентам.
