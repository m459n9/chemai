# ChemAI: Predict the Cure

Решение хакатона по предсказанию биологической активности химических соединений
против вируса гриппа. Регрессия трёх таргетов: IC50 (mM), CC50 (mM), SI = CC50 / IC50.

## Результат на Kaggle

| Public Score | Позиция | Команда | Сабмитов |
|---|---|---|---|
| **290.09325** | 7 / leaderboard | Тима Тимы | 51 |

Скриншот: [`kaggle_result.png`](kaggle_result.png)

OOF-метрики финального пайплайна (5-fold, валидация на оригинальных строках):

| Таргет | log-RMSE |
|---|---|
| IC50 | 1.2665 |
| CC50 | 0.9618 |
| SI | 0.9381 |
| **Mean** | **1.0555** |

## Структура репозитория

```
.
├── chemai_final.ipynb         # финальный пайплайн (EDA + решение, ~49 ячеек)
├── train.csv                  # 751 строка, 210 признаков + 3 таргета
├── test.csv                   # 250 строк, 210 признаков
├── sample_submission.csv      # формат submission
├── submission.csv             # итоговое предсказание (public 290.09)
├── kaggle_result.png          # скриншот лидерборда
├── requirements.txt
└── README.md
```

Ноутбук находит данные автоматически — поддерживается и плоская структура
(данные в корне), и `./data/` поддиректория. Реализовано через
`DATA_DIR = './data' if os.path.exists('./data/train.csv') else '.'`.

## Подход

Стэкинг двух tree-based моделей через положительную Ridge-регрессию:

1. **Feature engineering.** Lipinski + Veber rules, нормализованные по размеру
   молекулы свойства (logP/MW, TPSA/MW и т.д.), агрегаты по семействам RDKit
   дескрипторов (VSA, fr_, Chi, BCUT, SlogP). Ключевая часть — **флаги
   цензурирования CC50**: `is_assay_ceiling` (CC50 ≥ 3284), `is_round_cc50`
   (значения 100/200/250/300/500), `is_si_one` (IC50 == CC50, неактивные
   соединения).
2. **Дедупликация** повторных биологических измерений (близкие в признаковом
   пространстве строки, евклидово расстояние < 0.01 после `StandardScaler`)
   с усреднением таргетов в `log1p`. На train сокращает 751 → 630 строк,
   убирает основной шум в CC50.
3. **Mixup-аугментация** `n=300, α=0.4` для борьбы с малой выборкой.
4. **Отдельный clean-subset для SI** — обрезка по 98-му перцентилю
   (тяжёлый правый хвост — шум, не сигнал).
5. **Базовые модели:** CatBoost + ExtraTrees через 5-fold OOF на
   `(train_fold ∪ mixup)`, валидация только на исходных строках.
6. **Stacking:** `Ridge(alpha=1.0, fit_intercept=False, positive=True)`
   отдельно на каждый таргет.

## Воспроизведение

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute chemai_final.ipynb \
    --output chemai_final.executed.ipynb
```

либо открыть `chemai_final.ipynb` в Jupyter и выполнить «Run All».

Все случайные источники зафиксированы константой `SEED = 42`:
`numpy.random.seed`, `KFold(random_state=SEED)`, `CatBoost(random_seed=SEED)`,
`ExtraTreesRegressor(random_state=SEED)`, генератор mixup
(`numpy.random.default_rng(SEED)`).

При повторных прогонах с чистого kernel — расхождение между submissions
порядка 1e-11 (машинное ε), на финальный RMSE не влияет.

## Гиперпараметры

CatBoost и ExtraTrees параметры найдены через Optuna на исходном
(не аугментированном) train-наборе. Тюнить совместно с mixup-аугментацией
некорректно — параметры подгоняются под синтетику, а не под целевое распределение.

| Модель | Ключевые параметры |
|---|---|
| CatBoost | `iterations=939, depth=5, lr=0.1015, l2_leaf_reg=7.40, subsample=0.68, bootstrap=Bernoulli` |
| ExtraTrees | `n_estimators=1000, max_depth=10, min_samples_leaf=4, max_features=0.383` |

## Что было отброшено по результатам экспериментов

| Подход | OOF Δ | Public Δ | Решение |
|---|---|---|---|
| LightGBM-huber на SI-голову | −0.01 | +1 | отброшено |
| Per-target Optuna вместо общего | ≈0 | +0.4 | отброшено |
| PCA-фичи / Smart PCA Augmentation | 0 | +1 | отброшено |
| Формула SI = pred(CC50) / pred(IC50) | −0.01 | +2–3 | отброшено |
| Каскад IC50 → CC50 | нестабильно | разная | отброшено |
| Adversarial dropping drift-фичей | +0.005 | ±0 | отброшено |
| Multi-seed bagging (5–8 сидов) | ±шум | ±0.5 | отброшено |
| pIC50 (−log10) вместо log1p | −0.02 | +1 | отброшено |
| Mixup `n=500` | ±шум | +1 | отброшено |
