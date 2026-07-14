### Problema
Predecir la **probabilidad de que un cliente incumpla** en el pago de un préstamo (`default = 1`)  
basándose en variables financieras y crediticias históricas.

### Objetivo
Maximizar el **AUC-ROC** en el conjunto de test oculto de Kaggle utilizando  
un modelo de **Regresión Logística** con transformación WOE (Weight of Evidence).

### Dataset
| Conjunto | Registros | Variables | Target |
|:---:|:---:|:---:|:---:|
| Train | ~10,000 | 11 + target | `default` (0/1) |
| Test  | ~6,000  | 11          | — |

### Metodología
1. Análisis exploratorio (EDA)
2. Pre-procesamiento y limpieza
3. WOE Binning + Selección por IV
4. Balanceo con SMOTE
5. Regresión Logística + GridSearchCV
6. Evaluación y generación de submission
