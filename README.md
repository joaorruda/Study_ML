# Study_ML

# Credit Card Fraud Detection

Projeto de machine learning para detecção de fraudes em transações de cartão de crédito, desenvolvido com foco em boas práticas de pré-processamento, tratamento de desbalanceamento de classes e avaliação de modelos.

---

## Experimentos realizados

### Experimento 1 — Logistic Regression (baseline)

Modelo inicial para estabelecer uma baseline de desempenho.

**Problemas identificados e corrigidos durante o desenvolvimento:**

- `StandardScaler` não estava sendo aplicado, deixando `Amount` e `Time` sem normalização
- `precision_recall_curve` recebia `y_pred` binário em vez de `y_score` (probabilidades), gerando uma curva degenerada com apenas 3 pontos

**Pipeline final:**

```
Carga dos dados → StandardScaler (Amount + Time) → train_test_split (70/30) → LogisticRegression → classification_report → Curva PR
```

---

### Experimento 2 — Random Forest + SMOTE

Evolução do baseline com modelo mais robusto e tratamento do desbalanceamento de classes (~0.17% de fraudes).

**Melhorias implementadas:**

- `Amount` substituído por `Amount_Log` (`np.log1p`) para reduzir o efeito de outliers
- SMOTE aplicado **apenas no treino** após o split, evitando data leakage
- `StandardScaler` com `fit_transform` no treino e `transform` no teste

**Pipeline final:**

```
Carga dos dados → Amount_Log → drop(Amount) → train_test_split (70/30 ou 80/20)
→ StandardScaler (Amount_Log + Time) → SMOTE (só no treino)
→ RandomForestClassifier → classification_report → Curva PR
```

---

## Estrutura do projeto

```
.
├── frauds_LR.py          # Experimento 1 — Logistic Regression
├── frauds_RF.py          # Experimento 2 — Random Forest + SMOTE
├── .env                  # Variáveis de ambiente (não versionado)
├── .gitignore
└── README.md
```

---

### Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — disponível no Kaggle.

---

## Lições aprendidas

| Problema | Impacto | Solução |
|---|---|---|
| Scaler sem `fit`/`transform` separados | Data leakage | `fit_transform` no treino, `transform` no teste |
| SMOTE antes do split | Data leakage nas métricas | Aplicar SMOTE somente após o split |
| `predict` em vez de `predict_proba` na curva PR | Curva degenerada | Usar `predict_proba(x_test)[:, 1]` |
| `Amount` sem transformação | Escala desproporcional vs V1–V28 | `np.log1p` + StandardScaler |
| Threshold fixo em 0.5 | Recall de fraude baixo | Otimizar threshold pela curva PR |

---

## Próximos passos

- [ ] Threshold ótimo via curva Precision-Recall
- [ ] Validação cruzada estratificada (5-fold, métrica AUC-PR)
- [ ] XGBoost com `scale_pos_weight`
- [ ] Comparação de curvas PR e ROC entre modelos
- [ ] Feature importance do Random Forest
