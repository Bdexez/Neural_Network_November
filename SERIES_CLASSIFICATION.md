# Classification temporelle standardisée

Les changements de niveau et d'unités peuvent modifier la classification du modèle, même lorsque la forme du signal reste identique. Une tête facultative ajuste un classifieur linéaire sur la représentation de l'encodeur temporel après standardisation de chaque variable. Elle ne modifie pas les poids du checkpoint et n'agit que sur des exemples dont la seule modalité est `series`.

Pour chaque exemple et variable, la moyenne et l'écart type sont calculés sur les observations finies et les positions valides. L'écart type est borné inférieurement par `scale_floor`. Les valeurs absentes restent absentes ; le padding n'entre ni dans les statistiques ni dans la moyenne finale des représentations temporelles. L'encodeur du checkpoint, en mode évaluation et sans gradient, traite cet historique standardisé. Une régression ridge multiclasse ajuste ensuite les scores sur les étiquettes one-hot : elle régularise les poids, mais pas le biais. Les scores sont des **scores de classification, pas des probabilités calibrées**.

La standardisation supprime les informations de niveau et d'échelle : elle peut être utile si ces quantités représentent seulement un changement d'unités, et nuisible si elles portent une information pertinente pour la classe. Elle n'assure pas une invariance numérique exacte lorsque le plancher est actif, que des valeurs sont arrondies à l'entrée ou que la précision est insuffisante. Une variable entièrement absente reste représentée comme absente. Le nombre d'exemples par classe, le plancher, la précision AMP, les coefficients et les empreintes sont enregistrés dans le JSON de la tête.

## Utilisation

Préparer un JSONL de séries contenant `class_label`, distinct des jeux de sélection et de test. Les lignes non étiquetées sont ignorées ; chaque classe de la configuration doit posséder au moins un exemple étiqueté. Les autres modalités sont refusées pour éviter de laisser croire que la tête les exploite.

```bash
.venv/bin/python -m chimera refine-series-classifier \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --data data/series-head-train.jsonl --ridge 0.001 --scale-floor 1e-4 \
  --output runs/series-classifier.json

.venv/bin/python -m chimera evaluate \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --data data/series-head-test.jsonl --series-classifier runs/series-classifier.json \
  --output runs/series-head-evaluation.json

.venv/bin/python -m chimera predict \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --data data/new-series.jsonl --series-classifier runs/series-classifier.json \
  --output runs/series-head-predictions.jsonl
```

La prédiction conserve `class_id`, `class_probabilities` et `forecast` du modèle d'origine, et ajoute `series_classifier_class_id` et `series_classifier_scores`. L'évaluation conserve ses métriques habituelles et ajoute `series_classifier_accuracy`, l'effectif et une confusion creuse sous forme de triplets `actual`, `predicted`, `count`. `series_classifier_data_reused` vaut `true` si les octets JSONL sont identiques à ceux de l'ajustement, sinon `null` : des fichiers différents peuvent partager des exemples. La tête est liée au SHA-256 exact du checkpoint, à sa configuration et à la précision AMP utilisée à l'ajustement. L'extraction nécessite un passage supplémentaire dans l'encodeur temporel.

## Comparaison synthétique indépendante

Le checkpoint `tiny` du contrôle joint-affine est fixé après 180 étapes. La tête est ajustée par ridge sur 3 072 exemples de séries seulement, en alternant des historiques de 8, 16 et 32 pas. Les 512 exemples de test par condition utilisent d'autres seeds ; les transformations et masques d'une même ligne sont appariés. Les observations manquantes sont masquées indépendamment des valeurs ; l'ajustement ne comportait pas ces masques.

| Historique | Transformation | Accuracy du modèle | Accuracy de la tête |
| --- | --- | ---: | ---: |
| 8 pas | Identité | 69,53 % | 51,37 % |
| 8 pas | Décalage +20 | 17,38 % | 51,37 % |
| 16 pas | Identité | 73,83 % | 55,66 % |
| 16 pas | Décalage +20 | 17,38 % | 55,66 % |
| 32 pas | Identité | 93,55 % | 61,72 % |
| 32 pas | Décalage +20 | 17,38 % | 61,72 % |
| 8 pas, 30 % masqués | Identité | 69,53 % | 43,55 % |
| 8 pas, 30 % masqués | Décalage +20 | 17,38 % | 43,55 % |
| 16 pas, 30 % masqués | Identité | 73,05 % | 48,83 % |
| 16 pas, 30 % masqués | Décalage +20 | 17,38 % | 48,83 % |

Dans cet essai, la tête améliore les conditions décalées mais perd nettement en accuracy sur les données d'origine. Ses prédictions ne changent pas pour les cinq transformations positives testées sur les mêmes exemples ; cela ne constitue pas une garantie pour d'autres valeurs, d'autres précisions ou un plancher actif. Les données sont synthétiques, un seul checkpoint est étudié et les capacités multimodales ne sont pas évaluées par cette tête. Les [résultats détaillés](artifacts/series-classifier-study.json) incluent toutes les transformations, les confusions, les paramètres et l'empreinte du checkpoint.

```bash
.venv/bin/python tools/benchmark_series_classifier.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output runs/series-classifier-study.json
```
