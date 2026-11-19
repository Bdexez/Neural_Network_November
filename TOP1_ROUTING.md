# Routage avec un expert sélectionné

Avec `top_k: 1`, le mode historique `top1_gate: "normalized"` divise la probabilité de l'expert retenu par elle-même. Le coefficient appliqué à cet expert vaut donc toujours un. La sélection top-1 est discrète et, à l'arrondi près, la perte de tâche ne donne pas de gradient au routeur par ce coefficient. Les pertes d'équilibrage et de régularisation restent actives lorsqu'elles ont un poids non nul.

Le mode optionnel `top1_gate: "probability"` conserve la probabilité de l'expert retenu comme coefficient. La perte de tâche peut alors modifier le routeur, même si la sélection de l'expert reste discrète. L'expert partagé reste ajouté séparément. Ce choix modifie les prédictions et ne garantit pas une amélioration ; il exige `top_k: 1` et au moins deux experts. Les configurations et checkpoints sans ce champ conservent `normalized`.

Sur une tâche jouet à deux experts fixes, le routeur commence par choisir le mauvais expert pour deux entrées. En 80 étapes SGD, la perte passe de 3,5374 à 0,00185 avec `probability` ; elle reste à 4 avec `normalized`. Cet essai isole le signal de gradient, sans démontrer de qualité sur un problème réel.

## Comparaison synthétique

Les deux variantes utilisent le preset `tiny` avec quatre experts et `top_k: 1`. Pour chaque seed 42, 43 et 44, elles partent des mêmes poids initiaux et voient les mêmes lots synthétiques et décisions de retrait des modalités. Chaque entraînement exécute 80 étapes de huit exemples, avec un taux principal de 0,001, dix étapes d'échauffement et un dropout des modalités de 0,3. Le checkpoint final est retenu sans sélection sur le test.

Les mesures utilisent les mêmes 128 exemples synthétiques distincts de l'entraînement, répartis en 16 lots avec les seeds 3100000 à 3100015. Les nombres ci-dessous sont des moyennes sur trois seeds d'entraînement ; les écarts-types sont des écarts-types d'échantillon.

| Mesure | `normalized` | `probability` |
| --- | ---: | ---: |
| Accuracy, séries seules | 18,49 % ± 7,83 points | 15,89 % ± 4,30 points |
| MAE, séries seules | 0,2886 ± 0,0110 | 0,2895 ± 0,0101 |
| MAE, texte + image + séries | 0,2806 ± 0,0138 | 0,2836 ± 0,0174 |
| Perte de texte, trois modalités | 2,4961 ± 0,1463 | 2,5072 ± 0,1364 |

La porte probabiliste ne montre pas de gain général dans cet essai : les MAE sont proches et l'accuracy moyenne avec les séries seules est plus basse. Les trois entraînements sont courts, l'accuracy sur les séries est faible et le nombre de seeds est trop limité pour estimer précisément la variabilité. L'accuracy avec texte atteint 100 % dans les deux variantes sur ce générateur, dont le texte contient la classe. Ces mesures ne démontrent pas de généralisation à des données réelles. Le mode historique reste le défaut.

Les mesures complètes pour les sept combinaisons, commandes, durées et versions sont conservées dans [top1-routing-study.json](artifacts/top1-routing-study.json). Les checkpoints restent locaux sous `runs/`.

## Répartition des experts

Les six checkpoints ont été réévalués sur les mêmes exemples avec les diagnostics de routage. Le nombre effectif d'experts est l'exponentielle de l'entropie des parts de sélection : il vaut quatre pour une répartition uniforme entre les quatre experts et un si un seul est sélectionné. Le tableau donne la moyenne de cette mesure par checkpoint, puis sur les trois seeds ; il ne calcule pas l'entropie des parts moyennées entre seeds.

| Modalités | Mode | Couche 1 | Couche 2 | Couche 3 |
| --- | --- | ---: | ---: | ---: |
| Séries seules | `normalized` | 2,11 | 2,02 | 2,54 |
| Séries seules | `probability` | 2,00 | 1,92 | 1,67 |
| Texte + image + séries | `normalized` | 3,98 | 3,93 | 3,97 |
| Texte + image + séries | `probability` | 3,60 | 3,48 | 3,18 |

Dans cet essai, la sélection est donc plus concentrée avec `probability`, particulièrement dans la troisième couche. Cette observation ne démontre pas que la concentration cause les écarts de qualité. Sans texte, le décodeur utilise deux requêtes de tâche par exemple : les séries seules représentent ici 256 tokens routés, contre 2 688 avec les trois modalités. Les deux conditions ne portent donc pas sur les mêmes positions de décodeur. L'expert partagé est exclu de ces statistiques.

Les parts par expert, ratios de charge maximale, écarts-types et effectifs pour chaque condition et seed figurent dans [top1-expert-routing.json](artifacts/top1-expert-routing.json). Les six entraînements répétés pour ces mesures ont produit des tenseurs de poids identiques bit à bit aux entraînements de la comparaison précédente ; les mesures de qualité recalculées sont inchangées.

```bash
.venv/bin/python tools/compare_top1_routing.py --output-dir runs/nouveau-top1
```

Pour affiner un checkpoint existant avec ce mode :

```bash
.venv/bin/python -m chimera train \
  --init-from runs/workstation/model.pt \
  --model-overrides configs/top1-probability-overrides.json \
  --steps 100 --output runs/top1-refinement
```

Le transfert conserve les tenseurs de poids compatibles, sans ajouter de paramètres ; le changement de nombre d'experts sélectionnés et de porte change immédiatement la fonction calculée. Il convient donc de comparer et de recalibrer le modèle affiné sur les données visées.
