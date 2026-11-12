# Historiques courts et observations manquantes

Le modèle accepte des historiques incomplets, mais ses premiers entraînements utilisaient toujours 32 pas entièrement observés. Le diagnostic sur 64 exemples synthétiques a montré que sa MAE avec les séries seules passait de 0.2279 à 0.5680 en ne gardant que les 16 derniers pas, et à 0.6938 avec 8 pas. Masquer 30 % des valeurs donnait une MAE de 0.4327. Ce diagnostic a servi à choisir un axe d'apprentissage ; le test final utilise d'autres seeds.

## Augmentation à l'entraînement

L'augmentation est **désactivée par défaut**. Elle s'applique uniquement aux historiques, avant la sélection des modalités. Les textes, images, classes et futurs cibles restent inchangés.

- `--series-crop-probability` sélectionne les exemples dont la longueur est tirée uniformément entre `--series-min-history` et leur longueur disponible. Les positions valides les plus récentes sont conservées. Le minimum vaut 8 par défaut et est borné par la longueur réellement disponible.
- `--series-missing-rate` donne la proportion de valeurs à masquer dans les exemples sélectionnés par `--series-mask-probability`, qui vaut 0.5 par défaut. Le masquage utilise des NaN, déjà pris en charge par l'encodeur temporel.
- Les probabilités et le taux appartiennent à `[0, 1]`. La longueur minimale est strictement positive. Une valeur de 1 pour le taux et la probabilité de masquage cache toutes les valeurs, tout en gardant un masque de positions temporelles valide.

Avec une probabilité de troncature de 0.5, une probabilité de masquage de 0.5 et un taux de valeurs manquantes de 0.2, environ un quart des exemples conserve un historique complet sans masquage ajouté. La longueur minimale porte sur les positions temporelles valides, pas sur le nombre de valeurs finies après masquage.

Les deux perturbations utilisent des générateurs dédiés, indexés par la seed et l'index global du micro-lot. Elles ne consomment pas le RNG du dropout du réseau. La partition DDP utilise des index distincts par rang. Les réglages sont sauvegardés dans le checkpoint, restaurés lorsqu'ils sont omis à la reprise et vérifiés lorsqu'ils sont explicitement fournis. Changer ces réglages nécessite un nouvel entraînement avec `--init-from`.

```bash
.venv/bin/python -m chimera --threads 8 train \
  --init-from runs/workstation-refined/model.pt \
  --steps 180 --seed 314 --batch-size 8 --lr 0.00006 --warmup 10 \
  --modality-policy balanced --series-crop-probability 0.5 \
  --series-min-history 8 --series-missing-rate 0.2 --series-mask-probability 0.5 \
  --save-every 180 --eval-every 180 --output runs/workstation-robust
```

`metrics.jsonl` compte, sur le rang 0, les exemples avec série, les positions temporelles valides, les valeurs possibles et les valeurs finies effectivement observées après augmentation et sélection des modalités. La validation du trainer utilise les données de validation fournies, sans leur appliquer l'augmentation d'entraînement.

## Comparaison contrôlée

Le modèle de départ contient 549 485 298 paramètres. Deux branches partent des mêmes poids : un témoin sans augmentation et une branche avec les réglages ci-dessus. Elles reçoivent les mêmes exemples synthétiques, seeds 314 à 493, avec 180 étapes de 8 exemples, le même optimiseur neuf, le même planning et la même politique de modalités. Les checkpoints finaux sont comparés ; aucune étape n'est choisie en fonction du test.

Le budget en étapes et exemples est identique ; l'information observée diffère volontairement. Les données de test sont synthétiques : le masquage est indépendant des valeurs. Ces essais ne prouvent pas la robustesse sur des données réelles ou lorsque l'absence d'une observation dépend de sa valeur.

```bash
# Construire le témoin avec le même budget et les mêmes exemples
.venv/bin/python -m chimera --threads 8 train \
  --init-from runs/workstation-refined/model.pt \
  --steps 180 --seed 314 --batch-size 8 --lr 0.00006 --warmup 10 \
  --modality-policy balanced --save-every 180 --eval-every 180 \
  --output runs/workstation-robust-control

# Comparer les trois checkpoints sur les mêmes cas de test
.venv/bin/python tools/benchmark_robustness.py \
  --reference-checkpoint runs/workstation-refined/model.pt \
  --control-checkpoint runs/workstation-robust-control/latest.pt \
  --checkpoint runs/workstation-robust/model.pt \
  --output runs/robustness-benchmark.json
```

Le test final utilise 256 exemples, seeds 9000000 à 9000015, par lots de 16. Un second flux, seeds 9100000 à 9100015, produit les masques. Pour une même longueur, les valeurs masquées à 10 % forment un sous-ensemble de celles masquées à 30 %. Les modèles reçoivent les mêmes masques. Les conditions raccourcies conservent les derniers pas et le même futur cible. La persistance utilise la dernière valeur finie de chaque variable, ou zéro si elle est entièrement absente.

## Résultats et compromis

MAE avec les séries seules, sur les mêmes 256 exemples :

| Historique | Poids de départ | Témoin | Augmentation |
| --- | ---: | ---: | ---: |
| Complet, 32 pas | 0.2319 | **0.2110** | 0.2203 |
| 16 derniers pas | 0.5363 | 0.5258 | **0.5049** |
| 8 derniers pas | 0.6648 | 0.6612 | **0.6048** |
| 10 % de valeurs masquées | 0.2760 | 0.2661 | **0.2427** |
| 30 % de valeurs masquées | 0.3933 | 0.3910 | **0.3250** |
| 16 pas, 30 % masqués | 0.5577 | 0.5508 | **0.5349** |
| 8 pas, 30 % masqués | 0.6508 | 0.6480 | **0.6156** |

MAE avec texte, image et série :

| Historique de la série | Poids de départ | Témoin | Augmentation |
| --- | ---: | ---: | ---: |
| Complet, 32 pas | 0.2305 | **0.2048** | 0.2573 |
| 16 derniers pas | 0.4552 | 0.4797 | **0.4116** |
| 8 derniers pas | 0.5708 | 0.5918 | **0.5637** |
| 10 % de valeurs masquées | 0.2542 | **0.2367** | 0.2699 |
| 30 % de valeurs masquées | 0.3571 | 0.3595 | **0.3268** |
| 16 pas, 30 % masqués | 0.4959 | 0.5051 | **0.4655** |
| 8 pas, 30 % masqués | 0.5741 | 0.5900 | **0.5688** |

Par rapport au témoin, l'augmentation réduit la MAE de **16.9 %** avec les séries seules masquées à 30 %. Avec 8 pas, l'accuracy de classification passe de **21.09 % à 47.27 %** ; avec 30 % de valeurs masquées, de 48.83 % à 58.59 %. Elle coûte toutefois de la précision sur les exemples complets : la MAE avec les trois modalités augmente de **25.6 %**, et l'accuracy sur les séries seules complètes passe de 97.66 % pour le témoin à 83.98 % pour la branche augmentée. Le défaut reste donc sans augmentation, et les deux versions sont conservées.

Le témoin améliore aussi les données complètes par rapport au checkpoint de départ : accuracy des séries seules de 86.33 % à **97.66 %**, MAE des trois modalités de 0.2305 à **0.2048**. Ses poids sont dans `runs/workstation-robust-control/model.pt` ; ceux de la branche augmentée dans `runs/workstation-robust/model.pt`. Ces fichiers restent locaux et sont exclus de Git.

**La prévision avec très peu d'historique reste une limite.** Pour 8 pas complets, la persistance atteint une MAE de 0.5211, meilleure que les 0.6048 du modèle augmenté avec les séries seules. L'amélioration par rapport au réseau témoin ne suffit donc pas à dépasser cette référence simple dans tous les cas. Ces comparaisons utilisent une seule seed d'entraînement ; leur variabilité entre entraînements n'est pas mesurée.

Un contrôle supplémentaire sur 64 autres exemples, seeds à partir de 10000000, vérifie les sept combinaisons de modalités. L'accuracy reste à 100 % avec texte ou image pour les deux branches sur ce jeu facile ; le texte contient explicitement la classe. La perte de texte avec les trois modalités baisse de 0.00306 à 0.00154 pour le témoin et à 0.00175 avec augmentation. Cela ne démontre pas une capacité générale de langage ou de vision.

Les rapports complets sont [robustness-benchmark.json](artifacts/robustness-benchmark.json), [robustness-clean-benchmark.json](artifacts/robustness-clean-benchmark.json) et [robustness-control-clean-benchmark.json](artifacts/robustness-control-clean-benchmark.json). Le [rapport d'entraînement](artifacts/robustness-training.json) conserve le planning et les compteurs : les deux branches voient 816 exemples contenant une série après sélection des modalités ; la branche augmentée conserve en moyenne 25.84 pas et masque 10.34 % des valeurs possibles restantes.

Les mesures portent sur les sorties brutes, sans calibration réutilisée depuis les anciens poids. Une nouvelle version des poids doit recevoir sa propre [calibration](CALIBRATION.md).

## Vérifications

Les **68 tests passent**. Ils vérifient notamment les cibles intactes, les masques avec trous et padding, le maintien des observations récentes, la conservation des entrées originales et du RNG global, l'équivalence entre une troncature masquée à l'entraînement et une série physiquement plus courte à l'inférence, ainsi que la reprise exacte avec dropout, accumulation et modalités équilibrées.

Un test à deux processus CPU exécute aussi l'augmentation avec des modalités différentes entre rangs et une validation finale. La CI reprend ce scénario sur Python 3.11 et 3.14. Aucun résultat GPU n'est revendiqué.

Les preuves locales sont dans [robustness-tests.txt](artifacts/robustness-tests.txt), [robustness-tests.json](artifacts/robustness-tests.json) et [robustness-acceptance.json](artifacts/robustness-acceptance.json).
