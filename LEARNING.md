# Apprentissage et modalités absentes

Ce cycle ajoute un fine-tuning depuis des poids existants, une exposition équilibrée aux modalités et un alignement optionnel tenant compte des classes. Les checkpoints précédents restent chargeables ; aucun paramètre n'est ajouté au réseau.

## Comparaison contrôlée des politiques

Les deux politiques partent des mêmes initialisations `tiny` et reçoivent les mêmes lots synthétiques avant sélection des modalités. Chaque exécution utilise 280 étapes, des lots de 16 exemples, un taux de 0.001, un échauffement de 10 étapes et les seeds 42, 43 et 44. Le dropout de référence vaut 0.3. Le checkpoint final est évalué sur les mêmes 256 exemples indépendants, à partir de la seed 3000000.

Le budget en étapes et exemples est identique ; le nombre de tokens de chaque modalité et le coût de calcul diffèrent volontairement. La politique équilibrée couvre chacune des sept combinaisons une fois par cycle, avec un ordre mélangé déterministe. L'ordre est reconstruit à partir de l'index global des micro-lots, de la seed et du numéro de cycle, sans consommer le RNG d'entraînement.

| Mesure, moyenne sur trois seeds | Dropout | Équilibrage |
| --- | ---: | ---: |
| Accuracy, séries seules | 57.16 % | 58.72 % |
| MAE, séries seules | 0.2661 | 0.3089 |
| MAE, trois modalités | 0.2631 | 0.3372 |
| MAE, texte seul | 0.6054 | 0.4942 |
| Perte de texte, trois modalités | 0.1424 | 0.2458 |

**L'équilibrage ne constitue pas un gain général dans cet essai.** L'écart de classification est faible devant la variabilité entre seeds : écart-type de 9.26 points pour le dropout et 7.22 pour l'équilibrage. La prévision avec historique et la perte de texte se dégradent ; la prévision sans historique s'améliore. Le défaut reste donc le dropout, et les mesures brutes sont conservées dans [modality-study.json](artifacts/modality-study.json).

```bash
.venv/bin/python tools/compare_modality_training.py \
  --seeds 42 43 44 --steps 280 --output-dir runs/new-modality-study
```

Le dossier de sortie doit être vide. Le protocole conserve les commandes, les compteurs d'exposition, les résultats de chaque seed et leur agrégation.

## Fine-tuning du modèle existant

Le checkpoint local de 549 485 298 paramètres, entraîné pendant 240 étapes, reçoit 140 étapes supplémentaires avec une exposition équilibrée, un taux principal de 0.0001, un échauffement de 10 étapes et des lots de 8. Le planning et AdamW repartent de zéro. La seed d'entraînement reste 42 : ces données synthétiques réutilisent les exemples du premier entraînement, avec une nouvelle répartition des modalités. Les jeux de mesure sont distincts.

```bash
.venv/bin/python -m chimera --threads 8 train \
  --init-from runs/workstation/model.pt --steps 140 --batch-size 8 \
  --lr 0.0001 --warmup 10 --modality-policy balanced \
  --eval-every 140 --save-every 140 --output runs/workstation-refined
```

Sur les 64 exemples du benchmark initial :

| Mesure | Poids initiaux | Après fine-tuning |
| --- | ---: | ---: |
| Accuracy, séries seules | 46.88 % | 90.62 % |
| MAE, séries seules | 0.2967 | 0.2270 |
| MAE, texte + séries | 0.2082 | 0.1987 |
| MAE, image + séries | 0.2048 | 0.2109 |
| MAE, trois modalités | 0.2146 | 0.2201 |

La classification avec texte ou image reste à 100 % sur ce jeu synthétique facile. Le texte contient explicitement la classe ; cette accuracy ne mesure donc pas une compréhension générale. Le gain sur les séries seules s'accompagne d'une petite régression de la prévision avec certaines combinaisons. Les deux checkpoints sont conservés. Le gain concerne l'ensemble du fine-tuning ; cet essai ne permet pas de l'attribuer uniquement à la politique équilibrée.

Le rapport complet est dans [refined-benchmark.json](artifacts/refined-benchmark.json). Les poids restent locaux sous `runs/workstation-refined/`, exclus de Git.

Un second ensemble de **256 exemples**, seeds 4000000 à 4000031, confirme le gain sans utiliser les exemples du benchmark initial :

| Mesure indépendante | Poids initiaux | Après fine-tuning |
| --- | ---: | ---: |
| Accuracy, séries seules | 60.16 % | 88.67 % |
| MAE, séries seules | 0.2838 | 0.2282 |
| MAE, trois modalités | 0.2268 | 0.2321 |

L'accuracy gagne 28.52 points et la MAE baisse de 19.6 % avec les séries seules. La MAE avec les trois modalités augmente de 2.35 %. Les intervalles restent imparfaitement calibrés : sur les séries seules, la couverture passe de 86.69 % à 89.75 % pour une couverture nominale de 80 %. Le fine-tuning améliore la médiane, sans résoudre la calibration des quantiles. Le [rapport indépendant](artifacts/refined-independent-benchmark.json) conserve toutes les combinaisons.

```bash
.venv/bin/python tools/benchmark.py \
  --checkpoint runs/workstation-refined/model.pt \
  --reference-checkpoint runs/workstation/model.pt \
  --batches 32 --seed-start 4000000 --threads 8 \
  --output runs/refined-independent.json
```

## Alignement tenant compte des classes

Avec `alignment_mode: "class"`, l'objectif d'alignement considère toutes les correspondances de classe connue comme positives, en plus de la paire d'origine. Cela évite de demander à deux descriptions de la même classe de se repousser uniquement parce qu'elles occupent des lignes différentes du lot. Les labels `-100` ne créent pas de groupe artificiel, et les labels ne sont jamais injectés dans les représentations ou les prédictions.

La perte moyenne les log-probabilités des positifs dans chaque direction. Elle adapte l'idée de [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) à nos paires de modalités. L'option reste désactivée par défaut : les tests vérifient ses gradients et la compatibilité des poids, mais aucun gain de qualité après entraînement n'est revendiqué pour cette option. La température est configurable et les scores sont calculés en FP32 sous AMP.

## Vérification

Les **52 tests passent**. Les nouveaux cas vérifient les sept combinaisons, le maintien des cibles supervisées, le RNG, la partition du calendrier entre rangs, la reprise exacte de l'entraînement équilibré, le fine-tuning avec un optimiseur neuf et les gradients de l'alignement multi-positif. Un test local DDP à deux processus exécute sept étapes avec deux micro-lots accumulés et une validation finale. La CI ajoute un démarrage depuis un checkpoint dans ce scénario distribué. Le [rapport de validation](artifacts/learning-acceptance.json) regroupe les vérifications locales et les compteurs d'exposition.

## Alignement distribué entre rangs

`alignment_scope: "global"` permet, pendant l'entraînement distribué, de réunir les représentations des rangs avant de calculer l'objectif contrastif. La valeur par défaut, `"local"`, conserve le comportement historique. Par exemple, un rang qui ne reçoit qu'un seul exemple ne fournit aucune perte contrastive locale ; deux rangs ayant chacun une paire de modalités peuvent en revanche fournir deux exemples à la perte globale. Cela augmente le nombre de négatifs disponibles, une propriété étudiée notamment dans [SimCLR](https://proceedings.mlr.press/v119/chen20j.html), sans garantir un gain pour ces données ou ce modèle.

Le rassemblement accepte des tailles de lots différentes et des modalités différentes selon le rang. Dans le mode historique, seuls les exemples possédant les deux modalités sur un même rang participent à leur paire. Avec `--alignment-cross-rank-views` et des identités de ligne disponibles, deux rangs peuvent aussi fournir des vues complémentaires de la même ligne ; les vues sans partenaire de ligne ou d'identité sont écartées. La matrice peut être rectangulaire et doit contenir au moins deux vues d'un côté pour contribuer. Les paires de modalités actives sont moyennées avec le même poids. En mode `alignment_mode: "class"`, les étiquettes connues peuvent partager leurs positifs ; les étiquettes absentes ou `-100` restent individuelles. Quand aucune paire ne peut contribuer, la perte globale vaut zéro et tous les rangs participent quand même au backward.

Pour un fine-tuning depuis un checkpoint existant :

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m chimera --threads 1 train --init-from model.pt \
  --model-overrides configs/global-alignment-overrides.json \
  --batch-size 1 --steps 100 --output runs/global-alignment
```

Tous les rangs du groupe distribué par défaut doivent utiliser la même configuration et exécuter les appels d'entraînement et de backward dans le même ordre ; un sous-groupe DDP isolé n'est pas pris en charge par ce mode. Les dimensions, dtypes, modes et températures sont comparés entre rangs avant le rassemblement des représentations. Le mode global utilise un groupe de communication séparé de celui du réducteur DDP ; il ajoute des collectives au forward et au backward, y compris pendant `no_sync`. Il ne réunit pas les exemples de micro-lots différents lors d'une accumulation de gradients. Les représentations rassemblées et les matrices de similarité augmentent la mémoire avec le nombre d'exemples globaux ; les matrices peuvent croître quadratiquement. La détermination de la taille maximale et des paires actives peut synchroniser le device.

En mode évaluation, ou sans groupe distribué à plusieurs rangs, le calcul reste local. Cela permet notamment à la validation effectuée sur le seul rang 0 de fonctionner sans collective. Les groupes de communication ne sont ni sauvegardés dans le checkpoint ni copiés avec le modèle ; ils sont recréés collectivement au prochain entraînement distribué.

Les tests comparent la perte et les gradients à un calcul sur lot global, en modes `paired` et `class`, avec lots inégaux, étiquettes inconnues, modalités absentes, accumulation, rang sans gradient local et rang sans cible supervisée. Une reprise DDP sur deux processus CPU avec checkpointing des activations reproduit les poids d'une exécution continue. Les essais couvrent Gloo sur CPU ; aucun test NCCL/GPU ni gain de qualité après entraînement réel n'est revendiqué.

### Essai synthétique apparié

Deux processus CPU affinent le même checkpoint `tiny` pendant 80 étapes, avec un exemple par rang et par étape, pour trois seeds d'entraînement. Les deux modes reçoivent les mêmes exemples et le même budget ; `alignment_mode` vaut `class` et le poids d'alignement 0,2. Avec ce lot local de taille 1, la perte contrastive locale est absente aux 80 étapes, tandis que le mode global la calcule aux 80 étapes. Les données d'entraînement contiennent texte, image et série ; le texte synthétique indique explicitement la classe.

Le test utilise 512 autres exemples, avec des seeds distinctes de l'entraînement et de sa validation pour cet essai. Il ne fournit **pas de texte** au modèle : chaque image interroge les 32 séries de son lot et la recherche est correcte si la série sélectionnée porte la même classe. La MAE porte sur les prévisions médianes avec image et série.

| Seed | Recherche de classe, local | Recherche de classe, global | MAE, local | MAE, global |
| ---: | ---: | ---: | ---: | ---: |
| 101000000 | 75,39 % | 95,90 % | 0,17366 | 0,17964 |
| 103000000 | 99,02 % | 91,60 % | 0,17204 | 0,21031 |
| 105000000 | 81,05 % | 80,86 % | 0,17941 | 0,17450 |
| Moyenne | 85,16 % | 89,45 % | 0,17504 | 0,18815 |

Le gain moyen apparié de recherche est de 4,30 points, avec un écart-type de 14,50 points entre les trois différences ; la MAE moyenne augmente d'environ 7,5 %. Ces trois entraînements, un seul checkpoint initial et des modalités synthétiques faciles ne démontrent pas une amélioration générale. La recherche de classe accepte une autre série de même classe ; elle ne mesure pas l'identification unique d'un exemple. Les [mesures complètes](artifacts/global-alignment-study.json) incluent également la recherche de paire exacte, l'accuracy, les seeds et les empreintes.

```bash
.venv/bin/python tools/compare_global_alignment.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output-dir runs/global-alignment-study
```

Le dossier de sortie doit être nouveau. L'outil vérifie que les plages de seeds d'entraînement, de validation et de test de cet essai sont disjointes et conserve les logs et checkpoints locaux de chaque branche.

La répartition habituelle envoie généralement des lots distincts à chaque rang : les identités complémentaires peuvent alors ne pas coïncider pendant une étape donnée. `--shared-modality-batches` fait parcourir les mêmes lots JSONL par tous les rangs, dans le même ordre. Avec la politique `balanced`, le calendrier de modalités est indexé par rang et peut fournir différentes vues d'une même ligne au calcul global. L'option `--shared-modality-batch-interval N` partage seulement un micro-lot sur N, en repartant d'un partage au début de chaque époque ; le dernier micro-lot réparti peut nécessiter du remplissage. La fraction effectivement partagée dépend donc aussi de la longueur de l'époque. Cette stratégie traite moins de lignes distinctes par étape que des lots répartis ; les pertes supervisées portent sur des vues répétées et le coût d'encodage augmente pour un même nombre de lignes uniques. Le résultat dépend de la politique de modalités, des augmentations et des données : il faut mesurer les compromis. L'option nécessite `--mixed-modalities` et `--data` ; elle reste désactivée par défaut et son état est conservé à la reprise.

### Comparaison synthétique des lots partagés

Deux rangs CPU affinent le même checkpoint `tiny` pendant 60 étapes sur 48 lignes synthétiques texte–série, avec trois seeds appariées, quatre lignes par rang, la politique de modalités équilibrée, la vue textuelle isolée et l'alignement global par classe. Le test utilise 256 autres exemples en lots de 32 ; le texte indique explicitement la classe. Les données d'ajustement, de validation et de test de cet essai utilisent des seeds distinctes. L'indépendance des données d'entraînement du checkpoint initial n'est pas établie.

| Mesure moyenne | Lots répartis | Lots partagés |
| --- | ---: | ---: |
| Recherche texte–série : classe du premier voisin | 99,09 % | 93,10 % |
| Accuracy de classification, série seule | 87,50 % | 84,64 % |
| MAE de prévision, série seule | 0,18735 | 0,19573 |
| Étapes avec perte d'alignement non nulle, sur 60 | 37,3 | 57,7 |
| Occurrences de lignes distinctes par étape, cumulées | 480 | 240 |

Le partage active plus souvent l'alignement mais ne constitue pas un gain général dans cet essai : la différence moyenne appariée de recherche est de −5,99 points, avec un écart-type de 9,73 points entre trois seeds. La baisse est surtout liée à une seed ; la MAE augmente de 0,00838 en moyenne. Le même nombre de vues est traité dans les deux modes, mais les lots partagés contiennent deux fois moins de lignes distinctes par étape. La recherche accepte toute série de même classe : elle ne mesure pas l'identification d'une paire unique. Les données synthétiques faciles, un seul checkpoint et trois entraînements ne permettent pas de conclure pour des données réelles. Les [mesures complètes](artifacts/shared-modality-batches-study.json) incluent les seeds, empreintes et temps observés, qui comprennent le lancement des processus.

```bash
.venv/bin/python tools/compare_shared_modality_batches.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --output-dir runs/shared-modality-study
```

Le dossier de sortie doit être nouveau. La comparaison conserve les logs et checkpoints locaux des deux modes et vérifie les empreintes des données et du checkpoint parent.

### Comparaison du partage périodique

Une seconde exécution reprend les mêmes données, le même checkpoint, les trois seeds et le budget précédent, en ajoutant un partage tous les deux micro-lots. Les étapes avec perte d'alignement non nulle sont de 37,3 en moyenne pour les lots répartis, 47,0 pour le partage périodique et 57,7 pour le partage complet. Les occurrences de lignes distinctes par étape, cumulées sur les 60 étapes, valent respectivement 480, 360 et 240.

| Mesure moyenne | Répartis | Périodiques | Partagés |
| --- | ---: | ---: | ---: |
| Recherche texte–série : classe du premier voisin | 99,09 % | 99,74 % | 93,10 % |
| Accuracy de classification, série seule | 87,50 % | 85,03 % | 84,64 % |
| MAE de prévision, série seule | 0,18735 | 0,19223 | 0,19573 |

La différence appariée entre partage périodique et répartition est de +0,65 point pour la recherche, avec un écart-type de 0,81 point, de −2,47 points pour la classification et de +0,00487 pour la MAE. La recherche est proche de son plafond sur ces données où le texte révèle la classe. Trois seeds, un seul checkpoint et un seul jeu de test synthétique ne démontrent pas un gain général ; l'indépendance vis-à-vis de l'entraînement du checkpoint parent reste inconnue. Les [mesures complètes](artifacts/periodic-modality-batches-study.json) conservent les résultats et empreintes des neuf entraînements.

```bash
.venv/bin/python tools/compare_shared_modality_batches.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt \
  --partial-interval 2 --output-dir runs/periodic-modality-study
```

Les nouveaux entraînements hétérogènes activent également `--alignment-cross-rank-views` : si le choix des modalités diffère entre rangs pour une ligne répétée, les vues de cette ligne peuvent former une paire même si aucun rang ne les possède simultanément. Cette option nécessite `--alignment-duplicate-positives`, agit seulement avec l'alignement global distribué et change l'objectif d'apprentissage. Elle est conservée dans le checkpoint ; les anciens checkpoints la gardent désactivée à la reprise. Les comparaisons incluent les identités de ligne puis, en mode `class`, les classes connues parmi les vues appariables. Les tests couvrent les matrices rectangulaires, les modalités manquantes et les gradients sur deux rangs CPU ; aucun gain sur données réelles n'est établi.

Pour des données étiquetées mais non appariées, `--alignment-cross-class-views` autorise en plus les vues de lignes différentes à devenir positives dès que leur classe connue coïncide. L'option exige `alignment_mode: "class"`, ne nécessite pas d'identifiant et agit uniquement avec l'alignement global distribué. Elle reste désactivée par défaut et son état est restauré à la reprise. Les classes doivent désigner les mêmes catégories dans chaque source ; cette supervision peut créer des positifs inadéquats si les catégories sont larges. Les vues sans partenaire admissible sont écartées, y compris celles de classe inconnue, sauf si leur ligne ou leur identité fournit une correspondance. Comme pour les autres objectifs à plusieurs positifs, la valeur numérique de la perte dépend de leur nombre et de la composition du lot.

En alignement global, les rangs vérifient ensemble cette option avant le rassemblement des représentations : une option différente selon les rangs ou son emploi commun en mode `paired` produit une erreur sur chaque rang participant. Tous les rangs doivent néanmoins entrer dans le même appel d'alignement distribué.

### Essai sur des sources synthétiques non appariées

Deux sources indépendantes contiennent chacune 48 lignes : la première fournit seulement le texte et la classe, la seconde seulement la série, son futur et la classe. Le même checkpoint `tiny` est affiné pendant 60 étapes sur deux rangs CPU, avec trois seeds appariées, un lot local de 4 et un poids d'alignement de 0,2. Les deux variantes voient les mêmes lignes ; seule `--alignment-cross-class-views` change. Les 256 exemples de test utilisent d'autres tirages. La recherche accepte toute série du lot de 32 ayant la même classe que le texte ; la classification et la prévision médiane utilisent seulement la série.

| Seed | Recherche classe, témoin | Recherche classe, option | Exactitude série, témoin | Exactitude série, option | MAE série, témoin | MAE série, option |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211000000 | 97,27 % | 87,11 % | 87,89 % | 88,28 % | 0,19740 | 0,19005 |
| 213000000 | 100 % | 100 % | 84,38 % | 83,98 % | 0,19891 | 0,19247 |
| 215000000 | 100 % | 100 % | 82,81 % | 83,20 % | 0,18914 | 0,18829 |
| Moyenne | 99,09 % | 95,70 % | 85,03 % | 85,16 % | 0,19515 | 0,19027 |

L'alignement est non nul à 22 à 31 étapes sur 60 avec l'option, et à aucune étape chez le témoin. La différence moyenne appariée est de −3,39 points pour la recherche, avec un écart-type de 5,86 points ; elle est de +0,13 point pour la classification et de −0,00488 pour la MAE. Le résultat de recherche se dégrade donc sur une graine et reste inchangé sur les deux autres ; il ne justifie pas d'activer l'option par défaut. Le texte synthétique énonce explicitement la classe, le témoin est déjà proche du plafond et l'indépendance du checkpoint parent vis-à-vis du générateur n'est pas établie. Ces résultats ne démontrent pas de performance sur des sources réelles. Les [données agrégées et empreintes](artifacts/unpaired-class-alignment-study.json) conservent aussi les mesures par graine.

```bash
.venv/bin/python tools/compare_unpaired_class_alignment.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output-dir runs/unpaired-class-study
```

Le dossier de sortie doit être nouveau ; l'outil conserve les journaux et checkpoints de chaque variante et vérifie les seeds des sources et du test.

Un second essai exploratoire reprend exactement les mêmes données et seeds avec un poids de 0,05. La recherche moyenne atteint 96,22 %, soit −2,86 points face au témoin, la classification 84,64 % et la MAE 0,19293. La baisse de recherche persiste sur la première seed ; les deux autres restent à 100 %. Les [mesures de ce second essai](artifacts/unpaired-class-alignment-weight005.json) utilisent le même test déjà observé : elles ne valident pas indépendamment ce choix de poids.

Les lots répétés pour équilibrer les rangs peuvent présenter plusieurs vues de la même ligne. Les nouveaux entraînements avec `--mixed-modalities` transmettent l'indice stable de chaque ligne comme identité à la perte : les copies de cette ligne sont des positifs supplémentaires, plutôt que des négatifs dans l'alignement `paired`. `--alignment-duplicate-positives` active aussi ce comportement pour un jeu homogène ; l'option négative le désactive et les anciens checkpoints conservent leur réglage à la reprise. Les étiquettes de classe positives restent combinées aux identités lorsqu'elles sont activées. Ces répétitions affectent toujours les autres pertes et les poids des exemples ; les entrées augmentées sont supposées préserver leur correspondance.

## Vue textuelle isolée pour l'alignement

Par défaut, la représentation textuelle utilisée par la perte d'alignement est le dernier état du décodeur **après** attention croisée vers les latents multimodaux. Elle peut donc dépendre de l'image et de la série qu'elle est censée comparer : elle n'est pas une représentation du texte seul. Les approches contrastives à encodeurs séparés, comme [CLIP](https://proceedings.mlr.press/v139/radford21a.html), illustrent l'intérêt de vues calculées indépendamment.

L'option `alignment_text_view: "isolated"` calcule une seconde vue du texte avec les mêmes embeddings, le même décodeur et un contexte nul appris, sans lui fournir l'image ni la série de l'exemple. Cette vue sert uniquement à la perte d'alignement ; les logits de texte, les classes et les prévisions sont calculés par le passage multimodal habituel. À poids fixés, changer l'image ou la série ne change donc pas la vue textuelle isolée. Les gradients de cette vue mettent à jour les paramètres partagés, si bien que les prédictions peuvent évoluer après entraînement.

```bash
.venv/bin/python -m chimera train --init-from model.pt \
  --model-overrides configs/isolated-text-alignment-overrides.json \
  --steps 100 --output runs/isolated-text-alignment
```

Le défaut reste `"joint"`, compatible avec les anciens checkpoints ; aucun poids n'est ajouté. L'option fonctionne avec `alignment_mode` et `alignment_scope`, y compris le mode distribué. Dans ce dernier, tous les rangs doivent choisir la même vue : les paramètres sont vérifiés avant le rassemblement des représentations. Sans paire locale possible, le second passage est évité ; en mode global, un rang ayant du texte l'effectue même avec un seul exemple local, car d'autres rangs peuvent fournir les paires.

Cette vue réutilise le resampler et les blocs du décodeur : elle ajoute donc du calcul et des activations lorsque la perte d'alignement est calculée, y compris pour les diagnostics à poids nul, et consomme des tirages de dropout supplémentaires à l'entraînement. Les métriques et pénalités de routage existantes ne comptent que le passage principal ; elles ne décrivent pas le routage de cette seconde vue. Le checkpointing des activations et la reprise restent pris en charge. Elle ne transforme pas le décodeur en encodeur bidirectionnel et ne garantit pas une meilleure recherche sur des données réelles.

### Comparaison synthétique des vues textuelles

Les deux variantes affinent le même checkpoint `tiny` pendant 80 étapes sur deux processus CPU, avec un exemple par rang, trois seeds appariées, `alignment_scope: "global"`, `alignment_mode: "class"` et un poids d'alignement de 0,2. Les mêmes données synthétiques multimodales alimentent les deux variantes ; leur texte indique explicitement la classe. Le test utilise 512 autres exemples, avec des seeds distinctes de l'entraînement et de sa validation pour cet essai.

Pour les deux variantes, la recherche ci-dessous encode **le texte seul** avec le contexte nul, puis cherche parmi les 32 séries de son lot une série de même classe. La MAE est mesurée sur la prévision médiane à partir de l'image et de la série, sans texte en entrée.

| Seed | Recherche texte–série, vue jointe | Recherche texte–série, vue isolée | MAE, vue jointe | MAE, vue isolée |
| ---: | ---: | ---: | ---: | ---: |
| 111000000 | 81,05 % | 82,03 % | 0,18592 | 0,19708 |
| 113000000 | 80,27 % | 96,88 % | 0,18114 | 0,19281 |
| 115000000 | 79,88 % | 80,27 % | 0,18610 | 0,18437 |
| Moyenne | 80,40 % | 86,39 % | 0,18439 | 0,19142 |

L'écart moyen apparié de recherche est de 5,99 points, avec un écart-type de 9,19 points entre les trois différences ; la MAE moyenne augmente d'environ 3,8 %. Sur ce test, remplacer les images et séries associées change la représentation textuelle de la vue jointe d'une norme L2 moyenne de 0,95, contre zéro pour la vue isolée. Trois entraînements, un seul checkpoint et des données synthétiques faciles ne démontrent pas un gain général. La recherche de classe accepte une autre série de même classe et ne mesure pas l'identification unique d'un exemple. Les [mesures complètes](artifacts/isolated-text-alignment-study.json) comprennent les autres recherches, les seeds et les empreintes.

```bash
.venv/bin/python tools/compare_text_alignment_view.py \
  --checkpoint runs/joint-affine-pilot/control/latest.pt --output-dir runs/isolated-text-study
```

Le dossier de sortie doit être nouveau ; l'outil vérifie les plages de seeds et conserve les logs et checkpoints locaux de chaque branche.
