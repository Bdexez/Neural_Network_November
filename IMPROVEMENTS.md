# Améliorations implémentées

Quatre axes ont été retenus à partir des limites du code et des mesures du projet : mémoire de génération, chargement des poids, volume des données et suivi d'entraînement. Les poids du réseau existant restent compatibles ; aucune modification de la taille du modèle n'est nécessaire.

## Mesures avant/après

Les deux mesures de cache utilisent le preset `tiny`, 96 tokens de préremplissage, 64 tokens supplémentaires, la même seed et deux threads CPU. Le lecteur reçoit le même JSONL de 1 000 exemples contenant des images, soit 15 643 000 octets. Les mesures de chargement utilisent le même fichier de poids d'inférence de 549 485 298 paramètres, dans des processus distincts.

| Mesure | Avant | Après |
| --- | ---: | ---: |
| Clés/valeurs d'auto-attention conservées | 122 880 octets | 56 832 octets |
| Positions conservées, couches 1 / 2 / 3 | 160 / 160 / 160 | 31 / 160 / 31 |
| Projections du contexte par couche | 65 | 1 |
| Pic mémoire Python du lecteur JSONL | 120.804 Mo | 0.393 Mo |
| Pic RSS au chargement, avant inférence | 4457.5 Mio | 344.9 Mio |

Le cache K/V d'auto-attention diminue de **53.8 %** dans cette configuration. Les couches globales gardent toutes leurs positions ; seules les couches locales évacuent les anciennes clés. Les clés/valeurs d'attention croisée sont conservées séparément pour éviter les projections répétées.

La mesure du lecteur est obtenue avec `tracemalloc` et couvre la mémoire Python, pas la mémoire des tenseurs ni le cache du système de fichiers. Le nouveau lecteur garde un index compact et charge un exemple à la demande. Le gain dépend de la taille des lignes et du nombre d'exemples.

Le pic de chargement est mesuré avec `getrusage`. Le nouveau chargeur mappe le fichier en mémoire et instancie le module sur `meta`, puis attache les poids. **Cette mesure précède la première inférence** : elle ne signifie pas que le réseau de 549 millions de paramètres peut fonctionner entièrement dans 345 Mio. Les pages consultées occupent ensuite la mémoire nécessaire. Les temps ponctuels sont conservés dans les JSON mais ne constituent pas un benchmark de débit ; cache disque et nombre de threads peuvent influencer ces durées.

## Suivi de l'apprentissage

- Validation périodique configurable et sauvegarde du meilleur checkpoint dans `best.pt`.
- Arrêt anticipé avec patience et seuil de progrès, synchronisé en DDP. Le meilleur checkpoint reste celui de la plus petite perte, même si le progrès est inférieur à `min_delta`.
- Mélange déterministe des exemples à chaque époque, avec restauration de la position exacte du flux à la reprise.
- Empreintes SHA-256 des JSONL, vérifiées à la reprise ; modification détectée aussi pendant la lecture. Les images externes ne sont pas couvertes par cette empreinte.
- GradScaler peut ignorer un overflow et réduire son échelle ; une norme de gradient non finie n'interrompt plus ce mécanisme. Le test injecte un overflow réel dans GradScaler CPU puis vérifie la reprise des mises à jour. Le chemin FP16 sur GPU n'a pas été exécuté ici.
- Pertes de texte, de classification et de prévision pondérées par les cibles valides. Mesures de prévision par horizon, RMSE, couverture et largeur des intervalles. Les tâches sans cibles observées n'affichent pas une qualité artificiellement parfaite.
- Prédictions JSONL écrites progressivement dans un fichier temporaire, remplacé atomiquement à la fin. Export des poids avec conservation de la seed et de la provenance.

## Vérifications

**40 tests passent**, dont les 26 tests initiaux. Les nouveaux cas couvrent les caches locaux bornés, des chunks de tailles différentes, le padding, une fenêtre et une convolution de taille 1, la lecture parallèle, les poids mappés en mémoire, les cibles manquantes et l'arrêt anticipé.

La reprise avec mélange des données, dropout, accumulation et validations périodiques produit les mêmes poids bit à bit qu'une exécution continue. Une pause à une étape hors calendrier de validation ne modifie pas les décisions d'arrêt anticipé.

Deux processus CPU ont exécuté quatre étapes avec validation à chaque étape. Un second essai, avec des fichiers d'entraînement et de validation distincts et un seuil de progrès volontairement élevé, a vérifié l'arrêt synchronisé à l'étape 2 et la présence des checkpoints `best.pt` et `latest.pt`.

Le checkpoint `workstation` existant a été réévalué sur les mêmes 64 exemples pour les sept combinaisons de modalités. Accuracy, MAE et perte de texte restent identiques à une tolérance de 1e-5. La démonstration génère encore **« Signal de classe 3. »**, prédit les mêmes classes et produit les mêmes prévisions sur son exemple enregistré.

La mesure de couverture ajoutée révèle une couverture de **90.45 %** pour un intervalle nominal à **80 %** sur ce jeu synthétique. Les quantiles ne sont donc pas parfaitement calibrés ; cela reste une piste d'amélioration des poids, qui nécessite un entraînement et des données appropriés.

Une CI sur Python 3.11 et 3.14 exécute désormais les tests et un smoke test DDP. Son état distant doit être lu sur GitHub ; les preuves locales ci-dessous ne se substituent pas à son résultat.

## Reproduire

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/profile_runtime.py --cache --output runs/cache-profile.json
.venv/bin/python tools/profile_runtime.py --dataset votre-jeu.jsonl --output runs/data-profile.json
.venv/bin/python tools/profile_runtime.py --checkpoint runs/workstation/model.pt --output runs/load-profile.json
.venv/bin/python tools/benchmark.py --checkpoint runs/workstation/model.pt --output runs/quality.json
```

Les sorties brutes sont dans [runtime-before.json](artifacts/runtime-before.json), [runtime-after.json](artifacts/runtime-after.json), [checkpoint-before.json](artifacts/checkpoint-before.json), [checkpoint-after.json](artifacts/checkpoint-after.json), [improvement-tests.txt](artifacts/improvement-tests.txt), [improvement-tests.json](artifacts/improvement-tests.json), [improved-benchmark.json](artifacts/improved-benchmark.json) et [improvements-acceptance.json](artifacts/improvements-acceptance.json).
