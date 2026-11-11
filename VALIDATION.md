# Validation de CHIMERA

Mesures effectuées le 11 novembre 2026 sur CPU, avec Python 3.14.8 et PyTorch 2.15.0+cpu. Aucun GPU utilisable n'est disponible sur cette machine.

Ce rapport conserve les résultats de l'entraînement initial et de ses 26 tests. La version améliorée passe 40 tests ; ses mesures de mémoire, ses vérifications de reprise et la comparaison avec le checkpoint initial sont décrites dans [IMPROVEMENTS.md](IMPROVEMENTS.md).

## Ce qui a réellement été exécuté

- Réseau `workstation` de **549 485 298 paramètres** : entraînement conjoint pendant **240 étapes**, batch de 8, sans poids préentraînés.
- Temps d'entraînement enregistré : **435.3 secondes**, hors évaluation finale et export.
- Les 16 experts routés de chacune des 12 couches ont tous été utilisés pendant l'exécution.
- **26 tests réussis**, aucune erreur, aucun échec et aucun test ignoré.
- Entraînement DDP sur deux processus CPU, pendant 3 étapes avec accumulation de 2 microbatches.
- Entraînement depuis un fichier JSONL, validation sur un fichier distinct, prédiction et export des poids.
- Reprise exacte testée avec dropout et accumulation, sur données synthétiques et sur JSONL, en traversant une fin d'époque.
- Configurations `base`, `large` et `ultra` inspectées sur le device `meta` : comptage exact des paramètres sans allocation des poids. Elles n'ont pas été entraînées.

## Résultats sur des exemples indépendants

Le benchmark compare les poids initiaux et les poids entraînés sur les **mêmes 64 exemples**, générés avec des seeds à partir de 2 000 000, distincts des seeds d'entraînement et de la validation du trainer. Il utilise 8 lots de 8 exemples pour chacune des sept combinaisons de modalités.

| Entrées | Classification avant | Classification après | MAE futur avant | MAE futur après |
| --- | ---: | ---: | ---: | ---: |
| Texte | 23.44 % | 100.00 % | 1.2490 | 0.4566 |
| Image | 9.38 % | 100.00 % | 1.2524 | 0.5011 |
| Séries | 9.38 % | 46.88 % | 0.5479 | 0.2967 |
| Texte + image | 23.44 % | 100.00 % | 1.2512 | 0.4888 |
| Texte + séries | 9.38 % | 100.00 % | 0.5517 | 0.2082 |
| Image + séries | 9.38 % | 100.00 % | 0.5508 | 0.2048 |
| Trois modalités | 4.69 % | 100.00 % | 0.5541 | 0.2146 |

La référence de persistance répète la dernière valeur observée : sa MAE vaut **0.5424**. La prévision avec les trois modalités vaut **0.2146**, soit une réduction de **60.4 %**. La prévision avec les seules séries vaut **0.2967** et bat aussi cette référence.

Pour le texte conditionné sur les deux autres modalités, l'entropie croisée par octet passe de **5.6646** à **0.0039**. Cette mesure porte sur les courtes descriptions du générateur synthétique.

Ces données associent six classes à des images de motifs, des descriptions et des séries sinusoïdales avec tendance. Elles ne démontrent aucune compétence sur des photographies, des conversations ou des séries de production. La classification à partir des seules séries reste imparfaite ; les sept lignes sont conservées pour rendre ce résultat visible. Les résultats ne constituent pas une évaluation générale du langage ou de la vision.

## Garanties vérifiées par les tests

La suite vérifie le fonctionnement des sept combinaisons d'entrées, les formes de sorties, la propagation des gradients dans toutes les branches et les routeurs, l'effet des modalités sur les sorties, et la diminution des trois pertes supervisées.

Elle vérifie aussi la causalité du texte, l'équivalence entre calcul complet et cache au-delà de la fenêtre locale, le padding à gauche pendant l'inférence, l'invariance aux positions masquées, les valeurs temporelles manquantes, l'ordre des quantiles, l'apprentissage de la connexion temporelle au-delà de la persistance, le backward BF16 sur CPU, l'équivalence des gradients avec recalcul d'activations et les sauvegardes/reprises.

## Reproduire

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/benchmark.py --checkpoint runs/workstation/latest.pt
.venv/bin/python -m chimera demo --checkpoint runs/workstation/model.pt
```

Le checkpoint complet, destiné à la reprise, est [latest.pt](runs/workstation/latest.pt). Les [poids d'inférence](runs/workstation/model.pt) occupent **2.20 Go** et se chargent avec la même API. Le fichier d'inférence ne contient pas l'optimiseur ; utiliser le checkpoint complet pour reprendre l'entraînement.

Les configurations géantes demandent une infrastructure nettement plus grande. DDP réplique les poids et ne fournit aucun partitionnement d'experts. CUDA, FP16 sur GPU et les configurations au-delà de `workstation` n'ont pas fait l'objet d'une exécution d'entraînement ici.

Les preuves détaillées sont conservées dans [tests.txt](artifacts/tests.txt), [tests.json](artifacts/tests.json), [training.json](artifacts/training.json), [benchmark.json](artifacts/benchmark.json), [ddp-smoke.json](artifacts/ddp-smoke.json), [jsonl-smoke.json](artifacts/jsonl-smoke.json) et [architectures.json](artifacts/architectures.json). L'[audit final](artifacts/acceptance.json) vérifie notamment que les poids exportés sont exactement identiques à ceux du checkpoint d'entraînement. Les versions des dépendances se trouvent dans [environment-requirements.txt](artifacts/environment-requirements.txt).
