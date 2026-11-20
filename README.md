# CHIMERA

Un réseau PyTorch qui réunit les quatre usages demandés : modèle multimodal, génération de texte, vision et prévision de séries temporelles. Les trois modalités partagent un décodeur à mélange d'experts et peuvent être utilisées seules, par paires ou ensemble.

La complexité n'a pas de maximum absolu : on peut toujours ajouter des couches et des experts. CHIMERA fournit une architecture réellement entraînable, des configurations allant de **739 506 à 588 945 015 026 paramètres**, et un entraînement local de la configuration **549 485 298 paramètres**. Les grandes configurations sont décrites et comptées sans allocation des poids ; elles n'ont pas été entraînées ici.

Le checkpoint livré apprend sur des signaux synthétiques associés à des images de motifs et à de courtes descriptions. Il sert à démontrer l'apprentissage conjoint. Ce n'est pas un assistant préentraîné ni un modèle déjà performant sur des photos ou des textes quelconques. La complexité seule ne garantit pas la qualité.

Les checkpoints sous `runs/` sont disponibles dans l'espace de travail local et sont exclus de Git en raison de leur taille. Après un nouveau clone, les commandes d'entraînement ci-dessous permettent de les recréer ; les rapports de validation sont inclus dans le dépôt.

Les [améliorations mesurées](IMPROVEMENTS.md) décrivent le cache borné, le chargement des poids à la demande, le lecteur JSONL indexé et le suivi de validation.

Le [suivi de l'apprentissage](LEARNING.md) présente le fine-tuning du modèle existant, la comparaison des politiques de modalités et leurs compromis mesurés.

La [calibration des prévisions](CALIBRATION.md) ajuste les intervalles sur un jeu séparé, par coordonnée ou pour la couverture simultanée d'une trajectoire, et mesure leur couverture sur un test indépendant.

Les [historiques courts et observations manquantes](ROBUSTNESS.md) couvrent l'augmentation temporelle, sa reprise reproductible et les comparaisons avec un entraînement témoin.

Les [statistiques temporelles](TEMPORAL_STATISTICS.md) ajoutent une projection optionnelle du niveau, de l'échelle et de la durée observée, ainsi qu'un transfert explicite vers des configurations compatibles.

La [correction des prévisions courtes](FORECAST_REFINEMENT.md) ajuste une couche de 396 paramètres à partir des derniers pas de l'historique, en conservant les poids du modèle de base. Une porte facultative peut aussi l'activer pour les variables partiellement observées ; les mesures montrent des compromis selon la longueur d'historique.

Une [tête de classification temporelle standardisée](SERIES_CLASSIFICATION.md) peut être ajustée séparément pour tester la résistance aux changements de niveau et d'unités. Elle produit des scores distincts et présente un compromis de précision sur les séries d'origine.

La [prévision après changement de niveau ou d'unités](AFFINE_ROBUSTNESS.md) compare une augmentation affine et des caractéristiques relatives, avec un témoin ajusté sur les mêmes données.

## Démarrage

Python 3.11 ou plus récent. Dans cet espace de travail, `.venv` contient déjà PyTorch CPU.

```bash
# Installation sur une autre machine, avec PyTorch CPU
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .

# Exécuter les trois entrées et les trois sorties
.venv/bin/python -m chimera demo

# Utiliser les poids entraînés localement
.venv/bin/python -m chimera demo --checkpoint runs/workstation/model.pt

# Poids affinés pour les historiques complets, conservés localement
.venv/bin/python -m chimera demo --checkpoint runs/workstation-robust-control/model.pt

# Vérifier l'architecture sans allouer ses poids
.venv/bin/python -m chimera inspect --all

# Vérifier le fonctionnement et les gradients
.venv/bin/python -m unittest discover -s tests -v
```

Sur GPU, installer une distribution de PyTorch adaptée à CUDA avant le projet, puis utiliser `--device cuda --amp bf16`. Le chemin GPU est fourni mais n'a pas été exécuté sur cette machine sans GPU utilisable.

## Architecture

```mermaid
flowchart TD
    I[Images] --> V[Convolutions 3×3 et 5×5<br/>patches et encodeur de vision]
    S[Historique temporel] --> T[Normalisation par variable<br/>masque des valeurs manquantes<br/>convolution et encodeur temporel]
    V --> R[Fusion par attention croisée<br/>compression en latents appris]
    T --> R
    X[Texte UTF-8] --> E[Embedding partagé avec la sortie]
    E --> D[Décodeur hybride répété N fois<br/>GQA + RoPE + convolution causale<br/>attention croisée vers les latents<br/>MoE top-k + expert partagé]
    R --> D
    Q[Requêtes de tâches<br/>quand le texte est absent] --> D
    D --> L[Texte : prochain octet<br/>et prédictions auxiliaires futures]
    D --> C[Vision / compréhension : classes]
    D --> F[Séries : quantiles ordonnés<br/>par horizon et variable]
    R --> C
    R --> F
    S --> A[Connexion temporelle directe apprise<br/>référence : dernière observation]
    A --> F
```

Chaque bloc du décodeur combine :

- **Attention GQA** avec moins de têtes clé/valeur que de têtes requête, normalisation Q/K et positions rotatives RoPE.
- **Attention causale locale et globale** : fenêtre locale dans certains blocs, contexte global tous les `global_every` blocs.
- **Convolution causale à porte** en parallèle de l'attention, avec son propre état de génération.
- **Attention croisée multimodale** à porte vers les latents des images et de l'historique.
- **Experts SwiGLU** : routage top-k par token, experts partagés toujours actifs, aucune suppression de tokens, pénalités d'équilibrage et de stabilité du routeur.
- **RMSNorm**, connexions résiduelles et initialisation réduite des sorties d'experts.

Une [normalisation RMS stable optionnelle](NUMERICAL_PRECISION.md) évite le débordement des carrés sur les grandes activations finies ; le chemin historique reste le défaut.

L'[alignement contrastif peut utiliser les exemples des autres rangs](LEARNING.md#alignement-distribué-entre-rangs) pendant l'entraînement DDP et une [vue textuelle isolée](LEARNING.md#vue-textuelle-isolée-pour-lalignement) ; les modes locaux et joints restent les défauts.

Le modèle ajoute une prédiction auxiliaire de plusieurs tokens futurs, un alignement contrastif entre modalités, la normalisation temporelle réversible, une projection directe de l'historique vers les horizons futurs et une tête de prévision dont les quantiles ne peuvent pas se croiser. Le tokenizer par octets couvre tout Unicode sans téléchargement de vocabulaire ; il est moins compact qu'un tokenizer de sous-mots.

Les encodeurs d'images et d'historique sont bidirectionnels. L'historique donné en entrée doit donc se terminer **avant** le futur à prédire. Les futurs cibles ne sont jamais injectés dans les latents. Le décodeur de texte est causal, y compris sa convolution ; les tests modifient les tokens futurs et vérifient que les logits passés restent identiques.

## Tailles disponibles

Les comptes dédupliquent l'embedding et la tête de texte, qui partagent leurs poids.

| Configuration | Paramètres | Couches du décodeur | Experts par couche / sélectionnés | Poids BF16 seuls |
| --- | ---: | ---: | ---: | ---: |
| `tiny` | 739 506 | 3 | 4 / 2 | 0,001 Gio |
| `small` | 55 607 282 | 8 | 8 / 2 | 0,104 Gio |
| `workstation` | 549 485 298 | 12 | 16 / 2 | 1,023 Gio |
| `base` | 1 534 337 266 | 16 | 16 / 2 | 2,858 Gio |
| `large` | 39 285 370 354 | 32 | 32 / 2 | 73,175 Gio |
| `ultra` | 588 945 015 026 | 64 | 64 / 4 | 1 096,996 Gio |

Les experts partagés s'ajoutent aux experts routés : un pour tous les presets, deux pour `ultra`. Le nombre de paramètres sélectionnés par token n'est ni un compte de FLOPS ni une mesure du pic mémoire. Tous les poids restent en mémoire dans cette implémentation.

`inspect` indique aussi une estimation mémoire pour AdamW en FP32 : poids + gradients complets + deux moments, hors activations et allocations temporaires. Pour `workstation`, cette estimation est d'environ 8,19 Gio ; pour `ultra`, elle dépasse 8 775 Gio. Les paramètres restent FP32 pendant l'entraînement AMP : la colonne BF16 ne décrit pas la mémoire du trainer.

Les fichiers de [configuration](configs/) permettent de modifier dimensions, profondeur, experts, variables, classes, horizon et quantiles. Ils ne plafonnent pas artificiellement la taille. Inspecter un fichier personnalisé :

```bash
.venv/bin/python -m chimera inspect --config configs/workstation.json
```

## Entraînement conjoint

```bash
# Vérification rapide sur CPU
.venv/bin/python -m chimera train --preset tiny --steps 100 --output runs/tiny

# Exécution utilisée pour le modèle livré
.venv/bin/python -m chimera --threads 4 train \
  --preset workstation --steps 240 --batch-size 8 --warmup 20 \
  --lr 0.0002 --modality-dropout 0.3 \
  --output runs/workstation --log-every 20 --save-every 60

# Mesure avant/après sur 64 exemples synthétiques indépendants
.venv/bin/python tools/benchmark.py --checkpoint runs/workstation/latest.pt

# Produire un petit jeu JSONL multimodal reproductible
.venv/bin/python tools/make_dataset.py --preset tiny --output runs/data

# Entraîner sur ses propres données et valider sur un fichier distinct
.venv/bin/python -m chimera train --preset tiny --steps 100 \
  --data runs/data/train.jsonl --val-data runs/data/validation.jsonl \
  --eval-every 10 --patience 5 --min-delta 0.001 --output runs/custom

# Entraîner sur votre fichier mixed.jsonl dont les modalités varient selon les lignes
.venv/bin/python -m chimera train --preset tiny --steps 100 \
  --data runs/data/mixed.jsonl --mixed-modalities \
  --loss-normalization observed --output runs/mixed

# Mesurer le meilleur modèle sur un fichier indépendant
.venv/bin/python -m chimera evaluate --checkpoint runs/custom/best.pt \
  --data runs/data/validation.jsonl --output runs/custom/quality.json

# Classer, prévoir et générer sur un fichier de nouvelles entrées
.venv/bin/python -m chimera predict --checkpoint runs/custom/best.pt \
  --data runs/data/validation.jsonl --max-new-tokens 24 \
  --output runs/custom/predictions.jsonl

# Extraire les vecteurs contrastifs pour comparer les modalités
.venv/bin/python -m chimera embed --checkpoint runs/custom/best.pt \
  --data runs/data/validation.jsonl --batch-size 4 \
  --output runs/custom/embeddings.jsonl

# Rechercher les séries les plus proches de chaque texte
.venv/bin/python -m chimera search-embeddings \
  --queries runs/custom/embeddings.jsonl --query-modality text \
  --candidates runs/custom/embeddings.jsonl --candidate-modality series \
  --top-k 5 --exclude-self --output runs/custom/neighbors.jsonl

# Évaluer avec des annotations de pertinence fournies séparément
.venv/bin/python -m chimera evaluate-retrieval \
  --queries runs/custom/embeddings.jsonl --query-modality text \
  --candidates runs/custom/embeddings.jsonl --candidate-modality series \
  --relevance runs/custom/relevance.jsonl --top-k 5 \
  --output runs/custom/retrieval-quality.json

# Exporter les poids seuls pour réduire la taille et la mémoire de chargement
.venv/bin/python -m chimera export --checkpoint runs/custom/best.pt \
  --output runs/custom/model.pt
```

La perte totale combine les pertes de texte, classification et prévision, la prédiction auxiliaire de tokens, l'alignement et la régularisation des experts. Les poids sont configurables. La perte auxiliaire moyenne uniquement les horizons qui ont au moins une cible observée : ajouter du padding ne réduit donc pas son poids. La perte des quantiles est calculée au minimum en FP32, y compris sous AMP ; elle passe en FP64 si l'amplitude et le nombre de valeurs risquent de faire déborder son calcul. Pour les amplitudes FP64 extrêmes, le calcul utilise des demi-résidus et adapte la réduction afin d'éviter des débordements intermédiaires lorsque la moyenne reste représentable. Les prévisions aux positions sans cible finie sont ignorées. Ces vérifications peuvent synchroniser le GPU. La projection temporelle directe part de la dernière observation disponible par variable et apprend une correction, avec un taux d'apprentissage dix fois supérieur au réseau principal. `--modality-dropout` omet des entrées pour apprendre aussi avec des modalités absentes. La perte totale des logs peut alors porter sur des ensembles de tâches différents d'une étape à l'autre ; comparer les mesures de validation à modalités fixes.

`--modality-policy balanced` parcourt une fois chaque combinaison disponible par cycle mélangé : sept combinaisons si les trois entrées sont présentes. Le choix reste reproductible après une reprise et réparti entre les rangs DDP. Les sous-ensembles sans cible supervisée observée sont exclus, même si des champs de cibles entièrement manquantes sont présents. La politique `dropout` conserve également un signal supervisé. Cette option remplace le tirage par `--modality-dropout` ; la politique par défaut reste `dropout`. `metrics.jsonl` compte les micro-lots par combinaison sur le rang 0. L'équilibrage change les compromis entre tâches : consulter les mesures avant de le choisir.

Avec l'alignement global distribué, `--alignment-cross-rank-views` permet aussi d'apparier deux modalités complémentaires présentes sur des rangs différents lorsqu'elles partagent un identifiant de ligne. Seules les vues ayant un partenaire de ligne ou d'identité participent ; les autres sont exclues. Cette option nécessite `--alignment-duplicate-positives`. Elle est active par défaut dans les nouveaux entraînements hétérogènes et désactivée pour les anciens checkpoints qui n'enregistrent pas ce réglage. `--no-alignment-cross-rank-views` conserve le calcul historique ; l'option est sans effet sur l'alignement local.

L'argument Python facultatif `alignment_ids` permet de fournir un identifiant entier par exemple pour reconnaître les copies : les valeurs identiques non négatives sont positives et `-1` désigne une identité inconnue. Ces identifiants doivent appartenir au même espace sur tous les rangs ; ils ne modifient pas directement les prédictions. Le lecteur d'entraînement utilise l'indice stable de la ligne JSONL lorsque la reconnaissance des copies est activée. Cette option change l'objectif contrastif et peut donc modifier les poids appris ; elle ne déduplique pas les pertes supervisées ni l'échantillonnage. La moyenne des log-probabilités attribuées aux positifs dépend de leur nombre : la valeur de la perte n'est pas directement comparable entre lots de compositions différentes. La vérification et la comparaison des identités peuvent synchroniser le GPU et ajoutent un masque de taille quadratique au nombre de paires.

Dans une configuration JSON, `alignment_mode: "class"` fait des exemples de même classe des positifs supplémentaires pour l'alignement. Les classes ignorées (`-100`) restent appariées individuellement ; sans classes, l'objectif habituel est conservé. Choisir cette option lorsque les classes définissent les groupes sémantiques à rapprocher. `alignment_mode: "paired"` et `alignment_temperature: 0.1` conservent les réglages précédents.

Pour un routage à un seul expert, `top_k: 1` et `top1_gate: "probability"` multiplient la sortie de l'expert sélectionné par sa probabilité. La perte de tâche peut ainsi entraîner le routeur ; avec le mode historique `normalized`, son poids vaut toujours un et ce gradient est nul à l'arrondi près. Le mode `probability` exige au moins deux experts et modifie les prédictions : il faut affiner et évaluer les poids avant utilisation. Les anciennes configurations conservent `normalized`, et le routage avec plusieurs experts ne change pas. L'expert partagé reste ajouté séparément.

Le fichier [top1-probability-overrides.json](configs/top1-probability-overrides.json) permet d'essayer ce mode avec `train --init-from ... --model-overrides ...`. Une [comparaison synthétique et un test de gradient](TOP1_ROUTING.md) documentent son comportement ; ils ne démontrent pas de gain général sur des données réelles.

Le trainer propose AdamW, échauffement et décroissance cosinus du taux d'apprentissage, accumulation des gradients, écrêtage, AMP BF16/FP16 et recalcul d'activations. Il écrit `config.json`, `metrics.jsonl`, `validation.jsonl`, `evaluation.json`, `latest.pt` et `best.pt`. L'écrêtage utilise le calcul PyTorch habituel ; si sa norme déborde alors que tous les gradients sont finis, un repli FP64 met les gradients à l'échelle avant de les écrêter. Le contrôle de la norme peut synchroniser l'appareil ; le repli utilise des tampons temporaires proportionnels à la taille d'un tenseur. Les gradients réellement non finis provoquent toujours une erreur sans GradScaler ou une étape ignorée avec GradScaler. Si la norme totale dépasse même FP64, l'écrêtage reste calculé par mise à l'échelle mais `grad_norm` vaut `null` dans les logs. Les logs contiennent l'utilisation de chaque expert et les mises à jour ignorées par GradScaler en cas d'overflow. L'utilisation des experts est pondérée par les tokens valides de tous les micro-lots et de tous les rangs DDP de l'étape ; les anciens logs ne décrivaient que le dernier micro-lot du rang 0. Les compteurs de modalités et de séries restent ceux du rang 0. Une réduction distribuée supplémentaire est effectuée à chaque étape pour ce diagnostic. Les sauvegardes sont remplacées atomiquement et incluent le modèle, AdamW, le scaler, les RNG, le planning et l'état de validation.

Avec `train --mixed-modalities`, les lignes du JSONL d'entraînement et de validation peuvent avoir des modalités différentes. Chaque lot est composé d'exemples ayant la même combinaison ; les entrées appariées restent ensemble. Le groupe et l'ordre des lots sont mélangés de façon reproductible, puis restaurés à la reprise ; `--no-shuffle` désactive ce mélange mais regroupe toujours les signatures. En distribué, les lots sont répartis entre rangs et les premiers lots sont répétés si nécessaire pour que tous les rangs effectuent le même nombre d'étapes. Ces répétitions augmentent le poids des exemples concernés. Dans un nouvel entraînement hétérogène, les copies d'une même ligne sont reconnues et deviennent des positifs supplémentaires de l'alignement, y compris en mode global ; deux lignes distinctes restent distinctes même si leur contenu est identique. `--no-alignment-duplicate-positives` conserve l'ancien traitement, et les anciens checkpoints reprennent leur politique historique. Pour l'activer avec un jeu homogène, utiliser `--alignment-duplicate-positives`. Les derniers lots de chaque groupe peuvent être plus petits ; la pondération `batch` donne le même poids à chaque micro-lot, et `observed` pondère les pertes supervisées par cible observée. Les annotations absentes et les modalités qui ne sont pas appariées ne créent pas de paires contrastives artificielles. Le lecteur stocke un octet de signature supplémentaire par ligne, et le regroupement conserve un index et une permutation en mémoire proportionnels au nombre de lignes. Les commandes `predict`, `embed` et `evaluate` acceptent aussi les modalités hétérogènes : elles préservent l'ordre du fichier et forment seulement des lots consécutifs homogènes. Les calibrations et les autres commandes d'ajustement conservent leurs exigences de modalités ; un artefact utilisé avec `predict` ou `evaluate` doit être compatible avec chaque ligne.

`--loss-normalization observed` pondère chaque perte supervisée par son nombre de cibles observées sur toute l'étape d'optimisation, entre micro-lots et rangs DDP. Les horizons auxiliaires de texte sont d'abord pondérés séparément, puis moyennés sur les horizons présents. Cela évite de donner le même poids à deux micro-lots dont les nombres de cibles diffèrent. Un micro-lot sans cible conserve ses modalités et peut contribuer aux pertes de routage et d'alignement, mais l'étape complète doit comporter au moins une cible observée. Ces deux pertes restent calculées par micro-lot ; l'objectif contrastif dépend toujours de sa composition. Ce mode conserve tous les micro-lots d'une étape en mémoire jusqu'au backward et peut donc consommer davantage de mémoire. Dans `metrics.jsonl`, la perte est la contribution du rang 0 multipliée par le nombre de rangs pour les termes supervisés ; les autres pertes affichées restent des moyennes par micro-lot.

Le mode par défaut, `--loss-normalization batch`, conserve la pondération historique. Le mode choisi est enregistré dans le checkpoint, restauré à la reprise et ne peut pas être changé pendant celle-ci. Les anciens checkpoints utilisent `batch`.

`--eval-every` règle la fréquence de validation. `--monitor` choisit la mesure qui sélectionne `best.pt` : `loss` par défaut, ou `observed_loss`, `text`, `classification`, `forecast`, `forecast_mae`, `forecast_rmse` ou `accuracy`. Toutes sont minimisées sauf l'accuracy, qui est maximisée. Par exemple, `--monitor forecast_mae` sélectionne directement la prévision médiane la plus précise sur les cibles observées. La mesure choisie doit être présente et finie ; un jeu de validation sans cible observée est refusé. `--patience` active l'arrêt après un nombre de validations sans progrès significatif ; `--min-delta` définit ce progrès dans les unités de la mesure choisie. L'arrêt anticipé nécessite `--val-data`. Sans fichier de validation, les mesures portent sur un lot synthétique fixe et l'arrêt anticipé reste désactivé par défaut. Les décisions d'arrêt sont partagées entre les processus DDP.

`observed_loss` recompose la somme pondérée des pertes de texte, classification et prévision après agrégation sur leurs cibles observées. Les horizons auxiliaires de texte sont agrégés séparément avant leur moyenne. Les pénalités de routage et d'alignement utilisent leurs moyennes par lot déjà rapportées : elles peuvent dépendre de la taille et de la composition des lots, notamment parce que l'alignement contrastif compare les exemples d'un même lot. La mesure vaut `null` en l'absence de toute cible supervisée observée. La valeur historique `loss`, moyenne des pertes totales par exemple et toujours utilisée par défaut, peut varier lorsque le nombre de cibles observées diffère entre lots. Les reprises existantes conservent leur mesure de sélection ; choisir `--monitor observed_loss` lors d'un nouvel entraînement pour utiliser la nouvelle mesure.

Le choix de `--monitor` est enregistré dans le checkpoint et restauré à la reprise ; il ne peut pas être changé pendant celle-ci. Les anciens checkpoints utilisent `loss`. Pour préserver leur format, la valeur de la mesure sélectionnée reste enregistrée sous le nom historique `validation_state.best_loss`, y compris lorsque la mesure choisie n'est pas une perte.

Les fichiers d'entraînement sont mélangés de façon reproductible à chaque époque ; `--no-shuffle` conserve leur ordre. `--workers 2` active deux processus de lecture. Une reprise d'un ancien checkpoint conserve son ancien ordre de lecture par défaut.

Reprise exacte sur un processus, en conservant les mêmes arguments de planning et de données :

```bash
.venv/bin/python -m chimera train --preset tiny --steps 100 \
  --stop-after 40 --output runs/resumable
.venv/bin/python -m chimera train --preset tiny --steps 100 \
  --resume runs/resumable/latest.pt --output runs/resumable
```

La suite de tests compare les poids d'une exécution continue et d'une reprise avec dropout, accumulation, mélange des données et arrêt anticipé : ils doivent être identiques bit à bit. Les empreintes SHA-256 des JSONL d'entraînement et de validation sont conservées dans les nouveaux checkpoints et vérifiées à la reprise. Une empreinte distincte couvre le contenu des images externes référencées par `image_path` ; une modification détectée empêche la reprise. Les anciens checkpoints restent utilisables avec un avertissement si leur empreinte d'images manque.

Le lecteur indexé contrôle aussi l'identité et les métadonnées du JSONL lors de son indexation et autour de chaque lecture. Un remplacement ou une modification détectée interrompt la lecture, même si la taille et la date de modification ont été conservées. Ces contrôles dépendent des métadonnées du système de fichiers et ne remplacent pas le stockage stable des jeux de données pendant une exécution.

Pour adapter des poids existants avec un nouvel optimiseur, un nouveau planning et de nouvelles données :

```bash
.venv/bin/python -m chimera train --init-from runs/custom/model.pt \
  --data adaptation/train.jsonl --val-data adaptation/validation.jsonl \
  --steps 100 --lr 0.0001 --output runs/adaptation
```

`--init-from` accepte un checkpoint complet ou un export d'inférence et utilise sa configuration. L'étape repart de zéro et la provenance est enregistrée. `--resume` conserve l'état d'un entraînement interrompu ; les deux options sont exclusives.

Entraînement distribué DDP :

```bash
# Deux processus CPU ; chemin vérifié localement
.venv/bin/torchrun --standalone --nproc-per-node=2 -m chimera --threads 1 train \
  --preset tiny --steps 3 --batch-size 2 --warmup 1 --accumulation 2 \
  --output runs/ddp-smoke --log-every 1

# Exemple de deux GPU avec PyTorch CUDA installé
.venv/bin/torchrun --standalone --nproc-per-node=2 -m chimera train \
  --preset small --device cuda --amp bf16 --steps 1000 --output runs/gpu
```

DDP réplique le modèle sur chaque processus ; il ne répartit pas les experts ni les poids entre cartes. Les nouveaux checkpoints distribués sauvegardent les états aléatoires Python et PyTorch propres à chaque rang, y compris le générateur CUDA utilisé par ce rang. La reprise nécessite le même nombre de processus. Un test CPU à deux rangs compare bit à bit une exécution continue avec des reprises depuis `latest.pt` et `best.pt`, avec dropout, accumulation et données mélangées. Les anciens checkpoints distribués restent chargeables mais ne permettent pas une reprise exacte des états aléatoires ; un avertissement le signale. Sur GPU, la reproductibilité dépend aussi des algorithmes et de l'environnement et n'a pas été vérifiée ici.

Les configurations dépassant la mémoire d'un accélérateur nécessitent un système de partitionnement supplémentaire ; FSDP, parallélisme tensoriel et parallélisme d'experts ne sont pas implémentés ici. Les masques d'attention locale restent denses pendant le préremplissage et le traitement de plusieurs nouveaux tokens. Pendant la génération, les couches locales ne conservent que `sliding_window - 1` anciennes positions ; les couches globales gardent le contexte complet. Les clés/valeurs du contexte multimodal sont projetées une seule fois par couche.

## Données et API

Un fichier JSONL contient un objet par ligne. Les commandes d'ajustement exigent par défaut les mêmes modalités dans tout le fichier ; `train --mixed-modalities`, `predict`, `embed` et `evaluate` acceptent des combinaisons différentes. Chaque modalité est facultative, mais au moins une est requise. Les champs `class_label` et `future` peuvent être omis pour certains exemples : leurs cibles sont alors ignorées dans les pertes et les mesures. Avec la normalisation par défaut, chaque micro-lot d'entraînement doit contenir au moins une cible supervisée observée ; le mode `observed` impose cette condition à l'étape entière, tous rangs confondus.

La commande `embed` écrit une ligne JSON par exemple, dans l'ordre du fichier, avec `index` et un dictionnaire `vectors` contenant `text`, `image` et/ou `series`. Chaque vecteur est la projection contrastive en FP32, normalisée lorsque sa norme n'est pas négligeable ; une projection nulle reste nulle. Le texte est toujours encodé avec un contexte isolé, même si le checkpoint indique `alignment_text_view: "joint"` : dans ce dernier cas, cette vue textuelle n'a pas été directement optimisée par l'alignement et sa qualité pour la recherche doit être vérifiée. L'image et la série sont également encodées indépendamment des autres modalités. Des vecteurs peuvent être comparés par produit scalaire lorsqu'ils sont non nuls, mais un modèle non entraîné ou un alignement insuffisant ne garantit aucune pertinence sémantique.

Chaque ligne indique les empreintes SHA-256 du checkpoint et du JSONL, celle des images externes s'il y en a, ainsi que la vue textuelle d'entraînement, AMP, l'appareil, la taille des lots et la version de PyTorch. Comparer des vecteurs issus du même checkpoint et de la même précision AMP ; appareil et taille de lot peuvent introduire de petits écarts numériques. Le checkpoint et le jeu sont revérifiés à la fin, ce qui relit également les images externes. Une sortie fichier est remplacée seulement après succès ; sur stdout, des lignes peuvent déjà avoir été émises en cas d'erreur tardive. L'API `model.encode_modalities(...)` expose les mêmes vecteurs sans changer le mode train/eval ni désactiver les gradients.

`search-embeddings` compare deux sorties `embed`, éventuellement issues de jeux distincts, par cosinus exact. Chaque ligne de résultat contient l'indice de la requête, les modalités et les empreintes des deux fichiers de vecteurs, puis `matches`, classés par cosinus décroissant et indice candidat croissant en cas d'égalité. `--exclude-self` exclut les lignes de même indice uniquement lorsque les empreintes du jeu et de ses images correspondent ; cela peut exclure une paire texte/série effectivement associée. Les fichiers doivent partager checkpoint et AMP ; les dimensions, métadonnées et valeurs sont vérifiées, et les vecteurs nuls sont refusés. Par défaut, la modalité choisie doit être présente sur chaque ligne. `--skip-missing` ignore les lignes sans cette modalité et conserve leurs indices d'origine dans les résultats ; il faut au moins une requête et un candidat disponibles. Un vecteur présent de la modalité choisie mais invalide reste une erreur. Les calculs utilisent une normalisation FP64 et parcourent les candidats une fois par lot de requêtes : la mémoire dépend des tailles de lots, au prix de lectures répétées et d'un coût quadratique en nombre de comparaisons. La sortie fichier est atomique ; stdout peut contenir des résultats partiels en cas d'erreur tardive.

`evaluate-retrieval` calcule les moyennes par requête du taux de réussite (`hit`), du rappel, de la précision, du rang réciproque et du NDCG binaires parmi les `top-k` voisins exacts. Le JSONL de pertinence commence par un en-tête contenant `format_version: 1`, `query_dataset_sha256`, `query_dataset_images_sha256`, `candidate_dataset_sha256` et `candidate_dataset_images_sha256`. Ces empreintes doivent être identiques aux champs `dataset_sha256` et `dataset_images_sha256` des fichiers de vecteurs correspondants ; les empreintes des images valent `null` en leur absence. Suivent les jugements, une ligne pour chaque requête dans le même ordre, par exemple `{"index":0,"relevant":[0,3]}` : les entiers désignent les indices des candidats pertinents. Une liste vide signifie que la requête n'est pas jugée et l'exclut des moyennes ; le rapport en donne le nombre. Les candidats omis d'une liste non vide sont considérés non pertinents : des annotations incomplètes peuvent donc biaiser les mesures. Il faut utiliser des jeux indépendants de l'entraînement pour estimer la qualité hors échantillon.

Exemple de fichier de pertinence pour deux requêtes sans images externes, en remplaçant les deux empreintes par celles des vecteurs :

```jsonl
{"format_version":1,"query_dataset_sha256":"<sha256 des requêtes>","query_dataset_images_sha256":null,"candidate_dataset_sha256":"<sha256 des candidats>","candidate_dataset_images_sha256":null}
{"index":0,"relevant":[0,3]}
{"index":1,"relevant":[]}
```

Avec `evaluate-retrieval --skip-missing`, le fichier de pertinence garde une ligne par requête d'origine, y compris celles qui ne possèdent pas la modalité recherchée. Ces dernières sont vérifiées puis ignorées. Pour une requête retenue, chaque candidat pertinent doit posséder la modalité candidate : sinon l'évaluation échoue plutôt que de réduire silencieusement le rappel. Le rapport indique les nombres totaux et ceux exclus pour modalité absente, séparément des requêtes sans jugement. Le contrôle conserve un octet par ligne de chaque fichier de vecteurs en mémoire.

La précision divise par `top-k`, le rappel par le nombre de candidats pertinents annotés et le rang réciproque est nul si aucun pertinent n'apparaît dans les `top-k`. Les indices répétés, hors limites ou manquants sont refusés. Avec `--exclude-self` sur un même jeu, aucune annotation d'une requête retenue ne peut désigner la paire exclue ; `top-k` ne doit pas dépasser le nombre de candidats admissibles pour chaque requête retenue. Le rapport conserve les empreintes des fichiers et les modalités. Une évaluation nécessite une lecture préalable complète des vecteurs puis les mêmes lectures répétées et calculs que `search-embeddings` ; elle ne démontre que la qualité sur les requêtes et les jugements fournis.

Le lecteur indexe les positions des lignes et charge les exemples à la demande : il garde environ 16 octets d'index par exemple, plus un octet quand les modalités hétérogènes sont activées, au lieu de conserver tous les tableaux Python. Le fichier est parcouru une fois à l'ouverture pour valider les modalités et calculer son empreinte. Chaque image externe distincte est lue une fois pour calculer son empreinte, ce qui ajoute un coût d'ouverture et un index en mémoire proportionnel au nombre de chemins d'images distincts. Le lecteur vérifie aussi l'identité des fichiers d'image avant et après leur décodage. Les erreurs indiquent le fichier et la ligne. Les commandes `predict`, `embed`, `evaluate`, `calibrate` et `refine-forecast` refusent d'écrire sur une image du jeu, y compris via un lien physique ou symbolique. Les sorties de prédiction et d'évaluation remplacent leur fichier uniquement après une exécution réussie.

| Champ | Contenu |
| --- | --- |
| `text` | Chaîne UTF-8 ; BOS/EOS et cibles de texte ajoutés automatiquement |
| `image` | Tableau flottant `[canaux, hauteur, largeur]`, déjà à `image_size` |
| `image_path` | Alternative : PNG/JPEG relatif au JSONL, converti et redimensionné ; nécessite `pip install '.[vision]'` |
| `series` | Historique `[temps, series_features]`, du plus ancien au plus récent |
| `class_label` | Classe entière de `0` à `num_classes - 1` ; `-100` pour ignorer |
| `future` | Cible `[forecast_horizon, series_features]`, strictement après l'historique |

Les textes longs sont tronqués à `max_seq_len` en conservant EOS. Les historiques trop longs sont refusés pour éviter une troncature silencieuse. `null` dans un tableau numérique est converti en valeur manquante par l'outil de génération/chargement ; les tenseurs d'historique acceptent NaN. Les cibles de prévision non finies sont ignorées dans la perte.

Pour construire des fenêtres de séries sans réutiliser de positions entre entraînement, validation, calibration et test, `prepare-series` lit un JSONL contenant une série complète par ligne. Chaque objet contient un `series_id` unique, chaîne non vide ou entier, et `values`, tableau chronologique de vecteurs numériques ou `null`. Chaque vecteur doit avoir le nombre de variables de la configuration ; les valeurs numériques doivent être finies et représentables en FP32.

```bash
.venv/bin/python -m chimera prepare-series \
  --input data/series.jsonl --output-dir data/chronological-splits \
  --config configs/tiny.json --history 32 --stride 12 --gap 12 \
  --train-end 1000 --validation-end 1300 --calibration-end 1600
```

Les coupures sont des indices de positions à partir de zéro, avec fin exclusive. Pour chaque série, les périodes sont `[0, train_end[`, `[train_end + gap, validation_end[`, `[validation_end + gap, calibration_end[` et `[calibration_end + gap, longueur[`. Chaque fenêtre inclut `history` pas d'entrée et l'horizon de la configuration ; elle doit tenir entièrement dans une période. `stride` règle son décalage au sein de chaque période. Les fenêtres sans aucune cible observée sont ignorées et comptées ; chaque série doit fournir au moins une fenêtre supervisée par période. La calibration peut demander davantage de cibles observées par horizon et variable.

Le nouveau dossier contient `train.jsonl`, `validation.jsonl`, `calibration.jsonl`, `test.jsonl` et `manifest.json`. Le manifeste contient les réglages, les effectifs et les empreintes ; chaque ligne produite porte aussi `series_id`, `window_start`, `forecast_start` et `window_end`, ignorés par le modèle et utilisables par `audit-splits`. Le dossier de sortie ne doit pas déjà exister. Les fichiers sont préparés avant publication par liens physiques, ce qui nécessite leur prise en charge par le système de fichiers ; le manifeste est publié en dernier. Une interruption brutale peut laisser un dossier incomplet sans manifeste.

Ces indices suivent l'ordre fourni : le programme ne vérifie ni les horodatages ni leur régularité. Pour une chronologie commune entre plusieurs séries, leurs positions doivent correspondre aux mêmes dates et intervalles. Des fenêtres d'un même jeu peuvent se chevaucher, et des groupes différents peuvent être liés ; la séparation des positions ne prouve donc pas l'indépendance statistique ni les hypothèses d'une garantie conformelle.

Avant l'entraînement ou la calibration, comparer les jeux pour repérer des exemples dont les champs choisis sont identiques, même si les lignes sont réordonnées ou le JSON reformaté :

```bash
.venv/bin/python -m chimera audit-splits \
  --split train=data/train.jsonl --split calibration=data/calibration.jsonl --split test=data/test.jsonl \
  --fields series future --output runs/split-audit.json --fail-on-overlap
```

Par défaut, la comparaison utilise les champs lus par le modèle ; `--fields` sélectionne d'autres champs présents dans chaque ligne. Les nombres JSON égaux sont comparés sans perte décimale, les doublons sont comptés et les rapports indiquent des numéros de lignes sans recopier les valeurs. Le code de sortie vaut 2 si `--fail-on-overlap` détecte un partage ; le rapport est écrit avant la sortie. Cette option n'est pas automatiquement appliquée à `train` ou `calibrate`.

Pour détecter aussi des fenêtres différentes qui se recouvrent, les lignes peuvent porter des métadonnées numériques supplémentaires représentant **toute la fenêtre, historique et futur compris** : par exemple `series_id`, `window_start` et `window_end`. Ajouter `--interval-fields series_id window_start window_end` compare les intervalles semi-ouverts `[start, end[` par série et par paire de jeux. Les bornes doivent utiliser les mêmes unités ; deux fenêtres qui se touchent seulement en une borne ne se recouvrent pas. Le rapport compte les lignes concernées dans chaque jeu, pas toutes les paires possibles. Ces métadonnées ne sont pas utilisées comme entrées du modèle.

L'audit conserve des empreintes par exemple et, si demandé, les intervalles en mémoire : son coût mémoire grandit avec la taille des jeux. Il ne lit pas le contenu des images externes et ne valide pas les métadonnées temporelles contre les données brutes. Des images identiques portant des chemins différents, des fenêtres proches, des copies modifiées ou des sujets apparentés peuvent lui échapper. L'absence de correspondance ne prouve pas l'indépendance statistique.

Pour les séries d'amplitude courante, les statistiques temporelles sont calculées en FP32 comme auparavant. Si leur amplitude risque de faire déborder la somme des carrés, elles passent en FP64 ; les prévisions associées peuvent alors être en FP64. Les amplitudes dépassant la plage sûre du calcul FP64 sont refusées explicitement. Ce contrôle d'amplitude peut ajouter une synchronisation sur GPU.

Utiliser directement les tenseurs :

```python
import torch
from chimera import ByteTokenizer, Chimera, ChimeraConfig

config = ChimeraConfig.preset("tiny")
model = Chimera(config)
tokenizer = ByteTokenizer()
tokens = torch.tensor([tokenizer.encode("Décris ce signal.")])

result = model(
    tokens,
    images=torch.rand(1, 3, 32, 32),
    series=torch.randn(1, 24, 3),
    labels=tokens,                         # décalage causal effectué par le modèle
    class_labels=torch.tensor([2]),
    forecast_targets=torch.randn(1, 12, 3),
)
result.loss.backward()
print(result.logits.shape)                  # [1, tokens, 260]
print(result.classification_logits.shape)   # [1, 6]
print(result.forecast.shape)                # [1, 12, 3, 3] : q10, q50, q90

model.eval()
with torch.inference_mode():
    vision = model(images=torch.rand(1, 3, 32, 32))
    temporal = model(series=torch.randn(1, 24, 3))
    # Le futur peut aussi être conditionné sur une image et/ou du texte.
```

La classe de vision correspond aux étiquettes du jeu d'entraînement ; le modèle ne produit pas de nouvelles images. Les intervalles de prévision sont des quantiles appris, pas des garanties de couverture statistique. `attention_mask` et `series_mask` sont binaires `[B, T]`. Les valeurs manquantes par variable sont représentées explicitement. Les positions de padding ne participent pas au routage ni aux pertes de texte.

`evaluate` rapporte les pertes supervisées pondérées par leurs cibles réellement observées, l'accuracy, la MAE et la RMSE du futur, la MAE par horizon, la couverture et la largeur des intervalles. La RMSE met à l'échelle les erreurs avant de les élever au carré, puis combine les lots à échelle commune pour limiter les débordements et sous-dépassements intermédiaires ; ce calcul peut synchroniser le GPU. Une différence individuelle qui dépasse FP64 reste non finie ; les autres sommes de mesures conservent également leurs limites de précision FP64. Pour `multi_token`, la perte de chaque horizon est d'abord pondérée par ses cibles observées dans tout le jeu, puis les horizons observés sont moyennés. Une tâche sans cible valide a une perte et des mesures de qualité `null`. La perte totale reste la moyenne par exemple des pertes totales de chaque lot : elle dépend donc du découpage en lots lorsque les nombres de cibles observées varient. Les termes de routage/alignement sont également moyennés par exemple ; l'alignement contrastif dépend de la composition des lots.

L'option `evaluate --iid-intervals` ajoute des intervalles bilatéraux de Wilson à 95 % pour l'accuracy de classification et, si disponibles, l'accuracy de la tête temporelle et de sa porte de sélection, la couverture de ses ensembles et leur accuracy lorsqu'ils contiennent une seule classe. Les couvertures par classe ont chacune leur propre intervalle ; sans exemple étiqueté, la valeur est `null`. Ces intervalles approximatifs supposent des exemples de test étiquetés indépendants et identiquement distribués, pour un modèle et des calibrations fixés avant le test et indépendants du jeu de test. Ils ne corrigent ni une sélection biaisée des étiquettes, ni la dépendance temporelle, ni les comparaisons simultanées entre classes. Voir la [description de la méthode de Wilson par le NIST](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

Le rapport inclut aussi `expert_usage`, la part des sélections de chaque expert par couche, pondérée par les tokens valides ; l'expert partagé est exclu. `expert_routed_tokens` compte les tokens du décodeur, sans padding, et non les sélections multipliées par `top_k`. `expert_effective_count` vaut l'exponentielle de l'entropie de ces parts ; `expert_max_load_ratio` compare la part maximale à une répartition uniforme. Ces diagnostics décrivent la répartition des sélections, pas leur importance causale ni la qualité du modèle.

Génération avec cache clé/valeur **et** état de convolution :

```bash
.venv/bin/python -m chimera generate --checkpoint runs/workstation/model.pt \
  --prompt "Signal de classe " --max-new-tokens 32 --temperature 0
```

`model.generate(...)` accepte aussi `images=` et `series=` ; le contexte est calculé une seule fois. Température, top-k, top-p et arrêt EOS sont disponibles. Les batches de génération de longueurs différentes utilisent du padding à gauche. Le cache est réservé à l'inférence et sa longueur reste limitée par `max_seq_len`.

Pour l'échantillonnage, les logits sont centrés avant la division par la température ; les températures positives hors plage FP32 utilisent FP64. Une température nulle conserve la sélection gloutonne. Un maximum de logits non fini pendant l'échantillonnage produit une erreur explicite.

`load_checkpoint` utilise le mappage mémoire et une initialisation sur `meta` pour charger les poids sans copie complète et sans initialisation aléatoire préalable. Les pages sont lues à la demande ; la première inférence utilise ensuite la mémoire des poids consultés. `mmap=False` permet de charger un ancien fichier PyTorch qui n'utilise pas le format ZIP.

## Fichiers utiles

- [Modèle et pertes](chimera/model.py), [briques du réseau](chimera/modules.py), [configuration](chimera/config.py).
- [Entraînement et sauvegardes](chimera/training.py), [données](chimera/data.py), [CLI](chimera/cli.py).
- [Tests](tests/test_chimera.py), [protocole de mesure](tools/benchmark.py), [comptages exacts](artifacts/architectures.json).
- [Rapport de validation](VALIDATION.md).
- [Améliorations et mesures de mémoire](IMPROVEMENTS.md), [profilage reproductible](tools/profile_runtime.py).
- [CI Python 3.11 et 3.14](.github/workflows/tests.yml) : tests, inspection des presets et vérification DDP à chaque push ou pull request.

## Références

Cette implémentation combine des idées existantes ; ce n'est pas une reproduction des modèles de ces articles.

- [GQA — Ainslie et al.](https://arxiv.org/abs/2305.13245) : groupes de têtes clé/valeur.
- [RoFormer — Su et al.](https://arxiv.org/abs/2104.09864) : positions rotatives.
- [Switch Transformers — Fedus et al.](https://arxiv.org/abs/2101.03961) : routage et équilibrage des experts.
- [ST-MoE — Zoph et al.](https://arxiv.org/abs/2202.08906) : stabilité du routage.
- [Perceiver IO — Jaegle et al.](https://arxiv.org/abs/2107.14795) : compression et interrogation d'espaces latents.
- [Attention SDPA de PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) et [recalcul d'activations](https://docs.pytorch.org/docs/stable/checkpoint).
