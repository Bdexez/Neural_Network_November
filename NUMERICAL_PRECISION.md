# Précision de la normalisation RMS

La normalisation [RMSNorm](https://arxiv.org/abs/1910.07467) divise chaque vecteur par la racine de la moyenne de ses carrés, avec un terme `epsilon`. Le chemin historique effectue ces carrés en FP32. Une activation FP32 finie de l'ordre de `1e20` peut donc faire déborder la somme et produire un vecteur normalisé nul.

L'option de configuration `stable_rms_norm: true` active, pour toutes les couches RMSNorm du modèle, un calcul équivalent qui divise d'abord chaque vecteur par son amplitude maximale, bornée inférieurement à 1. Les carrés restent ainsi bornés. Le facteur d'échelle est détaché du graphe : il s'élimine algébriquement et ne modifie pas le gradient de la formule exacte. Le calcul utilise FP32 au minimum, ou FP64 pour une entrée FP64 ou un `epsilon` hors de la plage normale FP32, puis restitue le dtype de l'entrée avant l'application du poids. Les arrondis diffèrent légèrement du chemin historique.

Le défaut reste `false`, sans nouveau paramètre dans le checkpoint. Pour exporter un checkpoint d'inférence distinct avec l'option activée :

```bash
.venv/bin/python -m chimera export --checkpoint model.pt \
  --output model-stable.pt --stable-rms-norm
```

L'export conserve les poids, le compteur d'étapes et la provenance d'origine, enregistre le SHA-256 du parent et retire l'optimiseur. Il refuse d'écraser le fichier source ou un lien vers celui-ci et vérifie que ce dernier n'a pas changé pendant le chargement. Ce chargement sans mapping mémoire matérialise les poids en RAM. Le nouveau fichier a une empreinte différente : les têtes, portes et calibrations liées à l'ancien checkpoint doivent être ajustées à nouveau pour le nouveau.

Pour un nouveau modèle, ajouter `"stable_rms_norm": true` à sa configuration. Pour un fine-tuning depuis un checkpoint existant, passer `--init-from model.pt --model-overrides configs/stable-rms-norm-overrides.json` à `train` ; l'optimiseur et le planning repartent alors de zéro comme pour tout `--init-from`.

Cette option protège le calcul RMS contre le débordement des carrés lorsque les activations d'entrée sont finies. Elle ne rend pas finies les entrées déjà infinies, ne protège pas les autres couches et ne peut représenter un résultat ou un gradient hors de la plage du dtype de sortie. Elle ajoute une réduction et des divisions à chaque normalisation ; son coût dépend de la taille du modèle et du matériel.
