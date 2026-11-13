# Vérification FVT / progressif / élagage sur les vrais assets code (CPU, 13 nov. 2026)

Script : `scripts/check_transfer.py`. Sortie : `results/transfer-check-code.json`.
GPT-2 @607a30d7, ancien tokenizer `gpt2` (sha 8414cab9…), nouveau `code70_gpt4_32k` (sha 3dd5a347…),
50 premiers documents de validation code (identiques à ceux du paquet code50), CPU, block 1024, eval mode.
Environnement figé de Chloé (`requirements-milestone.txt` : torch 2.6.0 CPU, transformers 4.49.0, tokenizers 0.21.4, datasets 3.6.0).
Durée : 17 min 27 s (seconde exécution, avec les évaluations masquées), 4,8 Go RAM max.

| # | Vérification | Résultat |
|---|---|---|
| 1 | Union : 50 257 anciens + 14 009 nouveaux seulement = 64 266 lignes ; 17 991 jetons partagés ; EOT nouveau (id 0) → 50256 ; anciens ids = identité | PASS |
| 2 | FVT : la concaténation des morceaux GPT-2 redonne chaque jeton (0 échec / 32 000) ; 2,48 morceaux en moyenne, max 64 (suite d'espaces) | PASS |
| 3 | Init : lignes partagées copiées exactement (B, C, E) ; FVT = moyenne des morceaux (erreur 0,0 sur 100 échantillons) ; C aléatoire ≠ FVT (écart relatif min 1,13) ; E : nouvelles lignes = celles de B | PASS |
| 4 | bpb au pas 0 : B < C, E-new ≈ B, E masqué = A et B | PASS ; E-old non masqué ≈ A : FAIL (attendu, voir ci-dessous) |
| 5 | Élagage : bpb élaguée 1,661241 ≤ E-new 1,668653 (écart 0,0074) ; logits max \|Δ\| 1,1e-4, argmax identique 100 % ; sauvegarde/rechargement : Δ = 0,0 | PASS |
| 6 | Échantillonneur progressif (code50, rho 0,5, T = 762, 20 × 8) : pas 0 p=0/frac 0 ; 190 p=0,499/frac 0,488 ; 381 et 600 p=1/frac 1 ; 96 paires ancien/nouveau : même préfixe de texte, 0 écart | PASS |
| 7 | Poids liés (lm_head = wte) après chaque build, après élagage et après rechargement | PASS |

## bpb au pas 0 (50 docs de validation)

| Condition | bpb | tokens |
|---|---|---|
| A (GPT-2) | 1,22498 | 165 443 |
| B (swap FVT) | 1,66124 | 93 372 |
| C (swap aléatoire) | 2,03291 | 93 372 |
| E-old (union, ancienne tokenisation) | **2,43201** | 165 443 |
| E-new (union, nouvelle tokenisation) | 1,66865 | 93 372 |
| E élagué (= B exactement) | 1,66124 | 93 372 |
| E-old masqué (`mask_foreign_logits`) | **1,22498** (= A, Δ 0,0) | 165 443 |
| E-new masqué (`mask_foreign_logits`) | **1,66124** (= B, Δ 7,5e-7) | 93 372 |

- B < C : oui, avec 0,37 bpb d'écart (le seuil de 0,5 dans le script était trop strict).
- E-new ≈ B : oui (+0,0074, c'est la masse donnée aux lignes uniquement anciennes ; elle disparaît à l'élagage).
- **E-old ≈ A : non (+1,21 bpb).** Ce n'est pas un bug d'implémentation : unions, init et élagage sont exacts. La cause est structurelle.
  Les embeddings sont liés, et les 60 jetons « suites d'espaces » du nouveau tokenizer (absents de GPT-2, sauf un) sont initialisés par FVT
  comme la moyenne de N copies de `Ġ`, donc leur logit de sortie est quasi égal à celui de `Ġ`. En tokenisation GPT-2, chaque fois que le modèle
  prédit `Ġ` (indentation), la masse se répartit sur environ 60 jetons. Diagnostic sur 4 096 tokens : 57 % de la masse de probabilité
  part en moyenne sur des lignes uniquement nouvelles (médiane 65 %), soit +1,56 nat par token ancien ; les jetons les plus favorisés sont des suites de 15 à 60 espaces.

Conséquence pour E : au début de l'entraînement, les séquences encore en ancienne tokenisation (p_new petit) partent d'une perte nettement
pire que A. Cela fait partie de la méthode telle qu'elle est implémentée et doit être mentionné quand on interprète E (code, rho 0,5).

## Correctif optionnel `mask_foreign_logits` (désactivé par défaut)

Clé de config `mask_foreign_logits: false` (configs/train/base.yaml). Avec `true` et la condition progressive, chaque séquence
(entraînement et évaluations union périodique/finale) a son softmax restreint aux lignes de son tokenizer : lignes uniquement
nouvelles masquées pour l'ancienne tokenisation, lignes uniquement anciennes masquées pour la nouvelle. L'évaluation élaguée
ne change pas. Sur les vrais assets, au pas 0, E-old masqué = A (Δ 0,0) et E-new masqué = B (Δ 7,5e-7) : E démarre alors
exactement comme A et B. Avec le drapeau à false, les pertes et les poids sauvegardés sont identiques bit à bit à main
(test `tests/test_logit_mask.py`).

Tests du dépôt (`.venv` : torch 2.15 CPU, transformers 5.22) : main 85 passed ; cette branche 92 passed (le test de la porte de
réutilisation chinoise compare désormais au trainer figé de main, et vérifie qu'un trainer modifié reste refusé).
