# Ingestion de documents — `ingest.py`

Alimente la mémoire long-terme de Lythéa avec tes propres documents.
Une fois ingérés, ils deviennent consultables par Lythéa pendant ses
réponses, exactement comme ses souvenirs de conversation — mais
persistants et thématiques.

## À quoi ça sert

- **Spécialiser Lythéa sur une entreprise** : ingère procédures,
  rapports, specs, documentation interne. Lythéa répond ensuite en
  s'appuyant sur ce corpus.
- **Analyse scientifique** : charge des articles, des jeux de
  résultats, des notes de recherche. Lythéa peut alors aider à
  croiser les informations entre documents.
- **Base de connaissance de référence** : un socle stable de faits
  que Lythéa garde en mémoire indépendamment des conversations.

## Prérequis

Le script utilise la même base ChromaDB que Lythéa. Aucun modèle
Lythéa n'a besoin d'être chargé — ChromaDB calcule lui-même les
embeddings.

Dépendances (installées par `deploy.sh`) :
- `chromadb` — déjà requis par Lythéa
- `pdfplumber` — pour les PDF
- `python-docx` — pour les `.docx`

`.txt`, `.md`, `.rst` ne nécessitent aucune dépendance supplémentaire.

## Utilisation

### Ingérer des documents

```bash
# Un dossier entier (récursif)
python3 ingest.py mes_documents/

# Un seul fichier
python3 ingest.py rapport_annuel_2025.pdf

# Avec une étiquette thématique
python3 ingest.py --tag physique_quantique articles_scientifiques/
python3 ingest.py --tag rh procedures_internes/
```

Formats reconnus : `.pdf`, `.txt`, `.md`, `.markdown`, `.rst`, `.docx`.

### Gérer ce qui est ingéré

```bash
# Lister les documents déjà en mémoire
python3 ingest.py --list

# Purger les anciens documents puis réingérer (mise à jour propre)
python3 ingest.py --reset mes_documents/

# Tout supprimer (sans réingérer)
python3 ingest.py --purge
```

`--purge` et `--reset` ne suppriment **que** les documents ingérés
(`type=knowledge`). Les souvenirs de conversation de Lythéa restent
intacts.

## Comment ça marche

1. **Extraction** — le texte est extrait page par page (PDF) ou
   paragraphe par paragraphe (docx, txt, md).
2. **Assemblage** — toutes les pages sont recollées en un seul texte
   continu *avant* le découpage. C'est important : une phrase coupée
   en bas de page se reconstitue au lieu d'être tronquée. Les césures
   typographiques (`mot-\nsuite`) et les retours à la ligne en pleine
   phrase sont réparés ; les vraies ruptures (fin de phrase,
   paragraphes) sont conservées.
3. **Chunking** — le texte assemblé est découpé en morceaux d'environ
   800 caractères, avec un chevauchement de 150 caractères. Les
   coupures se font sur des frontières naturelles (fin de
   paragraphe, sinon fin de phrase, sinon espace) pour ne jamais
   casser une idée en plein milieu.
4. **Indexation** — chaque chunk est ajouté à ChromaDB avec des
   metadata : `type=knowledge`, `source` (nom du fichier), `page`
   (numéro de page où le chunk commence — retrouvé via une table
   d'offsets, donc l'info de localisation n'est pas perdue malgré
   l'assemblage), `tag` éventuel, et un horodatage.
5. **Disponibilité** — Lythéa exploite ces chunks dès sa requête
   suivante, via sa Phase B (RAG). L'index BM25 se reconstruit
   automatiquement.

## Idempotence

Réingérer le même fichier **remplace** ses chunks au lieu de créer
des doublons. L'identifiant de chaque chunk est dérivé d'un hash du
nom + contenu du fichier : si le fichier n'a pas changé, les ids
sont identiques et ChromaDB écrase proprement.

Tu peux donc relancer l'ingestion d'un dossier autant de fois que
tu veux : seuls les fichiers nouveaux ou modifiés ajoutent du
contenu.

## Conseils d'usage

- **Tags** : utilise `--tag` pour séparer logiquement tes corpus
  (`--tag client_A`, `--tag veille_techno`…). Pratique pour t'y
  retrouver avec `--list`.
- **PDF scannés** : si un PDF est une image scannée (pas de couche
  texte), l'extraction renverra du vide. Il faut d'abord l'OCR-iser.
- **Gros corpus** : l'ingestion est séquentielle et affiche sa
  progression. Pour des centaines de documents, lance-la dans un
  terminal détaché (`tmux`, `screen`).
- **Mise à jour** : quand un document source change, relance
  `ingest.py` dessus — le remplacement est automatique. Pour
  repartir totalement de zéro, `--reset`.

## Exemple de workflow — spécialisation entreprise

```bash
# 1. Rassembler les documents de référence dans un dossier
mkdir corpus_entreprise/
#    (y déposer les PDF, docx, etc.)

# 2. Ingérer avec un tag
python3 ingest.py --tag entreprise corpus_entreprise/

# 3. Vérifier
python3 ingest.py --list

# 4. Démarrer Lythéa — elle s'appuiera sur ce corpus
bash launch.sh

# 5. Plus tard, quand les documents évoluent
python3 ingest.py --reset --tag entreprise corpus_entreprise/
```
