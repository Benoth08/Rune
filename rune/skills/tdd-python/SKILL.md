---
name: tdd-python
description: Écrire et corriger du code Python en test-driven — écrire le test d'abord, le faire échouer, puis coder jusqu'au vert.
tags: [python, tests, pytest, tdd, debugging]
---

Quand la tâche consiste à créer ou corriger un module Python :

1. Écris d'abord le test (`test_<module>.py`) qui décrit le comportement
   attendu, AVANT le code. Couvre le cas nominal et au moins un cas limite.
2. Lance les tests : ils doivent ÉCHOUER pour la bonne raison (le module
   n'existe pas encore / le bug est reproduit). Un test qui passe d'emblée
   ne prouve rien.
3. Écris le minimum de code pour passer au vert. Pas de fonctionnalité non
   testée.
4. Relance. En cas d'échec, lis le message d'erreur en entier (type +
   ligne) avant de modifier ; corrige la cause, pas le symptôme.
5. Une fois vert, refactore si nécessaire — les tests restent le filet.

À éviter : coder puis « tester à la main » ; attraper une exception large
pour masquer l'erreur ; modifier le test pour qu'il passe au lieu de
corriger le code.
