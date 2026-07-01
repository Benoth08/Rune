"""Verrou de génération PARTAGÉ (chat + agent).

Module volontairement minuscule et SANS dépendance lourde (pas de torch) : il est
importé par model.py, agentic/workers.py et server/routes.py sans créer de cycle
ni tirer tout le moteur. Un seul GPU/modèle in-process — deux générations
simultanées entrelaceraient leurs forward-pass ET les hooks à état partagé
(latent_state, file de capture, hook d'entropie) → corruption. L'agent prend ce
verrou systématiquement ; le chat le prend aussi par défaut
(`agent_chat_shared_lock_enabled`, ON), ce qui permet de chatter pendant une
mission sans collision.
"""
from __future__ import annotations

import threading

GENERATION_LOCK = threading.Lock()
