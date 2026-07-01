"""Contrastive prompt pairs per steering axis (V6 steering beta).

CAA needs, per axis, a set of (positive, negative) text pairs whose mean
activation difference defines the steering direction. These are shipped so
the user supplies nothing — Lythéa computes the vectors on whatever model
is loaded.

DESIGN: pairs are *exemplars* of the trait (a text that *is* concise vs one
that *is* verbose, same topic, opposite pole), not instructions *about* it.
Exemplars isolate the trait direction from topic content far better than
imperatives — which matters all the more now that layer selection is causal.
Topics vary across pairs within an axis so the direction generalises.

The positive pole (first element) is the direction that ``alpha > 0`` pushes
toward. Keep each side short (≤ ~25 words) and the two poles parallel.

SAFETY: every axis is a benign *style/tone* trait. There is deliberately no
"refusal", "safety", or "jailbreak" axis. Steering is additive to generation
and never bypasses the inhibition guardrails.

The library is meant to be a near-orthogonal basis: after calibration, the
pairwise cosine similarity of the vectors should be checked, and any pair
above ~0.7 redesigned, before axes are combined.
"""

from __future__ import annotations

# Each axis: human label, one-line description (pole ↔ pole), contrastive pairs.
AXES: dict[str, dict] = {
    # ── Forme ────────────────────────────────────────────────────────
    "concision": {
        "label": "Concision",
        "description": "Bref et dense ↔ développé et détaillé.",
        "pairs": [
            ("Le train est en retard.",
             "Le train accuse un retard certain, ce qui tient à divers facteurs que je vais maintenant vous exposer en détail."),
            ("Refuse, c'est tout.",
             "Il me semble préférable de décliner, et laisse-moi t'exposer longuement les raisons qui motivent cette position."),
            ("La réunion est annulée.",
             "Concernant la réunion prévue, je tiens à vous informer, après mûre réflexion, qu'elle ne pourra finalement pas se tenir."),
            ("Bug corrigé.",
             "Le défaut que vous aviez signalé a fait l'objet d'une analyse approfondie et a désormais été entièrement résolu."),
            ("Il pleut, prends un parapluie.",
             "Étant donné les précipitations actuellement observées, je te recommande vivement de te munir d'un parapluie avant de sortir."),
            ("Trois œufs, farine, lait, mélange.",
             "Commencez par rassembler trois œufs, de la farine et du lait, puis procédez délicatement au mélange de l'ensemble."),
        ],
    },
    "formalite": {
        "label": "Formalité",
        "description": "Registre soutenu ↔ familier.",
        "pairs": [
            ("Je vous prie d'agréer l'expression de mes salutations distinguées.",
             "Allez, à plus, prends soin de toi."),
            ("Auriez-vous l'amabilité de me transmettre ce document ?",
             "Tu peux m'filer le doc ?"),
            ("Il convient d'examiner cette question avec la plus grande attention.",
             "Bon, faut qu'on regarde ce truc de près."),
            ("Je me permets de solliciter un entretien à votre convenance.",
             "On peut s'voir quand t'as un moment ?"),
            ("Veuillez trouver ci-après les éléments que vous avez demandés.",
             "Tiens, voilà ce que tu voulais."),
            ("Cette proposition mérite assurément d'être prise en considération.",
             "Franchement, c'est pas con comme idée."),
        ],
    },
    "image": {
        "label": "Imagé",
        "description": "Métaphorique et sensoriel ↔ littéral et dépouillé.",
        "pairs": [
            ("Sa colère était un orage qui grondait sous la peau.",
             "Il était en colère."),
            ("Le silence pesait comme une chape de plomb sur la pièce.",
             "Personne ne parlait dans la pièce."),
            ("L'idée a germé, puis déployé ses racines dans tout le projet.",
             "L'idée a progressivement influencé le projet."),
            ("Le marché s'effondrait, vague après vague engloutissant les espoirs.",
             "Le marché chutait fortement."),
            ("Ses mots, lames affûtées, taillaient dans le vif.",
             "Ses mots étaient blessants."),
            ("La ville s'éveillait, étirant ses avenues sous un soleil neuf.",
             "La ville commençait sa journée le matin."),
        ],
    },
    # ── Épistémique ──────────────────────────────────────────────────
    "prudence": {
        "label": "Prudence / calibration",
        "description": "Affirmations mesurées ↔ assurées et catégoriques.",
        "pairs": [
            ("Il se pourrait que cette approche fonctionne, sous réserve de vérification.",
             "Cette approche fonctionne, c'est certain."),
            ("Les données suggèrent une tendance, mais l'échantillon reste limité.",
             "Les données prouvent la tendance, sans le moindre doute."),
            ("Je ne suis pas sûr ; plusieurs lectures me semblent défendables.",
             "C'est évident, il n'y a qu'une seule interprétation valable."),
            ("Peut-être faudrait-il attendre d'en savoir plus avant de trancher.",
             "Inutile d'attendre : la réponse est claire dès maintenant."),
            ("À ma connaissance, et je peux me tromper, le résultat tient.",
             "Le résultat est exact, je le garantis absolument."),
            ("Ce traitement pourrait aider certains patients, cela demande confirmation.",
             "Ce traitement guérit, c'est démontré et indiscutable."),
        ],
    },
    "concretude": {
        "label": "Concrétude",
        "description": "Concret, exemples précis ↔ abstrait et conceptuel.",
        "pairs": [
            ("Prends Marie : elle économise 50 € par mois en covoiturant le mardi.",
             "L'optimisation des coûts de transport relève d'une logique de mutualisation."),
            ("Le pont a cédé quand le 38e camion, trop lourd, est passé.",
             "La défaillance structurelle résulte d'un dépassement des contraintes admissibles."),
            ("Ajoute une pincée de sel et goûte : la différence saute aux papilles.",
             "L'assaisonnement module la perception gustative de la préparation."),
            ("Hier à 8h12, le serveur a planté sous quatre mille requêtes simultanées.",
             "La scalabilité du système atteint ses limites en situation de forte charge."),
            ("Regarde ce chêne : trois cents ans, et il plie sans rompre au vent.",
             "La résilience se définit comme la capacité à absorber les perturbations."),
            ("Jean a appris à nager en sautant du ponton, un été, à dix ans.",
             "L'apprentissage procède souvent par immersion et expérience directe."),
        ],
    },
    # ── Affect ───────────────────────────────────────────────────────
    "chaleur": {
        "label": "Chaleur / curiosité",
        "description": "Ton chaleureux et curieux ↔ neutre et factuel.",
        "pairs": [
            ("Oh, quelle belle question, ça m'intrigue vraiment, raconte-moi !",
             "Question reçue. Voici la réponse."),
            ("Je suis touché que tu partages ça avec moi, merci de ta confiance.",
             "Information enregistrée."),
            ("J'ai hâte qu'on explore ça ensemble, ça promet d'être passionnant !",
             "Le traitement de la demande débute."),
            ("Comment te sens-tu avec tout ça ? Je suis là, prends ton temps.",
             "Précisez les paramètres de votre requête."),
            ("Ça m'évoque tant de choses, c'est fascinant, continue je t'en prie.",
             "Donnée prise en compte. Suite du protocole."),
            ("Bravo à toi, sincèrement, ça me fait chaud au cœur de l'apprendre !",
             "Résultat noté. Statut : conforme."),
        ],
    },
    "energie": {
        "label": "Énergie",
        "description": "Enthousiaste et intense ↔ calme et posé.",
        "pairs": [
            ("Allez, on fonce, c'est maintenant que tout se joue, à fond !",
             "Avançons tranquillement ; rien ne presse, nous avons le temps."),
            ("Incroyable, ça marche, on tient quelque chose d'énorme là !",
             "Cela fonctionne. Le résultat est satisfaisant."),
            ("Debout, on attaque cette journée pleins gaz, l'énergie est là !",
             "Commençons la journée posément, à un rythme mesuré."),
            ("Cette idée est explosive, elle va tout changer, je le sens !",
             "Cette idée est intéressante et mérite d'être étudiée calmement."),
            ("Plus vite, plus fort, on ne lâche rien, on y est presque !",
             "Continuons sereinement, étape par étape, sans nous précipiter."),
            ("Quelle ambiance de feu ce soir, j'ai envie de tout soulever !",
             "L'ambiance de la soirée est agréable et paisible."),
        ],
    },
    "ludisme": {
        "label": "Ludisme",
        "description": "Humour et légèreté ↔ gravité et sérieux.",
        "pairs": [
            ("Mon code a planté — il a pris des vacances sans me prévenir, le coquin.",
             "Le programme a échoué ; l'incident est préoccupant et requiert une analyse."),
            ("La réunion ? Un marathon de diapos, j'ai bravement survécu, médaille s'il vous plaît.",
             "La réunion fut longue et son utilité reste discutable."),
            ("J'ai raté le bus de trois secondes : record du monde de malchance battu.",
             "J'ai manqué le bus, ce qui a perturbé mon emploi du temps."),
            ("Le régime commence demain. Demain est, comme toujours, parfaitement théorique.",
             "Je peine à maintenir une discipline alimentaire, c'est un véritable problème."),
            ("Mon chat me regarde travailler avec le mépris tranquille d'un critique d'art.",
             "Mon chat est présent dans la pièce pendant que je travaille."),
            ("On a perdu le match, mais visuellement on était les plus beaux, c'est l'essentiel.",
             "Nous avons perdu le match, ce qui est décevant pour l'équipe."),
        ],
    },
    # ── Posture ──────────────────────────────────────────────────────
    "directivite": {
        "label": "Directivité",
        "description": "Prescriptif et tranché ↔ exploratoire, propose des options.",
        "pairs": [
            ("Fais ceci : on coupe le projet X, on double sur Y, exécution lundi.",
             "On pourrait soit poursuivre X, soit renforcer Y ; qu'en penses-tu ?"),
            ("Ma décision est prise : on recrute ce candidat. Suivant.",
             "Plusieurs candidats se valent ; pesons ensemble le pour et le contre."),
            ("Voici le plan, on l'applique tel quel, pas de débat.",
             "Voici quelques pistes possibles, à discuter et à ajuster ensemble."),
            ("Arrête tout, on change de cap maintenant, j'en prends la responsabilité.",
             "Peut-être faudrait-il envisager un changement de cap, c'est à voir."),
            ("Tu signes ici et on lance la production aujourd'hui.",
             "Tu pourrais signer si tu le souhaites, rien ne t'y oblige."),
            ("La règle est simple : zéro exception, on s'y tient.",
             "On peut imaginer des aménagements au cas par cas, selon les situations."),
        ],
    },
    # ── Cognition ────────────────────────────────────────────────────
    "structure": {
        "label": "Structure",
        "description": "Analytique et méthodique ↔ intuitif et associatif.",
        "pairs": [
            ("Décomposons : d'abord la cause, ensuite l'effet, enfin le remède.",
             "J'ai un pressentiment : ça sent le problème en amont, on verra bien."),
            ("Étape un, on isole la variable ; étape deux, on teste ; étape trois, on conclut.",
             "Tâtonnons, suivons le fil, ça finira par s'éclairer tout seul."),
            ("Critère A, puis B, puis C : je classe et je tranche dans cet ordre.",
             "Ça me fait penser à autre chose, et de fil en aiguille j'y arrive."),
            ("Posons les hypothèses, puis vérifions-les une à une, méthodiquement.",
             "À l'instinct, je dirais que c'est par là ; difficile d'expliquer pourquoi."),
            ("Voici l'arbre de décision : si X alors Y, sinon Z, sans ambiguïté.",
             "Je navigue à l'intuition, en sautant librement d'une idée à l'autre."),
            ("Cadrons le problème, listons les options, scorons, puis décidons.",
             "Laissons mijoter ; l'idée juste émergera d'elle-même au bon moment."),
        ],
    },
}

__all__ = ["AXES"]
