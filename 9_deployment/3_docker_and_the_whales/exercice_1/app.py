import random

NAMES = [
    "Michel",
    "Patrick",
    "Jean-Marc",
    "Chantal",
    "Marie-Chantal",
    "Baby",
    "Hugo",
    "Antoine",
    "Sophie",
    "Abigail",
    "Raphaël",
    "Laurence",
    "Jay-Z"
]

GROUPS_SIZE = 3

# Mélanger les noms aléatoirement
random.shuffle(NAMES)

# Diviser les noms en groupes
groups = [NAMES[i:i + GROUPS_SIZE] for i in range(0, len(NAMES), GROUPS_SIZE)]

# Si le dernier groupe contient une seule personne, la réattribuer au groupe précédent
if len(groups[-1]) == 1:
    alone_person = groups.pop()  # Supprime et récupère le dernier groupe
    groups[-1] += alone_person  # Ajoute la personne seule au dernier groupe précédent

# Afficher les groupes
for i, group in enumerate(groups):
    print(f"Group {i + 1} is composed of {', '.join(group)}")