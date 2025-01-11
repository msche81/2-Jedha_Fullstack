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

random.shuffle(NAMES)

groups = [NAMES[i:i+GROUPS_SIZE] for i in range(0, len(NAMES), GROUPS_SIZE)]

if len(groups[-1])== 1:
    alone_person = groups.pop[-1]
    groups[-1] += alone_person

for i,group in enumerate(groups):

    print('Group {} is composed of {}'.format(i.group))
