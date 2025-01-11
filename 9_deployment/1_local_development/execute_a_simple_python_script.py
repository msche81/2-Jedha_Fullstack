def split_into_groups(names, n, p):
    """
    Split a list of names into n groups with p people in each group.

    :param names: List of names (elements of type str).
    :param n: Number of groups to form.
    :param p: Number of people per group.
    :return: A list of n lists, each containing p names.
    :raises ValueError: If the number of groups and people per group exceed the number of names.
    """
    if len(names) < n * p:
        raise ValueError("Not enough names to form the requested groups.")
    
    groups = []
    for i in range(n):
        group = names[i * p:(i + 1) * p]
        groups.append(group)
    return groups

# Exemple d'utilisation
names_list = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
n_groups = 2  # Nombre de groupes
p_per_group = 4  # Nombre de personnes par groupe

try:
    result = split_into_groups(names_list, n_groups, p_per_group)
    print(result)
except ValueError as e:
    print(e)
