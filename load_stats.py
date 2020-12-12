import urllib.request
import json

with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/stats/age.json") as fp:
    split = [l.decode("utf-8").strip().split(",") for l in fp.readlines()]

    data = [(int(age), int(nb)) for age, nb in split]

    STATS_AGES = [nb for age, nb in data]


with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/stats/communities.json") as fp:
    data = json.load(fp)
    STATS_COMMUNITIES_POP = [None] * (len(data)+1)
    STATS_COMMUNITIES_FRAC = [None] * (len(data)+1)

    for d in data:
        STATS_COMMUNITIES_POP[ d['wardNo']] = d['totalPopulation']
        STATS_COMMUNITIES_FRAC[ d['wardNo']] = d['fracPopulation']


with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/stats/households.json") as fp:
    split = [line.decode("utf-8").strip().split(",")
             for line in fp.readlines()]
    data = [(int(size), int(nb)) for size, nb in split]

    # Households of size 0 don't exist, but I put
    # them in so that indexing works fine.
    STATS_HOUSEHOLDS = [0] + [nb for size, nb in data]


def parse_places(url):
    with urllib.request.urlopen(url) as fp:
        split = [line.decode("utf-8").strip().split(",")
                 for line in fp.readlines()]

        data = [([int(x) for x in size.split("-")], int(nb)) for size, nb in split]

    return data

STATS_SCHOOLS = parse_places("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/stats/schools.json")
STATS_WORKPLACES = parse_places("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/stats/workplaces.json")


if __name__ == "__main__":

    # STATS_AGES[ age ] = nb de personnes avec cet âge.
    print(STATS_AGES[10:15])
    print(f"Population = {sum(STATS_AGES)} persons")

    # STATS_COMMUNITIES_POP[i] = nbre de personne dans communauté i
    print(STATS_COMMUNITIES_POP[10:15])
    print(f"Population in communities = {sum(STATS_COMMUNITIES_POP[1:])} persons")
    print(STATS_COMMUNITIES_FRAC[10:15])

    # STATS_HOUSEHOLDS[i] = nbre de ménage de taille i
    print(STATS_HOUSEHOLDS[1:4])
    s = sum([i*STATS_HOUSEHOLDS[i] for i in range(len(STATS_HOUSEHOLDS))])
    print(f"Population in households = {s} persons")

    # STATS_SCHOOLS[i] = (bounds, nb) : nbre nb d'écoles dont le nbre d'élèves est ntre bound[0] et bound[1]

    print(STATS_SCHOOLS[10:15])
    s = 0
    for bounds, nb_schools in STATS_SCHOOLS:
        nb_students = (bounds[0] + bounds[1]) / 2
        s += nb_students * nb_schools
    print(f"Total students in schools = {s}")

    # WORKPLACES[i] = (bounds, nb) : nbre nb d'écoles dont le nbre d'élèves est ntre bound[0] et bound[1]
    print(STATS_WORKPLACES[10:15])
    s = 0
    for bounds, nb_workplaces in STATS_WORKPLACES:
        nb_workers = (bounds[0] + bounds[1]) / 2
        s += nb_workers * nb_workplaces
    print(f"Total workers at work = {s}")

    """
    # Case Isolation ---------------------------------------

    Isolated person infection rate is
    - 0.25 times less when at home
    - 0.90 times less when in communities
    - 1 times less when at work (so basically he doesn't go to school/work anymore)

    It has also a duration of isolation

    Assumptions :
    ------------

    This affects our "exposed" compartment.

        dEdt = beta * S * (A+SP) / N - rho * E

    When someone is in E and isolation comes in, rho is affected.

    I assume :
    - All workers are at home 16hours a day (and 8 hours a day at work)
    - Non workers are at home 16hours a day

    """
