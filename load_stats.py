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
    print(f"Population in {len(STATS_COMMUNITIES_POP)} communities = {sum(STATS_COMMUNITIES_POP[1:])} persons")
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

    When an individual becomes symptomatic.

        dSPdt = sigma * A - tau * SP - gamma1 * SP



    This affects our "exposed" and "asymptomatic" compartment.

        dEdt = beta * S * (A+SP) / N - rho * E
        dAdt = rho * E - sigma * A - gamma4 * A

    When someone is in E and isolation comes in, rho is affected.

    I assume :
    - All workers are at home 75% of time
    - Non workers are at home 75% of time


    new infections = beta * (A + SP)
    A don't move.
    SP changes

    => new infections = beta * (A + SP) = (beta * A + beta2 * SP)


             |------@ home------|------@ work / school---------------------------|---@communities---|
    normal       0.70*beta           0.25*beta*(prop_work + prop_school)                0.05*beta

    isolated     0.75*beta2*0.25     0.25*beta2*0             0.05*beta2*0.1

    company size =
    - each worker belong to one company
    - given the company C, a worker can potentially
      infect C_size colleagues
    - it will infect 0 colleague if isolated

    household size =
    - each person belong to one household
    - given H, it can potentially infect H_size person
    - given beta2, it will infect round(beta2*H_size) person


    Cas actuel :

    Si on a 350 personnes infectées aujourd'hui, on aura
    350* beta personne infectés en PLUS demain (total l350 + 350*beta)

    [00000000001010010001111110001010101010101010101010]

    Comanies A,B,C,D
    [AAABCDABA.........................................]

    Soit un vecteur S de 1000000 de booléen, tous True au début
    Household : S[0:4] représente un household; S[4:6] représente le household suivant, etc.

    soit ndx les indices où S[i] == 1 .
    on sélectionne beta indices qui passeront dans l'état suivant, vecteur E



       X X            X         X
    |-4p-----|-2p--|-4p----|...

    (faire un tirage pour déterminer la taille du household)

    Dans un household de 4 personnes :
    - il peut y avoir 1,2,3 ou 4 personnes infectées : soint n_i personnes infectées.
    - si aujourd'hui il y a n_i personnes infectées, demain, il y en aura min( 4, n_i + 0.7*beta*n_i) si pas d'isolation.
    - si isolation : min( 4, n_i + 0.75*0.7*beta*n_i)




    """


"""
class Person:
    def __init__(self):
        self._id = 12334
        self.company = XXX
        self.household = XXX
        self.school = XXX
        self.state = S/E/A/SP/H/R/C/F

    def infected(self):
        return self.state != S


infectious_symptomatic = [person...] # symptomatiques ET asymptomatiques
infectious_asymptomatic = [person...]

companies = dict( company_id => [person,...] )
school = dict( school_id => [person,...] )


dEdt = 0

quota =
infection_rate = beta*0.13 # or less/zero if isolated
for ip in infectious_symptomatic:

    nb_to_infect = binom(...len(colleagues) * infection_rate...)
    # not sick colleagues
    colleagues = [c for c companies[ip.company] if not c.infected]
    new_sicks = random.pick(colleagues, nb_to_infect)

    for ns in new_sicks:
        ns.state = E
        dEdt += 1

infection_rate = beta*0.02
for ip in infectious_asymptomatic:

    nb_to_infect = binom(...len(colleagues) * infection_rate...)
    # not sick colleagues
    colleagues = [c for c companies[ip.company] if not c.infected]
    random.pick(colleagues, nb_to_infect)

    for ns in new_sicks:
        ns.state = E
        dEdt += 1
"""
