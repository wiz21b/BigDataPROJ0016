from collections import defaultdict
from enum import Enum
import random
import numpy as np

from load_stats import STATS_HOUSEHOLDS, STATS_WORKPLACES, \
    STATS_SCHOOLS, STATS_COMMUNITIES_POP


class Places(Enum):
    Workplace = 1
    HouseHold = 2
    School = 3
    Community = 4

class Person:
    def __init__(self):
        self._id = 12334
        self.workplace = None
        self.household = None
        self.school = None
        self.community = None
        self.state = "S"

    @property
    def infected(self):
        return self.state in ("E", "A")

    def infectables(self):
        assert self.infected, "Only an infected person can infect others"

        # All people that this person can infect
        t = {}

        if self.school:
            t[Places.School] = self.school.infectable()
        else:
            t[Places.School] = []

        if self.workplace:
            t[Places.Workplace] = self.workplace.infectable()
        else:
            t[Places.Workplace] = []

        if self.household:
            t[Places.HouseHold] = self.household.infectable()
        else:
            t[Places.HouseHold] = []

        if self.community:
            t[Places.Community] = self.community.infectable()
        else:
            t[Places.Community] = []

        return t


class GroupBase:
    def __init__(self):
        self._persons = set()

    def add_person(self, p: Person):
        self._persons.add(p)

    def infectable(self):
        # Return the list of people that can still be
        # infected in this group
        # May be an empty list

        return [p for p in self._persons if not p.infected]


class HouseHold(GroupBase):
    def __init__(self):
        super().__init__()



class WorkPlace(GroupBase):
    def __init__(self):
        super().__init__()


class School(GroupBase):
    def __init__(self):
        super().__init__()


class Community(GroupBase):
    def __init__(self):
        super().__init__()


class Groups:
    def __init__(self):
        self.schools = set()
        self.workplaces = set()
        self.households = set()
        self.communities = set()

    def status(self):
        print(f"{len(self.schools)} schools")
        print(f"{len(self.workplaces)} workplaces")
        print(f"{len(self.households)} households")
        print(f"{len(self.communities)} communities")

all_groups = Groups()

# --------------------------------------------------------------------
# Create 1000000 infectious people

print("Creating people")
persons = [ Person() for i in range(1000324)]

# --------------------------------------------------------------------
# Dropping people in workplaces

print("Creating workplaces")
nb_wp = 0
p_ndx = 0
for size, nb in STATS_WORKPLACES:

    if nb > 0:
        # FIXME Could be more random
        mean_size = (size[0] + size[1])//2
        for i in range(nb):
            wp = WorkPlace()
            all_groups.workplaces.add(wp)
            nb_wp += 1
            for j in range(mean_size):
                persons[p_ndx].workplace = wp
                wp.add_person(persons[p_ndx])
                p_ndx += 1

persons_in_companies = sum(1 for _ in filter(lambda p: p.workplace, persons))
print(f"{persons_in_companies} persons in {nb_wp} workplaces")

# --------------------------------------------------------------------
# Dropping people in schools

print("Creating schools")
persons_not_in_schools = [_ for _ in filter(
    lambda p: p.workplace is not None, persons)]

nb_sc = 0
p_ndx = 0
for size, nb in STATS_SCHOOLS:

    if nb > 0:
        # FIXME Could be more random
        mean_size = (size[0] + size[1])//2
        for i in range(nb):
            school = School()
            all_groups.schools.add(school)
            nb_sc += 1
            for j in range(mean_size):
                persons_not_in_schools[p_ndx].school = school
                school.add_person(persons_not_in_schools[p_ndx])
                p_ndx += 1


# --------------------------------------------------------------------
# Dropping people in households

print("Creating households")
nb_hh = 0
p_ndx = 0
for size, nb in enumerate(STATS_HOUSEHOLDS):
    for i in range(nb):
        hh = HouseHold()
        all_groups.households.add(hh)
        nb_hh += 1
        for j in range(size):
            persons[p_ndx].household = hh
            hh.add_person(persons[p_ndx])
            p_ndx += 1
print(f"{p_ndx} persons in {nb_hh} households")


# --------------------------------------------------------------------
# Dropping people in communities

print(f"Creating communities of {len(STATS_COMMUNITIES_POP)-1} types.")
nb_com = 0
p_ndx = 0
for size in STATS_COMMUNITIES_POP:
    if size is not None:
        com = Community()
        all_groups.communities.add(com)
        nb_com += 1
        for j in range(size):
            persons[p_ndx].community = com
            com.add_person(persons[p_ndx])
            p_ndx += 1
print(f"{p_ndx} persons in {nb_com} communities")

all_groups.status()

# --------------------------------------------------------------------
# Infecting people

for person in random.sample(persons, 12340):
    person.state = "E"


# --------------------------------------------------------------------
# Dispatching the quota

class InfectablePool:
    def __init__(self, infected_people):
        self._targets = {Places.Workplace: defaultdict(lambda: 0),
                         Places.HouseHold: defaultdict(lambda: 0),
                         Places.School: defaultdict(lambda: 0),
                         Places.Community: defaultdict(lambda: 0)}


        # Si A & B peuvent infecter C sur le lieu de travail
        # alors C apparait DEUX fois dans le groupe workplace
        # Cela simule le fait qu'il a plus de chances de se faire
        # infecter.

        n = 0
        for infected in infected_people:
            for grp, infectables in infected.infectables().items():
                # t is the group, p is the person
                # We count the person once more

                tgts = self._targets[grp]
                for p in infectables:
                    tgts[p] += 1

            n += 1
            if n % 1000 == 0:
                print(n)

        print(f"Targets reachable by {len(infected_people)} infected people")
        for k, v in self._targets.items():
            print(f"   {k} : {len(v)} targets")


        # For optimisation
        self._keys = dict()
        for grp, v in self._targets.items():
            self._keys[grp] = list(v.keys())

    def has_targets_in(self, group: Places):
        return len(self._targets[group]) > 0

    def infect_one_in(self, group: Places):
        """ Infect one person in the given group.
        """

        person = None
        p_ndx = None
        while person is None:
            p_ndx = random.randint(0,len(self._keys[group])-1)
            person = self._keys[group][p_ndx]

        # print(f"Removing someone {person} from {group}")
        # assert isinstance(person, Person)
        # assert person in self._targets[group]

        # Remove the person from all the groups
        done = False
        for grp, infectables in self._targets.items():
            # print(grp)
            # print(type(infectables))

            if person in infectables:

                if infectables[person] > 1:
                    infectables[person] -= 1
                else:
                    del infectables[person]
                    self._keys[group][p_ndx] = None

                done = True

        assert done, "You tried to infect someone who's not a target"

        person.state = 'E'



infected_people = [p for p in filter(lambda p: p.infected, persons)]
targets = InfectablePool(infected_people)

beta = 0.3
N = 100324
mesure_house = 0
mesure_workplace = 0
mesure_community = 0



for day in range(10):

    # Infected people
    cnt_infected = sum(1 for _ in filter(lambda p: p.infected, persons))
    print(f"{cnt_infected} people infected at beginning of simulation")

    A_plus_SP = cnt_infected
    S = N - A_plus_SP
    quota_to_infect = round(beta*S/N *(A_plus_SP))

    print(f"On day {day} we'll infect {quota_to_infect} persons")

    actually_infected = 0
    for nb_infected in range(quota_to_infect): # on prend directement le quota = nombre de personne à infecter ce jour.

        infected_hour = random.randint(0,24) # donc on infectera la personne en fct de l'heure

        # Make sure we can infect people in households before even trying.
        if infected_hour < 13 \
           and targets.has_targets_in(Places.HouseHold) \
           and np.random.binomial(1,1-mesure_house): # il infecte une personne a la maison. 1-mesure % de chance

            targets.infect_one_in(Places.HouseHold)
            actually_infected += 1

        elif infected_hour < 21 \
           and targets.has_targets_in(Places.Workplace) \
           and np.random.binomial(1,1-mesure_workplace):

            targets.infect_one_in(Places.Workplace)
            actually_infected += 1

        elif targets.has_targets_in(Places.Community) \
           and np.random.binomial(1,1-mesure_community):

            targets.infect_one_in(Places.Community)
            actually_infected += 1


    print(f"{actually_infected} were infected")
