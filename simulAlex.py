import distutils.util
import argparse

from datetime import datetime
from collections import defaultdict
from enum import Enum
import random
from math import floor
import numpy as np

from load_stats import STATS_HOUSEHOLDS, STATS_WORKPLACES, \
    STATS_SCHOOLS, STATS_COMMUNITIES_POP


IS_QUARANTINE = True
IS_CASE_ISOLATION = False
ISOLATION_TIME = 7
NB_SIMULATION_DAYS = 20

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--quarantine", "-q", type=lambda v:bool(distutils.util.strtobool(v)), help=f"Set a quarantine. Default is {IS_QUARANTINE}.", default=IS_QUARANTINE)
args_parser.add_argument("--case-isolation", "-i", type=lambda v:bool(distutils.util.strtobool(v)), help=f"Set a case isolation policy. Default is {IS_CASE_ISOLATION}.", default=IS_CASE_ISOLATION)
args_parser.add_argument("--isolation-time", "-t", type=int, help=f"Duration of isolation or quarantine, in days. Default {ISOLATION_TIME}.", default=ISOLATION_TIME)
args_parser.add_argument("--simulations", "-s", type=int, help=f"Number of simulation days to run. Default {NB_SIMULATION_DAYS}.", default=NB_SIMULATION_DAYS)

parsed_args = args_parser.parse_args()

IS_QUARANTINE = parsed_args.quarantine
IS_CASE_ISOLATION = parsed_args.case_isolation
ISOLATION_TIME = parsed_args.isolation_time
NB_SIMULATION_DAYS = parsed_args.simulations

print(f"Parameters : quarantine={parsed_args.quarantine}, case_isolation={IS_CASE_ISOLATION}, isolation_time={ISOLATION_TIME}, simulation days={NB_SIMULATION_DAYS}")

class PeopleCounter:
    def __init__(self, people_dict):
        self._d = dict()
        self._d.update(people_dict)

        # Index to people
        self._people = [p for p in people_dict.keys()]

        # People to index
        self._people_to_ndx = dict(
            zip(self._people, range(len(self._people))))

    def pick(self):
        if len(self._people) == 0:
            return None

        ndx = random.randint(0, len(self._people)-1)

        while self._people[ndx] is None:
            ndx = (ndx+1) % len(self._people)

        return self._people[ndx]

    def decount(self, p):
        assert p in self._d

        if self._d[p] > 1:
            self._d[p] -= 1
        else:
            self.remove(p)

    def remove(self, p):
        self._d.pop(p)
        self._people[self._people_to_ndx[p]] = None
        self._people_to_ndx[p] = None

    def __contains__(self, k):
        return k in self._d

    def __len__(self):
        return len(self._d)

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, ndx, value):
        self._d[ndx] = value

    def __str__(self):
        return str(self._d)





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
        self.isolation_time = 0

    @property
    def susceptible(self):
        return self.state == "S"

    @property
    def infected_A(self):
        return self.state == "A"

    @property
    def infected_E(self):
        return self.state == "E"

    @property
    def infected_SP(self):
        return self.state == "SP"

    @property
    def infected(self):
        return self.state in ("SP", "E", "A")

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

    def set_isolation_time(self):
        if IS_QUARANTINE or IS_CASE_ISOLATION:
            if self.infected_SP:
                self.isolation_time = ISOLATION_TIME
                if IS_QUARANTINE:
                    for p in self.household.isolable():
                        p.isolation_time = ISOLATION_TIME


class GroupBase:
    def __init__(self):
        self._persons = set()

    def add_person(self, p: Person):
        self._persons.add(p)

    def infectable(self):
        # Return the list of people that can still be
        # infected in this group
        # May be an empty list

        return [p for p in self._persons if p.susceptible]

    def isolable(self):
    # Return the list of people that are infectious
    # May be an empty list

        return [p for p in self._persons if p.infected_A or p.infected_SP or p.infected_E or p.susceptible]


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
# Dispatching the quota

def partition_persons(persons, repartition):
    ndx = 0

    shuf_ndx = [i for i in range(len(persons))]
    random.shuffle(shuf_ndx)

    for status, cnt in repartition.items():
        for i in range(cnt):
            persons[shuf_ndx[ndx]].state = status
            persons[shuf_ndx[ndx]].set_isolation_time()
            ndx += 1

    print("Partitioned {}/{} persons".format(
        sum(repartition.values()), ndx))





class InfectablePool:
    """ Represents a group of person that can be
    infected by a given group of infected people.
    """

    def __init__(self, infected_people):

        # Si A & B peuvent infecter C sur le lieu de travail
        # alors C apparait DEUX fois dans le groupe workplace
        # Cela simule le fait qu'il a plus de chances de se faire
        # infecter.

        # infected schools

        print(f"Building target pools for {len(infected_people)} infected people")

        self._targets = dict()
        for group, field in [(Places.Workplace, "workplace"),
                             (Places.School, "school"),
                             (Places.Community, "community"),
                             (Places.HouseHold, "household")]:
            schools = set(
                filter(lambda s: s is not None,
                       [getattr(infected, field) for infected in infected_people]))

            print(f"{group} has {len(schools)} locations with infected people")

            dd = defaultdict(lambda: 0)
            for school in schools:
                for p in school.infectable():
                    dd[p] += 1

            self._targets[group] = PeopleCounter(dd)

        print(f"Targets reachable by {len(infected_people)} infected people")
        for k, v in self._targets.items():
            print(f"   {k} : {len(v)} targets")

    def has_targets_in(self, group: Places):
        return len(self._targets[group]) > 0

    def infect_one_in(self, group: Places):
        """ Infect one person in the given group.
        """
        person = self._targets[group].pick()
        if person == None:
            return
        # print(f"Removing someone {person} from {group}")
        # assert isinstance(person, Person)
        # assert person in self._targets[group]

        # Remove the person from all the groups
        done = False
        for grp, infectables in self._targets.items():
            # print(grp)
            # print(type(infectables))

            if person in infectables:
                infectables.remove(person)
                done = True

        assert done, "You tried to infect someone who's not a target"

        person.state = "E"




def simulation_model(persons,beta):#,infectedPool):

    N = 1000324

    work_perc = 495488.5/1000324
    school_perc = 162647.5/1000324

    # mesure_house_A = 0
    # mesure_work_A = 0.75
    # mesure_community_A = 0.75
    # mesure_school_A = 1
    #
    # mesure_house_SP = 0
    # mesure_work_SP = 0.75
    # mesure_community_SP = 0.75
    # mesure_school_SP = 1

    # ------- CASE ISOLATION [
    mesure_house_A = 0
    mesure_work_A = 0
    mesure_community_A = 0
    mesure_school_A = 0

    mesure_house_SP = 0
    mesure_work_SP = 0
    mesure_community_SP = 0
    mesure_school_SP = 0

    mesure_house_isolated = 0.25
    mesure_work_isolated = 1
    mesure_community_isolated = 0.90
    mesure_school_isolated = 1
    # ------- CASE ISOLATION ]



    masque = 0.2
    # X = age_apply_social_distancing
    # Y = DSPDT -> 0
    # Y = 00000000000000000000000
    # Y+1 = 11111111111111111
    # Y+1 = 000000000000001111111111111
    # Y+2 = 111111111111112222222222222
    # Y+2 = 00000000000000000111111112222222222 gamma+tau
    # Y+3 = 11111111111111111222222223333333333
    for p in persons:
        if p.isolation_time > 0:
            p.isolation_time -= 1

    infected_people_A = [p for p in filter(lambda p: p.infected_A, persons) if p.isolation_time == 0]
    targets_A = InfectablePool(infected_people_A)
    #targets_A = infectedPool

    # Infected people
    cnt_infected_A = sum(1 for p in filter(lambda p: p.infected_A, persons) if p.isolation_time == 0) # diviser en A et SP
    print(f"{cnt_infected_A} people A at beginning of simulation")
    cnt_infected_SP = sum(1 for p in filter(lambda p: p.infected_SP, persons) if p.isolation_time == 0) # diviser en A et SP
    print(f"{cnt_infected_SP} people SP at beginning of simulation")
    # Pour CASE ISOLATION:
    #cnt_infected_isolated = sum(1 for p in filter(lambda p: p.infected_SP, persons) if p.isolation_time > 0) # diviser en SP et SP_isolated
    # Pour QUARANTINE ou CASE ISOLATION:
    cnt_infected_isolated = sum(1 for p in filter(lambda p: p.infected_SP or p.infected_A, persons) if p.isolation_time > 0)
    print(f"{cnt_infected_isolated} people isolated at beginning of simulation")

    # A_plus_SP = cnt_infected
    S = sum(1 for _ in filter(lambda p: p.susceptible, persons))
    print(S)
    quota_to_infect_A = round(beta*S/N *(cnt_infected_A)) # ici on est en full deterministique

    print(f" A people will infect {quota_to_infect_A} persons")
    actually_infected = 0

    # on infecte les perosnnes infectée par A
    for nb_infected in range(quota_to_infect_A): # on prend directement le quota = nombre de personne à infecter ce jour.
        infected_hour = random.randint(0,24) # donc on infectera la personne en fct de l'heure
        # Make sure we can infect people in households before even trying.
        # age = tirage sur age population
        # day = tirage sur nombre de jour dans SP

        if infected_hour < 13 \
           and targets_A.has_targets_in(Places.HouseHold):
            if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                targets_A.infect_one_in(Places.HouseHold)
                actually_infected += 1
            elif np.random.binomial(1,(1-mesure_house_A)*(1-masque)): # soit il respecte les règles et il infecte une personne a la maison. 1-mesure % de chance
                targets_A.infect_one_in(Places.HouseHold)
                actually_infected += 1

        elif infected_hour < 21: # temps travail/school
            work_school_others = random.uniform(0,1) # a regarder
            if ( work_school_others < work_perc ): # personne est au boulot :
                if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte  or (age < X)
                    targets_A.infect_one_in(Places.Workplace)
                    actually_infected += 1
                elif np.random.binomial(1,(1-mesure_work_A)*(1-masque)): # soit il respecte les règles et il infecte une personne au bureau. mesure % de chance
                    targets_A.infect_one_in(Places.Workplace)
                    actually_infected += 1
            elif ( work_school_others < work_perc + school_perc): # personne est a l'école :
                if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                    targets_A.infect_one_in(Places.School)
                    actually_infected += 1
                elif np.random.binomial(1,(1-mesure_school_A)*(1-masque)): # soit il respecte les règles et il infecte une personne au bureau. 1-mesure % de chance
                    targets_A.infect_one_in(Places.School)
                    actually_infected += 1
            else : # la personne vagabonde... donc potentiellement infecte tout le monde? ou seuls les susceptible en dehors du  boulot et de l'école ?
                # a voir entre communauté et chez elle
                if ( np.random.binomial(1,0.5)) and targets_A.has_targets_in(Places.HouseHold):
                    if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                        targets_A.infect_one_in(Places.HouseHold)
                        actually_infected += 1
                    elif np.random.binomial(1,(1-mesure_house_A)*(1-masque)): # soit il respecte les règles et il infecte une personne a la maison. 1-mesure % de chance
                        targets_A.infect_one_in(Places.HouseHold)
                        actually_infected += 1
                elif targets_A.has_targets_in(Places.Community):
                    if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte d'office
                        targets_A.infect_one_in(Places.Community)
                        actually_infected += 1
                    elif np.random.binomial(1,(1-mesure_community_A)*(1-masque)): # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                        targets_A.infect_one_in(Places.Community)
                        actually_infected += 1

        elif targets_A.has_targets_in(Places.Community):
            if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte d'office
                targets_A.infect_one_in(Places.Community)
                actually_infected += 1
            elif np.random.binomial(1,(1-mesure_community_A)*(1-masque)): # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                targets_A.infect_one_in(Places.Community)
                actually_infected += 1

    #infected_people_isolated = [p for p in filter(lambda p: p.infected_SP, persons) if p.isolation_time > 0]
    infected_people_SP = [p for p in filter(lambda p: p.infected_SP, persons) if p.isolation_time == 0]
    targets_SP = InfectablePool(infected_people_SP)
    #targets_SP = infectedPool
    quota_to_infect_SP = round(beta*S/N *(cnt_infected_SP)) # ici on est en full deterministique

    print(f"SP people will infect {quota_to_infect_SP} persons")

    # on infecte les perosnnes infectée par SP
    for nb_infected in range(quota_to_infect_SP): # on prend directement le quota = nombre de personne à infecter ce jour.
        infected_hour = random.randint(0,24) # donc on infectera la personne en fct de l'heure
        # Make sure we can infect people in households before even trying.
        if infected_hour < 13 \
           and targets_SP.has_targets_in(Places.HouseHold):

            if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                targets_SP.infect_one_in(Places.HouseHold)
                actually_infected += 1
            elif np.random.binomial(1,(1-mesure_house_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne a la maison. 1-mesure % de chance
                targets_SP.infect_one_in(Places.HouseHold)
                actually_infected += 1

        elif infected_hour < 21 :
            work_school_others = random.uniform(0,1)
            if ( work_school_others < work_perc ): # personne est au boulot :
                if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                    targets_SP.infect_one_in(Places.Workplace)
                    actually_infected += 1
                elif np.random.binomial(1,(1-mesure_work_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne au bureau. 1-mesure % de chance
                    targets_SP.infect_one_in(Places.Workplace)
                    actually_infected += 1
            elif ( work_school_others < work_perc + school_perc): # personne est a l'école :
                if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                    targets_SP.infect_one_in(Places.School)
                    actually_infected += 1
                elif np.random.binomial(1,(1-mesure_school_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne au bureau. 1-mesure % de chance
                    targets_SP.infect_one_in(Places.School)
                    actually_infected += 1
            else : # la personne vagabonde... donc potentiellement infecte tout le monde? ou seuls les susceptible en dehors du  boulot et de l'école ?
                # a voir entre communauté et chez elle
                if ( np.random.binomial(1,0.5)) and targets_SP.has_targets_in(Places.HouseHold):
                    if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte
                        targets_SP.infect_one_in(Places.HouseHold) # A -> SP
                        actually_infected += 1
                    elif np.random.binomial(1,(1-mesure_house_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne a la maison. 1-mesure % de chance
                        targets_SP.infect_one_in(Places.HouseHold)
                        actually_infected += 1
                elif targets_SP.has_targets_in(Places.Community):
                    if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte d'office
                        targets_SP.infect_one_in(Places.Community)
                        actually_infected += 1
                    elif np.random.binomial(1,(1-mesure_community_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                        targets_SP.infect_one_in(Places.Community)
                        actually_infected += 1

        elif targets_SP.has_targets_in(Places.Community):
            if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte d'office
                targets_SP.infect_one_in(Places.Community)
                actually_infected += 1
            elif np.random.binomial(1,(1-mesure_community_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                targets_SP.infect_one_in(Places.Community)
                actually_infected += 1

    print(f"{actually_infected} were infected")

    infected_people_isolated = [p for p in filter(lambda p: p.infected_SP or p.infected_A, persons) if p.isolation_time > 0]

    targets_isolated = InfectablePool(infected_people_isolated)
    # targets_SP = infectedPool
    quota_to_infect_isolated = round(beta * S / N * cnt_infected_isolated)  # ici on est en full deterministique

    print(f"Isolated people will infect {quota_to_infect_isolated} persons")

    # on infecte les perosnnes infectée par SP
    for nb_infected in range(quota_to_infect_isolated):  # on prend directement le quota = nombre de personne à infecter ce jour.
        infected_hour = random.randint(0, 24)  # donc on infectera la personne en fct de l'heure
        # Make sure we can infect people in households before even trying.
        if infected_hour < 13 \
                and targets_isolated.has_targets_in(Places.HouseHold):

            if (np.random.binomial(1, 0.1)):  # soit il respecte pas les règles et il infecte
                targets_isolated.infect_one_in(Places.HouseHold)
                actually_infected += 1
            elif np.random.binomial(1, (1 - mesure_house_isolated) * (1 - masque)):  # soit il respecte les règles et il infecte une personne a la maison. 1-mesure % de chance
                targets_isolated.infect_one_in(Places.HouseHold)
                actually_infected += 1

        elif infected_hour < 21:
            work_school_others = random.uniform(0, 1)
            if (work_school_others < work_perc):  # personne est au boulot :
                if (np.random.binomial(1, 0.1)):  # soit il respecte pas les règles et il infecte
                    targets_isolated.infect_one_in(Places.Workplace)
                    actually_infected += 1
                elif np.random.binomial(1, (1 - mesure_work_isolated) * (1 - masque)):  # soit il respecte les règles et il infecte une personne au bureau. 1-mesure % de chance
                    targets_isolated.infect_one_in(Places.Workplace)
                    actually_infected += 1
            elif (work_school_others < work_perc + school_perc):  # personne est a l'école :
                if (np.random.binomial(1, 0.1)):  # soit il respecte pas les règles et il infecte
                    targets_isolated.infect_one_in(Places.School)
                    actually_infected += 1
                elif np.random.binomial(1, (1 - mesure_school_isolated) * (1 - masque)):  # soit il respecte les règles et il infecte une personne au bureau. 1-mesure % de chance
                    targets_isolated.infect_one_in(Places.School)
                    actually_infected += 1
            else:  # la personne vagabonde... donc potentiellement infecte tout le monde? ou seuls les susceptible en dehors du  boulot et de l'école ?
                # a voir entre communauté et chez elle
                if (np.random.binomial(1, 0.5)) and targets_SP.has_targets_in(Places.HouseHold):
                    if (np.random.binomial(1, 0.1)):  # soit il respecte pas les règles et il infecte
                        targets_isolated.infect_one_in(Places.HouseHold)  # A -> SP
                        actually_infected += 1
                    elif np.random.binomial(1, (1 - mesure_house_isolated) * (1 - masque)):  # soit il respecte les règles et il infecte une personne a la maison. 1-mesure % de chance
                        targets_isolated.infect_one_in(Places.HouseHold)
                        actually_infected += 1
                elif targets_isolated.has_targets_in(Places.Community):
                    if (np.random.binomial(1, 0.1)):  # soit il respecte pas les règles et il infecte d'office
                        targets_isolated.infect_one_in(Places.Community)
                        actually_infected += 1
                    elif np.random.binomial(1, (1 - mesure_community_isolated) * (1 - masque)):  # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                        targets_isolated.infect_one_in(Places.Community)
                        actually_infected += 1

        elif targets_isolated.has_targets_in(Places.Community):
            if (np.random.binomial(1, 0.1)):  # soit il respecte pas les règles et il infecte d'office
                targets_isolated.infect_one_in(Places.Community)
                actually_infected += 1
            elif np.random.binomial(1, (1 - mesure_community_isolated) * (1 - masque)):  # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                targets_isolated.infect_one_in(Places.Community)
                actually_infected += 1

    print(f"{actually_infected} were infected")

    return actually_infected

def model_update(persons,rhoE,sigmaA,gamma4A,tauSP,gamma1SP):

    infected_people_E = [p for p in filter(lambda p: p.infected_E, persons)]
    print("E : " + str(len(infected_people_E)) + "--------------" + str(rhoE))
    if len(infected_people_E) == 0 or rhoE > len(infected_people_E):
        print(f"infected_people_E:{len(infected_people_E)} rhoE:{rhoE}")

    for person in random.sample(infected_people_E, min(rhoE, len(infected_people_E))):
        person.state = "A"

    infected_people_A = [p for p in filter(lambda p: p.infected_A, persons)]
    print("A : " + str(len(infected_people_A)) + "--------------" + str(sigmaA))
    for person in random.sample(infected_people_A, sigmaA):
        person.state = "SP"
        person.set_isolation_time()

    infected_people_A = [p for p in filter(lambda p: p.infected_A, persons)]
    print("A : " + str(len(infected_people_A)) + "--------------" + str(gamma4A))
    for person in random.sample(infected_people_A, gamma4A):
        person.state = "HCRF"

    infected_people_SP = [p for p in filter(lambda p: p.infected_SP, persons)]
    print("A : " + str(len(infected_people_SP)) + "--------------" + str(gamma1SP) + " " + str(tauSP))
    for person in random.sample(infected_people_SP, gamma1SP + tauSP):
        person.state = "HCRF"



    return

if __name__ == "__main__":
    random.seed(1)


    start_time = datetime.now()

    a = {"a":0, "b":0, "c":0}
    pc = PeopleCounter(a)
    print(pc.pick())
    pc.remove("a")
    pc.remove("b")
    print(pc.pick())
    print(pc.pick())
    pc["c"] = 3
    print(pc)

    assert pc["c"] == 3

    print("c" in pc)
    #exit()

    repartition = {
        "S" : 930857,
        "E" : 5929,
        "A" : 6611,
        "SP" : 12445,
        "H" : 681,
        "C" : 91,
        "F" : 39,
        "R" : 43237
    }

    partition_persons(persons, repartition)
    targets = InfectablePool([p for p in
                              filter(lambda p: p.infected_A, persons)])

    beta = 0.3
    N = 100324
    mesure_house = 0
    mesure_workplace = 0
    mesure_community = 0



    for day in range(NB_SIMULATION_DAYS):

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

                targets.infect_one_in(Places.HouseHold, "A")
                actually_infected += 1

            elif infected_hour < 21 \
               and targets.has_targets_in(Places.Workplace) \
               and np.random.binomial(1,1-mesure_workplace):

                targets.infect_one_in(Places.Workplace, "A")
                actually_infected += 1

            elif targets.has_targets_in(Places.Community) \
               and np.random.binomial(1,1-mesure_community):

                targets.infect_one_in(Places.Community, "A")
                actually_infected += 1


        print(f"{actually_infected} were infected")

    print("Run for {:d} seconds".format((datetime.now() - start_time).seconds))
