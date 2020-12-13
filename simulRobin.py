from enum import Enum
import random
import numpy as np

from load_stats import STATS_HOUSEHOLDS, STATS_WORKPLACES, \
    STATS_SCHOOLS, STATS_COMMUNITIES_POP

from simul import persons, InfectablePool, partition_persons,Places

# class Places(Enum):
#     Workplace = 1
#     HouseHold = 2
#     School = 3
#     Community = 4

# class Person:
#     def __init__(self,state):
#         self._id = 12334
#         self.workplace = None
#         self.household = None
#         self.school = None
#         self.community = None
#         self.state = state

#     @property
#     def susceptible(self):
#         return self.state in ("S")
#     @property
#     def infected_A(self):
#         return self.state in ("A")
#     @property
#     def infected_E(self):
#         return self.state in ("E")
#     @property
#     def infected_SP(self):
#         return self.state in ("SP")

#     def infectables(self):
#         assert self.infected, "Only an infected person can infect others"

#         # All people that this person can infect
#         t = {}

#         if self.school:
#             t[Places.School] = self.school.infectable()
#         else:
#             t[Places.School] = []

#         if self.workplace:
#             t[Places.Workplace] = self.workplace.infectable()
#         else:
#             t[Places.Workplace] = []

#         if self.household:
#             t[Places.HouseHold] = self.household.infectable()
#         else:
#             t[Places.HouseHold] = []

#         if self.community:
#             t[Places.Community] = self.community.infectable()
#         else:
#             t[Places.Community] = []

#         return t


# class GroupBase:
#     def __init__(self):
#         self._persons = set()

#     def add_person(self, p: Person):
#         self._persons.add(p)

#     def infectable(self):
#         # Return the list of people that can still be
#         # infected in this group
#         # May be an empty list

#         return [p for p in self._persons if not p.infected]


# class HouseHold(GroupBase):
#     def __init__(self):
#         super().__init__()



# class WorkPlace(GroupBase):
#     def __init__(self):
#         super().__init__()


# class School(GroupBase):
#     def __init__(self):
#         super().__init__()


# class Community(GroupBase):
#     def __init__(self):
#         super().__init__()


# class Groups:
#     def __init__(self):
#         self.schools = set()
#         self.workplaces = set()
#         self.households = set()
#         self.communities = set()

#     def status(self):
#         print(f"{len(self.schools)} schools")
#         print(f"{len(self.workplaces)} workplaces")
#         print(f"{len(self.households)} households")
#         print(f"{len(self.communities)} communities")

# all_groups = Groups()

# # --------------------------------------------------------------------
# # Create 1000000 infectious people

# print("Creating people")
# S = 930857 + 324
# E = 5929
# A = 6611
# SP = 12445
# HCRF = 100324 - S - E - A - SP
# person_s = np.tile("S",np.round(S))
# person_e = np.tile("E",np.round(E))
# person_a = np.tile("A",np.round(A))
# person_sp = np.tile("SP",np.round(SP))
# person_hcrf = np.tile("HCRF",np.round(100324 - np.round(S) - np.round(E) - np.round(A) - np.round(SP)))

# x = np.concatenate([person_s,person_e,person_a,person_sp,person_hcrf])_
# random.shuffle(x)
# persons = [Person(i) for i in x ] # création de tous nos états

# # --------------------------------------------------------------------
# # Dropping people in workplaces

# print("Creating workplaces")
# nb_wp = 0
# p_ndx = 0
# for size, nb in STATS_WORKPLACES:

#     if nb > 0:
#         # FIXME Could be more random
#         mean_size = (size[0] + size[1])//2
#         for i in range(nb):
#             wp = WorkPlace()
#             all_groups.workplaces.add(wp)
#             nb_wp += 1
#             for j in range(mean_size):
#                 persons[p_ndx].workplace = wp
#                 wp.add_person(persons[p_ndx])
#                 p_ndx += 1

# persons_in_companies = sum(1 for _ in filter(lambda p: p.workplace, persons))
# print(f"{persons_in_companies} persons in {nb_wp} workplaces")

# # --------------------------------------------------------------------
# # Dropping people in schools

# print("Creating schools")
# persons_not_in_schools = [_ for _ in filter(
#     lambda p: p.workplace is not None, persons)]

# nb_sc = 0
# p_ndx = 0
# for size, nb in STATS_SCHOOLS:

#     if nb > 0:
#         # FIXME Could be more random
#         mean_size = (size[0] + size[1])//2
#         for i in range(nb):
#             school = School()
#             all_groups.schools.add(school)
#             nb_sc += 1
#             for j in range(mean_size):
#                 persons_not_in_schools[p_ndx].school = school
#                 school.add_person(persons_not_in_schools[p_ndx])
#                 p_ndx += 1


# # --------------------------------------------------------------------
# # Dropping people in households

# print("Creating households")
# nb_hh = 0
# p_ndx = 0
# for size, nb in enumerate(STATS_HOUSEHOLDS):
#     for i in range(nb):
#         hh = HouseHold()
#         all_groups.households.add(hh)
#         nb_hh += 1
#         for j in range(size):
#             persons[p_ndx].household = hh
#             hh.add_person(persons[p_ndx])
#             p_ndx += 1
# print(f"{p_ndx} persons in {nb_hh} households")


# # --------------------------------------------------------------------
# # Dropping people in communities

# print(f"Creating communities of {len(STATS_COMMUNITIES_POP)-1} types.")
# nb_com = 0
# p_ndx = 0
# for size in STATS_COMMUNITIES_POP:
#     if size is not None:
#         com = Community()
#         all_groups.communities.add(com)
#         nb_com += 1
#         for j in range(size):
#             persons[p_ndx].community = com
#             com.add_person(persons[p_ndx])
#             p_ndx += 1
# print(f"{p_ndx} persons in {nb_com} communities")

# all_groups.status()

# # --------------------------------------------------------------------
# # Infecting people
#     #  plus besoin d'infecté vu qu'ils le sont de base 
# # for person in random.sample(persons, 1234):
# #     person.state = "E"



# # --------------------------------------------------------------------
# # Dispatching the quota


# # Second, figure out the people they can infect
# # Edge case : if 2 infected are in the same company, they
# # have "infectable" persons in common => we must make sure
# # they are counted once. That's why we use sets intead of arrays.

# class InfectablePool:
#     def __init__(self, infected_people):
#         self._targets = {Places.Workplace: [],
#                          Places.HouseHold: [],
#                          Places.School: [],
#                          Places.Community: []}

#         # Si A & B peuvent infecter C sur le lieu de travail
#         # Alors C n'apparait qu'une seule fois dans WorkPlace

#         # Avec array:
#         # Si A & B peuvent infecter C sur le lieu de travail
#         # alors C apparait DEUX fois dans le groupe workplace

#         n = 0
#         for infected in infected_people:
#             for t, p in infected.infectables().items():
#                 # t is the group, p is the person
#                 self._targets[t].extend(p)

#             n += 1
#             if n % 1000 == 0:
#                 print(n)

#         print(f"Targets reachable by {len(infected_people)} infected people")
#         for k, v in self._targets.items():
#             print(f"   {k} : {len(v)} targets")

#     def has_targets_in(self, group: Places):
#         return len(self._targets[group]) > 0

#     def infect_one_in(self, group: Places):
#         """ Infect one person in the given group.
#         """

#         person = random.choice(self._targets[group])


#         # Remove the person from all the groups
#         done = False
#         for infectables in self._targets.values():
#             if person in infectables:
#                 while person in infectables:
#                     infectables.remove(person)
#                 done = True

#         assert done, "You tried to infect someone who's not a target"

#         person.state = 'E'


# # -----------------------


"appel pour un jour avec les mesures : "
# on incremente tout les days des Symptomatique,
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


# ---- 

#appel d'une fonction qui renvoye dSdt.

beta = 0.3
N = 100324

work_perc = 0.49
school_perc = 0.16

mesure_house_A = 0
mesure_work_A = 0
mesure_community_A = 0
mesure_school_A = 0

mesure_house_SP = 0
mesure_work_SP = 0
mesure_community_SP = 0
mesure_school_SP = 0

masque = 0
# X = age_apply_social_distancing
# Y = DSPDT -> 0
# Y = 00000000000000000000000
# Y+1 = 11111111111111111 
# Y+1 = 000000000000001111111111111
# Y+2 = 111111111111112222222222222
# Y+2 = 00000000000000000111111112222222222 gamma+tau
# Y+3 = 11111111111111111222222223333333333

for day in range(20):
    infected_people_A = [p for p in filter(lambda p: p.infected_A, persons)]
    targets_A = InfectablePool(infected_people_A)
    infected_people_SP = [p for p in filter(lambda p: p.infected_SP, persons)]
    targets_SP = InfectablePool(infected_people_SP)

    # Infected people
    cnt_infected_A = sum(1 for _ in filter(lambda p: p.infected_A, persons)) # diviser en A et SP
    print(f"{cnt_infected_A} people A at beginning of simulation")
    cnt_infected_SP = sum(1 for _ in filter(lambda p: p.infected_SP, persons)) # diviser en A et SP
    print(f"{cnt_infected_SP} people SP at beginning of simulation")

    # A_plus_SP = cnt_infected
    S = sum(1 for _ in filter(lambda p: p.susceptible, persons))
    print(S)
    quota_to_infect_A = round(beta*S/N *(cnt_infected_A)) # ici on est en full deterministique
    quota_to_infect_SP = round(beta*S/N *(cnt_infected_SP)) # ici on est en full deterministique

    print(f"On day {day} A people will infect {quota_to_infect_A} persons")
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
                targets_A.infect_one_in(Places.Workplace)
                actually_infected += 1

        elif targets_A.has_targets_in(Places.Community):
            if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte d'office
                targets_A.infect_one_in(Places.Community)
                actually_infected += 1
            elif np.random.binomial(1,(1-mesure_community_A)*(1-masque)): # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                targets_A.infect_one_in(Places.Community)
                actually_infected += 1

    print(f"On day {day} SP people will infect {quota_to_infect_SP} persons")

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
                pass
                # targets_SP.infect_one_in(Places.Workplace)
                # actually_infected += 1

        elif targets_SP.has_targets_in(Places.Community):
            if (np.random.binomial(1,0.1)):# soit il respecte pas les règles et il infecte d'office
                targets_SP.infect_one_in(Places.Community)
                actually_infected += 1
            elif np.random.binomial(1,(1-mesure_community_SP)*(1-masque)): # soit il respecte les règles et il infecte une personne a la communauté. 1-mesure % de chance
                targets_SP.infect_one_in(Places.Community)
                actually_infected += 1

    print(f"{actually_infected} were infected")

    return actually_infected


    # il y a eu actually_infected person infected today ->
    # model normal Faire passer actually_infected de S a E
    # dsdt

    # E = sum(1 for _ in filter(lambda p: p.infected_E, persons))
    # A = sum(1 for _ in filter(lambda p: p.infected_A, persons))
    # SP = sum(1 for _ in filter(lambda p: p.infected_SP, persons))



    # rhoE = (E-actually_infected) *rho # nombre d'arrivant vers A
    # sigmaA = sigma *A
    # gamma4A = gamma4 *A
    # deltaA

    # infected_people_E = [p for p in filter(lambda p: p.infected_E, persons)]
    # for person in random.sample(infected_people_E, rhoE):
    #     person.state = "A"

    # infected_people_A = [p for p in filter(lambda p: p.infected_A, persons)]
    # for person in random.sample(infected_people_A, sigmaA):
    #     person.state = "SP"

    


    # on peut faire tourner "l'autre modèle" un jour, puis reprendre dedt, ... et faire ce tirage aléatoirement ici dans simul
    
    # 2 facon soit en local avec chiffre donc on prend nombre E*rho -> A, puis tirage sur E dans simul
    # puis comme d habitude de E-> A, A-> SP,SP->H, A-> R, Sp->R

    # quand on fait le tirage des delta*A personne qui vont dans SP, on mets day_in_state = 0,


    # E->A, A-> SP, SP-> HCRF, A->HCRF
    # nombre de E, ajouter comme des personnes E ici,
    # nombre de Sp, ajouter comme SP ici
    # nombre de A ajouter comme A ici
    # nombre de H+C+R+F ajouter comme HCRF ici

    # boucle pour faire passer les gens de E à A et de A a SP, + faire sortir tau gens de SP et gamma gens de A et SP
    # DHDT = tau*SP
    # A -> R

