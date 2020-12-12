import urllib.request
import json

with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/stats/age.json") as fp:
    split = [ l.decode("utf-8").strip().split(",") for l in fp.readlines()]

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


print(STATS_AGES[10:15])
print(STATS_COMMUNITIES_POP[10:15])
print(STATS_COMMUNITIES_FRAC[10:15])
print(STATS_HOUSEHOLDS[10:15])
print(STATS_SCHOOLS[10:15])
print(STATS_WORKPLACES[10:15])
