"""The prototype: Los Angeles, circa 1937.

Fuller wanted to start by recreating the era of his youth, so the geography,
the venues, the wages and the streetcar lines are all period fixtures. A tick
is fifteen minutes of simulated time; ninety-six ticks make a day.
"""

from dataclasses import dataclass, field

TICKS_PER_HOUR = 4
TICKS_PER_DAY = 24 * TICKS_PER_HOUR

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday")

# The prototype opens on a Monday.
EPOCH = "March 1937"


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    district: str
    kind: str            # home | work | food | social | civic | transit | edge
    opens: int           # hour, 24h
    closes: int
    cost: float = 0.0    # dollars per visit, 1937 prices
    wage: float = 0.0    # dollars per hour for work venues

    def open_at(self, hour):
        if self.opens == self.closes:
            return True
        if self.opens < self.closes:
            return self.opens <= hour < self.closes
        return hour >= self.opens or hour < self.closes   # spans midnight


DISTRICTS = (
    "Bunker Hill",
    "Spring Street",
    "Olvera Street",
    "Boyle Heights",
    "Central Avenue",
    "Hollywood",
    "Wilshire",
    "Venice",
)

# Red Car lines. Travel cost in ticks between districts; the Pacific Electric
# put nearly all of this within an hour of downtown in 1937.
_ADJACENCY = {
    ("Bunker Hill", "Spring Street"): 1,
    ("Bunker Hill", "Olvera Street"): 1,
    ("Spring Street", "Olvera Street"): 1,
    ("Spring Street", "Central Avenue"): 2,
    ("Spring Street", "Boyle Heights"): 2,
    ("Olvera Street", "Boyle Heights"): 2,
    ("Central Avenue", "Wilshire"): 3,
    ("Spring Street", "Wilshire"): 3,
    ("Wilshire", "Hollywood"): 3,
    ("Bunker Hill", "Hollywood"): 4,
    ("Wilshire", "Venice"): 5,
    ("Hollywood", "Venice"): 6,
}

VENUES = (
    # --- homes -------------------------------------------------------------
    Venue("rooming_house", "the Alta Vista rooming house", "Bunker Hill", "home", 0, 0),
    Venue("bungalow", "a court bungalow off Bixel", "Bunker Hill", "home", 0, 0),
    Venue("flat", "a walk-up flat on Brooklyn Avenue", "Boyle Heights", "home", 0, 0),
    Venue("apartment", "the Dunbar-side apartments", "Central Avenue", "home", 0, 0),
    Venue("hollywood_court", "a stucco court on Yucca", "Hollywood", "home", 0, 0),
    Venue("beach_room", "a rented room above the colonnade", "Venice", "home", 0, 0),
    # --- work --------------------------------------------------------------
    Venue("bradbury", "the Bradbury Building offices", "Spring Street", "work", 8, 18, wage=0.62),
    Venue("city_hall", "City Hall", "Spring Street", "work", 8, 17, wage=0.70),
    Venue("grand_central", "Grand Central Market", "Spring Street", "work", 6, 18, wage=0.45),
    Venue("packing_house", "the Boyle Heights packing house", "Boyle Heights", "work", 7, 17, wage=0.40),
    Venue("studio_lot", "the picture lot on Gower", "Hollywood", "work", 7, 19, wage=0.85),
    Venue("bullocks", "Bullocks Wilshire", "Wilshire", "work", 9, 18, wage=0.50),
    Venue("pier_works", "the Venice pier works", "Venice", "work", 6, 16, wage=0.42),
    # --- food --------------------------------------------------------------
    Venue("cliftons", "Clifton's Brookdale cafeteria", "Spring Street", "food", 7, 20, cost=0.35),
    Venue("philippes", "Philippe's", "Olvera Street", "food", 6, 22, cost=0.25),
    Venue("musso", "Musso & Frank", "Hollywood", "food", 11, 23, cost=1.10),
    Venue("cafe_boyle", "a lunch counter on First", "Boyle Heights", "food", 6, 19, cost=0.20),
    Venue("beach_stand", "the chili stand on the boardwalk", "Venice", "food", 10, 21, cost=0.15),
    # --- social ------------------------------------------------------------
    Venue("club_alabam", "Club Alabam", "Central Avenue", "social", 20, 2, cost=0.75),
    Venue("cocoanut_grove", "the Cocoanut Grove", "Wilshire", "social", 19, 1, cost=1.50),
    Venue("graumans", "Grauman's Chinese", "Hollywood", "social", 12, 23, cost=0.40),
    Venue("olvera_plaza", "the plaza on Olvera Street", "Olvera Street", "social", 8, 22, cost=0.10),
    Venue("plunge", "the Venice plunge", "Venice", "social", 9, 20, cost=0.25),
    # --- civic / errands ---------------------------------------------------
    Venue("central_library", "Central Library", "Spring Street", "civic", 9, 21),
    Venue("angels_flight", "Angels Flight", "Bunker Hill", "transit", 6, 22, cost=0.01),
    Venue("observatory", "Griffith Observatory", "Hollywood", "civic", 14, 22, cost=0.25),
    Venue("pershing_square", "Pershing Square", "Spring Street", "civic", 0, 0),
)

VENUES_BY_KEY = {v.key: v for v in VENUES}


def _travel_table():
    """All-pairs shortest streetcar time, in ticks."""
    inf = 99
    t = {a: {b: (0 if a == b else inf) for b in DISTRICTS} for a in DISTRICTS}
    for (a, b), w in _ADJACENCY.items():
        t[a][b] = min(t[a][b], w)
        t[b][a] = min(t[b][a], w)
    for k in DISTRICTS:
        for i in DISTRICTS:
            for j in DISTRICTS:
                if t[i][k] + t[k][j] < t[i][j]:
                    t[i][j] = t[i][k] + t[k][j]
    return t


TRAVEL = _travel_table()


@dataclass
class Clock:
    tick: int = 0

    @property
    def day(self):
        return self.tick // TICKS_PER_DAY

    @property
    def hour(self):
        return (self.tick % TICKS_PER_DAY) // TICKS_PER_HOUR

    @property
    def minute(self):
        return (self.tick % TICKS_PER_HOUR) * (60 // TICKS_PER_HOUR)

    @property
    def weekday(self):
        return DAY_NAMES[self.day % 7]

    @property
    def is_workday(self):
        return self.day % 7 < 5

    def stamp(self):
        return f"{self.weekday} {self.hour:02d}:{self.minute:02d}"

    def advance(self):
        self.tick += 1


def venues_in(district, kind=None):
    return [v for v in VENUES
            if v.district == district and (kind is None or v.kind == kind)]


def venues_of(kind):
    return [v for v in VENUES if v.kind == kind]


def travel_ticks(a, b):
    return TRAVEL[a][b]
