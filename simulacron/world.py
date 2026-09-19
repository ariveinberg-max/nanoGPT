"""The prototype: Los Angeles, 2010.

The first build recreated the era of Fuller's youth. This one is pointed at a
year within living memory instead, which changes more than the signage: the
Gold Line runs to Boyle Heights, the Expo Line does not run to Santa Monica
yet, so the Westside is a drive, and a unit needs about forty dollars a day
rather than a dollar twenty.

A tick is fifteen minutes of simulated time; ninety-six ticks make a day.
"""

from dataclasses import dataclass

TICKS_PER_HOUR = 4
TICKS_PER_DAY = 24 * TICKS_PER_HOUR

DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday")

# The prototype opens on a Monday.
EPOCH = "March 2010"

# What it costs a unit to get through one day: food, transit, incidentals.
# Every money term in the reward is expressed against this, so the units'
# sense of a dollar travels with the era rather than being baked into the
# learner. The 1937 build ran at 1.20.
DAILY_COST = 18.0

HOME_MEAL_COST = 3.50      # something out of the refrigerator
ERRAND_COST = 6.00         # coffee, a bus card, a prescription
STARTING_FUNDS = 10 * DAILY_COST


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    district: str
    kind: str            # home | work | food | social | civic | transit | edge
    opens: int           # hour, 24h
    closes: int
    cost: float = 0.0    # dollars per visit, 2010 prices
    wage: float = 0.0    # dollars per hour for work venues

    def open_at(self, hour):
        if self.opens == self.closes:
            return True
        if self.opens < self.closes:
            return self.opens <= hour < self.closes
        return hour >= self.opens or hour < self.closes   # spans midnight


DISTRICTS = (
    "Downtown",
    "Koreatown",
    "Silver Lake",
    "Hollywood",
    "Boyle Heights",
    "Leimert Park",
    "Santa Monica",
    "Venice",
)

# Travel time in ticks. Metro Rail where it existed in 2010 -- the Red and
# Purple Lines under Wilshire and Hollywood, the Gold Line's Eastside
# Extension to Boyle Heights, open since November 2009. The Expo Line would
# not reach Santa Monica until 2016, so the Westside is the 10 at whatever
# speed the 10 is moving.
_ADJACENCY = {
    ("Downtown", "Koreatown"): 1,
    ("Downtown", "Boyle Heights"): 1,
    ("Downtown", "Silver Lake"): 2,
    ("Downtown", "Leimert Park"): 2,
    ("Koreatown", "Hollywood"): 2,
    ("Koreatown", "Silver Lake"): 2,
    ("Koreatown", "Leimert Park"): 2,
    ("Silver Lake", "Hollywood"): 2,
    ("Koreatown", "Santa Monica"): 4,
    ("Hollywood", "Santa Monica"): 4,
    ("Leimert Park", "Santa Monica"): 4,
    ("Santa Monica", "Venice"): 1,
    ("Leimert Park", "Venice"): 4,
}

VENUES = (
    # --- homes -------------------------------------------------------------
    Venue("artist_loft", "a converted loft in the Arts District", "Downtown", "home", 0, 0),
    Venue("ktown_unit", "a rent-stabilized unit off Western", "Koreatown", "home", 0, 0),
    Venue("silverlake_court", "a bungalow court above Sunset", "Silver Lake", "home", 0, 0),
    Venue("hollywood_apt", "a courtyard apartment on Yucca", "Hollywood", "home", 0, 0),
    Venue("boyle_duplex", "a duplex off Cesar Chavez", "Boyle Heights", "home", 0, 0),
    Venue("venice_studio", "a studio two blocks off the boardwalk", "Venice", "home", 0, 0),
    # --- work --------------------------------------------------------------
    Venue("law_firm", "a law firm on Bunker Hill", "Downtown", "work", 8, 19, wage=34.00),
    Venue("planning_dept", "the city planning department", "Downtown", "work", 8, 17, wage=26.00),
    Venue("market_stall", "a stall in Grand Central Market", "Downtown", "work", 8, 18, wage=9.50),
    Venue("call_center", "a call center on Wilshire", "Koreatown", "work", 6, 22, wage=13.00),
    Venue("post_house", "a post house off Santa Monica Boulevard", "Hollywood", "work", 9, 20, wage=22.00),
    Venue("startup", "a startup on 2nd Street", "Santa Monica", "work", 9, 20, wage=38.00),
    Venue("warehouse", "a warehouse south of the tracks", "Boyle Heights", "work", 6, 16, wage=10.50),
    Venue("surf_shop", "a shop on Abbot Kinney", "Venice", "work", 10, 19, wage=11.00),
    Venue("barbershop", "a barbershop on Degnan", "Leimert Park", "work", 9, 19, wage=12.00),
    # --- food --------------------------------------------------------------
    Venue("philippes", "Philippe the Original", "Downtown", "food", 6, 22, cost=7.00),
    Venue("central_market", "the taco counter at Grand Central Market", "Downtown", "food", 8, 18, cost=6.50),
    Venue("ktown_bbq", "a Korean barbecue on 6th", "Koreatown", "food", 11, 2, cost=18.00),
    Venue("kogi", "wherever the Kogi truck is parked", "Silver Lake", "food", 18, 2, cost=8.00),
    Venue("sunset_coffee", "a coffee bar on Sunset", "Silver Lake", "food", 6, 20, cost=5.50),
    Venue("soondubu", "a soondubu place on 8th", "Koreatown", "food", 8, 22, cost=11.00),
    Venue("promenade", "a counter off the Third Street Promenade", "Santa Monica", "food", 8, 21, cost=10.00),
    Venue("boyle_counter", "a lunch counter on First", "Boyle Heights", "food", 7, 19, cost=7.50),
    Venue("intelligentsia", "Intelligentsia on Abbot Kinney", "Venice", "food", 6, 20, cost=5.00),
    Venue("thai_town", "a Thai place east of Normandie", "Hollywood", "food", 11, 23, cost=9.00),
    Venue("leimert_soul", "a soul food kitchen on Crenshaw", "Leimert Park", "food", 11, 21, cost=11.00),
    # --- social ------------------------------------------------------------
    Venue("noraebang", "a noraebang room off Western", "Koreatown", "social", 19, 2, cost=15.00),
    Venue("the_echo", "the Echo", "Silver Lake", "social", 20, 2, cost=12.00),
    Venue("arclight", "the ArcLight", "Hollywood", "social", 11, 0, cost=14.00),
    Venue("la_live", "L.A. Live", "Downtown", "social", 11, 1, cost=20.00),
    Venue("world_stage", "the World Stage", "Leimert Park", "social", 19, 0, cost=10.00),
    Venue("sm_pier", "Santa Monica Pier", "Santa Monica", "social", 10, 23, cost=8.00),
    Venue("boardwalk", "the Venice boardwalk", "Venice", "social", 8, 21, cost=3.00),
    # --- civic / errands ---------------------------------------------------
    Venue("central_library", "Central Library", "Downtown", "civic", 10, 20),
    Venue("observatory", "Griffith Observatory", "Silver Lake", "civic", 12, 22),
    Venue("echo_park", "Echo Park Lake", "Silver Lake", "civic", 0, 0),
    Venue("mariachi_plaza", "Mariachi Plaza", "Boyle Heights", "civic", 0, 0),
    Venue("leimert_plaza", "Leimert Plaza Park", "Leimert Park", "civic", 0, 0),
    Venue("sm_beach", "the beach at Ocean Park", "Santa Monica", "civic", 0, 0),
)

VENUES_BY_KEY = {v.key: v for v in VENUES}

# Where a unit ends up when it goes looking for the edge of the prototype.
EDGE_DISTRICT = "Venice"
EDGE_NOTE = "stood where the boardwalk meets the water and could not say what lay past it"

# What the units ride. Used only for flavour in the link shell.
TRANSIT_NOUN = "Metro"


def _travel_table():
    """All-pairs shortest travel time, in ticks."""
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
