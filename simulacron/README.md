# Simulacron

A working prototype of the system described in *The Thirteenth Floor* (1999):

> **"So the whole thing's what, a giant computer game?"**
>
> "No, not at all. It doesn't need a user to interact with it to function. Its
> units are fully formed self-learning cyber beings. […] Electronic simulated
> characters. They populate the system. They think. They work. They eat. […]
> They're modeled after us. Right now we have a working prototype, Los Angeles
> circa 1937. […] While my mind is jacked in, I'm walking around experiencing
> 1937. My body stays here, and kind of holds the consciousness of the program
> link unit."

Everything in that speech is a requirement. This is the build — pointed at
**Los Angeles in 2010** rather than the era of Fuller's youth.

| Whitney's claim | How it is implemented |
| --- | --- |
| doesn't need a user to function | `Simulation.step()` drives every unit; no operator is required, and the CLI's `run` never links one |
| fully formed self-learning beings | each unit has episodic memory, appraisal-based emotion, a model of itself and a standing with everyone it has met; a learned habit layer sits underneath |
| they think | perceive → appraise → recall → generate concrete options → evaluate → choose, every simulated hour, with the reasons kept |
| they work / they eat | jobs at 2010 wages, meals at 2010 prices, and an economy a unit has to solve to stay fed |
| modeled after us | five sampled traits, a circadian rhythm, chronic stress, trauma that intrudes, and rent falling due whether or not there is work |
| Los Angeles, 2010 | eight districts, 39 venues, Metro Rail where it existed that year |
| jacked in / the body stays | `LinkSession` suspends a unit's policy and hands you its body; jacking out leaves a hole in its memory |

Pure Python and numpy. No torch, no downloads, no network.

## What a unit is called, and what world it lives in

**Units** is the lab's word — the film's own ("its units are fully formed
self-learning cyber beings"). It is clinical and external, and no unit would
ever use it about itself. So there are two registers: the API, `mind`, `units`
and `city` speak the lab's language; the jack-in shell and anything a unit says
about itself speak from inside, where they are people with names.

The world is eight districts and nothing else. There is no California around
it, no ocean floor, no moon at two hundred and forty thousand miles. But a
person is mostly made of things they have never checked — you believe in
Lisbon — so every unit carries **received knowledge of an Earth that was never
built**: New York, the war, the tide, the stars. None of it is reachable and
none of it can be verified.

The value is the seam. A unit that walks to the water's edge, looks down the
coast, stands at Griffith Observatory after dark, or asks someone what lies
past the county line is querying that account against a prototype that does not
extend that far, and gets back something that will not resolve. That is what
`dissonance` measures now. Worrying at one seam habituates — forty trips to the
water leaves a unit at 0.63 confidence — but finding a *second, independent*
one is what actually erodes: four different seams, once each, takes it to 0.50.

It turns the scope limit into the mechanism rather than an apology for it. I
cannot simulate Earth. I can simulate people who believe in one they cannot
reach, and who can find the edge of it.

## Mortality

Nothing in the simulation meant anything until `_reap` existed. Units die of
starvation, illness and age; `peril()` estimates how close the end is and feeds
deliberation directly, so fear of dying is a reason to eat and to go to work.
Death removes a unit and the people who knew it grieve — bereavement appraises
as sadness at maximum salience, the most memorable thing in a survivor's life.

It is also the sharpest validation the architecture has. Over 120 days the
population that deliberates buries nobody. The same bodies choosing at random
starve to death.

Surviving a bad stretch is evidence a unit can cope, which finally gives the
`coping` belief something to move on. It had been frozen at exactly 0.5 for
every unit in every run — and it is the courage term dividing fear, so every
unit was identically brave, permanently.

## A life, rather than a loop

Units used to spring into existence as interchangeable thirty-year-olds. Now
capacity rises through childhood, holds, and falls after sixty-five; frailty
raises illness risk and slows mending; work has an age floor and ceiling.
Mortality is Gompertz — doubling roughly every eight years — giving life
expectancy 71 at birth and 80 at 65.

A child's world is small and grows: at two it can eat, rest and sleep, and
nothing else; at ten it has its own district; at twenty-five it has the city.
Left with the adult option set a one-year-old rode across the county alone and
stood at the edge of town questioning the nature of reality, which is a
memorable image and completely wrong.

## Family, and belief that outlives the believer

Everything a unit knew used to die with it, so the population never accumulated
anything and nobody had ever been told something. Children are not spawned now;
they are born to particular people, and they begin with **their parents'
account of the world** rather than a blank one — which streets are dangerous,
who can be trusted, what a person is for, and how much of it to doubt. None of
it is learned, because a child has lived no days. It is inherited, and it is
wrong in the specific ways its parents were wrong.

That is the only mechanism here that produces something nobody wrote.

## Crime, policing, and how fear gets around

Robbery is not a mode a unit enters. It is an option on the same list as going
to bed, scored by the same machinery, and it mostly loses. The same unit, the
same victim:

```
comfortable, in work   peril 0.00  ->  rob scores -3.00
broke but well         peril 0.00  ->  rob scores -3.00
starving               peril 0.62  ->  rob scores -1.39
```

Desperation is what brings it into range. Being known to the victim pushes it
back out.

Policing is deliberately crude and deliberately reflexive: **patrol follows
reported crime, and nothing else.** That one line produces the loop everyone
already knows about — more patrol means more arrests means more recorded crime
means more patrol — and the drift is towards where offences are *seen* rather
than where they happen. Whether a crime is reported at all depends on the
victim, not the crime.

Fear travels much further than crime does. After 300 days with 20 units:

```
Koreatown      feared by 12/24  mean danger 0.25  (9 of them on hearsay)
Boyle Heights  feared by 12/24  mean danger 0.25  (5 of them on hearsay)
Hollywood      feared by  0/24  mean danger 0.07  (0 of them on hearsay)
```

Nine of the twelve units afraid of Koreatown have never been there. They were
told, by someone they trust, and they stay away.

## The inner life

Units are not scored on a handful of drives any more. Each one carries:

- **Episodic memory.** Events are encoded at a strength set by what they felt
  like. Ordinary ones fade; the worst do not, and can surface unbidden.
  Crucially, recall is *inside* the decision: before choosing, a unit cues its
  memory with where it is, who is there and what it is considering.
- **Emotion by appraisal.** Nothing sets an emotion directly. An event is
  scored on goal congruence, certainty, agency, coping potential, norm
  violation and irreversibility, and fear, anger, sadness, shame, guilt, joy,
  pride, hope and relief fall out of the pattern. Being robbed and being laid
  off are both bad; one has an agent and the other does not, so one produces
  anger and the other sadness. A single mood scalar cannot make that
  distinction, and the distinction is behavioural.
- **A self-model.** Beliefs about what it is competent at, where its life is
  going, and what it is — beliefs formed from its own history, which reality
  can contradict. It predicts how something will go, finds out, and the gap is
  surprise.
- **Other units as individuals.** Familiarity, trust, affection and grudges,
  per person, rather than a crowd count.
- **A body that can fail it.** Illness, pain, rent, debt and dismissal.

What that buys, in the sim's own words — one unit after ninety days:

```
Ruth Abernathy
  works   the city planning department at $26.00/hr  [OUT OF WORK]
  hunger      [##########]   fatigue [..........]
  funds       $0.00   debt $104.27
  feeling     sadness 0.96, fear 0.86 | stress 1.00 | trauma 0.20 | drive 0.15
  self        Sees itself as nobody in particular. Good at work (0.90). Worn down.
  chose       sleep at a rent-stabilized unit off Western
              -- fear -0.86, retreat +0.80, habit +0.17
```

Nothing in the code models despair. Job loss produced sadness and fear;
sustained negative affect became chronic stress; stress cut `drive`, which is
the multiplier on every effortful option; and the fear term made retreating
home score above going out. A withdrawal spiral, assembled from parts that
each do something simpler.

## Does the thinking do anything?

Improvement over time is the wrong test here: rent, illness and dismissal make
the world a grind, and against a grind holding steady is the achievement. So
the control is the same bodies in the same city choosing **uniformly at random
from the same options** (`Simulation(deliberate=False)`). After 140 days, 14
units:

| | deliberating | choosing at random |
| --- | --- | --- |
| wellbeing | 0.685 | 0.499 |
| hunger | 0.74 | 1.00 (starving) |
| fatigue | 0.43 | 0.95 |
| debt | $41 | $525 |
| savings | $591 | $0 |
| out of work | 1 | 4 |
| chronic stress | 0.07 | 0.93 |
| people known | 5.0 | 11.6 |

The last row is the honest one: the units who never think about anything have
many more friends, because they are out every night. They are also destitute.

## Run it

```bash
python -m simulacron --units 24 run --days 500      # unattended, reports as it goes
python -m simulacron --units 24 rhythm --days 500   # what the units actually do, by hour
python -m simulacron --units 24 units --days 500    # who is down there
python -m simulacron mind --days 500                # one unit's inner life in full
python -m simulacron city --days 500                # crime, policing, and what the city believes
python -m simulacron jackin --days 500              # walk around in 2010
python -m simulacron.test_simulacron                # 114 checks
```

## Why 2010 and not 1937

The era lives in `world.py`. Changing it is not a matter of signage:

- **Transit.** The Gold Line's Eastside Extension opened in November 2009, so
  Boyle Heights is one stop from Downtown. The Expo Line would not reach Santa
  Monica until 2016, so the Westside is the 10 at whatever speed the 10 is
  moving. Commutes came out *shorter* than the 1937 streetcar map, which
  turned out to matter (see below).
- **Money.** A meal is seven dollars rather than thirty-five cents. Every money
  term in the reward is now expressed against `world.DAILY_COST`, so the units'
  sense of a dollar travels with the era instead of being baked into the
  learner. It is a calibration constant: it should match what units actually
  spend per day, and there is a one-liner in the design notes to measure that.
- **Venues.** 2010 fixtures, checked against the year: the Kogi truck, the
  ArcLight, Intelligentsia on Abbot Kinney, L.A. Live, the World Stage on
  Degnan, Grand Central Market before the remodel, a startup on 2nd Street in
  what people had just started calling Silicon Beach. Nobody works below the
  2010 California minimum wage of $8.00, which is a test.

## What actually emerges

Units start flailing: they blow their stake in a week and then go hungry.
Given a few hundred simulated days of their own experience they find the shape
of a life — and nobody wrote the shape down.

```
  day   reward  wellbeing     funds  hunger  fatigue  work h  anom
  100   -0.829      0.375     26.59    0.69     0.89     0.9     5
  200   -0.552      0.435     68.31    0.61     0.79     1.6    13
  300   -0.268      0.450    138.79    0.74     0.54     0.0    21
  400   -0.170      0.460    309.47    0.66     0.64     2.0    24
  500   +0.154      0.517    499.72    0.58     0.52     2.0    24
```

`rhythm` reports what units *did*, not what they wanted — a unit that set out
for a shut office wanted to work but spent the hour idle, and only the second
of those is a fact about the day it has learned. After 500 days:

```
 hour  mostly       share
    0  asleep        42%
    3  asleep        41%
    7  asleep        33%
   11  working       21%
   14  working       21%
   15  working       22%
   22  asleep        33%
```

Nights are for sleeping and the middle of the day is for working, and the
simulation never said so. It knows only that some venues pay wages between
certain hours. Work hours also fall to zero on Saturdays and Sundays — the
units found the weekend by discovering that showing up to a shut workplace
wastes the hour.

**Where it is weak.** Roughly a quarter of all unit-hours still resolve to
`idle`, mostly units setting out to do something that turns out to be
impossible, and the peak working share is 21% rather than the half you would
want. The 1937 build hit a sharper working day, and not for a flattering
reason: its streetcar commutes were half again as long, so a misplaced
intention cost more and the schedule was partly geography doing the learner's
job. The 2010 numbers are the more honest read on how well these units learn.

## Jacking in

```
$ python -m simulacron --units 12 jackin --days 500
LINK ESTABLISHED -- Marguerite Ashgrove
March 2010. You are standing in L.A. Live, Downtown.
Your body remains in the lab. Type 'help' for what this body knows.
Thursday 00:00, March 2010 -- L.A. Live, Downtown.
The door is $20.00. You have $89.00.

2010> go philippe
Boarding for Philippe the Original. 1 stops.

2010> wait 2
2 hour(s) pass. Thursday 02:00, March 2010 -- Philippe the Original, Downtown.
A meal runs $7.00. You have $89.00.
It is shut.

2010> status
Marguerite Ashgrove
  lives   a converted loft in the Arts District (Downtown)
  works   a law firm on Bunker Hill at $34.00/hr
  hunger      [#######...]
  fatigue     [##........]
  dissonance  [##########]

2010> jackout
LINK SEVERED after 2.2 simulated hours.
Marguerite Ashgrove is walking again. Dissonance now 2.00.
```

While you hold a unit its own policy is suspended — it is not asleep, it is
simply not there. When you let go it resumes mid-stride and finds hours it
cannot account for. Those gaps accumulate as `dissonance`, and a unit whose
dissonance passes 1.0 starts probing the edges of the prototype:

```
[Thursday 09:00] ANOMALY: Selma Ashgrove is asking questions about the shape of the world.
```

Which is, of course, how the picture starts.

## Layout

```
world.py      2010 Los Angeles: districts, venues, wages, prices, transit
worldview.py  received knowledge of an Earth that was never built, and its seams
lifecourse.py age: capacity, frailty, and a Gompertz mortality curve
family.py     pairing, birth, kinship, and belief inherited rather than learned
crime.py      robbery as an option, reflexive policing, and rumour
unit.py       a cyber being: body, traits, circadian rhythm, and the systems below
affect.py     appraisal, the emotions it yields, chronic stress and trauma
memory.py     episodes, salience, cued recall, and beliefs about places
selfmodel.py  competence, prospects, identity, expectation and its violation
social.py     other units as individuals: trust, affection, grudges
cognition.py  deliberation: options, evaluation, choice, and experiencing outcomes
brain.py      the learned habit layer (policy and value heads, actor-critic)
sim.py        the system; performs choices, runs the grind, with or without an operator
link.py       jack-in, jack-out, and the interactive 2010 shell
```

## Design notes

### Building the inner life

Four bugs were worth the finding:

- **Weighing options rehearsed memory.** Recall strengthens what it touches,
  and deliberation cues memory several times an hour. Within days every
  trivial episode had maximum salience and units had equally unforgettable
  lunches. Only deliberate recall rehearses now.
- **Option-count bias.** Thirteen bars against one bed meant `socialize` won
  the softmax on sheer count: units socialised 38% of the time while starving.
  Units now bring to mind the best option of each kind and choose among those,
  which is both the fix and a better model of how anyone decides.
- **Conditions were never appraised.** Hunger, debt, pain and unemployment are
  states, not events, so nothing scored them and a unit could starve without
  ever becoming distressed. They are appraised weakly every hour, which is what
  turns a bad month into chronic stress rather than a run of unrelated bad
  hours.
- **The commute was charged to the first hour.** A shift across town scored
  exactly zero (`travel -0.98, money +0.98`) and units with a long ride stopped
  going to work. The fare is now paid against the whole visit — and jobs are
  sited near homes, because drawing them at random handed people a daily round
  trip across the county.

### Getting the habit layer to learn at all

Six things had to be got right before units learned anything at all. Most of
them were found by measuring a population that had stopped improving and
asking what it was actually doing all day.

1. **Hourly commitments.** Deciding afresh every quarter hour produced
   twitching sleepwalkers who spent the whole day in transit and never earned a
   cent. A decision now commits the unit for an hour, and travel is charged
   against the intent that started the trip.
2. **Purchases fire once per commitment.** The obvious way to run an hour-long
   commitment is to re-resolve it every tick, which quietly had units eating
   four meals an hour and spending twenty-four dollars on one errand. Spending
   money became the most productive-looking thing a unit could do with a day.
   Continuous activities — working, sleeping — still accrue per tick.
3. **A value head, with its own learning rate.** A unit's reward is dominated by
   slow-moving state: how hungry it is, what is in its pocket. Against a single
   moving-average baseline that drift drowns the part that depends on the choice
   just made. The first version folded the value head's step into the policy's
   learning rate, which left it ten times too slow to track returns of this
   magnitude; the baseline sat near zero and the units never learned to look at
   the clock at all. The symptom is worth knowing: P(work) was 22% whether the
   workplace was open or shut.
4. **A circadian rhythm.** Without one a unit slept as happily at two in the
   afternoon as at two in the morning, and the population never settled into
   nights. It is the plainest fact about being modeled after us, and it was
   missing.
5. **Dense reward, linear penalties.** A unit that only found out it had erred
   once it was starving would never trace the failure back to the morning it
   skipped work. Squared drive penalties, meanwhile, produced units that did
   nothing all day but manage their own fatigue.
6. **Private minds.** Sharing one set of weights across the population is the
   obvious fix for how little experience a single unit gets — twenty-four
   decisions a day. It collapses: units differ in where they live, where they
   work and what they want, and one policy serving all of them converged onto a
   single intent for every hour of the day. `Simulation(shared_mind=True)`
   still reproduces it. Fully formed beings turn out to need their own minds.

To recalibrate `DAILY_COST` after changing venues, measure what units actually
spend:

```python
s = Simulation(n_units=24); s.run_days(500)
prev = [u.funds for u in s.units]; spend = 0.0
for _ in range(world.TICKS_PER_DAY * 14):
    s.step()
    for i, u in enumerate(s.units):
        spend += max(0.0, prev[i] - u.funds); prev[i] = u.funds
print(spend / len(s.units) / 14)
```
