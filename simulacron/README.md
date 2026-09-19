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
| fully formed self-learning beings | each unit owns a policy-and-value network trained online from its own lived reward — nothing about its day is scripted |
| they think | an observation of 26 signals in, a distribution over eight intents out, every simulated hour |
| they work / they eat | jobs at 2010 wages, meals at 2010 prices, and an economy a unit has to solve to stay fed |
| modeled after us | five sampled traits (sociability, diligence, appetite, restlessness, thrift) and a circadian rhythm |
| Los Angeles, 2010 | eight districts, 39 venues, Metro Rail where it existed that year |
| jacked in / the body stays | `LinkSession` suspends a unit's policy and hands you its body; jacking out leaves a hole in its memory |

Pure Python and numpy. No torch, no downloads, no network.

## Run it

```bash
python -m simulacron --units 24 run --days 500      # unattended, reports as it goes
python -m simulacron --units 24 rhythm --days 500   # what the units actually do, by hour
python -m simulacron --units 24 units --days 500    # who is down there
python -m simulacron jackin --days 500              # walk around in 2010
python -m simulacron.test_simulacron                # 31 checks, ~13s
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
world.py   2010 Los Angeles: districts, venues, wages, prices, transit times
unit.py    a cyber being: drives, traits, circadian rhythm, episodic memory
brain.py   the policy and value heads, and the actor-critic learner
sim.py     the system; resolves intents, runs with or without an operator
link.py    jack-in, jack-out, and the interactive 2010 shell
```

## Design notes

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
