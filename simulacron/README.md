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

Everything in that speech is a requirement. This is the build.

| Whitney's claim | How it is implemented |
| --- | --- |
| doesn't need a user to function | `Simulation.step()` drives every unit; no operator is required, and the CLI's `run` never links one |
| fully formed self-learning beings | each unit owns a policy network trained online by REINFORCE on its own lived reward — nothing about its day is scripted |
| they think | an observation of 26 signals in, a distribution over eight intents out, every simulated hour |
| they work / they eat | jobs with 1937 wages, meals with 1937 prices, and an economy a unit has to solve to stay fed |
| modeled after us | five sampled traits (sociability, diligence, appetite, restlessness, thrift) feed the observation, so units diverge |
| Los Angeles circa 1937 | eight districts, 27 period venues, Red Car travel times |
| jacked in / the body stays | `LinkSession` suspends a unit's policy and hands you its body; jacking out leaves a hole in its memory |

Pure Python and numpy. No torch, no downloads, no network.

## Run it

```bash
python -m simulacron --units 24 run --days 500      # unattended, reports as it goes
python -m simulacron --units 24 rhythm --days 500   # the weekday the units invented
python -m simulacron --units 24 units --days 500    # who is down there
python -m simulacron jackin --days 500              # walk around in 1937
python -m simulacron.test_simulacron                # 27 checks, ~10s
```

## What actually emerges

Units start flailing: they blow their stake at the Cocoanut Grove in two days
and then starve in the street. Given a few hundred simulated days of their own
experience they find the shape of a life — and nobody wrote the shape down.

```
  day   reward  wellbeing     funds  hunger  fatigue  work h  anom
  100   -0.271      0.423      0.33    0.76     0.77     1.1     1
  200   +0.055      0.524      1.26    0.66     0.57     1.7     4
  300   +0.190      0.451      1.28    0.69     0.68     0.0     6
  400   +0.278      0.488      2.98    0.57     0.59     2.9    11
  500   +0.387      0.498      3.96    0.69     0.52     2.5    12
```

The clearest result is the **working day**, which is learned rather than coded.
The simulation knows only that some venues pay wages between certain hours; it
never tells a unit when to sleep. Sampling intents across five weekdays after
500 days:

```
 hour  dominant     share
    0  sleep         36%
    ...
    8  sleep         34%
    9  work          32%
   10  work          44%
   ...
   17  work          35%
   18  sleep         36%
   ...
```

Work takes 09:00–17:00 and sleep takes the night. On Saturdays and Sundays the
work hours go to zero — the units found the weekend by discovering that showing
up to a shut workplace wastes the hour.

## Jacking in

```
$ python -m simulacron jackin --units 12 --days 500
LINK ESTABLISHED -- Ottoline Castellane
March 1937. You are standing in Philippe's, Olvera Street.
Your body remains in the lab. Type 'help' for what this body knows.
Thursday 00:00, March 1937 -- Philippe's, Olvera Street.
A meal runs 0.25. You have $5.43.
It is shut.

1937> go clifton
Boarding for Clifton's Brookdale cafeteria. 1 stops.

1937> wait 8
8 hour(s) pass. Thursday 08:00, March 1937 -- Clifton's Brookdale cafeteria, Spring Street.
A meal runs 0.35. You have $5.43.
Reachable: Central Library, City Hall, Grand Central Market, Pershing
  Square, the Bradbury Building offices

1937> eat
You eat. funds +5.43->5.08, hunger +0.23->0.00

1937> jackout
LINK SEVERED after 8.2 simulated hours.
Ottoline Castellane is walking again. Dissonance now 1.82.
```

While you hold a unit its own policy is suspended — it is not asleep, it is
simply not there. When you let go it resumes mid-stride and finds hours it
cannot account for. Those gaps accumulate as `dissonance`, and a unit whose
dissonance passes 1.0 starts probing the edges of the prototype:

```
[Thursday 11:30] ANOMALY: Ottoline Castellane is asking questions about the shape of the world.
```

Which is, of course, how the picture starts.

## Layout

```
world.py   1937 Los Angeles: districts, venues, wages, prices, Red Car times
unit.py    a cyber being: drives, traits, episodic memory, perception
brain.py   the policy network and the REINFORCE learner
sim.py     the system; resolves intents, runs with or without an operator
link.py    jack-in, jack-out, and the interactive 1937 shell
```

## Design notes

Three things had to be got right before units learned anything at all:

1. **Hourly commitments.** Deciding afresh every quarter hour produced twitching
   sleepwalkers who spent the whole day riding the Red Cars and never earned a
   cent. A decision now commits the unit for an hour, and travel is charged
   against the intent that started the trip.
2. **Dense reward.** A unit that only found out it had erred once it was
   starving would never trace the failure back to the morning it skipped work.
   It feels the coin in its pocket the way it feels its stomach.
3. **Linear penalties, not cliffs.** Squared drive penalties produced units that
   did nothing all day but manage their own fatigue.
