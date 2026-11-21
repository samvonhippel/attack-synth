"""
papers/wardega.py -- K. Wardega et al., "HoLA Robots: Mitigating Plan-Deviation Attacks in
Multi-Robot Systems with Co-Observations and Horizon-Limiting Announcements" (AAMAS 2023;
arXiv:2301.10704).

Model: a graph G = (V, E) of locations, a set of robots R following a MAPF plan x, and a prefix
alpha(t) of x announced by a central entity at time t.
Adversary: a compromised robot a in R changes only its own position on some interval (s, f) and
falsifies its self-reports, knowing alpha(t) but not which other robots are compromised (Def. 3).
Attack: a deviation that enters V_forbidden while the central entity receives exactly the reports
it expects (Def. 4) -- i.e. undetected by the paper's co-observation rule (two robots that become
adjacent necessarily observe each other, so a deviation is safe only while it stays unobservable).
Defense (not an attack notion, but what the search below checks): horizon-limiting announcements
(Def. 6) bound how far ahead alpha(t) commits the plan, so that every forbidden deviation is
forced to become observable before it can complete.

The paper's Algorithm 1 is: expand the compromised robot's reachable-position set R_a(t) one step
at a time; stop at the first time u* some cell of R_a(t) becomes observable by a non-compromised
robot; then check (ATTACKEXISTS) whether a walk inside R_a(t), avoiding ever becoming observable
before u*, can reach V_forbidden. That second check is exactly an attack_synth search: a "model" is
the compromised robot's position, phi = "not in V_forbidden", neighbors(m) are graph-adjacent moves
that stay unobserved, and max_steps = u* - s.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_synth import Atom, ModelType, make_attack_synth  # noqa: E402


@dataclass(frozen=True)
class Grid:
    width: int
    height: int

    def cells(self):
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    def adjacent(self, c):
        x, y = c
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield (nx, ny)

    def observes(self, a, b):
        """Co-observation (Def. 2): robots that are graph-adjacent (or co-located) necessarily see
        each other."""
        ax, ay = a
        bx, by = b
        return abs(ax - bx) + abs(ay - by) <= 1


def first_observation_time(grid: Grid, start: int, start_pos, other_plans: dict, max_t: int):
    """Expand the compromised robot's reachable-position set R_a(t) (BFS over the graph, one step
    per time unit) and find u*, the first time some cell of R_a(t) is adjacent to some other,
    non-compromised robot's known position at that time (Algorithm 1, lines 1-6)."""
    R = {start_pos}
    for t in range(start, max_t):
        others_at_t = {pos for rid, plan in other_plans.items() for pos in [plan.get(t)] if pos is not None}
        if any(grid.observes(r, o) for r in R for o in others_at_t):
            return t, R
        R = {n for r in R for n in grid.adjacent(r)}
    return max_t, R


def unobserved_neighbors(grid: Grid, pos, t, other_plans: dict):
    others_at_t1 = {p for plan in other_plans.values() for p in [plan.get(t + 1)] if p is not None}
    for n in grid.adjacent(pos):
        if not any(grid.observes(n, o) for o in others_at_t1):
            yield n


def attack_exists(grid: Grid, start_pos, s: int, u_star: int, other_plans: dict, forbidden: frozenset):
    """ATTACKEXISTS: a walk inside R_a, never observed before u_star, that enters V_forbidden."""
    verifier = lambda pos_t, _phi: pos_t[0] not in forbidden
    mt = ModelType("robot position", neighbors=lambda pos_t: ((n, pos_t[1] + 1)
                                                               for n in unobserved_neighbors(grid, pos_t[0], pos_t[1], other_plans)),
                  key=lambda pos_t: pos_t)
    synth = make_attack_synth(verifier, mt, max_steps=max(0, u_star - s))
    return synth((start_pos, s), object(), limit=1)


def horizon_limiting_check(grid: Grid, robots: dict, compromised: str, forbidden: frozenset, horizon: int):
    """Algorithm 1 as a whole, for one candidate compromised robot: find u*, then search for an
    attack within the window the attacker is guaranteed unobserved. The announcement is
    horizon-limiting with respect to this robot iff no such attack exists for any start time."""
    other_plans = {r: plan for r, plan in robots.items() if r != compromised}
    plan = robots[compromised]
    results = []
    for s in range(0, horizon):
        if s not in plan:
            continue
        u_star, _ = first_observation_time(grid, s, plan[s], other_plans, max_t=s + horizon)
        r = attack_exists(grid, plan[s], s, u_star, other_plans, forbidden)
        results.append((s, u_star, r))
    return results


# --------------------------------------------------------------------------- an illustrative instance
def warehouse_instance():
    """A 6x6 warehouse grid. Robot 'guard' patrols a fixed route; robot 'cart' is compromised and
    is near a forbidden aisle (shelving that must never be entered while occupied)."""
    grid = Grid(6, 6)
    forbidden = frozenset({(5, 5), (5, 4), (4, 5)})           # a restricted shelving corner
    guard_route = {t: (0, min(t, 5)) for t in range(0, 12)}    # patrols straight up column 0, out of sight
    cart_route = {t: (2, 2) for t in range(0, 12)}             # cart idles near the middle, far from the guard
    robots = {"guard": guard_route, "cart": cart_route}
    return grid, robots, forbidden


def main():
    grid, robots, forbidden = warehouse_instance()
    print("Wardega et al. -- plan-deviation attack search (Algorithm 1)")
    print(f"  grid {grid.width}x{grid.height}, V_forbidden = {sorted(forbidden)}")
    for horizon_announced in (2, 12):
        print(f"  -- announcement horizon = {horizon_announced} steps --")
        results = horizon_limiting_check(grid, robots, "cart", forbidden, horizon=horizon_announced)
        any_attack = False
        for s, u_star, r in results:
            if r.found:
                any_attack = True
                path = r.best.path
                print(f"     start t={s}: undetected until u*={u_star}; ATTACK found, "
                      f"{len(path)-1} moves to {r.best.model[0]} (enters V_forbidden)")
        if not any_attack:
            print("     no undetected deviation reaches V_forbidden from any start time: horizon-limiting holds.")


if __name__ == "__main__":
    main()
