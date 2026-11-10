# Project 3: The Manipulator Jacobian and Velocity Control

**Full instructions:** [profshrestha/ee473-573 · Project 3](https://github.com/profshrestha/ee473-573/blob/main/projects/project-03-jacobian-control/README.md)

Consult Canvas for deliverables, deadlines, and the grading rubric.

This repository belongs to one team and holds one submission: a single package and a single
report, the same teams as Project 2.

---

## First, bring your Project 2 package across

This repository holds only what Project 3 adds. The forward kinematics, the parameter table and
the frame conventions all carry over unchanged, so copy them in from your team's Project 2
repository:

```
kinematics/transforms.py      from Project 2
kinematics/fk.py              from Project 2
robot/widowx.py               from Project 2
```

Nothing in them needs to change. If your Jacobian disagrees with finite differences in Part A, the
forward kinematics is the first place to look, not the second.

## Contents

```
kinematics/
  jacobian.py     jacobian(), jacobian_fd(), manipulability()      yours
  control.py      joint_rates(), straight_line(), follow()         yours
REPORT.md         the team report, and the document assessed
data/             the Part D hardware session
figures/          plots and photographs referenced by the report
```

Every function is a stub raising `NotImplementedError`, with the section it comes from named in
its docstring.

**Observe the `robot/` boundary.** Every quantity particular to this manipulator stays in
`robot/widowx.py`. The modules in `kinematics/` take numbers and return numbers.

---

## Order of work

1. **Part A.** `jacobian()`, then `jacobian_fd()`, then the comparison. Do not start Part B until
   they agree: everything downstream rests on this matrix.
2. **Part B.** `joint_rates()` with the direct inverse, `straight_line()`, `follow()`. Replay in
   RViz before trusting the plot.
3. **Part C.** The determinant sweep, `manipulability()`, then the same path with and without
   damping.
4. **Part D.** The hardware session.

Compose `REPORT.md` as the work proceeds.

## The manipulator

Unchanged from Project 2: the WidowX AI, addressed over Ethernet by the `trossen_arm` Python
driver, with `trossen-arm` pinned to **1.10.0** to match the controller firmware.

> **Part C never runs on hardware.** It drives the arm deliberately towards a singular
> configuration, where commanded joint rates grow without bound. That belongs in RViz only. Part D
> runs the Part B trajectory, nothing else.

Send the arm to the rest pose `[0, pi/2, pi/2, 0, 0, 0]` at the start and end of every session, and
raise `goal_time` on the first motion until you have seen the path at low speed.

## Team practice

Each member commits under their own account. For a team submission the commit history is the only
record of individual contribution, and the closing section of the report is read against it.

## Before committing

Review `git status` before each commit. Credentials, tokens, and `.env` files are never committed.
Log out of Codex and the GitHub CLI when leaving a station, which is shared.
