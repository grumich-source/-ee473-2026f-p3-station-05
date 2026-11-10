# Project 3 Report: The Manipulator Jacobian and Velocity Control

**Station:**
**Team:**
**Date:**

---

## About this document

This is the team report, one per team. It is read on GitHub, and should be composed as a technical
document that a reader can follow rather than as a form with its blanks completed. Every claim is
supported by evidence the reader can verify: a number, a plot, or pasted output.

Delete this section before submission.

---

## 1. The Jacobian

State the construction you used and the frame convention it is written in, linear velocity above
angular, both in the base frame.

Give the Jacobian at one configuration, with the configuration stated, so a reader can check a
column by hand.

## 2. Verification against finite differences

| | |
|---|---|
| Configurations tested | |
| Step size h used | |
| Largest disagreement | |

Account for the choice of h. The error does not fall monotonically as h shrinks, and the reason is
the point of the exercise.

## 3. Resolved-rate control

The commanded straight line, the two endpoint poses, and the integration step size.

```
![Realized path against commanded](figures/straight-line.png)
```

| | |
|---|---|
| Largest deviation from the line | |

The deviation is not zero. Say why, in one sentence.

Confirm the trajectory was replayed in RViz, and note anything the replay showed that the plot did
not.

## 4. The determinant sweep

Which joints does det J depend on? Report the range of det J as each joint is swept with the
others held fixed.

| Joint swept | det J range | depends on it |
|---|---|---|
| joint_0 | | |
| joint_1 | | |
| joint_2 | | |
| joint_3 | | |
| joint_4 | | |
| joint_5 | | |

An arm whose wrist axes intersect gives a determinant independent of the first and last joint
variables. State what you found, and what it implies about this arm.

## 5. The singular configuration

The configuration found, how it was found, and which joints are responsible.

```
![Manipulability along the path](figures/manipulability.png)
```

## 6. Damping

```
![Joint rates, direct inverse against damped least squares](figures/damping.png)
```

| | Direct inverse | Damped least squares |
|---|---|---|
| Largest commanded joint rate | | |
| Largest deviation from the path | | |
| λ used | n/a | |

**The trade-off.** Quantify both effects from the data above, then describe how λ would be chosen
for a given application. This is the central result of the project.

## 7. Hardware session

| Point | Commanded angles | RViz pose | Measured | Error (mm) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

State the measurement procedure before the numbers. The same joint angles were commanded in both
cases, so account for the difference.

## 8. Discussion

What you would do differently, what remains uncertain, and what did not work, with the reason.

## 9. Who did what

| Member | Contribution |
|---|---|
| | |
| | |
| | |

This section is read against the commit history.

---

<details>
<summary><strong>Getting equations to render on GitHub</strong></summary>

**Use fenced math blocks.** This form always works:

````
```math
J_i = \begin{bmatrix} z_{i-1} \times (o_n - o_{i-1}) \\ z_{i-1} \end{bmatrix}
```
````

**Inline math is where this goes wrong.** GitHub applies Markdown emphasis *before* it resolves
math, so two underscores in the same paragraph pair into italics and carry the subscripts away with
them. When an expression contains an underscore, delimit it with backticks inside the dollars:

````
$`z_{i-1}`$        renders correctly
$z_{i-1}$          breaks if another underscore appears nearby
````

**Verify in the browser.** Push the file, open it on github.com, and inspect the result. An
editor's preview uses a different renderer, as does the GitHub mobile application, which does not
render mathematics at all.

</details>
