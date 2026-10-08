# Validation on ~1,000 labelled pairs

**Status: verified — truth includes the human check**

Pairs scored: 962 (143 land-use, 819 not). Truth source: {'human': 535, 'cant_tell (left out)': 40, 'claude': 467}.

Question asked (no format, no hints): *Before and after images of the same place, 6 years apart. Has anything lasting changed (like buildings, roads, ponds, cleared land), or are the differences only seasonal?*

## Detection of land-use change

| Rule | Found (recall) | Precision | False alarms on no-change pairs | Pairs flagged |
|---|---|---|---|---|
| 32B | 137/143 = 96% (91–98) | 137/521 = 26% (23–30) | 384/819 = 47% (43–50) | 521 of 962 |
| 235B | 113/143 = 79% (72–85) | 113/233 = 48% (42–55) | 120/819 = 15% (12–17) | 233 of 962 |
| 32B and 235B agree | 111/143 = 78% (70–84) | 111/218 = 51% (44–57) | 107/819 = 13% (11–16) | 218 of 962 |
| 32B or 235B | 139/143 = 97% (93–99) | 139/536 = 26% (22–30) | 397/819 = 48% (45–52) | 536 of 962 |

Percentages with 95% Wilson intervals.

## Answers by true class

| True class | n | 32B says land-use | 235B says land-use |
|---|---|---|---|
| land_use | 143 | 137 | 113 |
| seasonal | 400 | 200 | 67 |
| none | 383 | 184 | 53 |
| atmospheric | 36 | 0 | 0 |

Answers marked unclear/unparsed: {'32B': 0, '235B': 0}.

## Human check

- Pairs checked: 535 of 535 (can't tell: 40).
- Claude's land-use label matched the human on 417/495 checked pairs = 84% (81–87).
- Pairs where Claude and both models agreed (random sample): the human disagreed on 9/45 = 20% (11–34). This is the error rate hidden in the unchecked agreements.
- On the disagreements, the 32B matched the human on 69/450; Claude on 381/450.
- On the disagreements, the 235B matched the human on 309/450; Claude on 381/450.

## Claude self-audit of the models' false alarms

24 random pairs where both 32B and 235B said land-use and Claude did not, looked at again: models wrong 17, Claude wrong 2, unsure 5.
So most false alarms are real (71% (51–85) model wrong). Even counting every 'unsure' as a real change, precision of the best rule stays far below 90%.

## Decision rule

Scale the API step only if a rule reaches **≥ 90% precision** (lower interval bound near 90%) with useful recall. Otherwise keep the dataset to verified pairs.
