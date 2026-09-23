# Publication verification — September 23, 2026

**Public repository and private-browser verification passed.**

Repository: [shahdevansh/pacman-dqn-class3](https://github.com/shahdevansh/pacman-dqn-class3).

The name follows the existing `custom-llm-class4` convention: topic followed by class number. This Pac-Man work is Class 3, listed as Assignment 2 in bCourses.

## Anonymous access and file checks

An independent HTTP check used no login, cookies, or authorization header. All **78 published files** returned HTTP 200 and matched the committed and local bytes. All **72 README-relative link occurrences** (64 unique targets), including **39 images** (36 GIFs and three dashboards), resolved to those files. There were no HTTP errors, missing files, extra files, or content mismatches.

The initial complete file audit used immutable commit `d0752967cdbc4f3211d87401a990c6dcdf01b842`. The subsequent notebook-gallery change is commit `73cab0c6e8dda8e7f22d9e4c987cd95165f3663c`; the same already-verified GIF files are linked at immutable URLs. A final anonymous check of the latest public commit is retained in the local publication audit.

## Dia incognito visual inspection

The repository was opened in a new **Incognito** window in Dia. GitHub displayed **Sign in** and **Sign up**, confirming that the public view did not rely on an authenticated account.

The browser inspection covered:

- The public repository, file listing, rendered README, run comparison and all five official score rows with mean **450**.
- Embedded training dashboards and representative baseline, intermediate, final-checkpoint, and best-game GIFs in the README.
- GitHub's actual notebook preview, including the completed **Episode 500/500** training log, saved before/after score table, training dashboard, and separate supplemental results.
- The notebook's added gameplay gallery, visibly rendering the saved baseline and final trained GIFs. All 22 final-run GIF files are included in that gallery and separately covered by the anonymous file checks.

## Rendering issue found and corrected

GitHub rendered the notebook's original `image/gif` outputs as text placeholders, although the GIF payloads were present. A clearly labeled gallery was added to the final explanatory markdown cell, referencing the same saved GIF files. The corrected gallery was then reloaded and visually checked in the incognito browser.

No training or evaluation was rerun. All **28 code cells**, **59 original outputs**, and execution counters remain unchanged from the raw Colab export. Only the final explanatory markdown cell differs. The preserved null execution counters and their verification boundary remain documented in [export verification](verification_500.md).

## Submission boundary

**bCourses submission is left to the student.** This publication workflow did not submit the assignment. The repository URL above is the deliverable to paste into bCourses; publication and access checks are not a submission receipt.
