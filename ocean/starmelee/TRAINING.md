# StarMelee training: sparse rewards, PFSP, and the exploiter league

How the duel policy is trained as of 2026-07-24, and why each piece exists.
The dense-shaping mode (`sparse_reward = 0`) is unchanged and remains the
fallback; everything below concerns the sparse + self-play setup that
`config/starmelee.ini` now ships.

## Reward economy (sparse mode)

`sparse_reward = 1` zeroes every dense shaping coefficient in `c_init`
(band/aim/strafe/cycle/spacing/step/loiter/fire-nudge) and disables the
APPROACH auto-fire assist — the trigger belongs to the policy's 4th action
head alone. What remains:

| signal | value | key |
|---|---|---|
| killing blow (projectile only) | +2.0 to the shooter | `sparse_kill_reward` |
| damage dealt | +dmg/hp_max x 1.0 | `sparse_damage_scale` |
| damage taken (incl. planet/rocks) | -dmg/hp_max x 0.6 | `sparse_damage_taken_scale` |
| death | -1.0 terminal | hardcoded |
| combat timeout | -0.3 x out-of-contact fraction | `sparse_timeout_penalty` |
| button toggle | -0.003 per change | `input_change_penalty` (survives sparse) |

Design invariants:

- Kill credit pays only for the killing blow, so a self-inflicted planet or
  asteroid death is the victim's loss alone (no reward for outlasting luck).
  Round *scoring* for the opponent pool counts any elimination as a win —
  reward and scoring deliberately differ here.
- Damage dealt above damage taken makes an exchange positive-sum for the
  pair: the shared policy's best response to aggression is fighting back,
  not fleeing.
- The timeout charge is a *cowardice tax*: scaled by the fraction of the
  round spent beyond `disengage_range` (`ticks_in_contact` clock,
  `contact_frac` metric). A ship that fought all round pays nothing at the
  buzzer.
- `input_change_penalty` is control-style pricing, not task shaping, so it
  is not zeroed by sparse mode. `action_repeat = 4` remains the structural
  anti-twitch mechanism.

## Failure history (why the economy looks like this)

Each rule above was paid for by a failed run:

1. **Entropy collapse** (ent_coef 0.01, damage scale 0.5): sparse rewards
   lost to the entropy bonus; entropy 0.9 -> 2.2, hits/ep 5 -> 0.08, shots
   tripled into spray. Sparse economies cannot outbid a dense-mode entropy
   coefficient — hence `ent_coef = 0.002` and damage scale 1.0.
2. **Defensive meta** (symmetric damage, no draw cost): dodging compounds —
   better defense lowers expected kill payoff, which breeds fewer attacks.
   Kills 19% -> 7%, timeouts 93%.
3. **Mutual avoidance** (flat -0.1 draw penalty): a flat penalty charges the
   fighter and the drifter alike, and drifting is the safer way to eat it;
   ships parted to opposite ends of the torus. Pure sparse has *no*
   gradient toward proximity — the cowardice tax and the positive-sum
   damage split are what restored contact (kills 39%, contact ~40%).
4. **Twitchy control**: fixed by letting `input_change_penalty` survive
   sparse (toggles/decision 0.22 -> 0.06; accuracy improved as a side
   effect — spray was partly twitch).

## Opponent league

`[selfplay]` + `[vec]` in `config/starmelee.ini` wire the native trainer's
frozen-bank pool (see `pufferlib/selfplay.py`):

- ~60% of envs per buffer are pure mirror self-play; 4 banks (~8% of envs
  each) hold frozen opponents rotated by **PFSP**: sampling weight
  `(1 - P(win))^pfsp_p` from Elo-implied odds, so the learner focuses on
  opponents it still loses to (`pfsp_p = 2`, newest `pfsp_exclude_newest`
  snapshots skipped as near-copies).
- One bank is a **fixed exploiter** (`fixed_bank_paths`): a dense-reward
  brawler pinned as a league guardrail. A homogeneous self-play population
  can drift into a degenerate meta together (see failure 3); an exploiter
  with different reward DNA keeps committed attacks in the opponent
  distribution. Its per-bank winrate (`hist_*_bank_<b>`) is the running
  guardrail readout. Keep exploiter share <= ~15% of envs.
- Swap discipline: a rotating bank advances once the primary proves
  `swap_winrate` over `min_games` rounds (draws 0.5), or on
  `opp_timeout_steps`. Round results come from `sm_record_game` (a death
  decides a round; only slot 0's timeout logs the draw).

Known-blunted guardrail: the shipped dense exploiter predates the 1.5x
cannon range and never used the free-fire trigger unassisted; the primary
beats it ~88%. Retrain a dense brawler under current physics for a sharper
one, and consider the run-3 "avoider" checkpoint as a hunting exploiter.

## Recipe

```
# seed from the latest good sparse checkpoint, or the dense 300M fighter
puffer train starmelee --load-model-path <ckpt.bin>
python ocean/starmelee/export_policy.py <final ckpt.bin>   # -> exe playback
./starmelee.exe
```

Bootstrap note: `--load-model-path` loads before `selfplay.setup`, so the
opponent pool seeds itself from the loaded weights. `latest` resolution
everywhere skips `pool/` snapshots.

Watch during a run: `contact_frac` (avoidance), `crash_rate` (kill rate,
in sparse deaths ~= kills), `attack_passes`/`shots_fired` (accuracy),
`input_changes` (twitch), `hist_*_bank_*` (per-opponent winrates), entropy
(healthy ~0.6-0.9 here; ~2+ means the reward signal is losing to the
entropy bonus, ~0.15 means near-deterministic — consider ent_coef 0.004).
