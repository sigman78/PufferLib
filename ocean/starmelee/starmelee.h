// StarMelee: Star Control Melee style ships around a gravity planet on a torus.
// Phase 1 task: fly from spawn (A) to a beacon (B) without smashing into the
// planet. Ships are torque/thrust driven: hold left/right to apply angular
// acceleration, hold engine to accelerate along the heading. The planet pulls
// with inverse-square gravity and bounces ships off with damage.
// Combat mode: duel with a real cannon — short-range projectiles fired in
// bursts, a reload that forces a flee-and-return rhythm, and asteroids that
// double as destructible cover.
#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define STARMELEE_OBS_SIZE 62
#define STARMELEE_OBS_ASTEROIDS 4  // rocks with their own stable obs slots
#define STARMELEE_MAX_SHIPS 8
#define STARMELEE_MAX_ASTEROIDS 8
#define STARMELEE_MAX_PROJECTILES 64
#define STARMELEE_TRAIL_LEN 48

const unsigned char SM_STATE_APPROACH = 0;
const unsigned char SM_STATE_FLEE = 1;     // magazine empty: reload while opening distance
const unsigned char SM_STATE_RETREAT = 2;  // low hp: survival first

const unsigned char SM_RESULT_SUCCESS = 0;
const unsigned char SM_RESULT_CRASH = 1;
const unsigned char SM_RESULT_TIMEOUT = 2;
const unsigned char SM_RESULT_NONE = 3;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float success_rate;
    float crash_rate;
    float timeout_rate;
    float final_distance;
    float planet_hits;
    float input_changes;  // button toggles per step (jitter measure)
    float attack_passes;  // combat: projectile hits landed on the enemy
    float shots_fired;    // combat: cannon rounds fired
    float full_cycles;    // combat: burst followed by a clean disengage
    float retreats;       // combat: times low hp forced a retreat
    float asteroid_hits;  // asteroid impacts taken
    float asteroid_kills; // rocks broken by this ship's cannon fire
    float n;
} Log;

// Cannon round: straight-line flight (gravity does not bend shots), inherits
// the shooter's velocity at launch, dissolves at end of life or on any
// contact (planet, rock, ship). Rocks it hits disintegrate.
typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    int owner;
    int life;    // remaining ticks; <= 0 means the slot is free
} Projectile;

// Straight-line hazard: deliberately ignores gravity, disintegrates on any
// contact (ship or planet) and respawns later with a fresh trajectory
typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float radius;
    int active;
    int respawn_timer;
    // render-only rock outline, generated at spawn
    Vector2 shape[10];
    int num_vertices;
    // render-only explosion at the disintegration point
    float boom_x;
    float boom_y;
    float boom_radius;
    int boom_timer;
} Asteroid;

typedef struct {
    float x;
    float y;
    float vx;
    float vy;
    float heading;
    float omega;
    float hp;
    float goal_x;
    float goal_y;
    float prev_goal_dist;
    float episode_return;
    int episode_tick;
    int planet_hits;
    int input_changes;
    int asteroid_hits;
    unsigned char thrusting;
    unsigned char turning_left;
    unsigned char turning_right;
    unsigned char firing;
    unsigned char last_result;
    int event_timer;  // render-only feedback countdown
    // combat mode
    unsigned char attack_state;  // APPROACH / FLEE / RETREAT
    int mag_rounds;              // this spawn's magazine size (trait-like)
    int ammo;                    // rounds left in the magazine
    int cannon_timer;            // ticks until the next round (or reload end)
    unsigned char reloading;     // 1 while the 2 s reload runs (FLEE driver)
    unsigned char cycle_paid;    // this FLEE already resolved its cycle
    int hits_this_mag;           // hits landed by the current magazine
    int attack_passes;           // projectile hits landed
    int shots_fired;
    int asteroid_kills;
    int full_cycles;
    int retreats;
    float prev_enemy_dist;       // < 0 until first combat tick
    unsigned char just_finished; // episode ended this decision window
    int fire_flash;              // render-only muzzle flash countdown
    int hit_flash;               // render-only got-hit countdown
    // Per-spawn trait multipliers (1.0 +- trait_variation), observable by
    // the ship itself so the shared policy can act on its own build
    float t_thrust;
    float t_turn;
    float t_speed;
    float t_fragility;  // damage (and its reward pain) taken multiplier
    float t_caution;    // how much being fired at hurts this ship's reward
} Ship;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;

    // config
    float size;            // torus side length in world units
    int num_ships;
    int max_ticks;         // per-ship episode length limit
    float thrust;          // forward acceleration = engine force / mass
    float turn_accel;      // angular acceleration = torque / inertia
    float max_turn_rate;   // hard clamp on |omega|
    float angular_damping; // omega *= damping each tick
    float linear_damping;  // velocity *= damping each tick
    float max_speed;       // thrust cannot push speed above this
    float gravity;         // gravity acceleration at the planet surface
    float gravity_falloff; // exponent p in a(r) = gravity * (Rp/r)^p
    float planet_radius;
    float ship_radius;
    float goal_radius;
    float hp_max;
    float damage_scale;    // hp lost per unit of normal impact speed
    float restitution;     // bounce elasticity, 0..1
    float step_penalty;
    float progress_scale;  // reward per (goal distance shrink / size)
    float spawn_clearance; // min spawn/beacon distance from the planet center
    float min_goal_frac;   // min spawn-to-beacon distance as a fraction of size
    float input_change_penalty; // reward lost per button state change per step
    int action_repeat;     // physics ticks per decision (latched buttons)

    // combat (duel) mode
    int combat;            // 0 = navigate to beacon, 1 = duel the other ship
    float attack_range_min;  // engagement band, world units
    float attack_range_max;
    float aim_cone;          // max bearing error that counts as tracking, rad
    float disengage_range;   // flee distance that completes a cycle
    float engage_scale;      // shaping scale for band-approach / flee distance
    float aim_reward;        // per tracking tick inside the band
    float strafe_reward;     // per tick, scaled by transverse speed while tracking
    float hit_reward;        // landing a projectile on the enemy
    float fire_nudge;        // +nudge per shot fired tracking-in-range, -nudge wild
    float cycle_reward;      // completing the disengage
    float targeted_penalty;  // charged to the ship a projectile hit
    float shot_damage;       // hp a projectile hit takes off the victim
    float duel_spawn_frac;   // spawn separation as a fraction of size
    float trait_variation;   // +- range of per-spawn trait multipliers, 0..0.5
    float loiter_reward;     // per tick spent beyond disengage_range while off duty
    float retreat_hp_frac;   // hp fraction that forces RETREAT; 0 disables
    float hp_regen;          // hp per tick while beyond disengage_range
    float close_range;       // keep-out bubble around the enemy, world units
    float too_close_penalty; // per tick inside the bubble, scaled by depth
    float avoid_range;       // collision-course deterrent active inside this
    float collision_course_penalty; // per tick on a course through the bubble

    // cannon (combat mode)
    int cannon_rounds;         // magazine size (min when cannon_rounds_max set)
    int cannon_rounds_max;     // per-spawn magazine sampled in [rounds, max]
    int cannon_interval_ticks; // ticks between rounds within a salvo
    int cannon_reload_ticks;   // ticks to refill an empty magazine
    float projectile_speed;    // muzzle speed added along the heading
    float projectile_range;    // flight distance at muzzle speed before dissolving

    // asteroids (0 = off)
    int num_asteroids;
    float asteroid_radius;       // nominal; each spawn varies +-25%
    float asteroid_damage;       // hp cost of an impact (scaled by fragility)
    float asteroid_speed_min;
    float asteroid_speed_max;    // clamped to max_speed / 2
    int asteroid_respawn_ticks;  // delay before a destroyed rock returns
    float danger_hp_weight;      // hazard pain grows this much as hp empties

    Ship ships[STARMELEE_MAX_SHIPS];
    Asteroid asteroids[STARMELEE_MAX_ASTEROIDS];
    Projectile projectiles[STARMELEE_MAX_PROJECTILES];
    unsigned int rng;

    // render-only state (untouched during training)
    Vector2 trails[STARMELEE_MAX_SHIPS][STARMELEE_TRAIL_LEN];
    int trail_idx[STARMELEE_MAX_SHIPS];
    int trail_count[STARMELEE_MAX_SHIPS];
} StarMelee;

static inline float sm_randf(StarMelee* env) {
    return rand_r(&env->rng) / (float)RAND_MAX;
}

static inline float sm_clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Shortest signed displacement between two coordinates on the torus
static inline float sm_wrap_delta(float d, float size) {
    return d - size * roundf(d / size);
}

// Wrap a coordinate into [0, size)
static inline float sm_wrap_pos(float x, float size) {
    x = fmodf(x, size);
    if (x < 0.0f) {
        x += size;
    }
    return x;
}

static inline float sm_torus_dist(StarMelee* env, float ax, float ay, float bx, float by) {
    float dx = sm_wrap_delta(bx - ax, env->size);
    float dy = sm_wrap_delta(by - ay, env->size);
    return sqrtf(dx*dx + dy*dy);
}

static inline float sm_planet_x(StarMelee* env) { return 0.5f * env->size; }
static inline float sm_planet_y(StarMelee* env) { return 0.5f * env->size; }

void c_init(StarMelee* env) {
    if (env->size < 64.0f) env->size = 360.0f;
    if (env->num_ships < 1) env->num_ships = 1;
    if (env->num_ships > STARMELEE_MAX_SHIPS) env->num_ships = STARMELEE_MAX_SHIPS;
    env->num_agents = env->num_ships;
    if (env->max_ticks < 16) env->max_ticks = 1500;
    if (env->thrust <= 0.0f) env->thrust = 0.05f;
    if (env->turn_accel <= 0.0f) env->turn_accel = 0.004f;
    if (env->max_turn_rate <= 0.0f) env->max_turn_rate = 0.06f;
    env->angular_damping = sm_clampf(env->angular_damping, 0.0f, 1.0f);
    if (env->angular_damping == 0.0f) env->angular_damping = 0.95f;
    env->linear_damping = sm_clampf(env->linear_damping, 0.0f, 1.0f);
    if (env->linear_damping == 0.0f) env->linear_damping = 0.999f;
    if (env->max_speed <= 0.0f) env->max_speed = 2.0f;
    if (env->gravity <= 0.0f) env->gravity = 0.22f;
    if (env->gravity_falloff <= 0.0f) env->gravity_falloff = 1.5f;
    if (env->planet_radius <= 0.0f) env->planet_radius = 12.0f;
    env->planet_radius = fminf(env->planet_radius, 0.15f * env->size);
    if (env->ship_radius <= 0.0f) env->ship_radius = 6.0f;
    env->ship_radius = fminf(env->ship_radius, env->planet_radius);
    if (env->goal_radius <= 0.0f) env->goal_radius = 14.0f;
    if (env->hp_max <= 0.0f) env->hp_max = 100.0f;
    if (env->damage_scale < 0.0f) env->damage_scale = 20.0f;
    env->restitution = sm_clampf(env->restitution, 0.0f, 1.0f);
    if (env->restitution == 0.0f) env->restitution = 0.5f;
    if (env->progress_scale == 0.0f) env->progress_scale = 2.0f;
    if (env->step_penalty == 0.0f) env->step_penalty = -0.002f;
    if (env->spawn_clearance <= 0.0f) env->spawn_clearance = 4.0f * env->planet_radius;
    // Feasibility: beacons must sit outside the collision radius (reachable)
    // and inside the torus (a point at spawn_clearance from the planet exists).
    float min_clear = env->planet_radius + env->ship_radius + env->goal_radius + 2.0f;
    env->spawn_clearance = sm_clampf(env->spawn_clearance, min_clear, 0.45f * env->size);
    env->min_goal_frac = sm_clampf(env->min_goal_frac, 0.0f, 0.45f);
    if (env->min_goal_frac == 0.0f) env->min_goal_frac = 0.3f;
    if (env->input_change_penalty < 0.0f) env->input_change_penalty = 0.0f;
    if (env->action_repeat < 1) env->action_repeat = 1;
    if (env->action_repeat > 8) env->action_repeat = 8;

    if (env->attack_range_min <= 0.0f) env->attack_range_min = 4.0f * env->ship_radius;
    // Ordering matters: cap max below the disengage cap so the chain
    // min < max < disengage <= 0.45*size always holds afterwards
    env->attack_range_max = fminf(env->attack_range_max, 0.40f * env->size);
    if (env->attack_range_max <= env->attack_range_min) {
        env->attack_range_max = fminf(2.5f * env->attack_range_min, 0.40f * env->size);
        env->attack_range_min = fminf(env->attack_range_min, 0.5f * env->attack_range_max);
    }
    env->aim_cone = sm_clampf(env->aim_cone, 0.0f, 1.0f);
    if (env->aim_cone == 0.0f) env->aim_cone = 0.15f;
    env->disengage_range = fminf(env->disengage_range, 0.45f * env->size);
    if (env->disengage_range <= env->attack_range_max) {
        env->disengage_range = fminf(2.0f * env->attack_range_max, 0.45f * env->size);
        if (env->disengage_range <= env->attack_range_max) {
            env->disengage_range = 0.5f * (env->attack_range_max + 0.45f * env->size);
        }
    }
    if (env->engage_scale == 0.0f) env->engage_scale = 2.0f;
    if (env->aim_reward < 0.0f) env->aim_reward = 0.0f;
    if (env->strafe_reward < 0.0f) env->strafe_reward = 0.0f;
    if (env->hit_reward == 0.0f) env->hit_reward = 0.75f;
    if (env->fire_nudge < 0.0f) env->fire_nudge = 0.0f;
    if (env->cycle_reward == 0.0f) env->cycle_reward = 0.5f;
    if (env->targeted_penalty < 0.0f) env->targeted_penalty = 0.0f;
    env->duel_spawn_frac = sm_clampf(env->duel_spawn_frac, 0.0f, 0.45f);
    if (env->duel_spawn_frac == 0.0f) env->duel_spawn_frac = 0.35f;
    if (env->combat && env->num_ships < 2) env->combat = 0;
    env->trait_variation = sm_clampf(env->trait_variation, 0.0f, 0.5f);
    if (env->loiter_reward < 0.0f) env->loiter_reward = 0.0f;
    env->retreat_hp_frac = sm_clampf(env->retreat_hp_frac, 0.0f, 0.9f);
    if (env->hp_regen < 0.0f) env->hp_regen = 0.0f;
    // Regen is RETREAT's only exit; without it ships would lock into
    // retreat forever and farm loiter_reward from hiding
    if (env->retreat_hp_frac > 0.0f && env->hp_regen == 0.0f) env->hp_regen = 0.1f;
    if (env->shot_damage < 0.0f) env->shot_damage = 0.0f;

    // Cannon: magazine bounded so all rounds of every ship fit the pool.
    if (env->cannon_rounds < 1) env->cannon_rounds = 3;
    int rounds_cap = STARMELEE_MAX_PROJECTILES / STARMELEE_MAX_SHIPS;
    if (env->cannon_rounds > rounds_cap) env->cannon_rounds = rounds_cap;
    if (env->cannon_rounds_max < env->cannon_rounds) {
        env->cannon_rounds_max = env->cannon_rounds;
    }
    if (env->cannon_rounds_max > rounds_cap) env->cannon_rounds_max = rounds_cap;
    if (env->cannon_interval_ticks < 1) env->cannon_interval_ticks = 10;
    if (env->cannon_reload_ticks < 1) env->cannon_reload_ticks = 120;
    if (env->projectile_speed <= 0.0f) env->projectile_speed = 1.5f * env->max_speed;
    if (env->projectile_range <= 0.0f) env->projectile_range = 8.0f * env->ship_radius;
    // Shots must be able to cross the engagement band they are fired in
    env->projectile_range = fmaxf(env->projectile_range, env->attack_range_max);

    // Spacing: keep-out bubble below the band, deterrent horizon above it
    if (env->close_range <= 0.0f) env->close_range = 4.0f * env->ship_radius;
    env->close_range = fminf(env->close_range, env->attack_range_min);
    if (env->too_close_penalty < 0.0f) env->too_close_penalty = 0.0f;
    if (env->avoid_range <= 0.0f) env->avoid_range = 12.0f * env->ship_radius;
    env->avoid_range = fmaxf(env->avoid_range, env->close_range);
    if (env->collision_course_penalty < 0.0f) env->collision_course_penalty = 0.0f;

    if (env->num_asteroids < 0) env->num_asteroids = 0;
    // Every rock must own an obs slot: an unobservable lethal hazard is
    // noise the policy cannot learn around
    if (env->num_asteroids > STARMELEE_OBS_ASTEROIDS) {
        env->num_asteroids = STARMELEE_OBS_ASTEROIDS;
    }
    if (env->asteroid_radius <= 0.0f) env->asteroid_radius = 7.0f;
    if (env->asteroid_damage < 0.0f) env->asteroid_damage = 20.0f;
    // Spec: asteroids stay slower than half the ship top speed
    float rock_cap = 0.5f * env->max_speed;
    env->asteroid_speed_max = sm_clampf(env->asteroid_speed_max, 0.0f, rock_cap);
    if (env->asteroid_speed_max == 0.0f) env->asteroid_speed_max = rock_cap;
    env->asteroid_speed_min = sm_clampf(env->asteroid_speed_min, 0.0f, env->asteroid_speed_max);
    if (env->asteroid_speed_min == 0.0f) {
        env->asteroid_speed_min = 0.3f * env->asteroid_speed_max;
    }
    if (env->asteroid_respawn_ticks < 1) env->asteroid_respawn_ticks = 300;
    env->danger_hp_weight = sm_clampf(env->danger_hp_weight, 0.0f, 3.0f);
}

// Hazard pain multiplier: the same hp loss should hurt more when hp is low,
// mirroring the convex value of health (death and retreat are nearby).
// Flat pain was measurably ignored: a blindfold test showed the trained
// policy paid zero attention to asteroid observations.
static inline float sm_pain_scale(StarMelee* env, Ship* s) {
    float hp_frac = sm_clampf(s->hp / env->hp_max, 0.0f, 1.0f);
    return 1.0f + env->danger_hp_weight * (1.0f - hp_frac);
}

static inline float sm_trait(StarMelee* env) {
    return 1.0f + env->trait_variation * (2.0f * sm_randf(env) - 1.0f);
}

// Gravity acceleration vector at (x, y), pointing toward the planet:
// a(r) = gravity * (Rp/r)^falloff, clamped at the surface value. True
// inverse-square (falloff 2) is barely felt at 10 Rp when thrust can fight
// it near the planet; the shallower default keeps the well relevant across
// the advertised ~10 Rp while leaving spawn range escapable.
static inline void sm_gravity_at(StarMelee* env, float x, float y, float* gx, float* gy) {
    float dx = sm_wrap_delta(sm_planet_x(env) - x, env->size);
    float dy = sm_wrap_delta(sm_planet_y(env) - y, env->size);
    float r2 = dx*dx + dy*dy;
    float rp2 = env->planet_radius * env->planet_radius;
    if (r2 < rp2) r2 = rp2;
    float r = sqrtf(r2);
    float ratio = env->planet_radius / r;
    float a;
    if (env->gravity_falloff == 2.0f) {
        a = env->gravity * ratio * ratio;
    } else if (env->gravity_falloff == 1.0f) {
        a = env->gravity * ratio;
    } else {
        a = env->gravity * powf(ratio, env->gravity_falloff);
    }
    *gx = a * dx / r;
    *gy = a * dy / r;
}

// True when the planet does not block the straight (minimal-image) segment
// between two points: aiming needs it, and it makes the planet usable as a
// shield. Segments are short (attack range) relative to the torus, so the
// planet image nearest to the first endpoint is the only one that matters.
static int sm_los_clear(StarMelee* env, float ax, float ay, float bx, float by) {
    float dx = sm_wrap_delta(bx - ax, env->size);
    float dy = sm_wrap_delta(by - ay, env->size);
    float px = sm_wrap_delta(sm_planet_x(env) - ax, env->size);
    float py = sm_wrap_delta(sm_planet_y(env) - ay, env->size);
    float len2 = dx*dx + dy*dy;
    float t = len2 > 1e-6f ? (px*dx + py*dy) / len2 : 0.0f;
    t = sm_clampf(t, 0.0f, 1.0f);
    float cx = t*dx - px;
    float cy = t*dy - py;
    return cx*cx + cy*cy >= env->planet_radius * env->planet_radius;
}

static int sm_nearest_enemy(StarMelee* env, int i) {
    Ship* s = &env->ships[i];
    int nearest = -1;
    float nearest_dist = 0.0f;
    for (int j = 0; j < env->num_ships; j++) {
        if (j == i) continue;
        float d = sm_torus_dist(env, s->x, s->y, env->ships[j].x, env->ships[j].y);
        if (nearest < 0 || d < nearest_dist) {
            nearest = j;
            nearest_dist = d;
        }
    }
    return nearest;
}

// Random point on the torus at least min_planet_dist from the planet center.
// Falls back to a point exactly min_planet_dist away in a random direction,
// which always exists since c_init clamps clearance to 0.45*size.
static void sm_sample_clear_point(StarMelee* env, float min_planet_dist, float* out_x, float* out_y) {
    for (int attempt = 0; attempt < 100; attempt++) {
        float x = env->size * sm_randf(env);
        float y = env->size * sm_randf(env);
        if (sm_torus_dist(env, x, y, sm_planet_x(env), sm_planet_y(env)) >= min_planet_dist) {
            *out_x = x;
            *out_y = y;
            return;
        }
    }
    float angle = 2.0f * PI * sm_randf(env);
    *out_x = sm_wrap_pos(sm_planet_x(env) + min_planet_dist * cosf(angle), env->size);
    *out_y = sm_wrap_pos(sm_planet_y(env) + min_planet_dist * sinf(angle), env->size);
}

static void sm_spawn_ship(StarMelee* env, int i) {
    Ship* s = &env->ships[i];
    float clearance = env->spawn_clearance;

    // Duel: spawn opposite an existing ship at duel_spawn_frac * size,
    // roughly facing it, so episodes open with two ships closing in.
    int anchor = -1;
    if (env->combat) {
        for (int j = 0; j < env->num_ships; j++) {
            if (j != i && env->ships[j].hp > 0.0f) {
                anchor = j;
                break;
            }
        }
    }
    if (anchor >= 0) {
        Ship* a = &env->ships[anchor];
        float sep = env->duel_spawn_frac * env->size;
        for (int attempt = 0; attempt < 100; attempt++) {
            float ang = 2.0f * PI * sm_randf(env);
            s->x = sm_wrap_pos(a->x + sep * cosf(ang), env->size);
            s->y = sm_wrap_pos(a->y + sep * sinf(ang), env->size);
            if (sm_torus_dist(env, s->x, s->y, sm_planet_x(env), sm_planet_y(env)) >= clearance) {
                break;
            }
        }
        float hdx = sm_wrap_delta(a->x - s->x, env->size);
        float hdy = sm_wrap_delta(a->y - s->y, env->size);
        s->heading = atan2f(hdy, hdx) + 0.6f * (sm_randf(env) - 0.5f);
        if (s->heading < 0.0f) s->heading += 2.0f * PI;
    } else {
        sm_sample_clear_point(env, clearance, &s->x, &s->y);
        s->heading = 2.0f * PI * sm_randf(env);
    }
    s->vx = 0.0f;
    s->vy = 0.0f;
    s->omega = 0.0f;
    s->hp = env->hp_max;
    s->episode_tick = 0;
    s->episode_return = 0.0f;
    s->planet_hits = 0;
    s->input_changes = 0;
    s->asteroid_hits = 0;
    s->thrusting = 0;
    s->turning_left = 0;
    s->turning_right = 0;
    s->firing = 0;
    s->attack_state = SM_STATE_APPROACH;
    // Magazine size is a per-spawn trait like thrust or fragility: sampled
    // in [cannon_rounds, cannon_rounds_max], readable through the ammo obs
    int mag_span = env->cannon_rounds_max - env->cannon_rounds + 1;
    s->mag_rounds = env->cannon_rounds + (int)(sm_randf(env) * mag_span);
    if (s->mag_rounds > env->cannon_rounds_max) s->mag_rounds = env->cannon_rounds_max;
    s->ammo = s->mag_rounds;
    s->cannon_timer = 0;
    s->reloading = 0;
    s->cycle_paid = 0;
    s->hits_this_mag = 0;
    s->attack_passes = 0;
    s->shots_fired = 0;
    s->asteroid_kills = 0;
    s->full_cycles = 0;
    s->prev_enemy_dist = -1.0f;
    s->fire_flash = 0;
    s->hit_flash = 0;
    s->retreats = 0;
    s->t_thrust = sm_trait(env);
    s->t_turn = sm_trait(env);
    s->t_speed = sm_trait(env);
    s->t_fragility = sm_trait(env);
    s->t_caution = sm_trait(env);

    if (env->combat) {
        // No beacon in a duel; the other ship is the target
        s->goal_x = s->x;
        s->goal_y = s->y;
        s->prev_goal_dist = 0.0f;
        return;
    }

    // Beacon: clear of the planet and not trivially close to the spawn
    float min_goal_dist = env->min_goal_frac * env->size;
    for (int attempt = 0; attempt < 100; attempt++) {
        sm_sample_clear_point(env, clearance, &s->goal_x, &s->goal_y);
        if (sm_torus_dist(env, s->x, s->y, s->goal_x, s->goal_y) >= min_goal_dist) {
            break;
        }
    }
    s->prev_goal_dist = sm_torus_dist(env, s->x, s->y, s->goal_x, s->goal_y);
}

// New rock: placed away from every ship and the planet, flying a straight
// random line at a speed within [asteroid_speed_min, asteroid_speed_max]
static void sm_spawn_asteroid(StarMelee* env, int k) {
    Asteroid* a = &env->asteroids[k];
    float ship_clearance = 0.2f * env->size;
    for (int attempt = 0; attempt < 100; attempt++) {
        a->x = env->size * sm_randf(env);
        a->y = env->size * sm_randf(env);
        if (sm_torus_dist(env, a->x, a->y, sm_planet_x(env), sm_planet_y(env))
                < env->planet_radius + env->asteroid_radius + 20.0f) {
            continue;
        }
        int clear = 1;
        for (int i = 0; i < env->num_ships; i++) {
            if (sm_torus_dist(env, a->x, a->y, env->ships[i].x, env->ships[i].y)
                    < ship_clearance) {
                clear = 0;
                break;
            }
        }
        if (clear) break;
    }
    float ang = 2.0f * PI * sm_randf(env);
    float speed = env->asteroid_speed_min
        + (env->asteroid_speed_max - env->asteroid_speed_min) * sm_randf(env);
    a->vx = speed * cosf(ang);
    a->vy = speed * sinf(ang);
    a->radius = env->asteroid_radius * (0.75f + 0.5f * sm_randf(env));
    a->active = 1;
    a->respawn_timer = 0;

    // Rock outline (kept in the deterministic rng stream so training and
    // rendering runs stay identical)
    a->num_vertices = 8;
    for (int v = 0; v < a->num_vertices; v++) {
        float va = 2.0f * PI * v / a->num_vertices;
        float vr = a->radius * (0.7f + 0.6f * sm_randf(env));
        a->shape[v] = (Vector2){cosf(va) * vr, sinf(va) * vr};
    }
}

static void sm_destroy_asteroid(StarMelee* env, int k) {
    Asteroid* a = &env->asteroids[k];
    a->active = 0;
    a->respawn_timer = env->asteroid_respawn_ticks;
    a->boom_x = a->x;
    a->boom_y = a->y;
    a->boom_radius = a->radius;
    a->boom_timer = 12;
}

// One straight-line tick per rock, plus disintegration on planet or ship
// contact. Ships hit take hp damage, a reward sting, and a momentum kick.
static void sm_asteroids_tick(StarMelee* env) {
    for (int k = 0; k < env->num_asteroids; k++) {
        Asteroid* a = &env->asteroids[k];
        if (!a->active) {
            a->respawn_timer -= 1;
            if (a->respawn_timer <= 0) {
                sm_spawn_asteroid(env, k);
            }
            continue;
        }

        a->x = sm_wrap_pos(a->x + a->vx, env->size);
        a->y = sm_wrap_pos(a->y + a->vy, env->size);

        if (sm_torus_dist(env, a->x, a->y, sm_planet_x(env), sm_planet_y(env))
                < env->planet_radius + a->radius) {
            sm_destroy_asteroid(env, k);
            continue;
        }

        for (int i = 0; i < env->num_ships; i++) {
            Ship* s = &env->ships[i];
            if (s->just_finished) continue;
            if (sm_torus_dist(env, a->x, a->y, s->x, s->y)
                    >= a->radius + env->ship_radius) {
                continue;
            }
            float damage = env->asteroid_damage * s->t_fragility;
            s->hp -= damage;
            s->asteroid_hits += 1;
            s->vx += 0.3f * a->vx;
            s->vy += 0.3f * a->vy;
            float pain = -fminf(damage, 0.6f * env->hp_max) / env->hp_max
                * sm_pain_scale(env, s);
            env->rewards[i] += pain;
            s->episode_return += pain;
            sm_destroy_asteroid(env, k);
            break;
        }
    }
}

// Chamber a round: the shot inherits the shooter's velocity plus muzzle
// speed along the heading, and lives long enough to cover projectile_range
// at muzzle speed. Pool never overflows: c_init caps cannon_rounds so all
// rounds of every ship fit.
static void sm_fire_projectile(StarMelee* env, int i) {
    Ship* s = &env->ships[i];
    for (int k = 0; k < STARMELEE_MAX_PROJECTILES; k++) {
        Projectile* p = &env->projectiles[k];
        if (p->life > 0) continue;
        float dirx = cosf(s->heading);
        float diry = sinf(s->heading);
        p->x = sm_wrap_pos(s->x + dirx * (env->ship_radius + 2.0f), env->size);
        p->y = sm_wrap_pos(s->y + diry * (env->ship_radius + 2.0f), env->size);
        p->vx = s->vx + env->projectile_speed * dirx;
        p->vy = s->vy + env->projectile_speed * diry;
        p->owner = i;
        // Life from the actual launch speed, not the nominal muzzle speed:
        // inherited velocity must not stretch (or starve) the world-frame
        // reach the ini advertises. At least one tick so no round is a dud.
        float launch = sqrtf(p->vx*p->vx + p->vy*p->vy);
        p->life = (int)(env->projectile_range / fmaxf(launch, 0.1f) + 0.5f);
        if (p->life < 1) p->life = 1;
        s->shots_fired += 1;
        s->fire_flash = 10;
        return;
    }
}

// True when the minimal-image segment from (ax, ay) along (dx, dy) passes
// within r of the point (cx, cy). Used as a swept collision test: closing
// speeds can exceed a ship radius per tick, so point sampling tunnels.
static inline int sm_sweep_hits(StarMelee* env, float ax, float ay,
        float dx, float dy, float cx, float cy, float r) {
    float px = sm_wrap_delta(cx - ax, env->size);
    float py = sm_wrap_delta(cy - ay, env->size);
    float len2 = dx*dx + dy*dy;
    float t = len2 > 1e-6f ? (px*dx + py*dy) / len2 : 0.0f;
    t = sm_clampf(t, 0.0f, 1.0f);
    float ex = t*dx - px;
    float ey = t*dy - py;
    return ex*ex + ey*ey < r*r;
}

// One straight-line tick per shot: dissolve at end of life, on the planet
// (the line-of-sight shield is physical), on a rock (which disintegrates),
// or on a ship (damage + sting to the victim, hit_reward to the shooter).
static void sm_projectiles_tick(StarMelee* env) {
    for (int k = 0; k < STARMELEE_MAX_PROJECTILES; k++) {
        Projectile* p = &env->projectiles[k];
        if (p->life <= 0) continue;
        p->life -= 1;
        float from_x = p->x;
        float from_y = p->y;
        p->x = sm_wrap_pos(p->x + p->vx, env->size);
        p->y = sm_wrap_pos(p->y + p->vy, env->size);

        if (sm_torus_dist(env, p->x, p->y, sm_planet_x(env), sm_planet_y(env))
                < env->planet_radius) {
            p->life = 0;
            continue;
        }

        int consumed = 0;
        for (int r = 0; r < env->num_asteroids; r++) {
            Asteroid* a = &env->asteroids[r];
            if (!a->active) continue;
            if (sm_sweep_hits(env, from_x, from_y, p->vx, p->vy, a->x, a->y, a->radius)) {
                sm_destroy_asteroid(env, r);
                env->ships[p->owner].asteroid_kills += 1;
                p->life = 0;
                consumed = 1;
                break;
            }
        }
        if (consumed) continue;

        for (int i = 0; i < env->num_ships; i++) {
            if (i == p->owner) continue;
            Ship* s = &env->ships[i];
            if (s->just_finished) continue;
            if (!sm_sweep_hits(env, from_x, from_y, p->vx, p->vy, s->x, s->y,
                    env->ship_radius)) {
                continue;
            }
            s->hp -= env->shot_damage * s->t_fragility;
            s->hit_flash = 12;
            float sting = env->targeted_penalty * s->t_caution;
            env->rewards[i] -= sting;
            s->episode_return -= sting;
            Ship* owner = &env->ships[p->owner];
            env->rewards[p->owner] += env->hit_reward;
            owner->episode_return += env->hit_reward;
            owner->attack_passes += 1;
            owner->hits_this_mag += 1;
            p->life = 0;
            break;
        }
    }
}

void compute_observations(StarMelee* env) {
    float half = 0.5f * env->size;
    float half_diag = 0.70711f * env->size;
    for (int i = 0; i < env->num_ships; i++) {
        Ship* s = &env->ships[i];
        float* obs = &env->observations[i * STARMELEE_OBS_SIZE];

        float gdx = sm_wrap_delta(s->goal_x - s->x, env->size);
        float gdy = sm_wrap_delta(s->goal_y - s->y, env->size);
        float gdist = sqrtf(gdx*gdx + gdy*gdy);
        float pdx = sm_wrap_delta(sm_planet_x(env) - s->x, env->size);
        float pdy = sm_wrap_delta(sm_planet_y(env) - s->y, env->size);
        float pdist = sqrtf(pdx*pdx + pdy*pdy);

        obs[0] = gdx / half;
        obs[1] = gdy / half;
        obs[2] = gdist / half_diag;
        obs[3] = pdx / half;
        obs[4] = pdy / half;
        obs[5] = pdist / half_diag;
        // Normalize by this build's own limits so ranges stay nominal
        obs[6] = s->vx / (env->max_speed * s->t_speed);
        obs[7] = s->vy / (env->max_speed * s->t_speed);
        obs[8] = cosf(s->heading);
        obs[9] = sinf(s->heading);
        obs[10] = s->omega / (env->max_turn_rate * s->t_turn);
        obs[11] = s->hp / env->hp_max;
        obs[12] = (float)s->episode_tick / (float)env->max_ticks;

        // Nearest other ship, if any
        int nearest = -1;
        float nearest_dist = 0.0f;
        for (int j = 0; j < env->num_ships; j++) {
            if (j == i) continue;
            float d = sm_torus_dist(env, s->x, s->y, env->ships[j].x, env->ships[j].y);
            if (nearest < 0 || d < nearest_dist) {
                nearest = j;
                nearest_dist = d;
            }
        }
        if (nearest >= 0) {
            Ship* o = &env->ships[nearest];
            float odx = sm_wrap_delta(o->x - s->x, env->size);
            float ody = sm_wrap_delta(o->y - s->y, env->size);
            float odist = sqrtf(odx*odx + ody*ody);
            obs[13] = odx / half;
            obs[14] = ody / half;
            // Both ships can move at up to 2*max_speed (hard cap)
            obs[15] = (o->vx - s->vx) / (2.0f * env->max_speed);
            obs[16] = (o->vy - s->vy) / (2.0f * env->max_speed);
            obs[17] = 1.0f;
            if (env->combat) {
                float bearing = atan2f(ody, odx);
                float aim_delta = s->heading - bearing;
                int los = sm_los_clear(env, s->x, s->y, o->x, o->y);
                obs[18] = cosf(aim_delta);
                obs[19] = sinf(aim_delta);
                obs[20] = los ? 1.0f : 0.0f;
                obs[21] = s->attack_state == SM_STATE_FLEE ? 1.0f : 0.0f;
                // Normalized by the global max so absolute counts stay
                // readable: a full 3-round magazine is 0.6, a full 5 is 1.0
                obs[22] = (float)s->ammo / (float)env->cannon_rounds_max;
                obs[23] = (float)o->ammo / (float)env->cannon_rounds_max;
                obs[24] = cosf(o->heading);
                obs[25] = sinf(o->heading);
                // Cannon clock at reload scale: covers both the reload and
                // the (much shorter) salvo cooldown, so the policy can see
                // when its own trigger is live
                obs[26] = (float)s->cannon_timer / (float)env->cannon_reload_ticks;
                obs[32] = o->hp / env->hp_max;  // smell blood: press a wounded enemy
                obs[34] = o->reloading ? 1.0f : 0.0f;  // their window of weakness
                // Incoming fire imminent: the enemy has a chambered round,
                // clear sight, weapon range, and its nose on us (free fire
                // means any state can shoot)
                float back = atan2f(-ody, -odx);
                float e_err = fabsf(atan2f(sinf(o->heading - back), cosf(o->heading - back)));
                obs[35] = (o->ammo > 0 && los && odist <= env->projectile_range
                    && e_err <= env->aim_cone) ? 1.0f : 0.0f;
                // Enemy range at combat scale: the arena-scale deltas in
                // obs[13..14] compress the whole engagement into < 0.09
                obs[36] = sm_clampf(odist / env->disengage_range, 0.0f, 2.0f);
            } else {
                for (int k = 18; k < 27; k++) obs[k] = 0.0f;
                obs[32] = 0.0f;
                obs[34] = 0.0f;
                obs[35] = 0.0f;
                obs[36] = 0.0f;
            }
        } else {
            for (int k = 13; k < 27; k++) obs[k] = 0.0f;
            obs[32] = 0.0f;
            obs[34] = 0.0f;
            obs[35] = 0.0f;
            obs[36] = 0.0f;
        }

        // Own build: the policy needs to know what ship it is flying
        obs[27] = s->t_thrust - 1.0f;
        obs[28] = s->t_turn - 1.0f;
        obs[29] = s->t_speed - 1.0f;
        obs[30] = s->t_fragility - 1.0f;
        obs[31] = s->t_caution - 1.0f;
        obs[33] = s->attack_state == SM_STATE_RETREAT ? 1.0f : 0.0f;

        // Nearest hostile shot in flight, normalized to weapon scale (not
        // arena scale: a dodge decision lives entirely inside ~one range)
        int best = -1;
        float best_dist = 0.0f;
        for (int k = 0; k < STARMELEE_MAX_PROJECTILES; k++) {
            Projectile* p = &env->projectiles[k];
            if (p->life <= 0 || p->owner == i) continue;
            float d = sm_torus_dist(env, p->x, p->y, s->x, s->y);
            if (best < 0 || d < best_dist) {
                best = k;
                best_dist = d;
            }
        }
        if (best >= 0 && best_dist <= 2.0f * env->projectile_range) {
            Projectile* p = &env->projectiles[best];
            float pv_norm = env->projectile_speed + 2.0f * env->max_speed;
            obs[37] = sm_wrap_delta(p->x - s->x, env->size) / env->projectile_range;
            obs[38] = sm_wrap_delta(p->y - s->y, env->size) / env->projectile_range;
            obs[39] = (p->vx - s->vx) / pv_norm;
            obs[40] = (p->vy - s->vy) / pv_norm;
            obs[41] = 1.0f;
        } else {
            for (int k = 37; k <= 41; k++) obs[k] = 0.0f;
        }

        // Rocks in fixed slots keyed by asteroid index: a nearest-rock obs
        // flickers identity when two rocks trade places, which shreds the
        // apparent-velocity signal the policy needs to dodge
        for (int slot = 0; slot < STARMELEE_OBS_ASTEROIDS; slot++) {
            int base = 42 + 5 * slot;
            Asteroid* a = slot < env->num_asteroids ? &env->asteroids[slot] : NULL;
            if (a != NULL && a->active) {
                obs[base + 0] = sm_wrap_delta(a->x - s->x, env->size) / half;
                obs[base + 1] = sm_wrap_delta(a->y - s->y, env->size) / half;
                obs[base + 2] = (a->vx - s->vx) / (2.0f * env->max_speed);
                obs[base + 3] = (a->vy - s->vy) / (2.0f * env->max_speed);
                obs[base + 4] = 1.0f;
            } else {
                obs[base + 0] = 0.0f;
                obs[base + 1] = 0.0f;
                obs[base + 2] = 0.0f;
                obs[base + 3] = 0.0f;
                obs[base + 4] = 0.0f;
            }
        }
    }
}

static void sm_add_log(StarMelee* env, Ship* s, unsigned char result) {
    if (env->combat) {
        // A duel episode is scored by completed passes and clean disengages
        env->log.perf += (float)s->attack_passes;
        env->log.score += (float)s->attack_passes + 0.5f * (float)s->full_cycles
            - (result == SM_RESULT_CRASH ? 1.0f : 0.0f);
        env->log.success_rate += s->full_cycles > 0 ? 1.0f : 0.0f;
    } else {
        env->log.perf += result == SM_RESULT_SUCCESS ? 1.0f : 0.0f;
        env->log.score += result == SM_RESULT_SUCCESS ? 1.0f
            : (result == SM_RESULT_CRASH ? -1.0f : 0.0f);
        env->log.success_rate += result == SM_RESULT_SUCCESS ? 1.0f : 0.0f;
    }
    env->log.episode_return += s->episode_return;
    env->log.episode_length += s->episode_tick;
    env->log.crash_rate += result == SM_RESULT_CRASH ? 1.0f : 0.0f;
    env->log.timeout_rate += result == SM_RESULT_TIMEOUT ? 1.0f : 0.0f;
    env->log.final_distance += s->prev_goal_dist / env->size;
    env->log.planet_hits += s->planet_hits;
    env->log.attack_passes += (float)s->attack_passes;
    env->log.shots_fired += (float)s->shots_fired;
    env->log.full_cycles += (float)s->full_cycles;
    env->log.retreats += (float)s->retreats;
    env->log.asteroid_hits += (float)s->asteroid_hits;
    env->log.asteroid_kills += (float)s->asteroid_kills;
    env->log.input_changes += s->episode_tick > 0
        ? (float)s->input_changes / (float)s->episode_tick : 0.0f;
    env->log.n += 1.0f;
}

// Ends ship i's episode: terminal reward, log, respawn in place.
static void sm_finish_ship(StarMelee* env, int i, float reward, unsigned char result) {
    Ship* s = &env->ships[i];
    env->rewards[i] += reward;
    s->episode_return += reward;
    env->terminals[i] = 1.0f;
    s->last_result = result;
    sm_add_log(env, s, result);
    sm_spawn_ship(env, i);
    env->ships[i].just_finished = 1;
    // All in-flight rounds die with the episode: the finisher's own shots
    // must not pay into its next episode, and shots chasing the finisher
    // must not strafe the fresh spawn it teleported into.
    for (int k = 0; k < STARMELEE_MAX_PROJECTILES; k++) {
        env->projectiles[k].life = 0;
    }
    // The respawn teleports this ship: opponents must re-baseline their
    // distance shaping, and a pending flee-cycle is forfeit — the new
    // spawn distance (duel_spawn_frac * size) can exceed disengage_range,
    // which would complete the cycle for free.
    for (int j = 0; j < env->num_ships; j++) {
        if (j == i) continue;
        env->ships[j].prev_enemy_dist = -1.0f;
        if (env->ships[j].attack_state == SM_STATE_FLEE) {
            env->ships[j].cycle_paid = 1;
        }
    }
    if (IsWindowReady()) {
        env->ships[i].event_timer = 70;
        env->ships[i].last_result = result;
    }
}

void c_reset(StarMelee* env) {
    for (int i = 0; i < env->num_ships; i++) {
        sm_spawn_ship(env, i);
        env->ships[i].last_result = SM_RESULT_NONE;
        env->ships[i].event_timer = 0;
    }
    for (int k = 0; k < env->num_asteroids; k++) {
        sm_spawn_asteroid(env, k);
    }
    for (int k = 0; k < STARMELEE_MAX_PROJECTILES; k++) {
        env->projectiles[k].life = 0;
    }
    compute_observations(env);
}

// One combat tick for ship i: the engage -> burst -> reload-flee cycle.
// The cannon fires by itself whenever the ship is on duty (APPROACH), has a
// chambered round off cooldown, and is tracking the enemy — bearing inside
// aim_cone with clear line of sight — within projectile_range. Emptying the
// magazine starts the reload and forces FLEE: open distance (cycle_reward at
// disengage_range) until the magazine refills. Low hp forces RETREAT on top.
static void sm_combat_tick(StarMelee* env, int i) {
    Ship* s = &env->ships[i];
    int j = sm_nearest_enemy(env, i);
    if (j < 0) return;
    Ship* e = &env->ships[j];

    float dx = sm_wrap_delta(e->x - s->x, env->size);
    float dy = sm_wrap_delta(e->y - s->y, env->size);
    float dist = sqrtf(dx*dx + dy*dy);
    float r = env->step_penalty;  // duels feel time pressure too
    float next_prev = dist;       // next tick's shaping baseline

    // Cannon clock runs in every state; a finished reload releases FLEE
    if (s->cannon_timer > 0) {
        s->cannon_timer -= 1;
        if (s->cannon_timer == 0 && s->reloading) {
            s->ammo = s->mag_rounds;
            s->reloading = 0;
            s->hits_this_mag = 0;
            if (s->attack_state == SM_STATE_FLEE) {
                s->attack_state = SM_STATE_APPROACH;
                // Approach shaping only pays for the way back to the band;
                // drifting far while reloading must not refill the well
                next_prev = fminf(dist, env->disengage_range);
            }
        }
    }

    // Low hp overrides everything: break off and survive. Hysteresis (exit
    // well above the entry threshold) stops flapping at the boundary.
    float retreat_enter = env->retreat_hp_frac * env->hp_max;
    if (retreat_enter > 0.0f && s->attack_state != SM_STATE_RETREAT
            && s->hp <= retreat_enter) {
        s->attack_state = SM_STATE_RETREAT;
        s->retreats += 1;
        // Escape shaping must start from the real current distance: the
        // stored baseline may still carry the release-time clamp, and the
        // gap would mint phantom escape reward on this very tick
        s->prev_enemy_dist = dist;
    }

    // Tracking: nose on the enemy with nothing in the way
    float bearing = atan2f(dy, dx);
    float aim_err = fabsf(atan2f(sinf(s->heading - bearing), cosf(s->heading - bearing)));
    int tracking = aim_err <= env->aim_cone
        && sm_los_clear(env, s->x, s->y, e->x, e->y);

    // Free fire: the trigger is the 4th action button, live in any state
    // with a chambered round off cooldown. A dumped magazine still costs
    // the reload (forcing the flee leg when on duty) and pays no cycle
    // unless it drew blood — that economy is the trigger discipline.
    if (s->firing && s->ammo > 0 && s->cannon_timer == 0) {
        sm_fire_projectile(env, i);
        s->ammo -= 1;
        // Nudge toward disciplined shots: firing with the nose on target
        // inside weapon range pays a little, spraying costs the same
        r += (tracking && dist <= env->projectile_range)
            ? env->fire_nudge : -env->fire_nudge;
        if (s->ammo > 0) {
            s->cannon_timer = env->cannon_interval_ticks;
        } else {
            s->reloading = 1;
            s->cannon_timer = env->cannon_reload_ticks;
            s->cycle_paid = 0;
            if (s->attack_state == SM_STATE_APPROACH) {
                s->attack_state = SM_STATE_FLEE;
            }
        }
    }

    // Spacing discipline, active in every state. Inside close_range the
    // keep-out penalty grows with intrusion depth; out to avoid_range a
    // relative velocity whose projected pass would enter the keep-out
    // bubble is a collision course and pays by closing speed — oblique
    // passes and orbits are free, rams are not.
    if (dist < env->close_range) {
        r -= env->too_close_penalty * (env->close_range - dist) / env->close_range;
    } else if (dist < env->avoid_range && env->collision_course_penalty > 0.0f) {
        float rvx = s->vx - e->vx;
        float rvy = s->vy - e->vy;
        float along = rvx * dx + rvy * dy;  // > 0 when closing
        if (along > 0.0f) {
            float v2 = rvx * rvx + rvy * rvy;
            float miss_sq = dist * dist - along * along / fmaxf(v2, 1e-6f);
            if (miss_sq < env->close_range * env->close_range) {
                float closing = along / fmaxf(dist, 1e-5f);
                r -= env->collision_course_penalty
                    * sm_clampf(closing / (2.0f * env->max_speed), 0.0f, 1.0f);
            }
        }
    }

    if (s->attack_state == SM_STATE_APPROACH) {
        // Shaping toward the engagement band [attack_range_min, attack_range_max]
        float band_now = fmaxf(0.0f, dist - env->attack_range_max)
            + fmaxf(0.0f, env->attack_range_min - dist);
        if (s->prev_enemy_dist >= 0.0f) {
            float band_prev = fmaxf(0.0f, s->prev_enemy_dist - env->attack_range_max)
                + fmaxf(0.0f, env->attack_range_min - s->prev_enemy_dist);
            r += env->engage_scale * (band_prev - band_now) / env->size;
        }

        int in_band = dist >= env->attack_range_min && dist <= env->attack_range_max;
        if (in_band && tracking) {
            r += env->aim_reward;
            // Transverse velocity while tracking: a drifting, strafing
            // shooter is a harder target than one hanging on the firing line
            float inv = dist > 1e-5f ? 1.0f / dist : 0.0f;
            float v_perp = fabsf(s->vx * (-dy * inv) + s->vy * (dx * inv));
            r += env->strafe_reward * sm_clampf(v_perp / env->max_speed, 0.0f, 1.0f);
        }

        // Fire only when tracking the opponent inside weapon range
        if (tracking && dist <= env->projectile_range
                && s->ammo > 0 && s->cannon_timer == 0) {
            sm_fire_projectile(env, i);
            s->ammo -= 1;
            if (s->ammo > 0) {
                s->cannon_timer = env->cannon_interval_ticks;
            } else {
                // Magazine dry: reload forces the flee leg of the cycle
                s->reloading = 1;
                s->cannon_timer = env->cannon_reload_ticks;
                s->cycle_paid = 0;
                s->attack_state = SM_STATE_FLEE;
            }
        }
    } else if (s->attack_state == SM_STATE_FLEE) {
        // FLEE: reloading and a target, open distance fast. The prev-dist
        // gate keeps an opponent-respawn teleport from completing a cycle.
        // Once the cycle resolves, loiter takes over: distance shaping must
        // not keep paying all the way to the far side of the torus.
        if (!s->cycle_paid) {
            if (s->prev_enemy_dist >= 0.0f) {
                r += env->engage_scale * (dist - s->prev_enemy_dist) / env->size;
                if (dist >= env->disengage_range) {
                    s->cycle_paid = 1;
                    // A cycle only counts if the magazine that forced it
                    // drew blood: dumping guaranteed misses and running is
                    // otherwise the shared policy's favorite payday
                    if (s->hits_this_mag > 0) {
                        r += env->cycle_reward;
                        s->full_cycles += 1;
                    }
                }
            }
        } else if (dist >= env->disengage_range) {
            r += env->loiter_reward;
        }
    } else {
        // RETREAT: wounded, escape shaping only until hp recovers
        if (s->prev_enemy_dist >= 0.0f) {
            r += env->engage_scale * (dist - s->prev_enemy_dist) / env->size;
        }
        if (dist >= env->disengage_range) {
            r += env->loiter_reward;
        }
        float retreat_exit = fminf(env->retreat_hp_frac + 0.15f, 1.0f) * env->hp_max;
        if (s->hp >= retreat_exit) {
            // Resume an interrupted reload-flee instead of skipping it
            s->attack_state = s->reloading ? SM_STATE_FLEE : SM_STATE_APPROACH;
            if (s->attack_state == SM_STATE_APPROACH) {
                next_prev = fminf(dist, env->disengage_range);
            }
        }
    }

    // Shields recharge only at a safe distance: backing off buys hp back
    if (env->hp_regen > 0.0f && dist >= env->disengage_range && s->hp < env->hp_max) {
        s->hp = fminf(env->hp_max, s->hp + env->hp_regen);
    }

    s->prev_enemy_dist = next_prev;
    env->rewards[i] += r;
    s->episode_return += r;
}

// Reads this decision's button states and charges the jitter penalty.
// Buttons stay latched on the ship for the whole action_repeat window.
static void sm_read_inputs(StarMelee* env) {
    for (int i = 0; i < env->num_ships; i++) {
        Ship* s = &env->ships[i];
        int left = env->actions[i*4 + 0] > 0.5f;
        int right = env->actions[i*4 + 1] > 0.5f;
        int engine = env->actions[i*4 + 2] > 0.5f;
        int fire = env->actions[i*4 + 3] > 0.5f;

        // Jitter penalty: charge every button toggle so dithering (rapid
        // on/off switching a human would never produce) costs reward while
        // sustained holds stay free.
        int changes = (left != s->turning_left) + (right != s->turning_right)
            + (engine != s->thrusting) + (fire != s->firing);
        if (changes > 0 && env->input_change_penalty > 0.0f) {
            float jitter = env->input_change_penalty * (float)changes;
            env->rewards[i] -= jitter;
            s->episode_return -= jitter;
        }
        s->input_changes += changes;
        s->turning_left = (unsigned char)left;
        s->turning_right = (unsigned char)right;
        s->thrusting = (unsigned char)engine;
        s->firing = (unsigned char)fire;
    }
}

// One 60 Hz physics tick using the latched button states
static void sm_physics_tick(StarMelee* env) {
    int n = env->num_ships;
    for (int i = 0; i < n; i++) {
        Ship* s = &env->ships[i];
        s->episode_tick += 1;

        int left = s->turning_left;
        int right = s->turning_right;
        int engine = s->thrusting;

        // Torque / inertia -> angular acceleration while held
        float turn_accel = env->turn_accel * s->t_turn;
        float max_turn = env->max_turn_rate * s->t_turn;
        if (left) s->omega -= turn_accel;
        if (right) s->omega += turn_accel;
        s->omega *= env->angular_damping;
        s->omega = sm_clampf(s->omega, -max_turn, max_turn);
        s->heading = fmodf(s->heading + s->omega, 2.0f * PI);
        if (s->heading < 0.0f) s->heading += 2.0f * PI;

        // Gravity always applies and may exceed max_speed (slingshots)
        float gx, gy;
        sm_gravity_at(env, s->x, s->y, &gx, &gy);
        s->vx += gx;
        s->vy += gy;

        // Thrust cannot push speed above max_speed, but does not bleed
        // off any excess the slingshot gave us
        if (engine) {
            float prev_speed = sqrtf(s->vx*s->vx + s->vy*s->vy);
            float cap = fmaxf(env->max_speed * s->t_speed, prev_speed);
            float thrust = env->thrust * s->t_thrust;
            s->vx += thrust * cosf(s->heading);
            s->vy += thrust * sinf(s->heading);
            float speed = sqrtf(s->vx*s->vx + s->vy*s->vy);
            if (speed > cap) {
                s->vx *= cap / speed;
                s->vy *= cap / speed;
            }
        }

        // Safety cap + configurable bleed keeps slingshots bounded
        float speed = sqrtf(s->vx*s->vx + s->vy*s->vy);
        float hard_cap = 2.0f * env->max_speed * s->t_speed;
        if (speed > hard_cap) {
            s->vx *= hard_cap / speed;
            s->vy *= hard_cap / speed;
        }
        s->vx *= env->linear_damping;
        s->vy *= env->linear_damping;

        s->x = sm_wrap_pos(s->x + s->vx, env->size);
        s->y = sm_wrap_pos(s->y + s->vy, env->size);
    }

    // Planet collision: push out, bounce, damage by normal impact speed
    for (int i = 0; i < n; i++) {
        Ship* s = &env->ships[i];
        float dx = sm_wrap_delta(s->x - sm_planet_x(env), env->size);
        float dy = sm_wrap_delta(s->y - sm_planet_y(env), env->size);
        float dist = sqrtf(dx*dx + dy*dy);
        float min_dist = env->planet_radius + env->ship_radius;
        if (dist >= min_dist) continue;

        float nx, ny;  // surface normal, away from the planet
        if (dist > 1e-5f) {
            nx = dx / dist;
            ny = dy / dist;
        } else {
            nx = 1.0f;
            ny = 0.0f;
        }
        s->x = sm_wrap_pos(sm_planet_x(env) + nx * min_dist, env->size);
        s->y = sm_wrap_pos(sm_planet_y(env) + ny * min_dist, env->size);

        float vn = s->vx * nx + s->vy * ny;
        if (vn < 0.0f) {
            s->vx -= (1.0f + env->restitution) * vn * nx;
            s->vy -= (1.0f + env->restitution) * vn * ny;
            float damage = env->damage_scale * -vn * s->t_fragility;
            s->hp -= damage;
            s->planet_hits += 1;
            // Pain cap: a terminal step already adds -1, and the trainer
            // clamps steps to [-1, 1]; uncapped pain just gets truncated
            float pain = -fminf(damage, 0.6f * env->hp_max) / env->hp_max
                * sm_pain_scale(env, s);
            env->rewards[i] += pain;
            s->episode_return += pain;
        }
    }

    sm_asteroids_tick(env);
    sm_projectiles_tick(env);

    // Ship vs ship: equal masses, exchange normal velocity components
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            Ship* a = &env->ships[i];
            Ship* b = &env->ships[j];
            float dx = sm_wrap_delta(b->x - a->x, env->size);
            float dy = sm_wrap_delta(b->y - a->y, env->size);
            float dist = sqrtf(dx*dx + dy*dy);
            float min_dist = 2.0f * env->ship_radius;
            if (dist >= min_dist) continue;

            float nx, ny;  // from a toward b
            if (dist > 1e-5f) {
                nx = dx / dist;
                ny = dy / dist;
            } else {
                nx = 1.0f;
                ny = 0.0f;
            }
            float push = 0.5f * (min_dist - dist);
            a->x = sm_wrap_pos(a->x - nx * push, env->size);
            a->y = sm_wrap_pos(a->y - ny * push, env->size);
            b->x = sm_wrap_pos(b->x + nx * push, env->size);
            b->y = sm_wrap_pos(b->y + ny * push, env->size);

            float rel_vn = (a->vx - b->vx) * nx + (a->vy - b->vy) * ny;
            if (rel_vn > 0.0f) {  // approaching
                float impulse = 0.5f * (1.0f + env->restitution) * rel_vn;
                a->vx -= impulse * nx;
                a->vy -= impulse * ny;
                b->vx += impulse * nx;
                b->vy += impulse * ny;
            }
        }
    }

    // Task rewards and episode endings
    for (int i = 0; i < n; i++) {
        Ship* s = &env->ships[i];

        // A ship that finished its episode earlier in this action_repeat
        // window flies on, but the new episode must not leak rewards or a
        // second terminal into the already-terminated training step
        if (s->just_finished) continue;

        if (s->hp <= 0.0f) {
            sm_finish_ship(env, i, -1.0f, SM_RESULT_CRASH);
            continue;
        }

        if (env->combat) {
            sm_combat_tick(env, i);
            if (s->episode_tick >= env->max_ticks) {
                // Surviving a duel to the buzzer is not a failure
                sm_finish_ship(env, i, 0.0f, SM_RESULT_TIMEOUT);
            }
            continue;
        }

        float gdist = sm_torus_dist(env, s->x, s->y, s->goal_x, s->goal_y);
        float shaped = env->progress_scale * (s->prev_goal_dist - gdist) / env->size
            + env->step_penalty;
        s->prev_goal_dist = gdist;
        env->rewards[i] += shaped;
        s->episode_return += shaped;

        if (gdist <= env->goal_radius) {
            sm_finish_ship(env, i, 1.0f, SM_RESULT_SUCCESS);
        } else if (s->episode_tick >= env->max_ticks) {
            sm_finish_ship(env, i, -0.25f, SM_RESULT_TIMEOUT);
        }
    }
}

void c_step(StarMelee* env) {
    for (int i = 0; i < env->num_ships; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0.0f;
    }
    for (int i = 0; i < env->num_ships; i++) {
        env->ships[i].just_finished = 0;
    }
    sm_read_inputs(env);
    for (int rep = 0; rep < env->action_repeat; rep++) {
        sm_physics_tick(env);
    }
    compute_observations(env);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const Color SM_BACKGROUND = (Color){6, 24, 24, 255};
const Color SM_WHITE = (Color){241, 241, 241, 255};
const Color SM_CYAN = (Color){0, 187, 187, 255};
const Color SM_RED = (Color){187, 0, 0, 255};
const Color SM_GREEN = (Color){60, 255, 140, 255};
const Color SM_YELLOW = (Color){255, 220, 120, 255};
const Color SM_PLANET = (Color){205, 125, 60, 255};
const Color SM_PLANET_DARK = (Color){140, 80, 40, 255};

static const Color SM_SHIP_COLORS[STARMELEE_MAX_SHIPS] = {
    {0, 205, 215, 255}, {255, 160, 60, 255}, {170, 120, 255, 255}, {255, 110, 160, 255},
    {120, 220, 100, 255}, {240, 240, 90, 255}, {90, 140, 255, 255}, {220, 220, 220, 255},
};

// Render camera: full-arena view, or following ship 0 zoomed in.
// Toggle with C, zoom with Z/X. Render-only state.
typedef struct {
    int follow;
    float zoom;
} SMCamera;
static SMCamera sm_camera = {0, 3.0f};

static float sm_cam_scale(StarMelee* env) {
    float base = fminf((float)GetScreenWidth(), (float)GetScreenHeight()) / env->size;
    return sm_camera.follow ? base * sm_camera.zoom : base;
}

// World position -> screen position. In follow mode the view is centered on
// ship 0 and coordinates come from minimal-image deltas, so the seam is never
// visible near the ship.
static Vector2 sm_world_to_screen(StarMelee* env, float x, float y) {
    float scale = sm_cam_scale(env);
    if (!sm_camera.follow) {
        return (Vector2){x * scale, y * scale};
    }
    Ship* s = &env->ships[0];
    float dx = sm_wrap_delta(x - s->x, env->size);
    float dy = sm_wrap_delta(y - s->y, env->size);
    return (Vector2){
        0.5f * GetScreenWidth() + dx * scale,
        0.5f * GetScreenHeight() + dy * scale,
    };
}

// Draws thing at its torus position plus wrapped duplicates near the edges
static void sm_draw_wrapped(StarMelee* env, float x, float y, float margin,
        void (*draw)(StarMelee*, float, float, float, int), int i) {
    float scale = sm_cam_scale(env);
    if (sm_camera.follow) {
        Vector2 p = sm_world_to_screen(env, x, y);
        if (p.x > -margin && p.x < GetScreenWidth() + margin
                && p.y > -margin && p.y < GetScreenHeight() + margin) {
            draw(env, p.x, p.y, scale, i);
        }
        return;
    }
    float side = env->size * scale;
    for (int ox = -1; ox <= 1; ox++) {
        for (int oy = -1; oy <= 1; oy++) {
            float sx = x * scale + ox * side;
            float sy = y * scale + oy * side;
            if (sx < -margin || sx > side + margin) continue;
            if (sy < -margin || sy > side + margin) continue;
            draw(env, sx, sy, scale, i);
        }
    }
}

static void sm_draw_ship_sprite(StarMelee* env, float sx, float sy, float scale, int i) {
    Ship* s = &env->ships[i];
    float len = fmaxf(11.0f, env->ship_radius * scale * 2.2f);
    float wid = 0.62f * len;
    Vector2 dir = {cosf(s->heading), sinf(s->heading)};
    Vector2 side = {-dir.y, dir.x};
    Vector2 c = {sx, sy};
    Vector2 nose = {c.x + dir.x * len, c.y + dir.y * len};
    Vector2 lwing = {c.x - dir.x * len * 0.55f + side.x * wid, c.y - dir.y * len * 0.55f + side.y * wid};
    Vector2 rwing = {c.x - dir.x * len * 0.55f - side.x * wid, c.y - dir.y * len * 0.55f - side.y * wid};
    // Wounded ships visibly fade (and flicker hard while retreating)
    float hp_frac = sm_clampf(s->hp / env->hp_max, 0.0f, 1.0f);
    float glow = 0.45f + 0.55f * hp_frac;
    if (s->attack_state == SM_STATE_RETREAT && ((int)(GetTime() * 10.0f) % 2)) {
        glow *= 0.55f;
    }
    Color color = Fade(SM_SHIP_COLORS[i], glow);

    if (s->thrusting) {
        Vector2 base = {c.x - dir.x * len * 0.62f, c.y - dir.y * len * 0.62f};
        Vector2 tip = {base.x - dir.x * len * 0.9f, base.y - dir.y * len * 0.9f};
        Vector2 fl = {base.x + side.x * wid * 0.35f, base.y + side.y * wid * 0.35f};
        Vector2 fr = {base.x - side.x * wid * 0.35f, base.y - side.y * wid * 0.35f};
        DrawTriangle(tip, fl, fr, (Color){255, 140, 40, 220});
    }
    if (s->turning_left || s->turning_right) {
        float jet = s->turning_left ? 1.0f : -1.0f;
        Vector2 jb = {nose.x - dir.x * len * 0.35f, nose.y - dir.y * len * 0.35f};
        Vector2 jt = {jb.x + side.x * jet * wid * 0.8f, jb.y + side.y * jet * wid * 0.8f};
        DrawLineEx(jb, jt, 2.0f, (Color){255, 220, 100, 200});
    }
    DrawTriangle(nose, lwing, rwing, color);
    DrawTriangleLines(nose, lwing, rwing, SM_WHITE);
}

static void sm_draw_asteroid_sprite(StarMelee* env, float sx, float sy, float scale, int k) {
    Asteroid* a = &env->asteroids[k];
    for (int v = 0; v < a->num_vertices; v++) {
        int w = (v + 1) % a->num_vertices;
        DrawLineV((Vector2){sx + a->shape[v].x * scale, sy + a->shape[v].y * scale},
                  (Vector2){sx + a->shape[w].x * scale, sy + a->shape[w].y * scale},
                  (Color){170, 160, 150, 255});
    }
}

// Cannon round: a short streak along its velocity with a bright head
static void sm_draw_projectile_sprite(StarMelee* env, float sx, float sy, float scale, int k) {
    Projectile* p = &env->projectiles[k];
    int owner = (p->owner >= 0 && p->owner < STARMELEE_MAX_SHIPS) ? p->owner : 0;
    float spd = sqrtf(p->vx*p->vx + p->vy*p->vy);
    float ux = spd > 1e-5f ? p->vx / spd : 1.0f;
    float uy = spd > 1e-5f ? p->vy / spd : 0.0f;
    float len = fmaxf(6.0f, 3.0f * scale);
    DrawLineEx((Vector2){sx - ux * len, sy - uy * len}, (Vector2){sx, sy}, 2.0f,
        Fade(SM_SHIP_COLORS[owner], 0.85f));
    DrawCircleV((Vector2){sx, sy}, fmaxf(1.5f, 1.1f * scale), SM_WHITE);
}

// Basic disintegration burst: expanding fading ring plus radial shards
static void sm_draw_boom_sprite(StarMelee* env, float sx, float sy, float scale, int k) {
    Asteroid* a = &env->asteroids[k];
    float t = 1.0f - a->boom_timer / 12.0f;  // 0 -> 1 over the burst
    float alpha = 1.0f - t;
    float r = a->boom_radius * scale * (1.0f + 2.5f * t);
    DrawCircleLines((int)sx, (int)sy, r, Fade((Color){255, 160, 60, 255}, alpha));
    DrawCircleLines((int)sx, (int)sy, r * 0.55f, Fade((Color){255, 220, 100, 255}, alpha * 0.8f));
    for (int v = 0; v < 6; v++) {
        float ang = 2.0f * PI * v / 6.0f + (float)k;
        Vector2 from = {sx + cosf(ang) * r * 0.6f, sy + sinf(ang) * r * 0.6f};
        Vector2 to = {sx + cosf(ang) * r * 1.25f, sy + sinf(ang) * r * 1.25f};
        DrawLineV(from, to, Fade((Color){200, 190, 180, 255}, alpha));
    }
}

static void sm_draw_goal_sprite(StarMelee* env, float sx, float sy, float scale, int i) {
    float r = env->goal_radius * scale;
    float pulse = 0.72f + 0.28f * sinf(GetTime() * 4.0f + i);
    Color color = SM_SHIP_COLORS[i];
    DrawCircleLines((int)sx, (int)sy, r * pulse, color);
    DrawCircleLines((int)sx, (int)sy, r * 0.45f * pulse, Fade(color, 0.6f));
    DrawLineEx((Vector2){sx - r * 0.3f, sy}, (Vector2){sx + r * 0.3f, sy}, 1.5f, SM_GREEN);
    DrawLineEx((Vector2){sx, sy - r * 0.3f}, (Vector2){sx, sy + r * 0.3f}, 1.5f, SM_GREEN);
}

void c_render(StarMelee* env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(760, 760, "PufferLib StarMelee");
        // One render per c_step; scale FPS so action_repeat > 1 (training
        // cadence) still plays back at real-time physics speed.
        SetTargetFPS(60 / (env->action_repeat > 0 ? env->action_repeat : 1));
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsKeyPressed(KEY_C)) {
        sm_camera.follow = !sm_camera.follow;
    }
    if (IsKeyPressed(KEY_Z)) {
        sm_camera.zoom = fminf(sm_camera.zoom * 1.5f, 8.0f);
    }
    if (IsKeyPressed(KEY_X)) {
        sm_camera.zoom = fmaxf(sm_camera.zoom / 1.5f, 1.0f);
    }

    float scale = sm_cam_scale(env);

    BeginDrawing();
    ClearBackground(SM_BACKGROUND);

    // Starfield anchored to world coordinates so the follow camera reads right
    for (int k = 0; k < 140; k++) {
        unsigned int h = (unsigned int)(k * 2654435761u);
        float wx = (float)(h % 997) / 997.0f * env->size;
        float wy = (float)((h / 997) % 991) / 991.0f * env->size;
        Vector2 p = sm_world_to_screen(env, wx, wy);
        unsigned char b = (unsigned char)(70 + (h % 120));
        DrawPixel((int)p.x, (int)p.y, (Color){b, b, b, 255});
    }

    Vector2 pc = sm_world_to_screen(env, sm_planet_x(env), sm_planet_y(env));

    // Gravity well rings out to ~10 planet radii
    for (int k = 1; k <= 4; k++) {
        float rr = env->planet_radius * (2.5f * k) * scale;
        DrawCircleLines((int)pc.x, (int)pc.y, rr, Fade(SM_CYAN, 0.10f + 0.02f * (4 - k)));
    }

    // Planet
    DrawCircle((int)pc.x, (int)pc.y, env->planet_radius * scale, SM_PLANET);
    DrawCircle((int)(pc.x - env->planet_radius * scale * 0.25f),
               (int)(pc.y - env->planet_radius * scale * 0.25f),
               env->planet_radius * scale * 0.62f, Fade(SM_PLANET_DARK, 0.5f));
    DrawCircleLines((int)pc.x, (int)pc.y, env->planet_radius * scale, SM_WHITE);

    for (int k = 0; k < env->num_asteroids; k++) {
        if (env->asteroids[k].active) {
            sm_draw_wrapped(env, env->asteroids[k].x, env->asteroids[k].y, 40.0f,
                sm_draw_asteroid_sprite, k);
        }
        if (env->asteroids[k].boom_timer > 0) {
            env->asteroids[k].boom_timer -= 1;
            sm_draw_wrapped(env, env->asteroids[k].boom_x, env->asteroids[k].boom_y,
                60.0f, sm_draw_boom_sprite, k);
        }
    }

    for (int k = 0; k < STARMELEE_MAX_PROJECTILES; k++) {
        if (env->projectiles[k].life > 0) {
            sm_draw_wrapped(env, env->projectiles[k].x, env->projectiles[k].y, 30.0f,
                sm_draw_projectile_sprite, k);
        }
    }

    for (int i = 0; i < env->num_ships; i++) {
        Ship* s = &env->ships[i];

        // Trails (render-only ring buffer)
        env->trail_idx[i] = (env->trail_idx[i] + 1) % STARMELEE_TRAIL_LEN;
        env->trails[i][env->trail_idx[i]] = (Vector2){s->x, s->y};
        if (env->trail_count[i] < STARMELEE_TRAIL_LEN) env->trail_count[i] += 1;
        if (env->terminals != NULL && env->terminals[i] > 0.5f) {
            env->trail_count[i] = 0;  // break the trail on respawn
        }
        for (int t = 1; t < env->trail_count[i]; t++) {
            int i0 = (env->trail_idx[i] - t + STARMELEE_TRAIL_LEN) % STARMELEE_TRAIL_LEN;
            int i1 = (env->trail_idx[i] - t + 1 + STARMELEE_TRAIL_LEN) % STARMELEE_TRAIL_LEN;
            Vector2 p0 = env->trails[i][i0];
            Vector2 p1 = env->trails[i][i1];
            // Skip trail segments that jump across the torus seam
            if (fabsf(p0.x - p1.x) > 0.5f * env->size) continue;
            if (fabsf(p0.y - p1.y) > 0.5f * env->size) continue;
            float alpha = 0.35f * (1.0f - (float)t / STARMELEE_TRAIL_LEN);
            DrawLineV(sm_world_to_screen(env, p0.x, p0.y),
                      sm_world_to_screen(env, p1.x, p1.y),
                      Fade(SM_SHIP_COLORS[i], alpha));
        }

        if (!env->combat) {
            sm_draw_wrapped(env, s->goal_x, s->goal_y, 40.0f, sm_draw_goal_sprite, i);
        }
        sm_draw_wrapped(env, s->x, s->y, 40.0f, sm_draw_ship_sprite, i);

        // Muzzle flash at the nose, hit flash around a struck ship
        if (s->fire_flash > 0) {
            s->fire_flash -= 1;
            Vector2 c = sm_world_to_screen(env, s->x, s->y);
            float nose = (env->ship_radius + 3.0f) * scale;
            Vector2 fp = {c.x + cosf(s->heading) * nose, c.y + sinf(s->heading) * nose};
            DrawCircleV(fp, 3.0f + 0.3f * s->fire_flash, Fade(SM_YELLOW, s->fire_flash / 10.0f));
        }
        if (s->hit_flash > 0) {
            s->hit_flash -= 1;
            Vector2 c = sm_world_to_screen(env, s->x, s->y);
            DrawCircleLines((int)c.x, (int)c.y,
                (env->ship_radius + 4.0f) * scale * (1.0f + 0.06f * (12 - s->hit_flash)),
                Fade(SM_RED, s->hit_flash / 12.0f));
        }

        // Episode-end feedback text above the ship
        if (s->event_timer > 0) {
            s->event_timer -= 1;
            const char* text = "TIMEOUT";
            Color color = SM_YELLOW;
            if (s->last_result == SM_RESULT_SUCCESS) { text = "ARRIVED"; color = SM_GREEN; }
            if (s->last_result == SM_RESULT_CRASH) { text = "DESTROYED"; color = SM_RED; }
            Vector2 tp = sm_world_to_screen(env, s->x, s->y);
            DrawText(text, (int)tp.x - 30, (int)tp.y - 28, 14,
                Fade(color, s->event_timer / 70.0f));
        }
    }

    // HUD for ship 0
    Ship* s0 = &env->ships[0];
    float speed = sqrtf(s0->vx*s0->vx + s0->vy*s0->vy);
    DrawRectangle(12, 10, 190, env->combat ? 140 : 104, Fade(BLACK, 0.35f));
    if (env->combat) {
        DrawText("Duel: close, shoot, disengage", 20, 16, 16, SM_WHITE);
        const char* state_name = "ATTACK";
        Color state_color = SM_GREEN;
        if (s0->attack_state == SM_STATE_FLEE) {
            state_name = "RELOAD-FLEE";
            state_color = SM_YELLOW;
        } else if (s0->attack_state == SM_STATE_RETREAT) {
            state_name = "RETREAT";
            state_color = SM_RED;
        }
        DrawText(TextFormat("%s  hits %d/%d  cycles %d", state_name,
            s0->attack_passes, s0->shots_fired, s0->full_cycles), 20, 112, 12, state_color);
        // Magazine pips, or the reload bar while it refills
        if (s0->reloading) {
            float frac = 1.0f - (float)s0->cannon_timer / (float)env->cannon_reload_ticks;
            DrawText("reload", 20, 128, 12, SM_YELLOW);
            DrawRectangle(70, 130, 100, 8, Fade(SM_WHITE, 0.15f));
            DrawRectangle(70, 130, (int)(100 * frac), 8, SM_YELLOW);
        } else {
            DrawText("ammo", 20, 128, 12, SM_WHITE);
            for (int a = 0; a < s0->mag_rounds; a++) {
                Color pip = a < s0->ammo ? SM_GREEN : Fade(SM_WHITE, 0.2f);
                DrawRectangle(70 + 14 * a, 130, 10, 8, pip);
            }
        }
    } else {
        DrawText("Fly to your beacon", 20, 16, 16, SM_WHITE);
    }
    DrawText(TextFormat("speed %5.2f / %.1f", speed, env->max_speed), 20, 38, 14, SM_WHITE);
    DrawText(TextFormat("step  %d / %d", s0->episode_tick, env->max_ticks), 20, 56, 14, SM_WHITE);
    DrawText(TextFormat("hp    %3.0f", s0->hp), 20, 74, 14, SM_WHITE);
    DrawText(sm_camera.follow
        ? TextFormat("C cam (follow %.1fx) Z/X zoom", sm_camera.zoom)
        : "C cam (arena) Z/X zoom", 20, 94, 12, Fade(SM_WHITE, 0.7f));
    float hp_frac = sm_clampf(s0->hp / env->hp_max, 0.0f, 1.0f);
    DrawRectangle(80, 76, 110, 10, Fade(SM_WHITE, 0.15f));
    DrawRectangle(80, 76, (int)(110 * hp_frac), 10,
        hp_frac > 0.5f ? SM_GREEN : (hp_frac > 0.25f ? SM_YELLOW : SM_RED));

    EndDrawing();
}

void c_close(StarMelee* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}
