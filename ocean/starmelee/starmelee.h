// StarMelee: Star Control Melee style ships around a gravity planet on a torus.
// Phase 1 task: fly from spawn (A) to a beacon (B) without smashing into the
// planet. Ships are torque/thrust driven: hold left/right to apply angular
// acceleration, hold engine to accelerate along the heading. The planet pulls
// with inverse-square gravity and bounces ships off with damage.
#pragma once

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"

#define STARMELEE_OBS_SIZE 18
#define STARMELEE_MAX_SHIPS 8
#define STARMELEE_TRAIL_LEN 48

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
    float n;
} Log;

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
    unsigned char thrusting;
    unsigned char turning_left;
    unsigned char turning_right;
    unsigned char last_result;
    int event_timer;  // render-only feedback countdown
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

    Ship ships[STARMELEE_MAX_SHIPS];
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

    sm_sample_clear_point(env, clearance, &s->x, &s->y);
    s->vx = 0.0f;
    s->vy = 0.0f;
    s->heading = 2.0f * PI * sm_randf(env);
    s->omega = 0.0f;
    s->hp = env->hp_max;
    s->episode_tick = 0;
    s->episode_return = 0.0f;
    s->planet_hits = 0;
    s->input_changes = 0;
    s->thrusting = 0;
    s->turning_left = 0;
    s->turning_right = 0;

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
        obs[6] = s->vx / env->max_speed;
        obs[7] = s->vy / env->max_speed;
        obs[8] = cosf(s->heading);
        obs[9] = sinf(s->heading);
        obs[10] = s->omega / env->max_turn_rate;
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
            obs[13] = sm_wrap_delta(o->x - s->x, env->size) / half;
            obs[14] = sm_wrap_delta(o->y - s->y, env->size) / half;
            // Both ships can move at up to 2*max_speed (hard cap)
            obs[15] = (o->vx - s->vx) / (2.0f * env->max_speed);
            obs[16] = (o->vy - s->vy) / (2.0f * env->max_speed);
            obs[17] = 1.0f;
        } else {
            obs[13] = 0.0f;
            obs[14] = 0.0f;
            obs[15] = 0.0f;
            obs[16] = 0.0f;
            obs[17] = 0.0f;
        }
    }
}

static void sm_add_log(StarMelee* env, Ship* s, unsigned char result) {
    env->log.perf += result == SM_RESULT_SUCCESS ? 1.0f : 0.0f;
    env->log.score += result == SM_RESULT_SUCCESS ? 1.0f
        : (result == SM_RESULT_CRASH ? -1.0f : 0.0f);
    env->log.episode_return += s->episode_return;
    env->log.episode_length += s->episode_tick;
    env->log.success_rate += result == SM_RESULT_SUCCESS ? 1.0f : 0.0f;
    env->log.crash_rate += result == SM_RESULT_CRASH ? 1.0f : 0.0f;
    env->log.timeout_rate += result == SM_RESULT_TIMEOUT ? 1.0f : 0.0f;
    env->log.final_distance += s->prev_goal_dist / env->size;
    env->log.planet_hits += s->planet_hits;
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
    compute_observations(env);
}

void c_step(StarMelee* env) {
    int n = env->num_ships;
    for (int i = 0; i < n; i++) {
        env->rewards[i] = 0.0f;
        env->terminals[i] = 0.0f;
    }

    // Controls + physics integration
    for (int i = 0; i < n; i++) {
        Ship* s = &env->ships[i];
        s->episode_tick += 1;

        int left = env->actions[i*3 + 0] > 0.5f;
        int right = env->actions[i*3 + 1] > 0.5f;
        int engine = env->actions[i*3 + 2] > 0.5f;

        // Jitter penalty: charge every button toggle so dithering (rapid
        // on/off switching a human would never produce) costs reward while
        // sustained holds stay free.
        int changes = (left != s->turning_left) + (right != s->turning_right)
            + (engine != s->thrusting);
        if (changes > 0 && env->input_change_penalty > 0.0f) {
            float jitter = env->input_change_penalty * (float)changes;
            env->rewards[i] -= jitter;
            s->episode_return -= jitter;
        }
        s->input_changes += changes;
        s->turning_left = (unsigned char)left;
        s->turning_right = (unsigned char)right;
        s->thrusting = (unsigned char)engine;

        // Torque / inertia -> angular acceleration while held
        if (left) s->omega -= env->turn_accel;
        if (right) s->omega += env->turn_accel;
        s->omega *= env->angular_damping;
        s->omega = sm_clampf(s->omega, -env->max_turn_rate, env->max_turn_rate);
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
            float cap = fmaxf(env->max_speed, prev_speed);
            s->vx += env->thrust * cosf(s->heading);
            s->vy += env->thrust * sinf(s->heading);
            float speed = sqrtf(s->vx*s->vx + s->vy*s->vy);
            if (speed > cap) {
                s->vx *= cap / speed;
                s->vy *= cap / speed;
            }
        }

        // Safety cap + configurable bleed keeps slingshots bounded
        float speed = sqrtf(s->vx*s->vx + s->vy*s->vy);
        float hard_cap = 2.0f * env->max_speed;
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
            float damage = env->damage_scale * -vn;
            s->hp -= damage;
            s->planet_hits += 1;
            float pain = -damage / env->hp_max;
            env->rewards[i] += pain;
            s->episode_return += pain;
        }
    }

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

        if (s->hp <= 0.0f) {
            sm_finish_ship(env, i, -1.0f, SM_RESULT_CRASH);
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
    Color color = SM_SHIP_COLORS[i];

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
        SetTargetFPS(60);
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

        sm_draw_wrapped(env, s->goal_x, s->goal_y, 40.0f, sm_draw_goal_sprite, i);
        sm_draw_wrapped(env, s->x, s->y, 40.0f, sm_draw_ship_sprite, i);

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
    DrawRectangle(12, 10, 190, 104, Fade(BLACK, 0.35f));
    DrawText("Fly to your beacon", 20, 16, 16, SM_WHITE);
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
