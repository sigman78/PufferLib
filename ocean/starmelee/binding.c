#include "starmelee.h"

#define OBS_SIZE STARMELEE_OBS_SIZE
#define NUM_ATNS 3
#define ACT_SIZES {2, 2, 2}
#define OBS_TENSOR_T FloatTensor

#define Env StarMelee
#define MY_VEC_INIT
#include "vecenv.h"

// Same layout as the default my_vec_init in vecenv.h, plus fail-fast checks:
// slots are assigned in fixed agents_per_buffer blocks, so num_ships must
// divide both total_agents and agents_per_buffer or envs alias each other's
// buffer slots (and the last env writes past the end of the global buffers).
Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;
    int num_ships = (int)dict_get(env_kwargs, "num_ships")->value;
    if (num_ships < 1) num_ships = 1;
    if (num_ships > STARMELEE_MAX_SHIPS) num_ships = STARMELEE_MAX_SHIPS;

    if (total_agents % num_buffers != 0) {
        fprintf(stderr,
            "starmelee: num_buffers (%d) must divide total_agents (%d)\n",
            num_buffers, total_agents);
        exit(1);
    }
    if (total_agents % num_ships != 0 || agents_per_buffer % num_ships != 0) {
        fprintf(stderr,
            "starmelee: num_ships (%d) must divide both total_agents (%d) and "
            "total_agents/num_buffers (%d)\n",
            num_ships, total_agents, agents_per_buffer);
        exit(1);
    }

    int num_envs = total_agents / num_ships;
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    for (int i = 0; i < num_envs; i++) {
        envs[i].rng = i;
        my_init(&envs[i], env_kwargs);
    }

    int envs_per_buffer = agents_per_buffer / num_ships;
    for (int buf = 0; buf < num_buffers; buf++) {
        buffer_env_starts[buf] = buf * envs_per_buffer;
        buffer_env_counts[buf] = envs_per_buffer;
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_init(Env* env, Dict* kwargs) {
    env->size = (float)dict_get(kwargs, "size")->value;
    env->num_ships = (int)dict_get(kwargs, "num_ships")->value;
    env->max_ticks = (int)dict_get(kwargs, "max_ticks")->value;
    env->thrust = (float)dict_get(kwargs, "thrust")->value;
    env->turn_accel = (float)dict_get(kwargs, "turn_accel")->value;
    env->max_turn_rate = (float)dict_get(kwargs, "max_turn_rate")->value;
    env->angular_damping = (float)dict_get(kwargs, "angular_damping")->value;
    env->linear_damping = (float)dict_get(kwargs, "linear_damping")->value;
    env->max_speed = (float)dict_get(kwargs, "max_speed")->value;
    env->gravity = (float)dict_get(kwargs, "gravity")->value;
    env->gravity_falloff = (float)dict_get(kwargs, "gravity_falloff")->value;
    env->planet_radius = (float)dict_get(kwargs, "planet_radius")->value;
    env->ship_radius = (float)dict_get(kwargs, "ship_radius")->value;
    env->goal_radius = (float)dict_get(kwargs, "goal_radius")->value;
    env->hp_max = (float)dict_get(kwargs, "hp_max")->value;
    env->damage_scale = (float)dict_get(kwargs, "damage_scale")->value;
    env->restitution = (float)dict_get(kwargs, "restitution")->value;
    env->step_penalty = (float)dict_get(kwargs, "step_penalty")->value;
    env->progress_scale = (float)dict_get(kwargs, "progress_scale")->value;
    env->spawn_clearance = (float)dict_get(kwargs, "spawn_clearance")->value;
    env->min_goal_frac = (float)dict_get(kwargs, "min_goal_frac")->value;
    env->input_change_penalty = (float)dict_get(kwargs, "input_change_penalty")->value;
    env->action_repeat = (int)dict_get(kwargs, "action_repeat")->value;
    env->combat = (int)dict_get(kwargs, "combat")->value;
    env->attack_range_min = (float)dict_get(kwargs, "attack_range_min")->value;
    env->attack_range_max = (float)dict_get(kwargs, "attack_range_max")->value;
    env->aim_cone = (float)dict_get(kwargs, "aim_cone")->value;
    env->disengage_range = (float)dict_get(kwargs, "disengage_range")->value;
    env->engage_scale = (float)dict_get(kwargs, "engage_scale")->value;
    env->aim_reward = (float)dict_get(kwargs, "aim_reward")->value;
    env->strafe_reward = (float)dict_get(kwargs, "strafe_reward")->value;
    env->hit_reward = (float)dict_get(kwargs, "hit_reward")->value;
    env->cycle_reward = (float)dict_get(kwargs, "cycle_reward")->value;
    env->targeted_penalty = (float)dict_get(kwargs, "targeted_penalty")->value;
    env->shot_damage = (float)dict_get(kwargs, "shot_damage")->value;
    env->duel_spawn_frac = (float)dict_get(kwargs, "duel_spawn_frac")->value;
    env->trait_variation = (float)dict_get(kwargs, "trait_variation")->value;
    env->loiter_reward = (float)dict_get(kwargs, "loiter_reward")->value;
    env->retreat_hp_frac = (float)dict_get(kwargs, "retreat_hp_frac")->value;
    env->hp_regen = (float)dict_get(kwargs, "hp_regen")->value;
    env->cannon_rounds = (int)dict_get(kwargs, "cannon_rounds")->value;
    env->cannon_interval_ticks = (int)dict_get(kwargs, "cannon_interval_ticks")->value;
    env->cannon_reload_ticks = (int)dict_get(kwargs, "cannon_reload_ticks")->value;
    env->projectile_speed = (float)dict_get(kwargs, "projectile_speed")->value;
    env->projectile_range = (float)dict_get(kwargs, "projectile_range")->value;
    env->num_asteroids = (int)dict_get(kwargs, "num_asteroids")->value;
    env->asteroid_radius = (float)dict_get(kwargs, "asteroid_radius")->value;
    env->asteroid_damage = (float)dict_get(kwargs, "asteroid_damage")->value;
    env->asteroid_speed_min = (float)dict_get(kwargs, "asteroid_speed_min")->value;
    env->asteroid_speed_max = (float)dict_get(kwargs, "asteroid_speed_max")->value;
    env->asteroid_respawn_ticks = (int)dict_get(kwargs, "asteroid_respawn_ticks")->value;
    env->danger_hp_weight = (float)dict_get(kwargs, "danger_hp_weight")->value;
    c_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success_rate", log->success_rate);
    dict_set(out, "crash_rate", log->crash_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "final_distance", log->final_distance);
    dict_set(out, "planet_hits", log->planet_hits);
    dict_set(out, "input_changes", log->input_changes);
    dict_set(out, "attack_passes", log->attack_passes);
    dict_set(out, "shots_fired", log->shots_fired);
    dict_set(out, "full_cycles", log->full_cycles);
    dict_set(out, "retreats", log->retreats);
    dict_set(out, "asteroid_hits", log->asteroid_hits);
    dict_set(out, "asteroid_kills", log->asteroid_kills);
}
