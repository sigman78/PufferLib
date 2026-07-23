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
}
