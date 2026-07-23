// Standalone StarMelee: manual play-test, with optional trained-policy playback.
//
// Controls: LEFT/A and RIGHT/D turn, UP/W/SPACE fires the engine, R respawns,
// C toggles the follow camera, Z/X zoom, ESC quits. If
// resources/starmelee/starmelee_weights.bin exists, the policy flies the ship
// and holding LEFT SHIFT hands control back to the keyboard.
//
// Physics/task parameters are read from the [env] section of
// config/starmelee.ini (or a path given as argv[1]) so play-testing and
// training always share one tuning source.
#include <stdio.h>
#include <string.h>
#include "starmelee.h"
#include "puffernet.h"

static char* sm_trim(char* s) {
    while (*s == ' ' || *s == '\t') s++;
    char* end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) {
        end--;
    }
    *end = '\0';
    return s;
}

// Minimal ini reader: only the [env] section, "key = value" lines.
static int sm_load_ini(StarMelee* env, const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) {
        return 0;
    }
    char line[256];
    int in_env = 0;
    while (fgets(line, sizeof(line), f)) {
        char* s = sm_trim(line);
        if (*s == ';' || *s == '#' || *s == '\0') continue;
        if (*s == '[') {
            in_env = strncmp(s, "[env]", 5) == 0;
            continue;
        }
        if (!in_env) continue;
        char* eq = strchr(s, '=');
        if (eq == NULL) continue;
        *eq = '\0';
        char* key = sm_trim(s);
        float value = strtof(sm_trim(eq + 1), NULL);

        if (strcmp(key, "size") == 0) env->size = value;
        else if (strcmp(key, "num_ships") == 0) env->num_ships = (int)value;
        else if (strcmp(key, "max_ticks") == 0) env->max_ticks = (int)value;
        else if (strcmp(key, "thrust") == 0) env->thrust = value;
        else if (strcmp(key, "turn_accel") == 0) env->turn_accel = value;
        else if (strcmp(key, "max_turn_rate") == 0) env->max_turn_rate = value;
        else if (strcmp(key, "angular_damping") == 0) env->angular_damping = value;
        else if (strcmp(key, "linear_damping") == 0) env->linear_damping = value;
        else if (strcmp(key, "max_speed") == 0) env->max_speed = value;
        else if (strcmp(key, "gravity") == 0) env->gravity = value;
        else if (strcmp(key, "gravity_falloff") == 0) env->gravity_falloff = value;
        else if (strcmp(key, "planet_radius") == 0) env->planet_radius = value;
        else if (strcmp(key, "ship_radius") == 0) env->ship_radius = value;
        else if (strcmp(key, "goal_radius") == 0) env->goal_radius = value;
        else if (strcmp(key, "hp_max") == 0) env->hp_max = value;
        else if (strcmp(key, "damage_scale") == 0) env->damage_scale = value;
        else if (strcmp(key, "restitution") == 0) env->restitution = value;
        else if (strcmp(key, "step_penalty") == 0) env->step_penalty = value;
        else if (strcmp(key, "progress_scale") == 0) env->progress_scale = value;
        else if (strcmp(key, "spawn_clearance") == 0) env->spawn_clearance = value;
        else if (strcmp(key, "min_goal_frac") == 0) env->min_goal_frac = value;
        else if (strcmp(key, "input_change_penalty") == 0) env->input_change_penalty = value;
        else if (strcmp(key, "action_repeat") == 0) env->action_repeat = (int)value;
        else if (strcmp(key, "combat") == 0) env->combat = (int)value;
        else if (strcmp(key, "attack_range_min") == 0) env->attack_range_min = value;
        else if (strcmp(key, "attack_range_max") == 0) env->attack_range_max = value;
        else if (strcmp(key, "aim_cone") == 0) env->aim_cone = value;
        else if (strcmp(key, "aim_ticks") == 0) env->aim_ticks = (int)value;
        else if (strcmp(key, "disengage_range") == 0) env->disengage_range = value;
        else if (strcmp(key, "engage_scale") == 0) env->engage_scale = value;
        else if (strcmp(key, "aim_reward") == 0) env->aim_reward = value;
        else if (strcmp(key, "strafe_reward") == 0) env->strafe_reward = value;
        else if (strcmp(key, "fire_reward") == 0) env->fire_reward = value;
        else if (strcmp(key, "cycle_reward") == 0) env->cycle_reward = value;
        else if (strcmp(key, "targeted_penalty") == 0) env->targeted_penalty = value;
        else if (strcmp(key, "shot_damage") == 0) env->shot_damage = value;
        else if (strcmp(key, "duel_spawn_frac") == 0) env->duel_spawn_frac = value;
        else if (strcmp(key, "trait_variation") == 0) env->trait_variation = value;
        else if (strcmp(key, "reengage_ticks") == 0) env->reengage_ticks = (int)value;
        else if (strcmp(key, "loiter_reward") == 0) env->loiter_reward = value;
        else if (strcmp(key, "retreat_hp_frac") == 0) env->retreat_hp_frac = value;
        else if (strcmp(key, "hp_regen") == 0) env->hp_regen = value;
        else if (strcmp(key, "num_asteroids") == 0) env->num_asteroids = (int)value;
        else if (strcmp(key, "asteroid_radius") == 0) env->asteroid_radius = value;
        else if (strcmp(key, "asteroid_damage") == 0) env->asteroid_damage = value;
        else if (strcmp(key, "asteroid_speed_min") == 0) env->asteroid_speed_min = value;
        else if (strcmp(key, "asteroid_speed_max") == 0) env->asteroid_speed_max = value;
        else if (strcmp(key, "asteroid_respawn_ticks") == 0) env->asteroid_respawn_ticks = (int)value;
        else if (strcmp(key, "danger_hp_weight") == 0) env->danger_hp_weight = value;
    }
    fclose(f);
    return 1;
}

int main(int argc, char** argv) {
    StarMelee env = {0};
    env.rng = 42;

    const char* ini_path = argc > 1 ? argv[1] : "config/starmelee.ini";
    if (sm_load_ini(&env, ini_path)) {
        printf("Loaded [env] parameters from %s\n", ini_path);
    } else {
        printf("No ini at %s: using built-in defaults\n", ini_path);
    }
    // Manual play reads the keyboard every frame; latching inputs across
    // repeated ticks (training cadence) would fast-forward the game 4x.
    env.action_repeat = 1;
    c_init(&env);

    int num_agents = env.num_agents;
    env.observations = (float*)calloc(num_agents * STARMELEE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(num_agents * 3, sizeof(float));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (float*)calloc(num_agents, sizeof(float));

    // Optional policy: only load when a checkpoint was exported
    const char* weights_path = "resources/starmelee/starmelee_weights.bin";
    Weights* weights = NULL;
    PufferNet* net = NULL;
    FILE* probe = fopen(weights_path, "rb");
    if (probe != NULL) {
        fclose(probe);
        weights = load_weights(weights_path);
        int logit_sizes[3] = {2, 2, 2};
        net = make_puffernet(weights, num_agents, STARMELEE_OBS_SIZE, 128, 1, logit_sizes, 3);
        printf("Loaded policy from %s (hold LEFT SHIFT for manual control)\n", weights_path);
    } else {
        printf("No weights at %s: manual control only\n", weights_path);
    }

    c_reset(&env);
    c_render(&env);
    while (!WindowShouldClose()) {
        int manual = net == NULL || IsKeyDown(KEY_LEFT_SHIFT);
        if (manual) {
            env.actions[0] = (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) ? 1.0f : 0.0f;
            env.actions[1] = (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : 0.0f;
            env.actions[2] = (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W) || IsKeyDown(KEY_SPACE))
                ? 1.0f : 0.0f;
        } else {
            forward_puffernet(net, env.observations, env.actions);
        }

        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
        } else {
            c_step(&env);
        }
        c_render(&env);
    }

    if (net != NULL) {
        free_puffernet(net);
        free(weights);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
