// Standalone StarMelee: manual play-test and trained-policy playback.
//
// Controls: LEFT/A and RIGHT/D turn, UP/W/SPACE fires the engine, R respawns,
// C toggles the follow camera, Z/X zoom, ESC quits. If
// resources/starmelee/starmelee_policy.bin exists (produced by
// ocean/starmelee/export_policy.py from a torch checkpoint), the policy
// flies every ship; holding LEFT SHIFT hands ship 0 to the keyboard, so the
// same exe is watch-the-AI by default and human-vs-AI on demand.
//
// Physics/task parameters are read from the [env] section of
// config/starmelee.ini (or a path given as argv[1]) so play-testing and
// training always share one tuning source.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "starmelee.h"

// ---------------------------------------------------------------------------
// SMP1 policy: linear encoder -> MinGRU stack -> per-head linear decoder.
// Mirrors pufferlib/models.py (DefaultEncoder + MinGRU + DefaultDecoder);
// the file layout is defined in ocean/starmelee/export_policy.py.
// ---------------------------------------------------------------------------

#define SMP_MAGIC 0x31504D53
#define SMP_MAX_HEADS 8

typedef struct {
    int obs_size;
    int hidden;
    int num_layers;
    int num_heads;
    int nvec[SMP_MAX_HEADS];
    int logit_sum;
    float* enc_w;
    float* enc_b;
    float** gru_w;   // per layer, [3*hidden][hidden]
    float* dec_w;
    float* dec_b;
    float* state;    // [num_agents][num_layers][hidden]
    int num_agents;
    // scratch
    float* h;
    float* f;
    float* mix;
} SMPolicy;

static float* smp_read(FILE* file, size_t count) {
    float* data = (float*)malloc(count * sizeof(float));
    if (fread(data, sizeof(float), count, file) != count) {
        free(data);
        return NULL;
    }
    return data;
}

static SMPolicy* smp_load(const char* path, int num_agents) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    int header[5];
    if (fread(header, sizeof(int), 5, file) != 5 || header[0] != SMP_MAGIC) {
        printf("Policy %s: bad header, ignoring\n", path);
        fclose(file);
        return NULL;
    }
    SMPolicy* net = (SMPolicy*)calloc(1, sizeof(SMPolicy));
    net->obs_size = header[1];
    net->hidden = header[2];
    net->num_layers = header[3];
    net->num_heads = header[4];
    net->num_agents = num_agents;
    int ok = net->obs_size == STARMELEE_OBS_SIZE
        && net->hidden > 0 && net->hidden <= 4096
        && net->num_layers > 0 && net->num_layers <= 8
        && net->num_heads == 3
        && fread(net->nvec, sizeof(int), net->num_heads, file) == (size_t)net->num_heads;
    for (int hd = 0; ok && hd < net->num_heads; hd++) {
        net->logit_sum += net->nvec[hd];
        ok = net->nvec[hd] > 0 && net->nvec[hd] <= 16;
    }
    if (!ok) {
        printf("Policy %s: unsupported shape (obs %d, expected %d), ignoring\n",
            path, net->obs_size, STARMELEE_OBS_SIZE);
        fclose(file);
        free(net);
        return NULL;
    }
    int H = net->hidden;
    net->enc_w = smp_read(file, (size_t)H * net->obs_size);
    net->enc_b = smp_read(file, H);
    net->gru_w = (float**)calloc(net->num_layers, sizeof(float*));
    for (int l = 0; l < net->num_layers; l++) {
        net->gru_w[l] = smp_read(file, (size_t)3 * H * H);
    }
    net->dec_w = smp_read(file, (size_t)net->logit_sum * H);
    net->dec_b = smp_read(file, net->logit_sum);
    fclose(file);

    ok = net->enc_w && net->enc_b && net->dec_w && net->dec_b;
    for (int l = 0; l < net->num_layers; l++) {
        ok = ok && net->gru_w[l] != NULL;
    }
    if (!ok) {
        printf("Policy %s: truncated file, ignoring\n", path);
        return NULL;  // leak on this one-shot error path is fine
    }
    net->state = (float*)calloc((size_t)num_agents * net->num_layers * H, sizeof(float));
    net->h = (float*)malloc(H * sizeof(float));
    net->f = (float*)malloc(3 * H * sizeof(float));
    net->mix = (float*)malloc(H * sizeof(float));
    return net;
}

static void smp_reset_state(SMPolicy* net, int agent) {
    memset(net->state + (size_t)agent * net->num_layers * net->hidden, 0,
        (size_t)net->num_layers * net->hidden * sizeof(float));
}

static inline float smp_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// One eval step for one agent: mirrors Policy.forward_eval + MinGRU math
static void smp_forward(SMPolicy* net, const float* obs, int agent,
        float* actions, unsigned int* rng) {
    int H = net->hidden;

    for (int o = 0; o < H; o++) {
        const float* w = net->enc_w + (size_t)o * net->obs_size;
        float acc = net->enc_b[o];
        for (int i = 0; i < net->obs_size; i++) {
            acc += w[i] * obs[i];
        }
        net->h[o] = acc;
    }

    for (int l = 0; l < net->num_layers; l++) {
        float* st = net->state + ((size_t)agent * net->num_layers + l) * H;
        for (int o = 0; o < 3 * H; o++) {
            const float* w = net->gru_w[l] + (size_t)o * H;
            float acc = 0.0f;
            for (int k = 0; k < H; k++) {
                acc += w[k] * net->h[k];
            }
            net->f[o] = acc;
        }
        for (int k = 0; k < H; k++) {
            float hid = net->f[k];
            float z = smp_sigmoid(net->f[H + k]);
            float p = smp_sigmoid(net->f[2 * H + k]);
            float g = hid >= 0.0f ? hid + 0.5f : smp_sigmoid(hid);
            float out = st[k] + z * (g - st[k]);
            st[k] = out;
            net->mix[k] = p * out + (1.0f - p) * net->h[k];
        }
        memcpy(net->h, net->mix, H * sizeof(float));
    }

    // Per-head softmax sample, like the training-time rollouts
    int off = 0;
    for (int hd = 0; hd < net->num_heads; hd++) {
        int A = net->nvec[hd];
        float logits[16];
        float max_logit = -1e30f;
        for (int a = 0; a < A; a++) {
            const float* w = net->dec_w + (size_t)(off + a) * H;
            float acc = net->dec_b[off + a];
            for (int k = 0; k < H; k++) {
                acc += w[k] * net->h[k];
            }
            logits[a] = acc;
            if (acc > max_logit) max_logit = acc;
        }
        float total = 0.0f;
        for (int a = 0; a < A; a++) {
            logits[a] = expf(logits[a] - max_logit);
            total += logits[a];
        }
        float pick = total * (rand_r(rng) / (float)RAND_MAX);
        int choice = A - 1;
        for (int a = 0; a < A; a++) {
            pick -= logits[a];
            if (pick <= 0.0f) {
                choice = a;
                break;
            }
        }
        actions[hd] = (float)choice;
        off += A;
    }
}

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
    // The policy still decides at its trained cadence via policy_repeat.
    int policy_repeat = env.action_repeat >= 1 ? env.action_repeat : 4;
    env.action_repeat = 1;
    c_init(&env);

    int num_agents = env.num_agents;
    env.observations = (float*)calloc(num_agents * STARMELEE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(num_agents * 3, sizeof(float));
    env.rewards = (float*)calloc(num_agents, sizeof(float));
    env.terminals = (float*)calloc(num_agents, sizeof(float));

    // Optional trained policy, exported by ocean/starmelee/export_policy.py
    const char* policy_path = "resources/starmelee/starmelee_policy.bin";
    SMPolicy* net = smp_load(policy_path, num_agents);
    if (net != NULL) {
        printf("Loaded policy %s (hidden %d, %d layer%s)\n",
            policy_path, net->hidden, net->num_layers, net->num_layers == 1 ? "" : "s");
        printf("Policy flies all ships; hold LEFT SHIFT to take ship 0\n");
    } else {
        printf("No policy at %s: manual ship 0, others drift\n", policy_path);
        printf("Export one with: python ocean/starmelee/export_policy.py latest\n");
    }
    unsigned int policy_rng = 7u;

    c_reset(&env);
    c_render(&env);
    long frame = 0;
    while (!WindowShouldClose()) {
        int manual0 = net == NULL || IsKeyDown(KEY_LEFT_SHIFT);
        int policy_turn = frame % policy_repeat == 0;
        frame += 1;
        for (int i = 0; i < num_agents; i++) {
            if (i == 0 && manual0) {
                env.actions[0] = (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) ? 1.0f : 0.0f;
                env.actions[1] = (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) ? 1.0f : 0.0f;
                env.actions[2] = (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W) || IsKeyDown(KEY_SPACE))
                    ? 1.0f : 0.0f;
            } else if (net != NULL) {
                // Trained cadence: decide every policy_repeat frames, latch between
                if (policy_turn) {
                    smp_forward(net, &env.observations[i * STARMELEE_OBS_SIZE], i,
                        &env.actions[i * 3], &policy_rng);
                }
            } else {
                env.actions[i * 3 + 0] = 0.0f;
                env.actions[i * 3 + 1] = 0.0f;
                env.actions[i * 3 + 2] = 0.0f;
            }
        }

        if (IsKeyPressed(KEY_R)) {
            c_reset(&env);
            if (net != NULL) {
                for (int i = 0; i < num_agents; i++) smp_reset_state(net, i);
            }
        } else {
            c_step(&env);
            if (net != NULL) {
                for (int i = 0; i < num_agents; i++) {
                    if (env.terminals[i] > 0.5f) smp_reset_state(net, i);
                }
            }
        }
        c_render(&env);
    }
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
