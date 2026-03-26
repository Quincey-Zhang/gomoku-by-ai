#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

// Gomoku (Five in a Row) on a 15x15 board.
// Action space: 225 discrete actions (row * 15 + col)
// Observation: 225 board cells + 1 current player indicator = 226
//   Each cell: 0 = empty, 1 = ego stone, 2 = opponent stone
//
// Board storage uses 1-indexed [1..15][1..15] within a 17x17 array
// to allow safe boundary access for win-checking without bounds checks.

#define N_PLAYERS 2
#define BOARD_W 15
#define BOARD_H 15
#define BOARD_SIZE (BOARD_W * BOARD_H)
#define NUM_MOVES BOARD_SIZE
#define MAX_X (BOARD_W + 2)
#define MAX_Y (BOARD_H + 2)
#define OBS_SIZE (BOARD_SIZE + 1)
#define STATE_SIZE (BOARD_SIZE)

namespace Gomoku {

  class Engine;

  enum class ExportID : uint32_t {
    WorldReset,
    ActiveAgent,
    Action,
    Observation,
    ActionMask,
    Reward,
    WorldID,
    AgentID,
    AgentState,
    NumExports,
  };

  // singletons

  struct WorldReset {
    int32_t resetNow;
  };

  // per-agent

  struct ActiveAgent {
    int32_t isActive;
  };

  struct Action {
    int32_t choice; // 0..224: row*15+col
  };

  struct Observation {
    int8_t bitvec[OBS_SIZE];
  };

  struct State {
    int8_t board[MAX_X][MAX_Y]; // 0=empty, 'X'=player0, 'O'=player1
    int32_t current_player;
    int32_t move_count;
    bool game_end;
  };

  struct AgentState {
    int8_t bitvec[OBS_SIZE];
  };

  struct AgentID {
    int32_t id;
  };

  struct ActionMask {
    int32_t isValid[NUM_MOVES];
  };

  struct Reward {
    float rew;
  };

  struct Agent : public madrona::Archetype<Action, Observation, AgentState, AgentID, ActionMask, ActiveAgent, Reward> {};

  struct Config {
    uint32_t numPlayers;
  };

  struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry, const Config &cfg);

    static void setupTasks(madrona::TaskGraphManager &taskgraph_mgr, const Config &cfg);

    Sim(Engine &ctx, const Config& cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;

    uint32_t players;

    madrona::Entity *agents;
  };

  class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
  };

}
