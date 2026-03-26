#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include <cmath>

using namespace madrona;
using namespace madrona::math;

namespace Gomoku {

  void Sim::registerTypes(ECSRegistry &registry, const Config &)
  {
    base::registerTypes(registry);

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<State>();

    registry.registerComponent<Action>();
    registry.registerComponent<Observation>();
    registry.registerComponent<AgentID>();
    registry.registerComponent<ActionMask>();
    registry.registerComponent<ActiveAgent>();
    registry.registerComponent<Reward>();
    registry.registerComponent<AgentState>();
    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>((uint32_t)ExportID::WorldReset);
    registry.exportColumn<Agent, ActiveAgent>((uint32_t)ExportID::ActiveAgent);
    registry.exportColumn<Agent, AgentState>((uint32_t)ExportID::AgentState);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Observation>((uint32_t)ExportID::Observation);
    registry.exportColumn<Agent, ActionMask>((uint32_t)ExportID::ActionMask);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, WorldID>((uint32_t)ExportID::WorldID);
    registry.exportColumn<Agent, AgentID>((uint32_t)ExportID::AgentID);
  }

  // Encode board from the perspective of the current player.
  // ego stones = 1, opponent stones = 2, empty = 0.
  // Last element = current_player id.
  inline int encodeBoardAndPlayer(Engine &ctx, Entity &agent, int offset)
  {
    Observation &obs = ctx.get<Observation>(agent);
    State &state = ctx.singleton<State>();

    int idx = 0;
    for (int i = 1; i <= BOARD_H; i++) {
      for (int ix = 1; ix <= BOARD_W; ix++) {
        if ((state.current_player == 0 && state.board[i][ix] == 'X') ||
            (state.current_player == 1 && state.board[i][ix] == 'O')) {
          obs.bitvec[idx] = 1; // ego stone
        } else if ((state.current_player == 1 && state.board[i][ix] == 'X') ||
                   (state.current_player == 0 && state.board[i][ix] == 'O')) {
          obs.bitvec[idx] = 2; // opponent stone
        } else {
          obs.bitvec[idx] = 0; // empty
        }
        idx += 1;
      }
    }

    obs.bitvec[OBS_SIZE - 1] = state.current_player;
    offset += OBS_SIZE;
    return offset;
  }

  inline void generateObsState(Engine &ctx, Entity &agent)
  {
    int offset = encodeBoardAndPlayer(ctx, agent, 0);
    (void) offset;
  }

  inline void generateActionMask(Engine &ctx, Entity &agent)
  {
    ActionMask &mask = ctx.get<ActionMask>(agent);
    State &state = ctx.singleton<State>();

    bool any_valid = false;
    for (int i = 0; i < NUM_MOVES; i++) {
      int row = i / BOARD_W + 1;
      int col = i % BOARD_W + 1;
      if (state.board[row][col] == 'X' || state.board[row][col] == 'O') {
        mask.isValid[i] = 0;
      } else {
        mask.isValid[i] = 1;
        any_valid = true;
      }
    }
    // If no valid moves (shouldn't happen before game_end), allow all
    if (!any_valid) {
      for (int i = 0; i < NUM_MOVES; i++) {
        mask.isValid[i] = 1;
      }
    }
  }

  static void resetWorld(Engine &ctx)
  {
    uint32_t num_players = ctx.data().players;

    // Reset observations
    for (uint32_t i = 0; i < num_players; i++) {
      Entity agent = ctx.data().agents[i];
      Observation &obs = ctx.get<Observation>(agent);
      for (uint32_t j = 0; j < OBS_SIZE; j++) {
        obs.bitvec[j] = 0;
      }
    }

    // Reset board
    State &state = ctx.singleton<State>();
    for (uint32_t i = 0; i < MAX_X; i++) {
      for (uint32_t j = 0; j < MAX_Y; j++) {
        state.board[i][j] = 0;
      }
    }

    state.current_player = 0;
    state.move_count = 0;
    state.game_end = false;

    for (uint32_t i = 0; i < num_players; i++) {
      Entity agent = ctx.data().agents[i];
      ctx.get<ActiveAgent>(agent).isActive = (i == (uint32_t)state.current_player);
      generateObsState(ctx, agent);
      generateActionMask(ctx, agent);
    }
  }

  inline void actionSystem(Engine &ctx, State &state)
  {
    Entity agent = ctx.data().agents[state.current_player];
    uint32_t action_id = ctx.get<Action>(agent).choice;
    uint32_t num_players = ctx.data().players;

    // Decode action into row, col (1-indexed)
    int row = (int)(action_id / BOARD_W) + 1;
    int col = (int)(action_id % BOARD_W) + 1;

    // Clamp to valid range
    if (row < 1) row = 1;
    if (row > BOARD_H) row = BOARD_H;
    if (col < 1) col = 1;
    if (col > BOARD_W) col = BOARD_W;

    // Place stone if cell is empty
    if (state.board[row][col] != 'X' && state.board[row][col] != 'O') {
      state.board[row][col] = (state.current_player == 0) ? 'X' : 'O';
      state.move_count++;
    }

    // Switch player
    state.current_player = (state.current_player + 1) % num_players;
  }

  inline void observationSystem(Engine &ctx, State &state)
  {
    uint32_t num_players = ctx.data().players;

    if (!state.game_end) {
      for (uint32_t i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        if (i == (uint32_t)state.current_player) {
          ctx.get<ActiveAgent>(agent).isActive = 1;
          generateObsState(ctx, agent);
          generateActionMask(ctx, agent);
        } else {
          ctx.get<ActiveAgent>(agent).isActive = 0;
        }
      }
    } else {
      for (uint32_t i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        ctx.get<ActiveAgent>(agent).isActive = 1;
        generateObsState(ctx, agent);
        generateActionMask(ctx, agent);
      }
    }
  }

  // Check for five-in-a-row in 4 directions from every cell.
  // After actionSystem, current_player has already been switched,
  // so the player who just moved is the opponent of current_player.
  inline void checkDone(Engine &ctx, WorldReset &reset)
  {
    if (reset.resetNow) {
      resetWorld(ctx);
    }
    reset.resetNow = false;

    State &state = ctx.singleton<State>();
    uint32_t num_players = ctx.data().players;

    // The player who just placed a stone
    char XO = (state.current_player == 0) ? 'O' : 'X';
    int win = 0;

    // Check all 4 directions: horizontal, vertical, diagonal(\), diagonal(/)
    // For each cell, check if 5 consecutive stones exist starting from it.
    for (int i = 1; i <= BOARD_H; i++) {
      for (int j = 1; j <= BOARD_W; j++) {
        if (state.board[i][j] != XO) continue;

        // Horizontal: (i, j) to (i, j+4)
        if (j + 4 <= BOARD_W) {
          if (state.board[i][j+1] == XO &&
              state.board[i][j+2] == XO &&
              state.board[i][j+3] == XO &&
              state.board[i][j+4] == XO) {
            win = 1;
          }
        }

        // Vertical: (i, j) to (i+4, j)
        if (i + 4 <= BOARD_H) {
          if (state.board[i+1][j] == XO &&
              state.board[i+2][j] == XO &&
              state.board[i+3][j] == XO &&
              state.board[i+4][j] == XO) {
            win = 1;
          }
        }

        // Diagonal \: (i, j) to (i+4, j+4)
        if (i + 4 <= BOARD_H && j + 4 <= BOARD_W) {
          if (state.board[i+1][j+1] == XO &&
              state.board[i+2][j+2] == XO &&
              state.board[i+3][j+3] == XO &&
              state.board[i+4][j+4] == XO) {
            win = 1;
          }
        }

        // Diagonal /: (i, j) to (i+4, j-4)
        if (i + 4 <= BOARD_H && j - 4 >= 1) {
          if (state.board[i+1][j-1] == XO &&
              state.board[i+2][j-2] == XO &&
              state.board[i+3][j-3] == XO &&
              state.board[i+4][j-4] == XO) {
            win = 1;
          }
        }

        if (win) break;
      }
      if (win) break;
    }

    if (win == 1) {
      reset.resetNow = true;
      state.game_end = true;
      for (uint32_t i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        // current_player is the one who DIDN'T just move (already switched)
        if ((uint32_t)state.current_player == i) {
          ctx.get<Reward>(agent).rew = -1; // loser
          ctx.get<ActiveAgent>(agent).isActive = 1;
        } else {
          ctx.get<Reward>(agent).rew = 1;  // winner
          ctx.get<ActiveAgent>(agent).isActive = 1;
        }
      }
    } else {
      // Check for draw (board full)
      bool board_filled = (state.move_count >= BOARD_SIZE);
      if (board_filled) {
        reset.resetNow = true;
        state.game_end = true;
      }
      for (uint32_t i = 0; i < num_players; i++) {
        Entity agent = ctx.data().agents[i];
        ctx.get<Reward>(agent).rew = 0;
      }
    }
  }

  void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &)
  {
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
                                                         State>>({});

    auto update_obs = builder.addToGraph<ParallelForNode<Engine, observationSystem,
                                                         State>>({action_sys});

    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, checkDone, WorldReset>>({update_obs});

    (void)terminate_sys;
  }

  Sim::Sim(Engine &ctx, const Config&, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      players(init.players)
  {
    agents = (Entity *)rawAlloc(players * sizeof(Entity));

    for (uint32_t i = 0; i < players; i++) {
      agents[i] = ctx.makeEntity<Agent>();
      ctx.get<Action>(agents[i]).choice = 0;
      ctx.get<AgentID>(agents[i]).id = i;
      ctx.get<Reward>(agents[i]).rew = 0.f;
      ctx.get<ActiveAgent>(agents[i]).isActive = true;
    }

    resetWorld(ctx);
    ctx.singleton<WorldReset>().resetNow = false;
  }

  MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
