#pragma once

#include <madrona/sync.hpp>

namespace Gomoku {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
    uint32_t players;
  };

}
