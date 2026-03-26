set(SIMULATOR_SRCS
  sim.hpp sim.cpp
)

add_library(gomoku_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(gomoku_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
)

add_library(gomoku_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(gomoku_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    gomoku_cpu_impl
    madrona_mw_cpu
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            gomoku_gpu_srcs
        SOURCES_DEFN
            GOMOKU_SRC_LIST
        FLAGS_DEFN
            GOMOKU_COMPILE_FLAGS
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(gomoku_mgr PRIVATE
    madrona_mw_gpu
    gomoku_gpu_srcs
    )
endif ()

target_compile_definitions(gomoku_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_gomoku_python
    bindings.cpp
)

target_link_libraries(madrona_gomoku_python PRIVATE
    madrona_hdrs
    gomoku_mgr
)
