SetTarget(
    TEST
    MODE EXE
    TARGET_NAME JobSystem_Test
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    LIB
        Core
)