CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

SET(EXECUTABLE_NAME patches_from_edges)

SET(SRC
	patches_from_edges.cpp
	${PROJECT_SOURCE_DIR}/config.h
)

ADD_EXECUTABLE(${EXECUTABLE_NAME} ${SRC})

TARGET_LINK_LIBRARIES(${EXECUTABLE_NAME} LIBS_ALLDEPS)