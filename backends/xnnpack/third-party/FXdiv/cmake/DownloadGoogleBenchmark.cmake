CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

PROJECT(googlebenchmark-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googlebenchmark
	URL https://github.com/google/benchmark/archive/v1.5.0.zip
	URL_HASH SHA256=2d22dd3758afee43842bb504af1a8385cccb3ee1f164824e4837c1c1b04d92a0
	SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source"
	BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ""
	TEST_COMMAND ""
)
