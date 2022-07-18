OUTNAME_RELEASE = sample_onnx_yolox
OUTNAME_DEBUG   = sample_onnx_yolox_debug
EXTRA_DIRECTORIES = common
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= MakefileYolox.config
include $(MAKEFILE)
