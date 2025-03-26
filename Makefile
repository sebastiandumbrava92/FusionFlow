# Makefile for building the FusionFlow C backend shared library

# Compiler
CC = gcc

# Source and Header Files
SRCS = fusionflow_backend.c
HDRS = fusionflow_backend.h

# Output Target (Shared Library)
TARGET = libfusionflow_backend.so

# Compilation and Linking Flags
# -Wall -Wextra: Enable extensive warnings (good practice)
# -O2: Optimization level 2
# -std=c11: Use the C11 standard
# -shared: Tell GCC to produce a shared object (.so) suitable for linking into other executables
# -fPIC: Generate Position-Independent Code (necessary for shared libraries on x86-64)
# -g: (Optional but recommended) Include debugging symbols
CFLAGS = -Wall -Wextra -O2 -std=c11 -shared -fPIC -g

# Libraries to Link Against
# -lm: Link the standard math library (needed for math.h functions like pow, exp, etc.)
LIBS = -lm

# Phony targets are targets that don't represent actual files
.PHONY: all clean

# Default target: Build the shared library
# This rule depends on the source and header files.
# If any of them change, the TARGET will be rebuilt.
all: $(TARGET)

# Rule to build the target shared library from sources and headers
$(TARGET): $(SRCS) $(HDRS) Makefile
	@echo "CCLD $(TARGET)"
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LIBS)

# Target to clean up build artifacts
clean:
	@echo "CLEAN"
	@rm -f $(TARGET)
