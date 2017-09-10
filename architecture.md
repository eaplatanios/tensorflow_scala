# Architecture

## Build Configuration

SBT commands:
- `clean / compile / nativeCompile / test`: Commands that will compile and run everything for 
  the local system.
- `cross:* / cross:nativeCrossCompile`: Commands that cross-compile native code using Docker container.
- `publish / publishLocal`: Never cross-compile.
- `publishCrossCompiled / publishLocalCrossCompiled`: For that purpose use these instead.

