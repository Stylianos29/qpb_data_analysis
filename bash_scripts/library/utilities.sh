#!/bin/bash

# MULTIPLE SOURCING GUARD

# Prevent multiple sourcing of this script by exiting if UTILITIES_SH_INCLUDED
# is already set. Otherwise, set UTILITIES_SH_INCLUDED to mark it as sourced.
[[ -n "${UTILITIES_SH_INCLUDED}" ]] && return
UTILITIES_SH_INCLUDED=1

